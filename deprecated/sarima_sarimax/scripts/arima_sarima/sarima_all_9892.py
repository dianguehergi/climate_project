"""Entraîne le meilleur modèle ARIMA/SARIMA sur les 9 892 points SAFRAN.

Le choix SARIMA est fondé sur les résultats existants du projet. Le traitement
est parallèle et reprenable : chaque métrique est écrite immédiatement et un
modèle actualisé avec toute la série est sauvegardé pour chaque point.
"""

import argparse
import csv
import json
import os
import warnings
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MONTHLY_FILE = PROJECT_ROOT / "archive_old/sarima_sarimax_results/all_grid_points/monthly_grid_temperature.csv"
COMPARISON_FILE = PROJECT_ROOT / "archive_old/sarima_sarimax_results/results_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "archive_old/sarima_sarimax_results/all_9892_best_model"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_FILE = OUTPUT_DIR / "metrics_by_point.csv"
FAILURES_FILE = OUTPUT_DIR / "failures.csv"
MANIFEST_FILE = OUTPUT_DIR / "run_manifest.json"

TRAIN_END = "1995-12-01"
TEST_START = "1996-01-01"

# Meilleure configuration SARIMA issue de la recherche existante.
ORDER = (1, 0, 2)
SEASONAL_ORDER = (0, 1, 1, 12)

RESULT_FIELDS = [
    "LAMBX", "LAMBY", "model", "order", "seasonal_order", "train_n",
    "test_n", "rmse", "mae", "bias", "aic", "bic", "model_file",
    "elapsed_seconds",
]

MONTHLY = None
SAVE_MODELS = True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre de points pour un essai.")
    parser.add_argument(
        "--save-full-models",
        action="store_true",
        help="Sauvegarder les objets Statsmodels complets (très volumineux).",
    )
    parser.add_argument("--fresh", action="store_true", help="Refuser la reprise si un résultat existe déjà.")
    return parser.parse_args()


def load_monthly_data():
    if not MONTHLY_FILE.is_file():
        raise FileNotFoundError(f"Fichier mensuel absent : {MONTHLY_FILE}")

    print(f"Chargement : {MONTHLY_FILE}", flush=True)
    frame = pd.read_csv(
        MONTHLY_FILE,
        usecols=["MONTH", "LAMBX", "LAMBY", "T"],
        parse_dates=["MONTH"],
        dtype={"LAMBX": "int32", "LAMBY": "int32", "T": "float32"},
    )
    frame = frame.dropna(subset=["MONTH", "LAMBX", "LAMBY", "T"])
    return frame.set_index(["LAMBX", "LAMBY"]).sort_index()


def completed_points():
    if not RESULTS_FILE.is_file():
        return set()
    done = pd.read_csv(RESULTS_FILE, usecols=["LAMBX", "LAMBY"])
    return set(map(tuple, done[["LAMBX", "LAMBY"]].astype(int).to_numpy()))


def model_path(lambx, lamby):
    return MODELS_DIR / f"sarima_lambx_{lambx}_lamby_{lamby}.pkl"


def fit_point(point):
    lambx, lamby = map(int, point)
    started = perf_counter()
    try:
        point_frame = MONTHLY.loc[(lambx, lamby)]
        series = point_frame.set_index("MONTH")["T"].sort_index().astype(float)
        series = series[~series.index.duplicated(keep="first")].asfreq("MS").dropna()
        train = series.loc[:TRAIN_END]
        test = series.loc[TEST_START:]
        if len(train) < 120 or len(test) < 12:
            raise ValueError(f"série insuffisante : train={len(train)}, test={len(test)}")

        fitted = SARIMAX(
            train,
            order=ORDER,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        prediction = fitted.get_forecast(steps=len(test)).predicted_mean
        prediction.index = test.index
        errors = prediction - test

        saved_path = ""
        if SAVE_MODELS:
            # Conserve les paramètres estimés et actualise l'état jusqu'à la fin
            # de la série, sans effectuer une deuxième optimisation coûteuse.
            final_result = fitted.append(test, refit=False)
            destination = model_path(lambx, lamby)
            temporary = destination.with_suffix(".tmp")
            final_result.save(temporary)
            os.replace(temporary, destination)
            saved_path = str(destination.relative_to(PROJECT_ROOT))

        return "ok", {
            "LAMBX": lambx,
            "LAMBY": lamby,
            "model": "SARIMA",
            "order": str(ORDER),
            "seasonal_order": str(SEASONAL_ORDER),
            "train_n": len(train),
            "test_n": len(test),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "mae": float(np.mean(np.abs(errors))),
            "bias": float(np.mean(errors)),
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
            "model_file": saved_path,
            "elapsed_seconds": round(perf_counter() - started, 3),
        }
    except Exception as exc:
        return "error", {
            "LAMBX": lambx,
            "LAMBY": lamby,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(perf_counter() - started, 3),
        }


def append_row(path, fields, row):
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def selection_evidence():
    evidence = {"selected_family": "SARIMA", "order": ORDER, "seasonal_order": SEASONAL_ORDER}
    if COMPARISON_FILE.is_file():
        comparison = pd.read_csv(COMPARISON_FILE)
        evidence["comparison"] = comparison[["model", "rmse", "mae", "aic", "bic"]].to_dict("records")
    return evidence


def main():
    global MONTHLY, SAVE_MODELS
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers doit être supérieur ou égal à 1")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_MODELS = args.save_full_models

    if args.fresh and RESULTS_FILE.exists():
        raise FileExistsError(f"Résultats existants : {RESULTS_FILE}. Retire --fresh pour reprendre.")

    MONTHLY = load_monthly_data()
    points = [tuple(map(int, values)) for values in MONTHLY.index.unique().tolist()]
    done = completed_points()
    points = [point for point in points if point not in done]
    if args.limit is not None:
        points = points[: args.limit]

    manifest = {
        **selection_evidence(),
        "monthly_file": str(MONTHLY_FILE.relative_to(PROJECT_ROOT)),
        "total_points_in_data": int(MONTHLY.index.unique().size),
        "already_completed": len(done),
        "points_scheduled": len(points),
        "workers": args.workers,
        "save_models": SAVE_MODELS,
        "train_end": TRAIN_END,
        "test_start": TEST_START,
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if not points:
        print("Tous les points sont déjà traités.")
        return

    print(f"Points à traiter : {len(points)} | workers : {args.workers}", flush=True)
    failures_fields = ["LAMBX", "LAMBY", "error", "elapsed_seconds"]
    started = perf_counter()
    successes = 0
    failures = 0

    # Linux/fork permet aux processus de partager le DataFrame en lecture sans
    # recopier ses centaines de mégaoctets.
    with get_context("fork").Pool(processes=args.workers) as pool:
        for index, (status, row) in enumerate(pool.imap_unordered(fit_point, points, chunksize=1), start=1):
            if status == "ok":
                append_row(RESULTS_FILE, RESULT_FIELDS, row)
                successes += 1
            else:
                append_row(FAILURES_FILE, failures_fields, row)
                failures += 1

            if index == 1 or index % 25 == 0 or index == len(points):
                elapsed = perf_counter() - started
                rate = index / elapsed if elapsed else 0.0
                remaining = (len(points) - index) / rate if rate else 0.0
                print(
                    f"[{index}/{len(points)}] succès={successes} échecs={failures} "
                    f"vitesse={rate:.2f} point/s reste≈{remaining / 3600:.2f} h",
                    flush=True,
                )

    print(f"Terminé : {successes} succès, {failures} échecs.")
    print(f"Métriques : {RESULTS_FILE}")
    if SAVE_MODELS:
        print(f"Modèles : {MODELS_DIR}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
