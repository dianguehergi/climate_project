"""Entraîne SARIMA sur 1960-1999 et prévoit 2000-2025 pour chaque point SAFRAN.

Le script sauvegarde uniquement les paramètres appris (format CSV compact) et
les prévisions mensuelles dans un CSV compressé. Il peut reprendre un calcul
interrompu : les points déjà présents dans le fichier de paramètres sont ignorés.
"""

import argparse
import csv
import gzip
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


ROOT = Path(__file__).resolve().parents[3]
INPUT_FILE = ROOT / "data/processed/safran_mens_clean.csv"
OUTPUT_DIR = ROOT / "archive_old/sarima_sarimax_results/predictions_2000_2025"
PARAMETERS_FILE = OUTPUT_DIR / "sarima_parameters_1960_1999.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "temperature_predictions_2000_2025.csv.gz"
FAILURES_FILE = OUTPUT_DIR / "failures.csv"
SUMMARY_FILE = OUTPUT_DIR / "run_summary.json"

ORDER = (1, 0, 2)
SEASONAL_ORDER = (0, 1, 1, 12)
TRAIN_START = pd.Timestamp("1960-01-01")
TRAIN_END = pd.Timestamp("1999-12-01")
FORECAST_INDEX = pd.date_range("2000-01-01", "2025-12-01", freq="MS")

PARAMETER_FIELDS = [
    "LAMBX", "LAMBY", "model", "order", "seasonal_order", "train_start",
    "train_end", "train_n", "forecast_start", "forecast_end", "forecast_n",
    "ar_L1", "ma_L1", "ma_L2", "seasonal_ma_L12", "sigma2", "aic", "bic",
    "converged", "elapsed_seconds",
]
PREDICTION_FIELDS = ["LAMBX", "LAMBY", "DATE", "T_PRED", "CI95_LOW", "CI95_HIGH"]

MONTHLY = None


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None, help="Nombre de points (test rapide).")
    parser.add_argument("--fresh", action="store_true", help="Refuse de démarrer si des sorties existent.")
    return parser.parse_args()


def load_data():
    frame = pd.read_csv(
        INPUT_FILE,
        usecols=["LAMBX", "LAMBY", "DATE", "T"],
        dtype={"LAMBX": "int32", "LAMBY": "int32", "T": "float32"},
    )
    frame["MONTH"] = pd.to_datetime(
        frame["DATE"].astype("Int64").astype(str), format="%Y%m", errors="coerce"
    )
    frame = frame.drop(columns="DATE").dropna()
    return frame.set_index(["LAMBX", "LAMBY"]).sort_index()


def completed_points():
    if not PARAMETERS_FILE.exists() or PARAMETERS_FILE.stat().st_size == 0:
        return set()
    done = pd.read_csv(PARAMETERS_FILE, usecols=["LAMBX", "LAMBY"])
    return set(map(tuple, done.astype(int).to_numpy()))


def fit_and_forecast(point):
    started = perf_counter()
    x, y = map(int, point)
    try:
        series = MONTHLY.loc[(x, y)].set_index("MONTH")["T"].sort_index()
        series = series[~series.index.duplicated(keep="first")].asfreq("MS")
        train = series.loc[TRAIN_START:TRAIN_END].dropna().astype(float)
        if len(train) != 480:
            raise ValueError(f"480 mois attendus sur 1960-1999, reçus={len(train)}")

        fitted = SARIMAX(
            train,
            order=ORDER,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        forecast = fitted.get_forecast(steps=len(FORECAST_INDEX))
        predicted = np.asarray(forecast.predicted_mean, dtype=float)
        interval = np.asarray(forecast.conf_int(alpha=0.05), dtype=float)
        params = fitted.params.to_dict()

        parameter_row = {
            "LAMBX": x, "LAMBY": y, "model": "SARIMA",
            "order": str(ORDER), "seasonal_order": str(SEASONAL_ORDER),
            "train_start": TRAIN_START.date(), "train_end": TRAIN_END.date(),
            "train_n": len(train), "forecast_start": FORECAST_INDEX.min().date(),
            "forecast_end": FORECAST_INDEX.max().date(), "forecast_n": len(FORECAST_INDEX),
            "ar_L1": float(params["ar.L1"]), "ma_L1": float(params["ma.L1"]),
            "ma_L2": float(params["ma.L2"]),
            "seasonal_ma_L12": float(params["ma.S.L12"]),
            "sigma2": float(params["sigma2"]), "aic": float(fitted.aic),
            "bic": float(fitted.bic),
            "converged": bool(fitted.mle_retvals.get("converged", False)),
            "elapsed_seconds": round(perf_counter() - started, 3),
        }
        prediction_rows = [
            (x, y, date.strftime("%Y%m"), float(value), float(low), float(high))
            for date, value, (low, high) in zip(FORECAST_INDEX, predicted, interval)
        ]
        return "ok", parameter_row, prediction_rows
    except Exception as exc:
        return "error", {"LAMBX": x, "LAMBY": y, "error": f"{type(exc).__name__}: {exc}"}, None


def append_csv(path, fields, row):
    fresh = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fresh:
            writer.writeheader()
        writer.writerow(row)


def append_predictions(rows):
    fresh = not PREDICTIONS_FILE.exists() or PREDICTIONS_FILE.stat().st_size == 0
    with gzip.open(PREDICTIONS_FILE, "at", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        if fresh:
            writer.writerow(PREDICTION_FIELDS)
        writer.writerows(rows)


def main():
    global MONTHLY
    args = arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [PARAMETERS_FILE, PREDICTIONS_FILE, FAILURES_FILE, SUMMARY_FILE]
    if args.fresh and any(path.exists() for path in outputs):
        raise FileExistsError("Des sorties existent déjà ; retirez --fresh pour reprendre le calcul.")

    print(f"Chargement : {INPUT_FILE}", flush=True)
    MONTHLY = load_data()
    all_points = [tuple(map(int, key)) for key in MONTHLY.index.unique()]
    done = completed_points()
    points = [point for point in all_points if point not in done]
    if args.limit is not None:
        points = points[:args.limit]
    print(
        f"Total={len(all_points)} déjà_faits={len(done)} à_traiter={len(points)} "
        f"workers={args.workers}", flush=True,
    )

    success = failure = 0
    started = perf_counter()
    if points:
        with get_context("fork").Pool(args.workers) as pool:
            iterator = pool.imap_unordered(fit_and_forecast, points, chunksize=1)
            for index, (status, row, predictions) in enumerate(iterator, 1):
                if status == "ok":
                    append_csv(PARAMETERS_FILE, PARAMETER_FIELDS, row)
                    append_predictions(predictions)
                    success += 1
                else:
                    append_csv(FAILURES_FILE, ["LAMBX", "LAMBY", "error"], row)
                    failure += 1
                if index == 1 or index % 50 == 0 or index == len(points):
                    elapsed = perf_counter() - started
                    rate = index / elapsed if elapsed else 0.0
                    remaining = (len(points) - index) / rate if rate else 0.0
                    print(
                        f"[{index}/{len(points)}] succès={success} échecs={failure} "
                        f"vitesse={rate:.2f}/s reste≈{remaining / 60:.1f} min",
                        flush=True,
                    )

    final_done = completed_points()
    summary = {
        "model": "SARIMA",
        "order": list(ORDER),
        "seasonal_order": list(SEASONAL_ORDER),
        "train_period": [str(TRAIN_START.date()), str(TRAIN_END.date())],
        "forecast_period": [str(FORECAST_INDEX.min().date()), str(FORECAST_INDEX.max().date())],
        "months_per_point": len(FORECAST_INDEX),
        "total_grid_points": len(all_points),
        "completed_grid_points": len(final_done),
        "success_this_run": success,
        "failures_this_run": failure,
        "observations_available_after_1999": False,
        "parameters_file": str(PARAMETERS_FILE.relative_to(ROOT)),
        "predictions_file": str(PREDICTIONS_FILE.relative_to(ROOT)),
        "elapsed_seconds_this_run": round(perf_counter() - started, 2),
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
