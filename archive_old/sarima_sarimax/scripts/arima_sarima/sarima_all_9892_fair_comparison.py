"""SARIMA sur 1970-1995 pour une comparaison équitable avec SARIMAX."""

import argparse
import csv
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
OUTPUT_DIR = ROOT / "archive_old/sarima_sarimax_results/sarima_all_9892_fair_comparison"
RESULTS_FILE = OUTPUT_DIR / "metrics_and_parameters.csv"
FAILURES_FILE = OUTPUT_DIR / "failures.csv"
ORDER = (1, 0, 2)
SEASONAL_ORDER = (0, 1, 1, 12)
TRAIN_END = "1995-12-01"
TEST_START = "1996-01-01"
FIELDS = [
    "LAMBX", "LAMBY", "model", "order", "seasonal_order", "train_start",
    "train_end", "test_start", "test_end", "train_n", "test_n", "rmse",
    "mae", "bias", "aic", "bic", "ar_L1", "ma_L1", "ma_L2",
    "seasonal_ma_L12", "sigma2", "elapsed_seconds",
]
MONTHLY = None


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_data():
    frame = pd.read_csv(
        INPUT_FILE, usecols=["LAMBX", "LAMBY", "DATE", "T"],
        dtype={"LAMBX": "int32", "LAMBY": "int32", "T": "float32"},
    )
    frame["MONTH"] = pd.to_datetime(frame.DATE.astype("Int64").astype(str), format="%Y%m", errors="coerce")
    return frame.drop(columns="DATE").dropna().set_index(["LAMBX", "LAMBY"]).sort_index()


def completed():
    if not RESULTS_FILE.exists(): return set()
    done = pd.read_csv(RESULTS_FILE, usecols=["LAMBX", "LAMBY"])
    return set(map(tuple, done.astype(int).to_numpy()))


def fit_point(point):
    started = perf_counter()
    x, y = map(int, point)
    try:
        series = MONTHLY.loc[(x, y)].set_index("MONTH")["T"].sort_index()
        series = series[~series.index.duplicated(keep="first")].asfreq("MS").dropna().astype(float)
        train, test = series.loc[:TRAIN_END], series.loc[TEST_START:]
        if len(train) < 120 or len(test) < 12:
            raise ValueError(f"données insuffisantes train={len(train)} test={len(test)}")
        fitted = SARIMAX(
            train, order=ORDER, seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        prediction = fitted.get_forecast(len(test)).predicted_mean
        prediction.index = test.index
        errors = prediction - test
        p = fitted.params.to_dict()
        return "ok", {
            "LAMBX": x, "LAMBY": y, "model": "SARIMA", "order": str(ORDER),
            "seasonal_order": str(SEASONAL_ORDER), "train_start": train.index.min().date(),
            "train_end": train.index.max().date(), "test_start": test.index.min().date(),
            "test_end": test.index.max().date(), "train_n": len(train), "test_n": len(test),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "mae": float(np.mean(np.abs(errors))), "bias": float(np.mean(errors)),
            "aic": float(fitted.aic), "bic": float(fitted.bic),
            "ar_L1": float(p["ar.L1"]), "ma_L1": float(p["ma.L1"]),
            "ma_L2": float(p["ma.L2"]), "seasonal_ma_L12": float(p["ma.S.L12"]),
            "sigma2": float(p["sigma2"]), "elapsed_seconds": round(perf_counter() - started, 3),
        }
    except Exception as exc:
        return "error", {"LAMBX": x, "LAMBY": y, "error": f"{type(exc).__name__}: {exc}"}


def append(path, fields, row):
    fresh = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fresh: writer.writeheader()
        writer.writerow(row)


def main():
    global MONTHLY
    args = arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Chargement : {INPUT_FILE}", flush=True)
    MONTHLY = load_data()
    all_points = [tuple(map(int, key)) for key in MONTHLY.index.unique()]
    done = completed()
    points = [point for point in all_points if point not in done]
    if args.limit is not None: points = points[:args.limit]
    print(f"Total={len(all_points)} déjà_faits={len(done)} à_traiter={len(points)} workers={args.workers}", flush=True)
    success = failure = 0
    started = perf_counter()
    with get_context("fork").Pool(args.workers) as pool:
        for index, (status, row) in enumerate(pool.imap_unordered(fit_point, points, chunksize=1), 1):
            if status == "ok":
                append(RESULTS_FILE, FIELDS, row); success += 1
            else:
                append(FAILURES_FILE, ["LAMBX", "LAMBY", "error"], row); failure += 1
            if index == 1 or index % 50 == 0 or index == len(points):
                rate = index / (perf_counter() - started)
                remaining = (len(points) - index) / rate if rate else 0
                print(f"[{index}/{len(points)}] succès={success} échecs={failure} vitesse={rate:.2f}/s reste≈{remaining/60:.1f} min", flush=True)
    print(f"Terminé : succès={success}, échecs={failure}")
    print(f"Résultats : {RESULTS_FILE}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
