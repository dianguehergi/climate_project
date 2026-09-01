"""SARIMAX ETP+SWI sur les 9 892 points avec sauvegarde compacte."""

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
OUTPUT_DIR = ROOT / "archive_old/sarima_sarimax_results/sarimax_all_9892_compact"
RESULTS_FILE = OUTPUT_DIR / "metrics_and_parameters.csv"
FAILURES_FILE = OUTPUT_DIR / "failures.csv"

TARGET = "T"
EXOG = ["ETP", "SWI"]
ORDER = (0, 0, 0)
SEASONAL_ORDER = (0, 1, 1, 12)
TRAIN_END = "1995-12-01"
TEST_START = "1996-01-01"

FIELDS = [
    "LAMBX", "LAMBY", "model", "order", "seasonal_order", "exog_vars",
    "train_start", "train_end", "test_start", "test_end", "train_n", "test_n",
    "rmse", "mae", "bias", "aic", "bic", "coef_ETP", "coef_SWI",
    "seasonal_ma_L12", "sigma2", "ETP_mean", "ETP_scale", "SWI_mean",
    "SWI_scale", "elapsed_seconds",
]

MONTHLY = None


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_data():
    print(f"Chargement : {INPUT_FILE}", flush=True)
    frame = pd.read_csv(
        INPUT_FILE,
        usecols=["LAMBX", "LAMBY", "DATE", TARGET, *EXOG],
        dtype={"LAMBX": "int32", "LAMBY": "int32", TARGET: "float32", "ETP": "float32", "SWI": "float32"},
    )
    frame["MONTH"] = pd.to_datetime(frame.DATE.astype("Int64").astype(str), format="%Y%m", errors="coerce")
    frame = frame.drop(columns="DATE").dropna(subset=["MONTH", TARGET])
    return frame.set_index(["LAMBX", "LAMBY"]).sort_index()


def completed():
    if not RESULTS_FILE.exists():
        return set()
    done = pd.read_csv(RESULTS_FILE, usecols=["LAMBX", "LAMBY"])
    return set(map(tuple, done.astype(int).to_numpy()))


def fit_point(point):
    started = perf_counter()
    x, y = map(int, point)
    try:
        data = MONTHLY.loc[(x, y)].set_index("MONTH")[[TARGET, *EXOG]].sort_index()
        data = data[~data.index.duplicated(keep="first")].asfreq("MS")
        data[EXOG] = data[EXOG].interpolate(method="time")
        data = data.dropna()
        train = data.loc[:TRAIN_END]
        test = data.loc[TEST_START:]
        if len(train) < 120 or len(test) < 12:
            raise ValueError(f"données insuffisantes train={len(train)} test={len(test)}")

        means = train[EXOG].mean()
        scales = train[EXOG].std(ddof=0).replace(0, 1.0)
        x_train = (train[EXOG] - means) / scales
        x_test = (test[EXOG] - means) / scales

        fitted = SARIMAX(
            train[TARGET], exog=x_train, order=ORDER, seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        prediction = fitted.get_forecast(steps=len(test), exog=x_test).predicted_mean
        prediction.index = test.index
        errors = prediction - test[TARGET]
        params = fitted.params.to_dict()

        return "ok", {
            "LAMBX": x, "LAMBY": y, "model": "SARIMAX",
            "order": str(ORDER), "seasonal_order": str(SEASONAL_ORDER),
            "exog_vars": ",".join(EXOG),
            "train_start": train.index.min().date(), "train_end": train.index.max().date(),
            "test_start": test.index.min().date(), "test_end": test.index.max().date(),
            "train_n": len(train), "test_n": len(test),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "mae": float(np.mean(np.abs(errors))), "bias": float(np.mean(errors)),
            "aic": float(fitted.aic), "bic": float(fitted.bic),
            "coef_ETP": float(params["ETP"]), "coef_SWI": float(params["SWI"]),
            "seasonal_ma_L12": float(params["ma.S.L12"]), "sigma2": float(params["sigma2"]),
            "ETP_mean": float(means.ETP), "ETP_scale": float(scales.ETP),
            "SWI_mean": float(means.SWI), "SWI_scale": float(scales.SWI),
            "elapsed_seconds": round(perf_counter() - started, 3),
        }
    except Exception as exc:
        return "error", {"LAMBX": x, "LAMBY": y, "error": f"{type(exc).__name__}: {exc}"}


def append(path, fields, row):
    fresh = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fresh:
            writer.writeheader()
        writer.writerow(row)


def main():
    global MONTHLY
    args = arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MONTHLY = load_data()
    all_points = [tuple(map(int, key)) for key in MONTHLY.index.unique()]
    done = completed()
    points = [point for point in all_points if point not in done]
    if args.limit is not None:
        points = points[:args.limit]
    print(f"Total={len(all_points)} déjà_faits={len(done)} à_traiter={len(points)} workers={args.workers}", flush=True)

    success = failure = 0
    started = perf_counter()
    error_fields = ["LAMBX", "LAMBY", "error"]
    with get_context("fork").Pool(args.workers) as pool:
        for index, (status, row) in enumerate(pool.imap_unordered(fit_point, points, chunksize=1), 1):
            if status == "ok":
                append(RESULTS_FILE, FIELDS, row)
                success += 1
            else:
                append(FAILURES_FILE, error_fields, row)
                failure += 1
            if index == 1 or index % 50 == 0 or index == len(points):
                rate = index / (perf_counter() - started)
                remaining = (len(points) - index) / rate if rate else 0
                print(f"[{index}/{len(points)}] succès={success} échecs={failure} vitesse={rate:.2f}/s reste≈{remaining/60:.1f} min", flush=True)

    print(f"Terminé : succès={success}, échecs={failure}")
    print(f"Résultats compacts : {RESULTS_FILE}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
