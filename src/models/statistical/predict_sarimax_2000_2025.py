"""Prévision opérationnelle de T par SARIMAX sur 2000-2025 aux 9 892 points.

ETP et SWI futurs ne sont jamais lus : ils sont d'abord prévus par deux modèles
SARIMA entraînés sur 1970-1999, puis injectés dans le SARIMAX de température.
Les paramètres appris sont sauvegardés en CSV compact et les trajectoires dans
un CSV gzip. Le calcul peut reprendre après une interruption.
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
OUTPUT_DIR = ROOT / "results/statistical/sarimax_predictions_2000_2025"
PARAMETERS_FILE = OUTPUT_DIR / "sarimax_parameters_1970_1999.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "temperature_predictions_2000_2025.csv.gz"
EXOG_FILE = OUTPUT_DIR / "etp_swi_predictions_2000_2025.csv.gz"
FAILURES_FILE = OUTPUT_DIR / "failures.csv"
SUMMARY_FILE = OUTPUT_DIR / "run_summary.json"

TARGET_ORDER = (0, 0, 0)
TARGET_SEASONAL_ORDER = (0, 1, 1, 12)
EXOG_ORDER = (1, 0, 1)
EXOG_SEASONAL_ORDER = (0, 1, 1, 12)
TRAIN_START = pd.Timestamp("1970-01-01")
TRAIN_END = pd.Timestamp("1999-12-01")
FORECAST_INDEX = pd.date_range("2000-01-01", "2025-12-01", freq="MS")

PARAMETER_FIELDS = [
    "LAMBX", "LAMBY", "model", "target_order", "target_seasonal_order",
    "exog_order", "exog_seasonal_order", "train_start", "train_end", "train_n",
    "forecast_start", "forecast_end", "forecast_n", "coef_ETP", "coef_SWI",
    "target_seasonal_ma_L12", "target_sigma2", "ETP_mean", "ETP_scale",
    "SWI_mean", "SWI_scale", "ETP_ar_L1", "ETP_ma_L1", "ETP_seasonal_ma_L12",
    "ETP_sigma2", "SWI_ar_L1", "SWI_ma_L1", "SWI_seasonal_ma_L12", "SWI_sigma2",
    "target_aic", "target_bic", "target_converged", "ETP_converged",
    "SWI_converged", "elapsed_seconds",
]
PREDICTION_FIELDS = ["LAMBX", "LAMBY", "DATE", "T_PRED", "CI95_LOW", "CI95_HIGH"]
EXOG_FIELDS = ["LAMBX", "LAMBY", "DATE", "ETP_PRED", "SWI_PRED"]
MONTHLY = None


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def load_data():
    frame = pd.read_csv(
        INPUT_FILE, usecols=["LAMBX", "LAMBY", "DATE", "T", "ETP", "SWI"],
        dtype={"LAMBX": "int32", "LAMBY": "int32", "T": "float32", "ETP": "float32", "SWI": "float32"},
    )
    frame["MONTH"] = pd.to_datetime(frame.DATE.astype("Int64").astype(str), format="%Y%m", errors="coerce")
    return frame.drop(columns="DATE").dropna().set_index(["LAMBX", "LAMBY"]).sort_index()


def completed_points():
    if not PARAMETERS_FILE.exists() or PARAMETERS_FILE.stat().st_size == 0:
        return set()
    done = pd.read_csv(PARAMETERS_FILE, usecols=["LAMBX", "LAMBY"])
    return set(map(tuple, done.astype(int).to_numpy()))


def fit_exog(series):
    fitted = SARIMAX(
        series, order=EXOG_ORDER, seasonal_order=EXOG_SEASONAL_ORDER,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=150)
    prediction = np.asarray(fitted.get_forecast(len(FORECAST_INDEX)).predicted_mean, dtype=float)
    return fitted, prediction


def fit_and_forecast(point):
    started = perf_counter()
    x, y = map(int, point)
    try:
        data = MONTHLY.loc[(x, y)].set_index("MONTH")[["T", "ETP", "SWI"]].sort_index()
        data = data[~data.index.duplicated(keep="first")].asfreq("MS").loc[TRAIN_START:TRAIN_END]
        data[["ETP", "SWI"]] = data[["ETP", "SWI"]].interpolate(method="time")
        data = data.dropna()
        if len(data) != 360:
            raise ValueError(f"360 mois attendus sur 1970-1999, reçus={len(data)}")

        etp_fit, etp_future = fit_exog(data["ETP"].astype(float))
        swi_fit, swi_future = fit_exog(data["SWI"].astype(float))
        # Respect des domaines physiques observés : ETP >= 0 et SWI dans [0, 1].
        etp_future = np.maximum(etp_future, 0.0)
        swi_future = np.clip(swi_future, 0.0, 1.0)

        means = data[["ETP", "SWI"]].mean()
        scales = data[["ETP", "SWI"]].std(ddof=0).replace(0, 1.0)
        x_train = (data[["ETP", "SWI"]] - means) / scales
        future_raw = pd.DataFrame({"ETP": etp_future, "SWI": swi_future}, index=FORECAST_INDEX)
        x_future = (future_raw - means) / scales
        target_fit = SARIMAX(
            data["T"].astype(float), exog=x_train,
            order=TARGET_ORDER, seasonal_order=TARGET_SEASONAL_ORDER,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False, maxiter=150)
        forecast = target_fit.get_forecast(len(FORECAST_INDEX), exog=x_future)
        predicted = np.asarray(forecast.predicted_mean, dtype=float)
        interval = np.asarray(forecast.conf_int(alpha=0.05), dtype=float)
        tp, ep, sp = target_fit.params.to_dict(), etp_fit.params.to_dict(), swi_fit.params.to_dict()

        parameter_row = {
            "LAMBX": x, "LAMBY": y, "model": "SARIMAX_ETP_SWI_PREVUS",
            "target_order": str(TARGET_ORDER), "target_seasonal_order": str(TARGET_SEASONAL_ORDER),
            "exog_order": str(EXOG_ORDER), "exog_seasonal_order": str(EXOG_SEASONAL_ORDER),
            "train_start": TRAIN_START.date(), "train_end": TRAIN_END.date(), "train_n": len(data),
            "forecast_start": FORECAST_INDEX.min().date(), "forecast_end": FORECAST_INDEX.max().date(),
            "forecast_n": len(FORECAST_INDEX), "coef_ETP": float(tp["ETP"]), "coef_SWI": float(tp["SWI"]),
            "target_seasonal_ma_L12": float(tp["ma.S.L12"]), "target_sigma2": float(tp["sigma2"]),
            "ETP_mean": float(means.ETP), "ETP_scale": float(scales.ETP),
            "SWI_mean": float(means.SWI), "SWI_scale": float(scales.SWI),
            "ETP_ar_L1": float(ep["ar.L1"]), "ETP_ma_L1": float(ep["ma.L1"]),
            "ETP_seasonal_ma_L12": float(ep["ma.S.L12"]), "ETP_sigma2": float(ep["sigma2"]),
            "SWI_ar_L1": float(sp["ar.L1"]), "SWI_ma_L1": float(sp["ma.L1"]),
            "SWI_seasonal_ma_L12": float(sp["ma.S.L12"]), "SWI_sigma2": float(sp["sigma2"]),
            "target_aic": float(target_fit.aic), "target_bic": float(target_fit.bic),
            "target_converged": bool(target_fit.mle_retvals.get("converged", False)),
            "ETP_converged": bool(etp_fit.mle_retvals.get("converged", False)),
            "SWI_converged": bool(swi_fit.mle_retvals.get("converged", False)),
            "elapsed_seconds": round(perf_counter() - started, 3),
        }
        predictions = [(x, y, d.strftime("%Y%m"), float(v), float(lo), float(hi)) for d, v, (lo, hi) in zip(FORECAST_INDEX, predicted, interval)]
        exog = [(x, y, d.strftime("%Y%m"), float(e), float(s)) for d, e, s in zip(FORECAST_INDEX, etp_future, swi_future)]
        return "ok", parameter_row, predictions, exog
    except Exception as exc:
        return "error", {"LAMBX": x, "LAMBY": y, "error": f"{type(exc).__name__}: {exc}"}, None, None


def append_csv(path, fields, row):
    fresh = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fresh:
            writer.writeheader()
        writer.writerow(row)


def append_gzip(path, fields, rows):
    fresh = not path.exists() or path.stat().st_size == 0
    with gzip.open(path, "at", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        if fresh:
            writer.writerow(fields)
        writer.writerows(rows)


def main():
    global MONTHLY
    args = arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [PARAMETERS_FILE, PREDICTIONS_FILE, EXOG_FILE, FAILURES_FILE, SUMMARY_FILE]
    if args.fresh and any(p.exists() for p in outputs):
        raise FileExistsError("Des sorties existent déjà ; retirez --fresh pour reprendre.")
    print(f"Chargement : {INPUT_FILE}", flush=True)
    MONTHLY = load_data()
    all_points = [tuple(map(int, key)) for key in MONTHLY.index.unique()]
    done = completed_points()
    points = [p for p in all_points if p not in done]
    if args.limit is not None:
        points = points[:args.limit]
    print(f"Total={len(all_points)} déjà_faits={len(done)} à_traiter={len(points)} workers={args.workers}", flush=True)
    success = failure = 0
    started = perf_counter()
    if points:
        with get_context("fork").Pool(args.workers) as pool:
            for i, (status, row, predictions, exog) in enumerate(pool.imap_unordered(fit_and_forecast, points, chunksize=1), 1):
                if status == "ok":
                    append_csv(PARAMETERS_FILE, PARAMETER_FIELDS, row)
                    append_gzip(PREDICTIONS_FILE, PREDICTION_FIELDS, predictions)
                    append_gzip(EXOG_FILE, EXOG_FIELDS, exog)
                    success += 1
                else:
                    append_csv(FAILURES_FILE, ["LAMBX", "LAMBY", "error"], row)
                    failure += 1
                if i == 1 or i % 50 == 0 or i == len(points):
                    elapsed = perf_counter() - started
                    rate = i / elapsed if elapsed else 0
                    remaining = (len(points) - i) / rate if rate else 0
                    print(f"[{i}/{len(points)}] succès={success} échecs={failure} vitesse={rate:.2f}/s reste≈{remaining/60:.1f} min", flush=True)
    summary = {
        "model": "SARIMAX avec ETP et SWI prévus par SARIMA",
        "target_order": list(TARGET_ORDER), "target_seasonal_order": list(TARGET_SEASONAL_ORDER),
        "exog_order": list(EXOG_ORDER), "exog_seasonal_order": list(EXOG_SEASONAL_ORDER),
        "train_period": [str(TRAIN_START.date()), str(TRAIN_END.date())],
        "forecast_period": [str(FORECAST_INDEX.min().date()), str(FORECAST_INDEX.max().date())],
        "months_per_point": len(FORECAST_INDEX), "total_grid_points": len(all_points),
        "completed_grid_points": len(completed_points()), "success_this_run": success,
        "failures_this_run": failure, "future_exog_source": "prévisions SARIMA basées uniquement sur 1970-1999",
        "elapsed_seconds_this_run": round(perf_counter() - started, 2),
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
