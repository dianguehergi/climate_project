import os
import warnings
import itertools
from pathlib import Path
import pandas as pd
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_FILE = PROJECT_ROOT / "data/processed/safran_quot_clean.csv"
SARIMA_RESULTS = PROJECT_ROOT / "archive_old/sarima_sarimax_results/all_grid_points/sarima_all_points_results_with_latlon.csv"
OUT_DIR = PROJECT_ROOT / "archive_old/sarima_sarimax_results/sarimax_all_points_auto"

MAX_POINTS = 500
TRAIN_END = "1995-12-01"
TEST_START = "1996-01-01"

TARGET = "T"
EXOG_VARS = ["PRELIQ", "PRENEI", "FF", "HU", "SSI"]

P_RANGE = range(0, 3)
D_RANGE = range(0, 2)
Q_RANGE = range(0, 3)

SP_RANGE = range(0, 2)
SD_RANGE = range(0, 2)
SQ_RANGE = range(0, 2)
SEASONAL_PERIOD = 12

os.makedirs(OUT_DIR, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def load_monthly_grid():
    usecols = ["LAMBX", "LAMBY", "DATE", TARGET] + EXOG_VARS

    print("Chargement des données SAFRAN...")
    df = pd.read_csv(INPUT_FILE, usecols=usecols, low_memory=False)

    df["DATE"] = pd.to_datetime(
        df["DATE"].astype(str).str.replace(".0", "", regex=False),
        format="%Y%m%d",
        errors="coerce"
    )

    for col in [TARGET] + EXOG_VARS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["DATE", TARGET])
    df["MONTH"] = df["DATE"].dt.to_period("M").dt.to_timestamp()

    agg_dict = {
        TARGET: "mean",
        "PRELIQ": "sum",
        "PRENEI": "sum",
        "FF": "mean",
        "HU": "mean",
        "SSI": "mean",
    }

    monthly = (
        df.groupby(["MONTH", "LAMBX", "LAMBY"], as_index=False)
          .agg(agg_dict)
          .sort_values(["LAMBX", "LAMBY", "MONTH"])
    )

    print("Période :", monthly["MONTH"].min(), "→", monthly["MONTH"].max())
    print("Nombre de points :", monthly[["LAMBX", "LAMBY"]].drop_duplicates().shape[0])
    print("Variables exogènes :", EXOG_VARS)

    return monthly


def get_points(monthly):
    if os.path.exists(SARIMA_RESULTS):
        print("Utilisation des mêmes points que SARIMA.")
        pts = pd.read_csv(SARIMA_RESULTS)[["LAMBX", "LAMBY"]].drop_duplicates()
        if MAX_POINTS is not None:
            pts = pts.head(MAX_POINTS)
        return pts

    pts = monthly[["LAMBX", "LAMBY"]].drop_duplicates()
    if MAX_POINTS is not None:
        pts = pts.sample(min(MAX_POINTS, len(pts)), random_state=42)
    return pts


def prepare_point_data(monthly, lambx, lamby):
    point = monthly[
        (monthly["LAMBX"] == lambx) &
        (monthly["LAMBY"] == lamby)
    ].copy()

    point = point.set_index("MONTH").sort_index()
    point = point.asfreq("MS")

    needed_cols = [TARGET] + EXOG_VARS
    point = point[needed_cols].interpolate(method="time").dropna()

    if len(point) < 120:
        return None

    y = point[TARGET]
    X = point[EXOG_VARS]

    y_train = y.loc[:TRAIN_END]
    y_test = y.loc[TEST_START:]

    X_train = X.loc[:TRAIN_END]
    X_test = X.loc[TEST_START:]

    if len(y_train) < 120 or len(y_test) < 12:
        return None

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )

    return y_train, y_test, X_train_scaled, X_test_scaled


def fit_model(y_train, X_train, order, seasonal_order):
    model = SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)


def grid_search_point(y_train, y_test, X_train, X_test):
    best = None
    rows = []

    for order in itertools.product(P_RANGE, D_RANGE, Q_RANGE):
        for seasonal in itertools.product(SP_RANGE, SD_RANGE, SQ_RANGE):
            seasonal_order = (
                seasonal[0],
                seasonal[1],
                seasonal[2],
                SEASONAL_PERIOD
            )

            try:
                result = fit_model(y_train, X_train, order, seasonal_order)

                pred = result.get_forecast(
                    steps=len(y_test),
                    exog=X_test
                ).predicted_mean

                pred.index = y_test.index

                score_rmse = rmse(y_test, pred)
                score_mae = mae(y_test, pred)
                score_bias = float((pred - y_test).mean())

                row = {
                    "order": str(order),
                    "seasonal_order": str(seasonal_order),
                    "rmse": score_rmse,
                    "mae": score_mae,
                    "bias": score_bias,
                    "aic": float(result.aic),
                    "bic": float(result.bic)
                }

                rows.append(row)

                if best is None or score_rmse < best["rmse"]:
                    best = {
                        "order": order,
                        "seasonal_order": seasonal_order,
                        "rmse": score_rmse,
                        "mae": score_mae,
                        "bias": score_bias,
                        "aic": float(result.aic),
                        "bic": float(result.bic)
                    }

            except Exception:
                continue

    return best, pd.DataFrame(rows)


def main():
    monthly = load_monthly_grid()
    points = get_points(monthly)

    print("\nNombre de points à traiter :", len(points))

    all_results = []
    all_grid_rows = []

    for i, row in enumerate(points.itertuples(index=False), start=1):
        lambx = row.LAMBX
        lamby = row.LAMBY

        print(f"\n[{i}/{len(points)}] Point ({lambx}, {lamby})")

        prepared = prepare_point_data(monthly, lambx, lamby)

        if prepared is None:
            print("Point ignoré : données insuffisantes.")
            continue

        y_train, y_test, X_train, X_test = prepared

        best, grid_df = grid_search_point(y_train, y_test, X_train, X_test)

        if best is None:
            print("Aucun modèle valide.")
            continue

        print(
            "Best:",
            best["order"],
            best["seasonal_order"],
            "| RMSE:",
            round(best["rmse"], 3),
            "| MAE:",
            round(best["mae"], 3),
            "| AIC:",
            round(best["aic"], 2)
        )

        result_row = {
            "LAMBX": lambx,
            "LAMBY": lamby,
            "best_order": str(best["order"]),
            "best_seasonal_order": str(best["seasonal_order"]),
            "rmse": best["rmse"],
            "mae": best["mae"],
            "bias": best["bias"],
            "aic": best["aic"],
            "bic": best["bic"],
            "train_n": len(y_train),
            "test_n": len(y_test),
            "exog_vars": ",".join(EXOG_VARS)
        }

        all_results.append(result_row)

        grid_df["LAMBX"] = lambx
        grid_df["LAMBY"] = lamby
        all_grid_rows.append(grid_df)

        pd.DataFrame(all_results).to_csv(
            os.path.join(OUT_DIR, "sarimax_best_results_by_point.csv"),
            index=False
        )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        os.path.join(OUT_DIR, "sarimax_best_results_by_point.csv"),
        index=False
    )

    if all_grid_rows:
        grid_all = pd.concat(all_grid_rows, ignore_index=True)
        grid_all.to_csv(
            os.path.join(OUT_DIR, "sarimax_full_grid_search_results.csv"),
            index=False
        )

    print("\n=== Résumé SARIMAX ===")
    print(results_df[["rmse", "mae", "bias"]].describe())

    print("\nFichiers générés :")
    print("- archive_old/sarima_sarimax_results/sarimax_all_points_auto/sarimax_best_results_by_point.csv")
    print("- archive_old/sarima_sarimax_results/sarimax_all_points_auto/sarimax_full_grid_search_results.csv")


if __name__ == "__main__":
    main()
