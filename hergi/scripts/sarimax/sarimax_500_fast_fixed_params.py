import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_FILE = PROJECT_ROOT / "data/processed/safran_quot_clean.csv"
SARIMA_POINTS_FILE = PROJECT_ROOT / "results_sarima/all_grid_points/sarima_all_points_results_with_latlon.csv"
OUT_DIR = PROJECT_ROOT / "results_sarima/sarimax_500_fast"

MAX_POINTS = 500

TARGET = "T"
EXOG_VARS = ["PRELIQ", "PRENEI", "FF", "HU", "SSI"]

TRAIN_END = "1995-12-01"
TEST_START = "1996-01-01"

# Paramètres choisis à partir des premiers tests
ORDER = (0, 0, 0)
SEASONAL_ORDER = (0, 1, 1, 12)

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

    return monthly


def get_same_points_as_sarima(monthly):
    if os.path.exists(SARIMA_POINTS_FILE):
        print("Utilisation des mêmes points que SARIMA.")
        points = pd.read_csv(SARIMA_POINTS_FILE)[["LAMBX", "LAMBY"]].drop_duplicates()
        return points.head(MAX_POINTS)

    print("Fichier SARIMA introuvable, échantillon aléatoire.")
    points = monthly[["LAMBX", "LAMBY"]].drop_duplicates()
    return points.sample(MAX_POINTS, random_state=42)


def prepare_point(monthly, lambx, lamby):
    point = monthly[
        (monthly["LAMBX"] == lambx) &
        (monthly["LAMBY"] == lamby)
    ].copy()

    point = point.set_index("MONTH").sort_index().asfreq("MS")

    point = point[[TARGET] + EXOG_VARS].interpolate(method="time").dropna()

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


def fit_predict_point(y_train, y_test, X_train, X_test):
    model = SARIMAX(
        y_train,
        exog=X_train,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    result = model.fit(disp=False)

    pred = result.get_forecast(
        steps=len(y_test),
        exog=X_test
    ).predicted_mean

    pred.index = y_test.index

    return result, pred


def main():
    monthly = load_monthly_grid()
    points = get_same_points_as_sarima(monthly)

    print("\nNombre de points à traiter :", len(points))
    print("Paramètres SARIMAX utilisés :")
    print("ORDER =", ORDER)
    print("SEASONAL_ORDER =", SEASONAL_ORDER)
    print("EXOG =", EXOG_VARS)

    results = []
    predictions_all = []

    for i, row in enumerate(points.itertuples(index=False), start=1):
        lambx = row.LAMBX
        lamby = row.LAMBY

        print(f"[{i}/{len(points)}] Point ({lambx}, {lamby})")

        prepared = prepare_point(monthly, lambx, lamby)

        if prepared is None:
            print("  ignoré : données insuffisantes")
            continue

        y_train, y_test, X_train, X_test = prepared

        try:
            result, pred = fit_predict_point(y_train, y_test, X_train, X_test)

            score_rmse = rmse(y_test, pred)
            score_mae = mae(y_test, pred)
            score_bias = float((pred - y_test).mean())

            results.append({
                "LAMBX": lambx,
                "LAMBY": lamby,
                "order": str(ORDER),
                "seasonal_order": str(SEASONAL_ORDER),
                "rmse": score_rmse,
                "mae": score_mae,
                "bias": score_bias,
                "aic": float(result.aic),
                "bic": float(result.bic),
                "train_n": len(y_train),
                "test_n": len(y_test),
                "exog_vars": ",".join(EXOG_VARS)
            })

            tmp_pred = pd.DataFrame({
                "LAMBX": lambx,
                "LAMBY": lamby,
                "DATE": y_test.index,
                "observed": y_test.values,
                "predicted": pred.values,
                "error": pred.values - y_test.values
            })

            predictions_all.append(tmp_pred)

            if i % 25 == 0:
                pd.DataFrame(results).to_csv(
                    os.path.join(OUT_DIR, "sarimax_500_results.csv"),
                    index=False
                )

                pd.concat(predictions_all, ignore_index=True).to_csv(
                    os.path.join(OUT_DIR, "sarimax_500_predictions.csv"),
                    index=False
                )

        except Exception as e:
            print("  échec :", e)
            continue

    results_df = pd.DataFrame(results)
    pred_df = pd.concat(predictions_all, ignore_index=True)

    results_df.to_csv(
        os.path.join(OUT_DIR, "sarimax_500_results.csv"),
        index=False
    )

    pred_df.to_csv(
        os.path.join(OUT_DIR, "sarimax_500_predictions.csv"),
        index=False
    )

    print("\n=== Résumé SARIMAX 500 points ===")
    print(results_df[["rmse", "mae", "bias"]].describe())

    plt.figure(figsize=(8, 5))
    plt.hist(results_df["rmse"], bins=30)
    plt.title("Distribution des RMSE - SARIMAX 500 points")
    plt.xlabel("RMSE (°C)")
    plt.ylabel("Nombre de points")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "hist_rmse_sarimax_500.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(9, 7))
    sc = plt.scatter(
        results_df["LAMBX"],
        results_df["LAMBY"],
        c=results_df["rmse"],
        s=18
    )
    plt.colorbar(sc, label="RMSE (°C)")
    plt.title("Carte grille RMSE - SARIMAX 500 points")
    plt.xlabel("LAMBX")
    plt.ylabel("LAMBY")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "map_rmse_sarimax_500.png"), dpi=150)
    plt.close()

    print("\nFichiers générés :")
    print(os.path.join(OUT_DIR, "sarimax_500_results.csv"))
    print(os.path.join(OUT_DIR, "sarimax_500_predictions.csv"))
    print(os.path.join(OUT_DIR, "hist_rmse_sarimax_500.png"))
    print(os.path.join(OUT_DIR, "map_rmse_sarimax_500.png"))


if __name__ == "__main__":
    main()