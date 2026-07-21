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
OUT_DIR = PROJECT_ROOT / "results_sarima/sarimax"

TARGET = "T"
EXOG_VARS = ["PRELIQ", "PRENEI", "FF", "HU", "SSI"]

SARIMAX_ORDER = (1, 0, 2)
SARIMAX_SEASONAL_ORDER = (0, 1, 1, 12)

TRAIN_END = "1995-12-01"
TEST_START = "1996-01-01"

os.makedirs(OUT_DIR, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def load_data_for_one_point():
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

    px = df["LAMBX"].iloc[0]
    py = df["LAMBY"].iloc[0]

    print("Point choisi :", px, py)

    df = df[(df["LAMBX"] == px) & (df["LAMBY"] == py)].copy()
    df = df.sort_values("DATE")

    print("Période :", df["DATE"].min(), "→", df["DATE"].max())
    print("Nombre de lignes :", len(df))

    return df, px, py


def build_monthly_dataset(df):
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
        df.groupby("MONTH")
        .agg(agg_dict)
        .sort_index()
    )

    monthly = monthly.asfreq("MS")

    monthly = monthly.interpolate(method="time")
    monthly = monthly.dropna()

    print("\n=== Dataset mensuel SARIMAX ===")
    print(monthly.head())
    print("Période :", monthly.index.min(), "→", monthly.index.max())
    print("Nb mois :", len(monthly))
    print("Variables exogènes :", EXOG_VARS)

    return monthly


def train_test_split(monthly):
    y = monthly[TARGET]
    X = monthly[EXOG_VARS]

    y_train = y.loc[:TRAIN_END]
    y_test = y.loc[TEST_START:]

    X_train = X.loc[:TRAIN_END]
    X_test = X.loc[TEST_START:]

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

    print("\n=== Split ===")
    print("Train :", y_train.index.min(), "→", y_train.index.max(), "|", len(y_train), "mois")
    print("Test  :", y_test.index.min(), "→", y_test.index.max(), "|", len(y_test), "mois")

    return y_train, y_test, X_train_scaled, X_test_scaled, scaler


def fit_sarimax(y_train, X_train):
    print("\nEntraînement SARIMAX...")

    model = SARIMAX(
        y_train,
        exog=X_train,
        order=SARIMAX_ORDER,
        seasonal_order=SARIMAX_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    result = model.fit(disp=False)

    return result


def forecast_sarimax(result, y_test, X_test):
    forecast = result.get_forecast(
        steps=len(y_test),
        exog=X_test
    )

    pred = forecast.predicted_mean
    pred.index = y_test.index

    conf_int = forecast.conf_int()
    conf_int.index = y_test.index

    return pred, conf_int


def save_outputs(result, monthly, y_train, y_test, pred, conf_int, px, py):
    score_rmse = rmse(y_test, pred)
    score_mae = mae(y_test, pred)
    bias = float((pred - y_test).mean())

    print("\n=== Résultats SARIMAX ===")
    print(f"RMSE : {score_rmse:.3f} °C")
    print(f"MAE  : {score_mae:.3f} °C")
    print(f"Biais moyen : {bias:.3f} °C")
    print(f"AIC : {result.aic:.2f}")
    print(f"BIC : {result.bic:.2f}")

    predictions = pd.DataFrame({
        "observed": y_test,
        "predicted": pred,
        "lower_ci": conf_int.iloc[:, 0],
        "upper_ci": conf_int.iloc[:, 1],
        "error": pred - y_test,
    })

    predictions.to_csv(os.path.join(OUT_DIR, "sarimax_predictions.csv"))

    metrics = pd.DataFrame([{
        "model": f"SARIMAX{SARIMAX_ORDER}{SARIMAX_SEASONAL_ORDER}",
        "LAMBX": px,
        "LAMBY": py,
        "exog_vars": ",".join(EXOG_VARS),
        "rmse": score_rmse,
        "mae": score_mae,
        "bias": bias,
        "aic": result.aic,
        "bic": result.bic,
        "train_start": y_train.index.min(),
        "train_end": y_train.index.max(),
        "test_start": y_test.index.min(),
        "test_end": y_test.index.max()
    }])

    metrics.to_csv(os.path.join(OUT_DIR, "sarimax_metrics.csv"), index=False)

    with open(os.path.join(OUT_DIR, "sarimax_summary.txt"), "w") as f:
        f.write(result.summary().as_text())

    monthly.to_csv(os.path.join(OUT_DIR, "monthly_sarimax_dataset.csv"))


def make_plots(monthly, y_train, y_test, pred, conf_int, result):
    plt.figure(figsize=(12, 4))
    monthly[TARGET].plot()
    plt.title("Température mensuelle SAFRAN - Point de grille choisi")
    plt.xlabel("Date")
    plt.ylabel("Température moyenne mensuelle (°C)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_monthly_temperature.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    y_train.plot(label="Train")
    y_test.plot(label="Observé")
    pred.plot(label="Prédit SARIMAX")
    plt.fill_between(
        y_test.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        alpha=0.2,
        label="Intervalle de confiance"
    )
    plt.title("Prévision SARIMAX - Température mensuelle")
    plt.xlabel("Date")
    plt.ylabel("Température (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_sarimax_forecast.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    y_test.plot(label="Observé")
    pred.plot(label="Prédit SARIMAX")
    plt.fill_between(
        y_test.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        alpha=0.2,
        label="Intervalle de confiance"
    )
    plt.title("Zoom test - Observé vs Prédit SARIMAX")
    plt.xlabel("Date")
    plt.ylabel("Température (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_zoom_test_sarimax.png"), dpi=150)
    plt.close()

    errors = pred - y_test
    plt.figure(figsize=(12, 4))
    errors.plot()
    plt.axhline(0, linestyle="--")
    plt.title("Erreurs de prédiction SARIMAX")
    plt.xlabel("Date")
    plt.ylabel("Erreur (°C)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_sarimax_errors.png"), dpi=150)
    plt.close()

    residuals = result.resid
    plt.figure(figsize=(12, 4))
    residuals.plot()
    plt.axhline(0, linestyle="--")
    plt.title("Résidus du modèle SARIMAX")
    plt.xlabel("Date")
    plt.ylabel("Résidu")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_sarimax_residuals.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    monthly[EXOG_VARS].plot(figsize=(12, 6))
    plt.title("Variables exogènes mensuelles SAFRAN")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_exogenous_variables.png"), dpi=150)
    plt.close()


def main():
    df, px, py = load_data_for_one_point()
    monthly = build_monthly_dataset(df)

    y_train, y_test, X_train, X_test, scaler = train_test_split(monthly)

    result = fit_sarimax(y_train, X_train)

    pred, conf_int = forecast_sarimax(result, y_test, X_test)

    save_outputs(result, monthly, y_train, y_test, pred, conf_int, px, py)

    make_plots(monthly, y_train, y_test, pred, conf_int, result)

    print("\n=== Fichiers générés ===")
    print("Dossier :", OUT_DIR)
    print("- 01_monthly_temperature.png")
    print("- 02_sarimax_forecast.png")
    print("- 03_zoom_test_sarimax.png")
    print("- 04_sarimax_errors.png")
    print("- 05_sarimax_residuals.png")
    print("- 06_exogenous_variables.png")
    print("- sarimax_predictions.csv")
    print("- sarimax_metrics.csv")
    print("- sarimax_summary.txt")
    print("- monthly_sarimax_dataset.csv")


if __name__ == "__main__":
    main()