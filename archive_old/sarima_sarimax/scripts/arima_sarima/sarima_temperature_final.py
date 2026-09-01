import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data/processed/first_point_T_daily.csv"
OUT_DIR = PROJECT_ROOT / "results_sarima"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def load_daily_temperature(path):
    df = pd.read_csv(path)

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["T"] = pd.to_numeric(df["T"], errors="coerce")

    df = df.dropna(subset=["DATE", "T"])
    df = df.sort_values("DATE")

    serie = df.groupby("DATE")["T"].mean()
    serie = serie.asfreq("D")

    return serie


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Chargement des données journalières
    daily = load_daily_temperature(DATA_PATH)

    print("=== Série journalière ===")
    print("Début :", daily.index.min())
    print("Fin   :", daily.index.max())
    print("Nb jours :", len(daily))
    print("Nb NaN :", daily.isna().sum())

    # 2. Passage en mensuel
    monthly = daily.resample("MS").mean().dropna()

    print("\n=== Série mensuelle ===")
    print("Début :", monthly.index.min())
    print("Fin   :", monthly.index.max())
    print("Nb mois :", len(monthly))

    # 3. Split temporel propre
    train = monthly.loc[: "1995-12-01"]
    test = monthly.loc["1996-01-01":]

    print("\n=== Split ===")
    print("Train :", train.index.min(), "→", train.index.max(), "|", len(train), "mois")
    print("Test  :", test.index.min(), "→", test.index.max(), "|", len(test), "mois")

    # 4. Modèle SARIMA
    model = SARIMAX(
        train,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    result = model.fit(disp=False)

    # 5. Prédiction sur la période test
    forecast = result.get_forecast(steps=len(test))
    pred = forecast.predicted_mean
    pred.index = test.index

    conf_int = forecast.conf_int()
    conf_int.index = test.index

    # 6. Métriques
    score_rmse = rmse(test, pred)
    score_mae = mae(test, pred)
    bias = float((pred - test).mean())

    print("\n=== Résultats SARIMA ===")
    print(f"RMSE : {score_rmse:.3f} °C")
    print(f"MAE  : {score_mae:.3f} °C")
    print(f"Biais moyen : {bias:.3f} °C")
    print(f"AIC : {result.aic:.2f}")
    print(f"BIC : {result.bic:.2f}")

    # 7. Sauvegarde des prédictions
    predictions = pd.DataFrame({
        "observed": test,
        "predicted": pred,
        "lower_ci": conf_int.iloc[:, 0],
        "upper_ci": conf_int.iloc[:, 1]
    })

    predictions.to_csv(os.path.join(OUT_DIR, "sarima_predictions.csv"))

    metrics = pd.DataFrame([{
        "model": "SARIMA(1,0,1)(1,0,1,12)",
        "rmse": score_rmse,
        "mae": score_mae,
        "bias": bias,
        "aic": result.aic,
        "bic": result.bic,
        "train_start": train.index.min(),
        "train_end": train.index.max(),
        "test_start": test.index.min(),
        "test_end": test.index.max()
    }])

    metrics.to_csv(os.path.join(OUT_DIR, "sarima_metrics.csv"), index=False)

    with open(os.path.join(OUT_DIR, "sarima_summary.txt"), "w") as f:
        f.write(result.summary().as_text())

    # ----------------------------
    # VISUALISATIONS
    # ----------------------------

    # Graphique 1 : série mensuelle complète
    plt.figure(figsize=(12, 4))
    monthly.plot()
    plt.title("Température mensuelle SAFRAN - Point de grille choisi")
    plt.xlabel("Date")
    plt.ylabel("Température moyenne mensuelle (°C)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_monthly_series.png"), dpi=150)
    plt.close()

    # Graphique 2 : train/test/prédiction
    plt.figure(figsize=(12, 5))
    train.plot(label="Train")
    test.plot(label="Observé (test)")
    pred.plot(label="Prédiction SARIMA")
    plt.fill_between(
        test.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        alpha=0.2,
        label="Intervalle de confiance"
    )
    plt.title("Prévision SARIMA de la température mensuelle")
    plt.xlabel("Date")
    plt.ylabel("Température (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_sarima_forecast.png"), dpi=150)
    plt.close()

    # Graphique 3 : zoom sur la période test
    plt.figure(figsize=(12, 5))
    test.plot(label="Observé")
    pred.plot(label="Prédit SARIMA")
    plt.fill_between(
        test.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        alpha=0.2,
        label="Intervalle de confiance"
    )
    plt.title("Zoom - Observé vs Prédit sur la période test")
    plt.xlabel("Date")
    plt.ylabel("Température (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_zoom_test_forecast.png"), dpi=150)
    plt.close()

    # Graphique 4 : erreurs de prédiction
    errors = pred - test

    plt.figure(figsize=(12, 4))
    errors.plot()
    plt.axhline(0, linestyle="--")
    plt.title("Erreurs de prédiction SARIMA (Prédit - Observé)")
    plt.xlabel("Date")
    plt.ylabel("Erreur (°C)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_prediction_errors.png"), dpi=150)
    plt.close()

    # Graphique 5 : résidus du modèle
    residuals = result.resid

    plt.figure(figsize=(12, 4))
    residuals.plot()
    plt.axhline(0, linestyle="--")
    plt.title("Résidus du modèle SARIMA sur la période d'entraînement")
    plt.xlabel("Date")
    plt.ylabel("Résidu")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_model_residuals.png"), dpi=150)
    plt.close()

    print("\n=== Fichiers générés ===")
    print("Dossier :", OUT_DIR)
    print("- 01_monthly_series.png")
    print("- 02_sarima_forecast.png")
    print("- 03_zoom_test_forecast.png")
    print("- 04_prediction_errors.png")
    print("- 05_model_residuals.png")
    print("- sarima_predictions.csv")
    print("- sarima_metrics.csv")
    print("- sarima_summary.txt")


if __name__ == "__main__":
    main()