import os
import warnings
import itertools
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
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def load_monthly_series(path):
    df = pd.read_csv(path)

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["T"] = pd.to_numeric(df["T"], errors="coerce")

    df = df.dropna(subset=["DATE", "T"])
    df = df.sort_values("DATE")

    daily = df.groupby("DATE")["T"].mean().asfreq("D")
    monthly = daily.resample("MS").mean().dropna()

    return daily, monthly


def fit_sarima(train, order, seasonal_order):
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def sarima_grid_search(train):
    p = d = q = range(0, 3)
    P = D = Q = range(0, 2)
    s = 12

    best_result = None
    results = []

    for order in itertools.product(p, d, q):
        for seasonal in itertools.product(P, D, Q):
            seasonal_order = (seasonal[0], seasonal[1], seasonal[2], s)

            try:
                result = fit_sarima(train, order, seasonal_order)

                row = {
                    "order": str(order),
                    "seasonal_order": str(seasonal_order),
                    "aic": result.aic,
                    "bic": result.bic,
                }

                results.append(row)

                if best_result is None or result.aic < best_result["aic"]:
                    best_result = {
                        "order": order,
                        "seasonal_order": seasonal_order,
                        "aic": result.aic,
                        "bic": result.bic,
                        "result": result,
                    }

                print(f"Testé SARIMA{order}{seasonal_order} | AIC={result.aic:.2f}")

            except Exception:
                continue

    results_df = pd.DataFrame(results).sort_values("aic")
    return best_result, results_df


def temporal_validation(monthly, order, seasonal_order):
    rows = []

    for year in range(1990, 2000):
        train = monthly.loc[:f"{year-1}-12-01"]
        test = monthly.loc[f"{year}-01-01":f"{year}-12-01"]

        if len(test) != 12 or len(train) < 120:
            continue

        try:
            result = fit_sarima(train, order, seasonal_order)
            forecast = result.get_forecast(steps=len(test)).predicted_mean
            forecast.index = test.index

            rows.append({
                "test_year": year,
                "train_start": train.index.min(),
                "train_end": train.index.max(),
                "rmse": rmse(test, forecast),
                "mae": mae(test, forecast),
                "bias": float((forecast - test).mean()),
            })

        except Exception as e:
            print(f"Validation échouée pour {year}: {e}")

    return pd.DataFrame(rows)


def plot_series(monthly):
    plt.figure(figsize=(13, 4))
    monthly.plot()
    plt.title("Température mensuelle SAFRAN - Point de grille choisi")
    plt.xlabel("Date")
    plt.ylabel("Température moyenne mensuelle (°C)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_monthly_series.png"), dpi=150)
    plt.close()


def plot_forecast(train, test, pred, conf_int):
    plt.figure(figsize=(13, 5))
    train.plot(label="Train")
    test.plot(label="Observé")
    pred.plot(label="Prédit SARIMA")

    plt.fill_between(
        test.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        alpha=0.2,
        label="Intervalle de confiance",
    )

    plt.title("Prévision SARIMA optimisée - Température mensuelle")
    plt.xlabel("Date")
    plt.ylabel("Température (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_forecast_optimized_sarima.png"), dpi=150)
    plt.close()


def plot_zoom(test, pred, conf_int):
    plt.figure(figsize=(13, 5))
    test.plot(label="Observé")
    pred.plot(label="Prédit SARIMA")

    plt.fill_between(
        test.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        alpha=0.2,
        label="Intervalle de confiance",
    )

    plt.title("Zoom - Observé vs Prédit sur la période test")
    plt.xlabel("Date")
    plt.ylabel("Température (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_zoom_test_forecast.png"), dpi=150)
    plt.close()


def plot_errors(test, pred):
    errors = pred - test

    plt.figure(figsize=(13, 4))
    errors.plot()
    plt.axhline(0, linestyle="--")
    plt.title("Erreurs de prédiction SARIMA optimisé (Prédit - Observé)")
    plt.xlabel("Date")
    plt.ylabel("Erreur (°C)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_prediction_errors.png"), dpi=150)
    plt.close()


def plot_residuals(result):
    residuals = result.resid

    plt.figure(figsize=(13, 4))
    residuals.plot()
    plt.axhline(0, linestyle="--")
    plt.title("Résidus du modèle SARIMA optimisé")
    plt.xlabel("Date")
    plt.ylabel("Résidu")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_model_residuals.png"), dpi=150)
    plt.close()


def plot_validation(validation_df):
    plt.figure(figsize=(10, 4))
    plt.plot(validation_df["test_year"], validation_df["rmse"], marker="o", label="RMSE")
    plt.plot(validation_df["test_year"], validation_df["mae"], marker="o", label="MAE")
    plt.title("Validation temporelle SARIMA par année test")
    plt.xlabel("Année test")
    plt.ylabel("Erreur (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_temporal_validation.png"), dpi=150)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    daily, monthly = load_monthly_series(DATA_PATH)

    print("\n=== Données chargées ===")
    print("Daily :", daily.index.min(), "→", daily.index.max(), "|", len(daily), "jours")
    print("Monthly :", monthly.index.min(), "→", monthly.index.max(), "|", len(monthly), "mois")

    plot_series(monthly)

    train = monthly.loc[: "1995-12-01"]
    test = monthly.loc["1996-01-01":]

    print("\n=== Recherche du meilleur SARIMA sur AIC ===")
    best, grid_results = sarima_grid_search(train)

    grid_results.to_csv(os.path.join(OUT_DIR, "sarima_grid_search_results.csv"), index=False)

    print("\n=== Meilleur modèle trouvé ===")
    print("Order :", best["order"])
    print("Seasonal order :", best["seasonal_order"])
    print("AIC :", best["aic"])
    print("BIC :", best["bic"])

    best_result = best["result"]

    forecast_obj = best_result.get_forecast(steps=len(test))
    pred = forecast_obj.predicted_mean
    pred.index = test.index

    conf_int = forecast_obj.conf_int()
    conf_int.index = test.index

    final_rmse = rmse(test, pred)
    final_mae = mae(test, pred)
    final_bias = float((pred - test).mean())

    print("\n=== Performance finale ===")
    print(f"RMSE : {final_rmse:.3f} °C")
    print(f"MAE  : {final_mae:.3f} °C")
    print(f"Biais : {final_bias:.3f} °C")

    predictions = pd.DataFrame({
        "observed": test,
        "predicted": pred,
        "lower_ci": conf_int.iloc[:, 0],
        "upper_ci": conf_int.iloc[:, 1],
        "error": pred - test,
    })

    predictions.to_csv(os.path.join(OUT_DIR, "sarima_optimized_predictions.csv"))

    metrics = pd.DataFrame([{
        "model": f"SARIMA{best['order']}{best['seasonal_order']}",
        "order": str(best["order"]),
        "seasonal_order": str(best["seasonal_order"]),
        "rmse": final_rmse,
        "mae": final_mae,
        "bias": final_bias,
        "aic": best["aic"],
        "bic": best["bic"],
        "train_start": train.index.min(),
        "train_end": train.index.max(),
        "test_start": test.index.min(),
        "test_end": test.index.max(),
    }])

    metrics.to_csv(os.path.join(OUT_DIR, "sarima_optimized_metrics.csv"), index=False)

    with open(os.path.join(OUT_DIR, "sarima_optimized_summary.txt"), "w") as f:
        f.write(best_result.summary().as_text())

    plot_forecast(train, test, pred, conf_int)
    plot_zoom(test, pred, conf_int)
    plot_errors(test, pred)
    plot_residuals(best_result)

    print("\n=== Validation temporelle 1990–1999 ===")
    validation_df = temporal_validation(monthly, best["order"], best["seasonal_order"])
    validation_df.to_csv(os.path.join(OUT_DIR, "sarima_temporal_validation.csv"), index=False)

    print(validation_df)
    print("\nMoyenne validation :")
    print(validation_df[["rmse", "mae", "bias"]].mean())

    plot_validation(validation_df)

    print("\n=== Fichiers générés dans results_sarima ===")
    print("- 01_monthly_series.png")
    print("- 02_forecast_optimized_sarima.png")
    print("- 03_zoom_test_forecast.png")
    print("- 04_prediction_errors.png")
    print("- 05_model_residuals.png")
    print("- 06_temporal_validation.png")
    print("- sarima_grid_search_results.csv")
    print("- sarima_optimized_predictions.csv")
    print("- sarima_optimized_metrics.csv")
    print("- sarima_optimized_summary.txt")
    print("- sarima_temporal_validation.csv")


if __name__ == "__main__":
    main()