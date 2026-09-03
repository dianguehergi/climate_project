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
DATA_DAILY = PROJECT_ROOT / "data/processed/first_point_T_daily.csv"
OUT_DIR = PROJECT_ROOT / "results_sarima"


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def parse_dates_safe(date_series: pd.Series) -> pd.Series:
    """
    Convertit une colonne DATE en datetime.
    Accepte :
    - 19600101
    - 19600101.0
    - 1960-01-01
    - 1970-01-01
    """
    s = date_series.astype(str).str.strip()

    # Cas 19600101.0
    s = s.str.replace(".0", "", regex=False)

    # Essai 1 : format SAFRAN YYYYMMDD
    parsed = pd.to_datetime(s, format="%Y%m%d", errors="coerce")

    # Essai 2 : format ISO YYYY-MM-DD pour les dates non reconnues
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(s.loc[mask], errors="coerce")

    return parsed


def load_daily_series(csv_path: str) -> pd.Series:
    """
    Charge la série journalière exportée (DATE,T)
    et retourne une Series indexée par DATE.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    df = pd.read_csv(csv_path)

    if "DATE" not in df.columns:
        raise ValueError(f"La colonne DATE est absente. Colonnes trouvées : {list(df.columns)}")

    if "T" not in df.columns:
        raise ValueError(f"La colonne T est absente. Colonnes trouvées : {list(df.columns)}")

    df["DATE"] = parse_dates_safe(df["DATE"])
    df["T"] = pd.to_numeric(df["T"], errors="coerce")

    df = df.dropna(subset=["DATE", "T"])
    df = df.sort_values("DATE")

    # Si plusieurs lignes pour une même date, on moyenne
    s = df.groupby("DATE")["T"].mean()

    # Fréquence journalière
    s = s.asfreq("D")

    return s


def fit_and_forecast_sarima(
    series: pd.Series,
    train_ratio: float = 0.8,
    order=(1, 0, 1),
    seasonal_order=(0, 0, 0, 0),
    enforce_stationarity=False,
    enforce_invertibility=False,
):
    series = series.dropna()
    n = len(series)

    if n < 10:
        raise ValueError(f"Série trop courte pour faire un test : {n} points.")

    split = int(np.floor(n * train_ratio))
    split = min(max(split, n - 5), n - 1)

    train = series.iloc[:split]
    test = series.iloc[split:]

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
    )

    res = model.fit(disp=False)

    pred = res.get_forecast(steps=len(test)).predicted_mean
    pred.index = test.index

    return res, train, test, pred


def plot_train_test_forecast(train, test, pred, title, out_path):
    plt.figure(figsize=(10, 4))
    train.plot(label="train")
    test.plot(label="test")
    pred.plot(label="forecast")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_series(series, title, out_path):
    plt.figure(figsize=(10, 4))
    series.plot()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    daily = load_daily_series(DATA_DAILY)

    print("\n=== DAILY (point de grille) ===")
    print("Début :", daily.index.min(), "| Fin :", daily.index.max())
    print("Nb jours :", len(daily))
    print("Nb NaN :", int(daily.isna().sum()))
    print(daily.head())

    if daily.dropna().empty:
        raise ValueError("La série journalière est vide après nettoyage.")

    plot_series(
        daily,
        "SAFRAN - Température journalière (1 point de grille)",
        os.path.join(OUT_DIR, "daily_series.png"),
    )

    monthly = daily.resample("MS").mean().dropna()

    print("\n=== MONTHLY (moyenne mensuelle) ===")
    print("Début :", monthly.index.min(), "| Fin :", monthly.index.max())
    print("Nb mois :", len(monthly))
    print(monthly.head())

    if len(monthly) < 10:
        raise ValueError(f"Série mensuelle trop courte : {len(monthly)} mois.")

    plot_series(
        monthly,
        "SAFRAN - Température mensuelle (moyenne)",
        os.path.join(OUT_DIR, "monthly_series.png"),
    )

    train_ratio = 0.8
    results_rows = []

    print("\n--- RUN 1: ARIMA mensuel (sans saisonnalité) ---")
    res1, train1, test1, pred1 = fit_and_forecast_sarima(
        monthly,
        train_ratio=train_ratio,
        order=(1, 0, 1),
        seasonal_order=(0, 0, 0, 0),
    )

    rmse1 = rmse(test1.values, pred1.values)
    mae1 = mae(test1.values, pred1.values)

    print(
        f"Train months: {len(train1)} | Test months: {len(test1)} | "
        f"RMSE: {rmse1:.3f} | MAE: {mae1:.3f}"
    )

    results_rows.append({
        "model": "ARIMA(1,0,1)",
        "order": "(1,0,1)",
        "seasonal_order": "(0,0,0,0)",
        "train_ratio": train_ratio,
        "train_n": len(train1),
        "test_n": len(test1),
        "rmse": rmse1,
        "mae": mae1,
        "aic": float(res1.aic),
        "bic": float(res1.bic),
    })

    pd.DataFrame({"y_true": test1, "y_pred": pred1}).to_csv(
        os.path.join(OUT_DIR, "pred_arima_monthly.csv"),
        index=True,
    )

    plot_train_test_forecast(
        train1,
        test1,
        pred1,
        title="ARIMA mensuel - train/test/forecast",
        out_path=os.path.join(OUT_DIR, "forecast_arima_monthly.png"),
    )

    with open(os.path.join(OUT_DIR, "arima_summary.txt"), "w") as f:
        f.write(res1.summary().as_text())

    print("\n--- RUN 2: SARIMA mensuel (saisonnalité=12) ---")

    if len(monthly) >= 36:
        try:
            res2, train2, test2, pred2 = fit_and_forecast_sarima(
                monthly,
                train_ratio=train_ratio,
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, 12),
            )

            rmse2 = rmse(test2.values, pred2.values)
            mae2 = mae(test2.values, pred2.values)

            print(
                f"Train months: {len(train2)} | Test months: {len(test2)} | "
                f"RMSE: {rmse2:.3f} | MAE: {mae2:.3f}"
            )

            results_rows.append({
                "model": "SARIMA(1,0,1)(1,0,1,12)",
                "order": "(1,0,1)",
                "seasonal_order": "(1,0,1,12)",
                "train_ratio": train_ratio,
                "train_n": len(train2),
                "test_n": len(test2),
                "rmse": rmse2,
                "mae": mae2,
                "aic": float(res2.aic),
                "bic": float(res2.bic),
            })

            pd.DataFrame({"y_true": test2, "y_pred": pred2}).to_csv(
                os.path.join(OUT_DIR, "pred_sarima_monthly.csv"),
                index=True,
            )

            plot_train_test_forecast(
                train2,
                test2,
                pred2,
                title="SARIMA mensuel (s=12) - train/test/forecast",
                out_path=os.path.join(OUT_DIR, "forecast_sarima_monthly.png"),
            )

            with open(os.path.join(OUT_DIR, "sarima_summary.txt"), "w") as f:
                f.write(res2.summary().as_text())

        except Exception as e:
            print("SARIMA saisonnier a échoué. Raison :", str(e))
    else:
        print(f"Skipping SARIMA saisonnier : seulement {len(monthly)} mois disponibles (<36).")

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(os.path.join(OUT_DIR, "results_summary.csv"), index=False)

    print("\n=== Résumé sauvegardé ===")
    print("Dossier :", OUT_DIR)
    print(results_df)

    print("\nFichiers générés :")
    print("- daily_series.png")
    print("- monthly_series.png")
    print("- forecast_arima_monthly.png")
    print("- pred_arima_monthly.csv")
    print("- arima_summary.txt")
    print("- forecast_sarima_monthly.png si SARIMA a fonctionné")
    print("- pred_sarima_monthly.csv si SARIMA a fonctionné")
    print("- sarima_summary.txt si SARIMA a fonctionné")
    print("- results_summary.csv")


if __name__ == "__main__":
    main()