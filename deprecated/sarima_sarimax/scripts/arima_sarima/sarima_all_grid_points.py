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
INPUT_FILE = PROJECT_ROOT / "data/processed/safran_quot_clean.csv"
OUT_DIR = PROJECT_ROOT / "archive_old/sarima_sarimax_results/all_grid_points"

MAX_POINTS = 100   # mets None pour tous les points, mais commence avec 100
TRAIN_END = "1995-12-01"
TEST_START = "1996-01-01"

SARIMA_ORDER = (1, 0, 2)
SARIMA_SEASONAL_ORDER = (0, 1, 1, 12)

os.makedirs(OUT_DIR, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def load_monthly_grid():
    print("Chargement des données propres...")

    usecols = ["LAMBX", "LAMBY", "DATE", "T"]

    df = pd.read_csv(INPUT_FILE, usecols=usecols, low_memory=False)

    df["DATE"] = pd.to_datetime(
        df["DATE"].astype(str).str.replace(".0", "", regex=False),
        format="%Y%m%d",
        errors="coerce"
    )

    df["T"] = pd.to_numeric(df["T"], errors="coerce")
    df = df.dropna(subset=["DATE", "T"])

    df["MONTH"] = df["DATE"].dt.to_period("M").dt.to_timestamp()

    print("Période :", df["DATE"].min(), "→", df["DATE"].max())
    print("Nombre de lignes :", len(df))
    print("Nombre de points :", df[["LAMBX", "LAMBY"]].drop_duplicates().shape[0])

    print("Agrégation mensuelle par point de grille...")

    monthly = (
        df.groupby(["MONTH", "LAMBX", "LAMBY"], as_index=False)["T"]
        .mean()
        .sort_values(["LAMBX", "LAMBY", "MONTH"])
    )

    return monthly


def fit_one_point(point_df, lambx, lamby):
    serie = point_df.set_index("MONTH")["T"].sort_index()
    serie = serie.asfreq("MS").dropna()

    if len(serie) < 120:
        return None

    train = serie.loc[:TRAIN_END]
    test = serie.loc[TEST_START:]

    if len(train) < 120 or len(test) < 12:
        return None

    try:
        model = SARIMAX(
            train,
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        result = model.fit(disp=False)

        pred = result.get_forecast(steps=len(test)).predicted_mean
        pred.index = test.index

        return {
            "LAMBX": lambx,
            "LAMBY": lamby,
            "train_n": len(train),
            "test_n": len(test),
            "rmse": rmse(test, pred),
            "mae": mae(test, pred),
            "bias": float((pred - test).mean()),
            "aic": float(result.aic),
            "bic": float(result.bic)
        }

    except Exception as e:
        print(f"Échec point ({lambx}, {lamby}) : {e}")
        return None


def run_all_points(monthly):
    points = monthly[["LAMBX", "LAMBY"]].drop_duplicates()

    if MAX_POINTS is not None:
        points = points.sample(MAX_POINTS, random_state=42)

    results = []

    total = len(points)
    print(f"\nNombre de points à modéliser : {total}")

    for i, row in enumerate(points.itertuples(index=False), start=1):
        lambx = row.LAMBX
        lamby = row.LAMBY

        print(f"[{i}/{total}] SARIMA point ({lambx}, {lamby})")

        point_df = monthly[
            (monthly["LAMBX"] == lambx) &
            (monthly["LAMBY"] == lamby)
        ]

        res = fit_one_point(point_df, lambx, lamby)

        if res is not None:
            results.append(res)

    results_df = pd.DataFrame(results)

    out_csv = os.path.join(OUT_DIR, "sarima_all_points_results.csv")
    results_df.to_csv(out_csv, index=False)

    print("\nRésultats sauvegardés :", out_csv)
    print(results_df.describe())

    return results_df


def plot_rmse_grid(results_df):
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        results_df["LAMBX"],
        results_df["LAMBY"],
        c=results_df["rmse"],
        s=20
    )
    plt.colorbar(sc, label="RMSE (°C)")
    plt.title("Performance SARIMA par point de grille SAFRAN")
    plt.xlabel("LAMBX")
    plt.ylabel("LAMBY")
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "map_rmse_grid_coordinates.png")
    plt.savefig(out, dpi=150)
    plt.close()

    print("Carte grille sauvegardée :", out)


def plot_mae_grid(results_df):
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        results_df["LAMBX"],
        results_df["LAMBY"],
        c=results_df["mae"],
        s=20
    )
    plt.colorbar(sc, label="MAE (°C)")
    plt.title("MAE SARIMA par point de grille SAFRAN")
    plt.xlabel("LAMBX")
    plt.ylabel("LAMBY")
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "map_mae_grid_coordinates.png")
    plt.savefig(out, dpi=150)
    plt.close()

    print("Carte MAE sauvegardée :", out)


def try_plot_real_map(results_df):
    try:
        from pyproj import Transformer

        # Hypothèse courante SAFRAN : Lambert II étendu EPSG:27572.
        # Les coordonnées LAMBX/LAMBY semblent en hectomètres, donc x100.
        transformer = Transformer.from_crs("EPSG:27572", "EPSG:4326", always_xy=True)

        x = results_df["LAMBX"].astype(float) * 100
        y = results_df["LAMBY"].astype(float) * 100

        lon, lat = transformer.transform(x.values, y.values)

        results_df = results_df.copy()
        results_df["lon"] = lon
        results_df["lat"] = lat

        results_df.to_csv(
            os.path.join(OUT_DIR, "sarima_all_points_results_with_latlon.csv"),
            index=False
        )

        plt.figure(figsize=(8, 8))
        sc = plt.scatter(
            results_df["lon"],
            results_df["lat"],
            c=results_df["rmse"],
            s=20
        )
        plt.colorbar(sc, label="RMSE (°C)")
        plt.title("Performance SARIMA - coordonnées latitude/longitude")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()

        out = os.path.join(OUT_DIR, "map_rmse_latlon.png")
        plt.savefig(out, dpi=150)
        plt.close()

        print("Carte latitude/longitude sauvegardée :", out)

    except Exception as e:
        print("\nCarte lat/lon non générée.")
        print("Raison :", e)
        print("Installe pyproj si besoin : pip install pyproj")


def plot_best_worst_points(results_df, monthly):
    if results_df.empty:
        return

    best = results_df.sort_values("rmse").iloc[0]
    worst = results_df.sort_values("rmse").iloc[-1]

    for label, point in [("best", best), ("worst", worst)]:
        lambx = point["LAMBX"]
        lamby = point["LAMBY"]

        point_df = monthly[
            (monthly["LAMBX"] == lambx) &
            (monthly["LAMBY"] == lamby)
        ]

        serie = point_df.set_index("MONTH")["T"].sort_index().asfreq("MS").dropna()

        train = serie.loc[:TRAIN_END]
        test = serie.loc[TEST_START:]

        model = SARIMAX(
            train,
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        result = model.fit(disp=False)
        forecast = result.get_forecast(steps=len(test))
        pred = forecast.predicted_mean
        pred.index = test.index

        plt.figure(figsize=(12, 5))
        train.plot(label="Train")
        test.plot(label="Observé")
        pred.plot(label="Prédit SARIMA")
        plt.title(
            f"{label.upper()} point SARIMA | LAMBX={lambx}, LAMBY={lamby} | RMSE={point['rmse']:.2f}°C"
        )
        plt.xlabel("Date")
        plt.ylabel("Température (°C)")
        plt.legend()
        plt.tight_layout()

        out = os.path.join(OUT_DIR, f"forecast_{label}_point.png")
        plt.savefig(out, dpi=150)
        plt.close()

        print(f"Graphique {label} point sauvegardé :", out)


def main():
    monthly = load_monthly_grid()

    monthly.to_csv(os.path.join(OUT_DIR, "monthly_grid_temperature.csv"), index=False)

    results_df = run_all_points(monthly)

    if results_df.empty:
        print("Aucun résultat exploitable.")
        return

    plot_rmse_grid(results_df)
    plot_mae_grid(results_df)
    try_plot_real_map(results_df)
    plot_best_worst_points(results_df, monthly)

    print("\n=== Terminé ===")
    print("Dossier résultats :", OUT_DIR)
    print("Fichiers principaux :")
    print("- sarima_all_points_results.csv")
    print("- monthly_grid_temperature.csv")
    print("- map_rmse_grid_coordinates.png")
    print("- map_mae_grid_coordinates.png")
    print("- map_rmse_latlon.png si pyproj fonctionne")
    print("- forecast_best_point.png")
    print("- forecast_worst_point.png")


if __name__ == "__main__":
    main()
