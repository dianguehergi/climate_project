import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pyproj import Transformer

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_FILE = PROJECT_ROOT / "data/processed/safran_quot_clean.csv"
OUT_DIR = PROJECT_ROOT / "results_sarima/all_grid_points"

# Mets 500 pour un test avancé, None pour tous les points
MAX_POINTS = 500

TRAIN_END = "1995-12-01"
TEST_START = "1996-01-01"

SARIMA_ORDER = (1, 0, 2)
SARIMA_SEASONAL_ORDER = (0, 1, 1, 12)

os.makedirs(OUT_DIR, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def convert_lambert_to_latlon(df):
    """
    Conversion approximative SAFRAN LAMBX/LAMBY vers lat/lon.
    Hypothèse : coordonnées en hectomètres, projection Lambert II étendu EPSG:27572.
    """
    transformer = Transformer.from_crs("EPSG:27572", "EPSG:4326", always_xy=True)

    x = df["LAMBX"].astype(float) * 100
    y = df["LAMBY"].astype(float) * 100

    lon, lat = transformer.transform(x.values, y.values)

    df = df.copy()
    df["lon"] = lon
    df["lat"] = lat

    return df


def load_monthly_grid():
    print("Chargement des données propres...")

    usecols = ["LAMBX", "LAMBY", "DATE", "T"]

    df = pd.read_csv(INPUT_FILE, usecols=usecols, low_memory=False)

    df["DATE"] = pd.to_datetime(
        df["DATE"].astype(str).str.replace(".0", "", regex=False),
        format="%Y%m%d",
        errors="coerce",
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
            enforce_invertibility=False,
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
            "bic": float(result.bic),
        }

    except Exception as e:
        print(f"Échec point ({lambx}, {lamby}) : {e}")
        return None


def run_all_points(monthly):
    points = monthly[["LAMBX", "LAMBY"]].drop_duplicates()

    if MAX_POINTS is not None:
        points = points.sample(min(MAX_POINTS, len(points)), random_state=42)

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

    if results_df.empty:
        raise ValueError("Aucun résultat exploitable.")

    results_df = convert_lambert_to_latlon(results_df)

    out_csv = os.path.join(OUT_DIR, "sarima_all_points_results_with_latlon.csv")
    results_df.to_csv(out_csv, index=False)

    print("\nRésultats sauvegardés :", out_csv)
    print(results_df[["rmse", "mae", "bias"]].describe())

    return results_df


def plot_rmse_grid(results_df):
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        results_df["LAMBX"],
        results_df["LAMBY"],
        c=results_df["rmse"],
        s=18
    )
    plt.colorbar(sc, label="RMSE (°C)")
    plt.title("RMSE SARIMA par point de grille SAFRAN")
    plt.xlabel("LAMBX")
    plt.ylabel("LAMBY")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "map_rmse_grid_coordinates.png"), dpi=150)
    plt.close()


def plot_rmse_latlon_simple(results_df):
    plt.figure(figsize=(8, 8))
    sc = plt.scatter(
        results_df["lon"],
        results_df["lat"],
        c=results_df["rmse"],
        s=18
    )
    plt.colorbar(sc, label="RMSE (°C)")
    plt.title("RMSE SARIMA - Latitude / Longitude")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "map_rmse_latlon_simple.png"), dpi=150)
    plt.close()


def plot_real_map_geopandas(results_df):
    """
    Carte plus propre avec fond géographique.
    Si geopandas/contextily échoue, le script continue.
    """
    try:
        import geopandas as gpd
        import contextily as ctx

        gdf = gpd.GeoDataFrame(
            results_df,
            geometry=gpd.points_from_xy(results_df["lon"], results_df["lat"]),
            crs="EPSG:4326"
        )

        gdf_web = gdf.to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(9, 9))

        gdf_web.plot(
            ax=ax,
            column="rmse",
            cmap="viridis",
            markersize=25,
            legend=True,
            legend_kwds={"label": "RMSE (°C)"}
        )

        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        ax.set_title("Carte réelle - RMSE SARIMA par point SAFRAN")
        ax.set_axis_off()

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "map_rmse_real_france.png"), dpi=150)
        plt.close()

        print("Carte réelle sauvegardée : map_rmse_real_france.png")

    except Exception as e:
        print("Carte réelle non générée :", e)


def plot_best_worst_points(results_df, monthly):
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
            enforce_invertibility=False,
        )

        result = model.fit(disp=False)
        pred = result.get_forecast(steps=len(test)).predicted_mean
        pred.index = test.index

        plt.figure(figsize=(12, 5))
        train.plot(label="Train")
        test.plot(label="Observé")
        pred.plot(label="Prédit SARIMA")

        plt.title(
            f"{label.upper()} point SARIMA | "
            f"LAMBX={lambx}, LAMBY={lamby} | "
            f"RMSE={point['rmse']:.2f}°C"
        )
        plt.xlabel("Date")
        plt.ylabel("Température (°C)")
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(OUT_DIR, f"forecast_{label}_point.png"), dpi=150)
        plt.close()


def main():
    monthly = load_monthly_grid()

    monthly.to_csv(
        os.path.join(OUT_DIR, "monthly_grid_temperature.csv"),
        index=False
    )

    results_df = run_all_points(monthly)

    plot_rmse_grid(results_df)
    plot_rmse_latlon_simple(results_df)
    plot_real_map_geopandas(results_df)
    plot_best_worst_points(results_df, monthly)

    print("\n=== Terminé ===")
    print("Dossier résultats :", OUT_DIR)
    print("- sarima_all_points_results_with_latlon.csv")
    print("- map_rmse_grid_coordinates.png")
    print("- map_rmse_latlon_simple.png")
    print("- map_rmse_real_france.png si disponible")
    print("- forecast_best_point.png")
    print("- forecast_worst_point.png")


if __name__ == "__main__":
    main()