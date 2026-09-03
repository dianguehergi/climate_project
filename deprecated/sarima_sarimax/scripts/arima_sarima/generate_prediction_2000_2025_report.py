"""Crée les chiffres, diagnostics et figures des prévisions SARIMA 2000-2025."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = ROOT / "data/processed/safran_mens_clean.csv"
METRICS_FILE = ROOT / "archive_old/sarima_sarimax_results/sarima_all_9892_fair_comparison/metrics_and_parameters.csv"
PREDICTIONS_FILE = ROOT / "archive_old/sarima_sarimax_results/predictions_2000_2025/temperature_predictions_2000_2025.csv.gz"
OUTPUT_DIR = ROOT / "archive_old/sarima_sarimax_results/predictions_2000_2025/report"


def save_figure(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path.relative_to(ROOT))


def load_inputs():
    historical = pd.read_csv(
        DATA_FILE,
        usecols=["LAMBX", "LAMBY", "DATE", "T"],
        dtype={"LAMBX": "int32", "LAMBY": "int32", "DATE": "int32", "T": "float32"},
    )
    historical["DATE_DT"] = pd.to_datetime(historical.DATE.astype(str), format="%Y%m")
    forecasts = pd.read_csv(
        PREDICTIONS_FILE,
        usecols=["LAMBX", "LAMBY", "DATE", "T_PRED", "CI95_LOW", "CI95_HIGH"],
        dtype={"LAMBX": "int32", "LAMBY": "int32", "DATE": "int32", "T_PRED": "float32", "CI95_LOW": "float32", "CI95_HIGH": "float32"},
    )
    forecasts["DATE_DT"] = pd.to_datetime(forecasts.DATE.astype(str), format="%Y%m")
    metrics = pd.read_csv(METRICS_FILE)
    return historical, forecasts, metrics


def seasonal_naive_metrics(historical):
    ordered = historical.sort_values(["LAMBX", "LAMBY", "DATE"]).copy()
    ordered["NAIVE_PRED"] = ordered.groupby(["LAMBX", "LAMBY"], sort=False)["T"].shift(12)
    test = ordered[(ordered.DATE >= 199601) & (ordered.DATE <= 199912)].dropna(subset=["NAIVE_PRED"])
    test["ERROR"] = test["NAIVE_PRED"] - test["T"]
    naive = test.groupby(["LAMBX", "LAMBY"], sort=False).ERROR.agg(
        rmse=lambda x: float(np.sqrt(np.mean(np.square(x)))),
        mae=lambda x: float(np.mean(np.abs(x))),
        bias="mean",
    ).reset_index()
    return naive.rename(columns={"rmse": "naive_rmse", "mae": "naive_mae", "bias": "naive_bias"})


def diagnostic_numbers(metrics, naive, forecasts):
    paired = metrics.merge(naive, on=["LAMBX", "LAMBY"], validate="one_to_one")
    rmse_gain = (paired.naive_rmse - paired.rmse) / paired.naive_rmse * 100
    mae_gain = (paired.naive_mae - paired.mae) / paired.naive_mae * 100
    ci_width = forecasts.CI95_HIGH - forecasts.CI95_LOW
    by_date = forecasts.groupby("DATE", sort=True).agg(
        spatial_mean=("T_PRED", "mean"),
        spatial_min=("T_PRED", "min"),
        spatial_max=("T_PRED", "max"),
        mean_ci_width=("CI95_HIGH", lambda x: 0.0),
    )
    # L'agrégation séparée évite de confondre intervalle moyen et intervalle de la moyenne.
    by_date["mean_ci_width"] = ci_width.groupby(forecasts.DATE).mean()
    first = forecasts[(forecasts.DATE >= 200001) & (forecasts.DATE <= 200412)].T_PRED.mean()
    last = forecasts[(forecasts.DATE >= 202101) & (forecasts.DATE <= 202512)].T_PRED.mean()
    numbers = {
        "backtest_period": "1996-01 à 1999-12",
        "backtest_points": int(len(paired)),
        "sarima_rmse_mean": float(paired.rmse.mean()),
        "sarima_rmse_median": float(paired.rmse.median()),
        "sarima_rmse_p05": float(paired.rmse.quantile(0.05)),
        "sarima_rmse_p95": float(paired.rmse.quantile(0.95)),
        "sarima_mae_mean": float(paired.mae.mean()),
        "sarima_mae_median": float(paired.mae.median()),
        "sarima_bias_mean": float(paired.bias.mean()),
        "sarima_points_rmse_below_1c_pct": float((paired.rmse < 1).mean() * 100),
        "sarima_points_rmse_below_1_5c_pct": float((paired.rmse < 1.5).mean() * 100),
        "naive_rmse_mean": float(paired.naive_rmse.mean()),
        "naive_mae_mean": float(paired.naive_mae.mean()),
        "sarima_better_rmse_points_pct": float((paired.rmse < paired.naive_rmse).mean() * 100),
        "sarima_better_mae_points_pct": float((paired.mae < paired.naive_mae).mean() * 100),
        "median_rmse_gain_vs_naive_pct": float(rmse_gain.median()),
        "mean_rmse_gain_vs_naive_pct": float(rmse_gain.mean()),
        "median_mae_gain_vs_naive_pct": float(mae_gain.median()),
        "forecast_spatial_mean_2000_2025_c": float(forecasts.T_PRED.mean()),
        "forecast_spatial_min_c": float(forecasts.T_PRED.min()),
        "forecast_spatial_max_c": float(forecasts.T_PRED.max()),
        "mean_pointwise_ci95_width_2000_c": float(by_date.loc[200001:200012, "mean_ci_width"].mean()),
        "mean_pointwise_ci95_width_2025_c": float(by_date.loc[202501:202512, "mean_ci_width"].mean()),
        "forecast_mean_2000_2004_c": float(first),
        "forecast_mean_2021_2025_c": float(last),
        "forecast_change_last5_minus_first5_c": float(last - first),
    }
    return paired, by_date, numbers


def plot_backtest(paired, numbers):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    bins = np.linspace(0, max(paired.rmse.quantile(0.995), paired.naive_rmse.quantile(0.995)), 45)
    axes[0].hist(paired.naive_rmse, bins=bins, alpha=.55, label="Naïf saisonnier", color="#999999")
    axes[0].hist(paired.rmse, bins=bins, alpha=.72, label="SARIMA", color="#1769aa")
    axes[0].axvline(paired.rmse.median(), color="#0d47a1", ls="--", lw=1.5)
    axes[0].set(title="Distribution du RMSE par point (1996–1999)", xlabel="RMSE (°C)", ylabel="Nombre de points")
    axes[0].legend()
    limit = max(paired.naive_rmse.quantile(.995), paired.rmse.quantile(.995))
    axes[1].scatter(paired.naive_rmse, paired.rmse, s=5, alpha=.22, color="#1769aa", rasterized=True)
    axes[1].plot([0, limit], [0, limit], "--", color="black", lw=1, label="Égalité")
    axes[1].set(xlim=(0, limit), ylim=(0, limit), xlabel="RMSE naïf saisonnier (°C)", ylabel="RMSE SARIMA (°C)", title="Comparaison point par point")
    axes[1].text(.04, .95, f"SARIMA meilleur : {numbers['sarima_better_rmse_points_pct']:.1f} % des points", transform=axes[1].transAxes, va="top", bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
    axes[1].legend(loc="lower right")
    fig.suptitle("Validation hors échantillon du modèle", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "01_backtest_sarima_vs_naive.png")


def plot_timeline(historical, forecasts, by_date):
    hist_monthly = historical.groupby("DATE_DT")["T"].mean()
    pred_monthly = by_date.copy()
    pred_monthly.index = pd.to_datetime(pred_monthly.index.astype(str), format="%Y%m")
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.plot(hist_monthly.index, hist_monthly.values, color="#555555", lw=.8, label="Température observée moyenne")
    ax.plot(pred_monthly.index, pred_monthly.spatial_mean, color="#d62728", lw=1, label="Prévision SARIMA moyenne")
    annual_hist = hist_monthly.resample("YS").mean()
    annual_pred = pred_monthly.spatial_mean.resample("YS").mean()
    ax.plot(annual_hist.index, annual_hist, color="#111111", lw=2, label="Moyenne annuelle observée")
    ax.plot(annual_pred.index, annual_pred, color="#8b0000", lw=2.2, label="Moyenne annuelle prévue")
    ax.axvline(pd.Timestamp("2000-01-01"), color="#1769aa", ls="--", lw=1.5)
    ax.text(pd.Timestamp("2000-05-01"), ax.get_ylim()[1]-.5, "Début des prévisions", color="#1769aa")
    ax.set(title="Température moyenne spatiale : observations et projections", ylabel="Température (°C)", xlabel="Année")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=.2)
    fig.tight_layout()
    save_figure(fig, "02_historique_et_previsions_1960_2025.png")


def plot_seasonality(historical, forecasts):
    h = historical[historical.DATE >= 197001].copy()
    h["month"] = h.DATE % 100
    f = forecasts.copy()
    f["month"] = f.DATE % 100
    hs = h.groupby("month").T.agg(["mean", "std"])
    fs = f.groupby("month").T_PRED.agg(["mean", "std"])
    months = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(months, hs["mean"], marker="o", color="#333333", label="Historique 1970–1999")
    ax.plot(months, fs["mean"], marker="o", color="#d62728", label="Prévision 2000–2025")
    ax.fill_between(months, fs["mean"]-fs["std"], fs["mean"]+fs["std"], color="#d62728", alpha=.12, label="±1 écart-type spatial/temporel")
    ax.set(xticks=months, xlabel="Mois", ylabel="Température (°C)", title="Cycle saisonnier moyen")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "03_cycle_saisonnier.png")


def plot_maps(forecasts):
    point = forecasts.groupby(["LAMBX", "LAMBY"], sort=False).T_PRED.mean().rename("mean_2000_2025").to_frame()
    early = forecasts[(forecasts.DATE >= 200001) & (forecasts.DATE <= 200412)].groupby(["LAMBX", "LAMBY"]).T_PRED.mean()
    late = forecasts[(forecasts.DATE >= 202101) & (forecasts.DATE <= 202512)].groupby(["LAMBX", "LAMBY"]).T_PRED.mean()
    point["change"] = late - early
    point = point.reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    s1 = axes[0].scatter(point.LAMBX, point.LAMBY, c=point.mean_2000_2025, s=7, cmap="coolwarm", rasterized=True)
    fig.colorbar(s1, ax=axes[0], label="°C")
    axes[0].set_title("Température moyenne prévue 2000–2025")
    vmax = max(abs(point.change.quantile(.01)), abs(point.change.quantile(.99)), .01)
    s2 = axes[1].scatter(point.LAMBX, point.LAMBY, c=point.change, s=7, cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=True)
    fig.colorbar(s2, ax=axes[1], label="°C")
    axes[1].set_title("2021–2025 moins 2000–2004")
    for ax in axes:
        ax.set(xlabel="LAMBX", ylabel="LAMBY", aspect="equal")
    fig.suptitle("Structure spatiale des projections SARIMA", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "04_cartes_previsions.png")


def plot_representative_point(historical, forecasts, paired):
    target = paired.iloc[(paired.rmse - paired.rmse.median()).abs().argsort().iloc[0]]
    x, y = int(target.LAMBX), int(target.LAMBY)
    h = historical[(historical.LAMBX == x) & (historical.LAMBY == y) & (historical.DATE >= 199001)]
    f = forecasts[(forecasts.LAMBX == x) & (forecasts.LAMBY == y)]
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.plot(h.DATE_DT, h["T"], color="#222222", lw=1, label="Observé")
    ax.plot(f.DATE_DT, f.T_PRED, color="#d62728", lw=1, label="Prévision SARIMA")
    ax.fill_between(f.DATE_DT, f.CI95_LOW, f.CI95_HIGH, color="#d62728", alpha=.16, label="Intervalle prédictif 95 %")
    ax.axvline(pd.Timestamp("2000-01-01"), color="#1769aa", ls="--")
    ax.set(title=f"Point représentatif ({x}, {y}) — RMSE backtest = {target.rmse:.2f} °C", xlabel="Année", ylabel="Température (°C)")
    ax.legend(ncol=3)
    ax.grid(alpha=.2)
    fig.tight_layout()
    save_figure(fig, "05_point_representatif.png")
    return {"LAMBX": x, "LAMBY": y, "backtest_rmse": float(target.rmse), "backtest_mae": float(target.mae)}


def write_interpretation(numbers, representative):
    n = numbers
    text = f"""# Rapport SARIMA — prévisions de température 2000–2025

## Protocole

- Modèle : SARIMA(1,0,2)(0,1,1,12), un modèle par point.
- Apprentissage final : 1960–1999 sur 9 892 points.
- Prévision : 2000–2025, soit 312 mois par point.
- Mesure réelle de performance : backtest 1996–1999 après apprentissage sur 1960–1995.
- Référence : prévision saisonnière naïve, qui reprend la température du même mois un an plus tôt.

## Résultats du backtest

- RMSE SARIMA moyen : **{n['sarima_rmse_mean']:.3f} °C** ; médian : **{n['sarima_rmse_median']:.3f} °C**.
- MAE SARIMA moyen : **{n['sarima_mae_mean']:.3f} °C** ; médian : **{n['sarima_mae_median']:.3f} °C**.
- Biais moyen : **{n['sarima_bias_mean']:+.3f} °C**.
- 90 % centraux des points : RMSE entre **{n['sarima_rmse_p05']:.3f} et {n['sarima_rmse_p95']:.3f} °C**.
- **{n['sarima_points_rmse_below_1c_pct']:.1f} %** des points ont un RMSE inférieur à 1 °C ; **{n['sarima_points_rmse_below_1_5c_pct']:.1f} %** sont sous 1,5 °C.
- Référence naïve : RMSE moyen **{n['naive_rmse_mean']:.3f} °C**, MAE moyen **{n['naive_mae_mean']:.3f} °C**.
- SARIMA bat le naïf sur le RMSE pour **{n['sarima_better_rmse_points_pct']:.1f} %** des points.
- Gain RMSE médian par rapport au naïf : **{n['median_rmse_gain_vs_naive_pct']:.1f} %**.

## Projections 2000–2025

- Température moyenne prévue, tous points et mois : **{n['forecast_spatial_mean_2000_2025_c']:.2f} °C**.
- Moyenne 2000–2004 : **{n['forecast_mean_2000_2004_c']:.2f} °C**.
- Moyenne 2021–2025 : **{n['forecast_mean_2021_2025_c']:.2f} °C**.
- Écart fin moins début de projection : **{n['forecast_change_last5_minus_first5_c']:+.3f} °C**.
- Largeur moyenne des intervalles prédictifs ponctuels : **{n['mean_pointwise_ci95_width_2000_c']:.2f} °C** en 2000, contre **{n['mean_pointwise_ci95_width_2025_c']:.2f} °C** en 2025.

## Interprétation

Le RMSE et le MAE du backtest quantifient une vraie prévision hors échantillon. Le biais indique si le modèle surestime (positif) ou sous-estime (négatif) systématiquement la température. La comparaison au naïf saisonnier est essentielle : un modèle saisonnier n'est utile que s'il apporte un gain face à cette référence simple.

Les valeurs 2000–2025 sont des **projections**, pas une validation : aucune température observée postérieure à 1999 n'est présente dans le projet. L'élargissement des intervalles à 95 % montre que l'incertitude augmente avec l'horizon. De plus, ce SARIMA prolonge la dynamique historique ; sans variables ou scénarios climatiques externes, il ne doit pas être présenté comme un modèle de projection du changement climatique à long terme.

Le point représentatif de la figure 5 est ({representative['LAMBX']}, {representative['LAMBY']}), avec RMSE={representative['backtest_rmse']:.3f} °C et MAE={representative['backtest_mae']:.3f} °C sur le backtest.
"""
    (OUTPUT_DIR / "interpretation.md").write_text(text, encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    historical, forecasts, metrics = load_inputs()
    naive = seasonal_naive_metrics(historical)
    paired, by_date, numbers = diagnostic_numbers(metrics, naive, forecasts)
    plot_backtest(paired, numbers)
    plot_timeline(historical, forecasts, by_date)
    plot_seasonality(historical, forecasts)
    plot_maps(forecasts)
    representative = plot_representative_point(historical, forecasts, paired)
    numbers["representative_point"] = representative
    (OUTPUT_DIR / "numeric_summary.json").write_text(json.dumps(numbers, indent=2, ensure_ascii=False), encoding="utf-8")
    paired.to_csv(OUTPUT_DIR / "backtest_sarima_vs_seasonal_naive.csv", index=False)
    write_interpretation(numbers, representative)
    print(json.dumps(numbers, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
