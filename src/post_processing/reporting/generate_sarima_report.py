"""Génère les graphiques et le rapport d'évaluation du SARIMA sur 9 892 points."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "results/statistical/all_9892_best_model"
FIGURES = BASE / "figures"
METRICS = BASE / "metrics_by_point.csv"
PARAMETERS = BASE / "model_parameters.csv"
MONTHLY = ROOT / "results/statistical/all_grid_points/monthly_grid_temperature.csv"
SARIMAX_500 = ROOT / "results/statistical/sarimax_500_fast/sarimax_500_results.csv"
ORDER = (1, 0, 2)
SEASONAL_ORDER = (0, 1, 1, 12)


def save(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close()


def distributions(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, column, color in zip(axes, ["rmse", "mae"], ["#2a6fbb", "#31a354"]):
        ax.hist(df[column], bins=45, color=color, alpha=.85, edgecolor="white")
        mean = df[column].mean()
        median = df[column].median()
        ax.axvline(mean, color="#d7301f", lw=2, label=f"Moyenne : {mean:.3f} °C")
        ax.axvline(median, color="#252525", lw=2, ls="--", label=f"Médiane : {median:.3f} °C")
        ax.set(title=f"Distribution de {column.upper()}", xlabel="Erreur (°C)", ylabel="Nombre de points")
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle("SARIMA sur 9 892 points SAFRAN — distribution des erreurs", fontweight="bold")
    save("01_distributions_erreurs.png")


def spatial_maps(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    first = axes[0].scatter(df.LAMBX, df.LAMBY, c=df.rmse, s=8, cmap="viridis")
    fig.colorbar(first, ax=axes[0], label="RMSE (°C)")
    axes[0].set_title("Erreur spatiale RMSE")
    limit = np.quantile(np.abs(df.bias), .98)
    second = axes[1].scatter(df.LAMBX, df.LAMBY, c=df.bias, s=8, cmap="coolwarm", vmin=-limit, vmax=limit)
    fig.colorbar(second, ax=axes[1], label="Biais (°C)")
    axes[1].set_title("Biais spatial (prédit − observé)")
    for ax in axes:
        ax.set(xlabel="LAMBX", ylabel="LAMBY")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Comportement spatial du SARIMA", fontweight="bold")
    save("02_cartes_spatiales.png")


def cumulative_and_bias(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ordered = np.sort(df.rmse.to_numpy())
    cumulative = np.arange(1, len(ordered) + 1) / len(ordered) * 100
    axes[0].plot(ordered, cumulative, color="#2a6fbb", lw=2.5)
    for threshold in [1.0, 1.25, 1.5, 2.0]:
        pct = (df.rmse <= threshold).mean() * 100
        axes[0].scatter([threshold], [pct], label=f"≤ {threshold:g} °C : {pct:.1f} %")
    axes[0].set(xlabel="RMSE (°C)", ylabel="Points cumulés (%)", title="Courbe cumulative de RMSE")
    axes[0].legend()
    axes[0].grid(alpha=.25)
    hb = axes[1].hexbin(df.bias, df.rmse, gridsize=45, mincnt=1, cmap="magma")
    fig.colorbar(hb, ax=axes[1], label="Nombre de points")
    axes[1].axvline(0, color="white", ls="--", lw=1.5)
    axes[1].set(xlabel="Biais (°C)", ylabel="RMSE (°C)", title="Erreur selon le biais")
    fig.suptitle("Robustesse globale et biais", fontweight="bold")
    save("03_robustesse_et_biais.png")


def compare_sarimax(df):
    other = pd.read_csv(SARIMAX_500)
    paired = df.merge(other, on=["LAMBX", "LAMBY"], suffixes=("_sarima", "_sarimax"))
    paired["delta_rmse"] = paired.rmse_sarimax - paired.rmse_sarima
    paired["delta_mae"] = paired.mae_sarimax - paired.mae_sarima

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    low = min(paired.rmse_sarima.min(), paired.rmse_sarimax.min())
    high = max(paired.rmse_sarima.max(), paired.rmse_sarimax.max())
    axes[0].scatter(paired.rmse_sarima, paired.rmse_sarimax, s=18, alpha=.55)
    axes[0].plot([low, high], [low, high], "k--", label="Égalité")
    axes[0].set(xlabel="RMSE SARIMA (°C)", ylabel="RMSE SARIMAX (°C)", title="Comparaison point par point")
    axes[0].legend()
    axes[0].grid(alpha=.2)
    means = [paired.rmse_sarima.mean(), paired.rmse_sarimax.mean(), paired.mae_sarima.mean(), paired.mae_sarimax.mean()]
    labels = ["RMSE\nSARIMA", "RMSE\nSARIMAX", "MAE\nSARIMA", "MAE\nSARIMAX"]
    bars = axes[1].bar(labels, means, color=["#2a6fbb", "#f28e2b", "#2a6fbb", "#f28e2b"])
    axes[1].bar_label(bars, fmt="%.3f")
    axes[1].set(ylabel="Erreur moyenne (°C)", title=f"Mêmes {len(paired)} points, même période")
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("SARIMA face à SARIMAX avec variables exogènes", fontweight="bold")
    save("04_comparaison_sarimax_500.png")
    return paired


def forecast_examples(df):
    choices = {
        "Meilleur": df.loc[df.rmse.idxmin()],
        "Médian": df.iloc[(df.rmse - df.rmse.median()).abs().argsort().iloc[0]],
        "Plus difficile": df.loc[df.rmse.idxmax()],
    }
    keys = {(int(row.LAMBX), int(row.LAMBY)) for row in choices.values()}
    chunks = []
    for chunk in pd.read_csv(MONTHLY, parse_dates=["MONTH"], chunksize=250_000):
        mask = pd.Series(False, index=chunk.index)
        for x, y in keys:
            mask |= (chunk.LAMBX == x) & (chunk.LAMBY == y)
        if mask.any():
            chunks.append(chunk.loc[mask])
    monthly = pd.concat(chunks, ignore_index=True)
    params = pd.read_csv(PARAMETERS).set_index(["LAMBX", "LAMBY"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for ax, (label, row) in zip(axes, choices.items()):
        key = (int(row.LAMBX), int(row.LAMBY))
        selected = monthly[(monthly.LAMBX == key[0]) & (monthly.LAMBY == key[1])]
        series = selected.set_index("MONTH")["T"].sort_index().asfreq("MS").dropna()
        train, test = series.loc[:"1995-12-01"], series.loc["1996-01-01":]
        model = SARIMAX(train, order=ORDER, seasonal_order=SEASONAL_ORDER, enforce_stationarity=False, enforce_invertibility=False)
        p = params.loc[key, ["ar.L1", "ma.L1", "ma.L2", "ma.S.L12", "sigma2"]].to_numpy(float)
        fitted = model.filter(p)
        prediction = fitted.get_forecast(len(test)).predicted_mean
        prediction.index = test.index
        ax.plot(train.index[-60:], train.iloc[-60:], color="#777777", label="Entraînement (fin)")
        ax.plot(test.index, test, color="#252525", lw=2, label="Observé")
        ax.plot(prediction.index, prediction, color="#d7301f", lw=2, label="SARIMA")
        ax.set_title(f"{label} — point ({key[0]}, {key[1]}) — RMSE={row.rmse:.3f} °C")
        ax.set_ylabel("Température (°C)")
        ax.grid(alpha=.2)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Prévisions mensuelles : meilleur, médian et point le plus difficile", fontweight="bold")
    save("05_previsions_exemples.png")


def write_report(df, paired):
    quantiles = df.rmse.quantile([.05, .25, .5, .75, .95]).to_dict()
    summary = {
        "points": len(df),
        "rmse_mean": df.rmse.mean(), "rmse_median": df.rmse.median(),
        "rmse_std": df.rmse.std(), "rmse_min": df.rmse.min(), "rmse_max": df.rmse.max(),
        "mae_mean": df.mae.mean(), "bias_mean": df.bias.mean(),
        "absolute_bias_mean": df.bias.abs().mean(),
        "rmse_quantiles": quantiles,
        "share_rmse_le_1": (df.rmse <= 1).mean(),
        "share_rmse_le_1_25": (df.rmse <= 1.25).mean(),
        "share_rmse_le_1_5": (df.rmse <= 1.5).mean(),
        "share_rmse_le_2": (df.rmse <= 2).mean(),
        "share_abs_bias_le_0_5": (df.bias.abs() <= .5).mean(),
        "sarimax_matched_points": len(paired),
        "sarima_rmse_matched": paired.rmse_sarima.mean(),
        "sarimax_rmse_matched": paired.rmse_sarimax.mean(),
        "sarima_mae_matched": paired.mae_sarima.mean(),
        "sarimax_mae_matched": paired.mae_sarimax.mean(),
        "sarima_win_rate_rmse": (paired.rmse_sarima < paired.rmse_sarimax).mean(),
    }
    summary["sarimax_rmse_improvement_vs_sarima"] = 1 - summary["sarimax_rmse_matched"] / summary["sarima_rmse_matched"]
    (BASE / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pct = lambda value: f"{100 * value:.1f} %"
    report = f"""# Évaluation du SARIMA sur 9 892 points

## Résultats principaux

- RMSE moyenne : **{summary['rmse_mean']:.3f} °C** ; médiane : **{summary['rmse_median']:.3f} °C**.
- MAE moyenne : **{summary['mae_mean']:.3f} °C**.
- Biais moyen : **{summary['bias_mean']:.3f} °C** : le modèle sous-estime légèrement la température.
- RMSE minimale / maximale : **{summary['rmse_min']:.3f} / {summary['rmse_max']:.3f} °C**.
- Points avec RMSE ≤ 1,25 °C : **{pct(summary['share_rmse_le_1_25'])}**.
- Points avec RMSE ≤ 1,50 °C : **{pct(summary['share_rmse_le_1_5'])}**.
- Points avec |biais| ≤ 0,50 °C : **{pct(summary['share_abs_bias_le_0_5'])}**.

## Comparaison équitable avec SARIMAX

Sur les mêmes {len(paired)} points et la même séparation 1960–1995 / 1996–1999 :

- SARIMA : RMSE **{summary['sarima_rmse_matched']:.3f} °C**, MAE **{summary['sarima_mae_matched']:.3f} °C**.
- SARIMAX : RMSE **{summary['sarimax_rmse_matched']:.3f} °C**, MAE **{summary['sarimax_mae_matched']:.3f} °C**.
- Gain moyen de RMSE de SARIMAX face à SARIMA : **{pct(summary['sarimax_rmse_improvement_vs_sarima'])}**.
- SARIMAX gagne point par point dans **{pct(1 - summary['sarima_win_rate_rmse'])}** des cas.

## Peut-on dire que le modèle est bon ?

Oui, on peut dire que le modèle est **solide comme référence mensuelle spatiale** : il a été évalué hors échantillon sur les 9 892 points, sans échec, avec une RMSE moyenne de {summary['rmse_mean']:.3f} °C. En revanche, on ne peut pas dire qu'il est le meilleur modèle du projet : sur les 500 points comparables, SARIMAX réduit la RMSE moyenne de {pct(summary['sarimax_rmse_improvement_vs_sarima'])} et gagne sur {pct(1 - summary['sarima_win_rate_rmse'])} des points.

On ne peut pas encore affirmer qu'il est meilleur que les modèles modernes « du marché » (TCN, ConvLSTM, PatchTST, Prophet, XGBoost, modèles fondation) : les résultats locaux TCN/ConvLSTM prédisent le **quotidien à J+1**, souvent sur un seul point, tandis que ce SARIMA prédit **48 mois** sur toute la grille. Une comparaison scientifique exige exactement la même cible, le même horizon, les mêmes points et les mêmes dates de test.

## Limites à annoncer

- Le biais moyen négatif indique une sous-estimation d'environ {abs(summary['bias_mean']):.2f} °C.
- Les paramètres SARIMA sont identiques en structure pour tous les points ; seule leur estimation varie.
- Les extrêmes climatiques et les intervalles de confiance ne sont pas encore évalués globalement.
- Les données s'arrêtent en 1999 : une validation sur une période plus récente est nécessaire pour parler de déploiement actuel.
"""
    (BASE / "INTERPRETATION.md").write_text(report, encoding="utf-8")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(METRICS)
    distributions(df)
    spatial_maps(df)
    cumulative_and_bias(df)
    paired = compare_sarimax(df)
    forecast_examples(df)
    write_report(df, paired)
    print(f"Rapport généré dans {BASE}")


if __name__ == "__main__":
    main()
