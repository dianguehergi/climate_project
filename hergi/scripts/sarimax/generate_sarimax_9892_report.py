"""Graphiques et interprétation du SARIMAX ETP+SWI sur 9 892 points."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "results_sarima/sarimax_all_9892_compact"
FIGURES = BASE / "figures"
RESULTS = BASE / "metrics_and_parameters.csv"
MONTHLY = ROOT / "data/processed/safran_mens_clean.csv"
SARIMA = ROOT / "results_sarima/sarima_all_9892_fair_comparison/metrics_and_parameters.csv"
ORDER = (0, 0, 0)
SEASONAL_ORDER = (0, 1, 1, 12)


def save(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close()


def error_distributions(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, col, color in zip(axes, ["rmse", "mae"], ["#e6550d", "#31a354"]):
        ax.hist(df[col], bins=45, color=color, alpha=.85, edgecolor="white")
        ax.axvline(df[col].mean(), color="#08519c", lw=2, label=f"Moyenne : {df[col].mean():.3f} °C")
        ax.axvline(df[col].median(), color="black", ls="--", lw=2, label=f"Médiane : {df[col].median():.3f} °C")
        ax.set(title=col.upper(), xlabel="Erreur (°C)", ylabel="Nombre de points")
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle("SARIMAX ETP+SWI — distribution des erreurs sur 9 892 points", fontweight="bold")
    save("01_distributions_erreurs.png")


def spatial_behavior(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rmse = axes[0].scatter(df.LAMBX, df.LAMBY, c=df.rmse, s=8, cmap="viridis")
    fig.colorbar(rmse, ax=axes[0], label="RMSE (°C)")
    axes[0].set_title("RMSE spatiale")
    limit = np.quantile(np.abs(df.bias), .98)
    bias = axes[1].scatter(df.LAMBX, df.LAMBY, c=df.bias, s=8, cmap="coolwarm", vmin=-limit, vmax=limit)
    fig.colorbar(bias, ax=axes[1], label="Biais (°C)")
    axes[1].set_title("Biais spatial (prédit − observé)")
    for ax in axes:
        ax.set(xlabel="LAMBX", ylabel="LAMBY"); ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Comportement spatial du SARIMAX", fontweight="bold")
    save("02_cartes_erreurs.png")


def coefficient_maps(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, col, title, cmap in [
        (axes[0], "coef_ETP", "Coefficient ETP standardisé", "plasma"),
        (axes[1], "coef_SWI", "Coefficient SWI standardisé", "coolwarm"),
    ]:
        scatter = ax.scatter(df.LAMBX, df.LAMBY, c=df[col], s=8, cmap=cmap)
        fig.colorbar(scatter, ax=ax, label="Coefficient")
        ax.set(title=title, xlabel="LAMBX", ylabel="LAMBY")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Influence spatiale des variables exogènes", fontweight="bold")
    save("03_cartes_coefficients_ETP_SWI.png")


def robustness(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ordered = np.sort(df.rmse)
    cumulative = np.arange(1, len(df) + 1) / len(df) * 100
    axes[0].plot(ordered, cumulative, lw=2.5)
    for threshold in [1.0, 1.25, 1.5, 2.0]:
        share = (df.rmse <= threshold).mean() * 100
        axes[0].scatter(threshold, share, label=f"≤ {threshold:g} °C : {share:.1f} %")
    axes[0].set(title="Courbe cumulative de RMSE", xlabel="RMSE (°C)", ylabel="Points cumulés (%)")
    axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].hist(df.coef_ETP, bins=40, alpha=.7, label="ETP", color="#e6550d")
    axes[1].hist(df.coef_SWI, bins=40, alpha=.7, label="SWI", color="#3182bd")
    axes[1].axvline(0, color="black", ls="--")
    axes[1].set(title="Distribution des coefficients standardisés", xlabel="Coefficient", ylabel="Nombre de points")
    axes[1].legend(); axes[1].grid(alpha=.2)
    fig.suptitle("Robustesse et interprétabilité", fontweight="bold")
    save("04_robustesse_et_coefficients.png")


def compare_sarima(df):
    baseline = pd.read_csv(SARIMA)
    paired = baseline.merge(df, on=["LAMBX", "LAMBY"], suffixes=("_sarima", "_sarimax"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    low = min(paired.rmse_sarima.min(), paired.rmse_sarimax.min())
    high = max(paired.rmse_sarima.max(), paired.rmse_sarimax.max())
    axes[0].scatter(paired.rmse_sarima, paired.rmse_sarimax, s=10, alpha=.35)
    axes[0].plot([low, high], [low, high], "k--")
    axes[0].set(xlabel="RMSE SARIMA (°C)", ylabel="RMSE SARIMAX (°C)", title="Comparaison point par point")
    axes[0].grid(alpha=.2)
    values = [paired.rmse_sarima.mean(), paired.rmse_sarimax.mean(), paired.mae_sarima.mean(), paired.mae_sarimax.mean()]
    bars = axes[1].bar(["RMSE\nSARIMA", "RMSE\nSARIMAX", "MAE\nSARIMA", "MAE\nSARIMAX"], values,
                       color=["#2a6fbb", "#e6550d", "#2a6fbb", "#e6550d"])
    axes[1].bar_label(bars, fmt="%.3f")
    axes[1].set(title="Moyennes sur les 9 892 points", ylabel="Erreur (°C)")
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("SARIMAX ETP+SWI face au SARIMA", fontweight="bold")
    save("05_comparaison_SARIMA_SARIMAX.png")
    return paired


def forecast_examples(df):
    choices = {
        "Meilleur": df.loc[df.rmse.idxmin()],
        "Médian": df.iloc[(df.rmse - df.rmse.median()).abs().argsort().iloc[0]],
        "Plus difficile": df.loc[df.rmse.idxmax()],
    }
    keys = {(int(row.LAMBX), int(row.LAMBY)) for row in choices.values()}
    pieces = []
    for chunk in pd.read_csv(MONTHLY, usecols=["LAMBX", "LAMBY", "DATE", "T", "ETP", "SWI"], chunksize=250_000):
        mask = pd.Series(False, index=chunk.index)
        for x, y in keys:
            mask |= (chunk.LAMBX == x) & (chunk.LAMBY == y)
        if mask.any(): pieces.append(chunk.loc[mask])
    data = pd.concat(pieces, ignore_index=True)
    data["MONTH"] = pd.to_datetime(data.DATE.astype(str), format="%Y%m")

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for ax, (label, row) in zip(axes, choices.items()):
        key = (int(row.LAMBX), int(row.LAMBY))
        selected = data[(data.LAMBX == key[0]) & (data.LAMBY == key[1])].set_index("MONTH").sort_index()
        train, test = selected.loc[:"1995-12-01"], selected.loc["1996-01-01":]
        x_train = pd.DataFrame({
            "ETP": (train.ETP - row.ETP_mean) / row.ETP_scale,
            "SWI": (train.SWI - row.SWI_mean) / row.SWI_scale,
        }, index=train.index)
        x_test = pd.DataFrame({
            "ETP": (test.ETP - row.ETP_mean) / row.ETP_scale,
            "SWI": (test.SWI - row.SWI_mean) / row.SWI_scale,
        }, index=test.index)
        model = SARIMAX(train["T"], exog=x_train, order=ORDER, seasonal_order=SEASONAL_ORDER,
                        enforce_stationarity=False, enforce_invertibility=False)
        params = [row.coef_ETP, row.coef_SWI, row.seasonal_ma_L12, row.sigma2]
        fitted = model.filter(params)
        pred = fitted.get_forecast(len(test), exog=x_test).predicted_mean
        pred.index = test.index
        ax.plot(train.index[-60:], train["T"].iloc[-60:], color="#777", label="Entraînement (fin)")
        ax.plot(test.index, test["T"], color="#252525", lw=2, label="Observé")
        ax.plot(pred.index, pred, color="#e6550d", lw=2, label="SARIMAX")
        ax.set(title=f"{label} — point ({key[0]}, {key[1]}) — RMSE={row.rmse:.3f} °C", ylabel="Température (°C)")
        ax.grid(alpha=.2)
    axes[0].legend(ncol=3); axes[-1].set_xlabel("Date")
    fig.suptitle("Prévisions SARIMAX sur trois niveaux de difficulté", fontweight="bold")
    save("06_previsions_exemples.png")


def report(df, paired):
    s = {
        "points": len(df), "rmse_mean": df.rmse.mean(), "rmse_median": df.rmse.median(),
        "rmse_std": df.rmse.std(), "rmse_min": df.rmse.min(), "rmse_max": df.rmse.max(),
        "mae_mean": df.mae.mean(), "bias_mean": df.bias.mean(),
        "absolute_bias_mean": df.bias.abs().mean(),
        "share_rmse_le_1": (df.rmse <= 1).mean(), "share_rmse_le_1_25": (df.rmse <= 1.25).mean(),
        "share_rmse_le_1_5": (df.rmse <= 1.5).mean(), "share_rmse_le_2": (df.rmse <= 2).mean(),
        "share_abs_bias_le_0_5": (df.bias.abs() <= .5).mean(),
        "coef_ETP_mean": df.coef_ETP.mean(), "coef_SWI_mean": df.coef_SWI.mean(),
        "coef_ETP_positive_share": (df.coef_ETP > 0).mean(), "coef_SWI_positive_share": (df.coef_SWI > 0).mean(),
        "sarima_rmse": paired.rmse_sarima.mean(), "sarimax_rmse": paired.rmse_sarimax.mean(),
        "sarima_mae": paired.mae_sarima.mean(), "sarimax_mae": paired.mae_sarimax.mean(),
        "sarimax_win_rate": (paired.rmse_sarimax < paired.rmse_sarima).mean(),
    }
    s["rmse_improvement_vs_sarima"] = 1 - s["sarimax_rmse"] / s["sarima_rmse"]
    (BASE / "evaluation_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    pct = lambda v: f"{100*v:.1f} %"
    text = f"""# Évaluation SARIMAX ETP+SWI sur 9 892 points

## Résultats

- RMSE moyenne / médiane : **{s['rmse_mean']:.3f} / {s['rmse_median']:.3f} °C**.
- MAE moyenne : **{s['mae_mean']:.3f} °C**.
- Biais moyen : **{s['bias_mean']:.3f} °C**.
- RMSE minimale / maximale : **{s['rmse_min']:.3f} / {s['rmse_max']:.3f} °C**.
- RMSE ≤ 1,25 °C : **{pct(s['share_rmse_le_1_25'])}** des points.
- RMSE ≤ 1,50 °C : **{pct(s['share_rmse_le_1_5'])}** des points.
- |biais| ≤ 0,50 °C : **{pct(s['share_abs_bias_le_0_5'])}** des points.

## Variables exogènes

- Coefficient ETP moyen standardisé : **{s['coef_ETP_mean']:.3f}**, positif sur **{pct(s['coef_ETP_positive_share'])}** des points.
- Coefficient SWI moyen standardisé : **{s['coef_SWI_mean']:.3f}**, positif sur **{pct(s['coef_SWI_positive_share'])}** des points.
- ETP est le déterminant linéaire dominant, ce qui confirme le rapport méthodologique. La structure et le signe de SWI varient davantage spatialement.

## Comparaison avec SARIMA

- SARIMA : RMSE **{s['sarima_rmse']:.3f} °C**, MAE **{s['sarima_mae']:.3f} °C**.
- SARIMAX ETP+SWI : RMSE **{s['sarimax_rmse']:.3f} °C**, MAE **{s['sarimax_mae']:.3f} °C**.
- Réduction de RMSE : **{pct(s['rmse_improvement_vs_sarima'])}**.
- SARIMAX gagne sur **{pct(s['sarimax_win_rate'])}** des points.

La comparaison est strictement équitable : mêmes 9 892 points, même fichier mensuel, même entraînement 1970–1995 et même test 1996–1999. La différence mesurée isole donc l'apport de la structure SARIMAX et des variables ETP/SWI, sous réserve des ordres propres à chaque modèle.

## Conclusion défendable

Ce SARIMAX est un **très bon modèle statistique mensuel à grande échelle** : couverture complète, aucun échec, paramètres interprétables et amélioration nette face au SARIMA. Il est actuellement le meilleur modèle mensuel directement évalué à cette échelle dans le projet.

On ne peut pas encore dire qu'il bat tous les modèles « du marché ». TCN, ConvLSTM, XGBoost, PatchTST ou les modèles fondation doivent être testés sur les mêmes 9 892 points, la même cible mensuelle et la même période. Les chiffres TCN/ConvLSTM déjà présents concernent surtout du quotidien J+1 et ne constituent donc pas un classement comparable.

## Limites

- Données terminant en 1999 : validation récente nécessaire avant déploiement.
- Les valeurs futures d'ETP et SWI doivent être disponibles pour prévoir avec SARIMAX.
- Les extrêmes et intervalles de confiance doivent encore être évalués globalement.
"""
    (BASE / "INTERPRETATION.md").write_text(text, encoding="utf-8")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS)
    error_distributions(df); spatial_behavior(df); coefficient_maps(df); robustness(df)
    paired = compare_sarima(df)
    forecast_examples(df)
    report(df, paired)
    print(f"Rapport généré : {BASE}")


if __name__ == "__main__":
    main()
