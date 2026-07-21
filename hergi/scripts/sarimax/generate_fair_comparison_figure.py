"""Figure du gain SARIMAX face au SARIMA sur le protocole strictement commun."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SARIMA = ROOT / "results_sarima/sarima_all_9892_fair_comparison/metrics_and_parameters.csv"
SARIMAX = ROOT / "results_sarima/sarimax_all_9892_compact/metrics_and_parameters.csv"
OUTPUT = ROOT / "results_sarima/sarimax_all_9892_compact/figures/07_gains_SARIMAX_point_par_point.png"


def main():
    sarima = pd.read_csv(SARIMA)
    sarimax = pd.read_csv(SARIMAX)
    paired = sarima.merge(sarimax, on=["LAMBX", "LAMBY"], suffixes=("_sarima", "_sarimax"))
    paired["gain_rmse"] = paired.rmse_sarima - paired.rmse_sarimax
    paired["gain_percent"] = paired.gain_rmse / paired.rmse_sarima * 100
    win_rate = (paired.gain_rmse > 0).mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    limit = np.quantile(np.abs(paired.gain_rmse), .99)
    scatter = axes[0].scatter(
        paired.LAMBX, paired.LAMBY, c=paired.gain_rmse, s=8,
        cmap="RdYlGn", vmin=-limit, vmax=limit,
    )
    fig.colorbar(scatter, ax=axes[0], label="RMSE SARIMA − RMSE SARIMAX (°C)")
    axes[0].set(title="Gain spatial de SARIMAX", xlabel="LAMBX", ylabel="LAMBY")
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].hist(paired.gain_percent, bins=50, color="#31a354", edgecolor="white")
    axes[1].axvline(0, color="black", ls="--", lw=2, label="Égalité")
    axes[1].axvline(
        paired.gain_percent.mean(), color="#d7301f", lw=2,
        label=f"Gain relatif moyen : {paired.gain_percent.mean():.1f} %",
    )
    axes[1].set(
        title="Distribution du gain relatif",
        xlabel="Réduction de RMSE par SARIMAX (%)", ylabel="Nombre de points",
    )
    axes[1].legend(); axes[1].grid(alpha=.2)
    fig.suptitle(
        f"Comparaison équitable 1970–1995 → 1996–1999 — SARIMAX gagne sur {win_rate:.1f} % des points",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
