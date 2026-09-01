from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from config import FIGURE_DIR, METRICS_DIR, create_output_directories


def save_map(df, column, title, label, filename):
    fig, ax = plt.subplots(figsize=(10, 9))

    scatter = ax.scatter(
        df["LAMBX"],
        df["LAMBY"],
        c=df[column],
        s=8,
        alpha=0.9,
    )

    ax.set_title(title)
    ax.set_xlabel("LAMBX")
    ax.set_ylabel("LAMBY")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(label)

    fig.tight_layout()

    output = FIGURE_DIR / filename
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Figure créée :", output)


def main():
    create_output_directories()

    path = (
        METRICS_DIR
        / "comparison_1x1_vs_masked_5x5_by_point.csv"
    )

    df = pd.read_csv(path)

    save_map(
        df,
        "MAE_delta_5x5_minus_1x1",
        "Différence MAE : ConvLSTM 5×5 masqué - ConvLSTM 1×1",
        "Δ MAE en °C",
        "map_delta_mae_5x5_minus_1x1.png",
    )

    save_map(
        df,
        "RMSE_delta_5x5_minus_1x1",
        "Différence RMSE : ConvLSTM 5×5 masqué - ConvLSTM 1×1",
        "Δ RMSE en °C",
        "map_delta_rmse_5x5_minus_1x1.png",
    )

    bad_5x5 = df[df["not_improved_5x5"]].copy()

    fig, ax = plt.subplots(figsize=(10, 9))

    ax.scatter(
        df["LAMBX"],
        df["LAMBY"],
        s=5,
        alpha=0.18,
        label="Tous les points",
    )

    ax.scatter(
        bad_5x5["LAMBX"],
        bad_5x5["LAMBY"],
        s=35,
        marker="x",
        label="Points non améliorés en 5×5",
    )

    ax.set_title("Localisation des 56 points non améliorés par le 5×5")
    ax.set_xlabel("LAMBX")
    ax.set_ylabel("LAMBY")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    output = FIGURE_DIR / "map_56_bad_points_5x5.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Figure créée :", output)


if __name__ == "__main__":
    main()
