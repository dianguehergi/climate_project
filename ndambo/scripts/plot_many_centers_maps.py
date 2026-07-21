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


from config import (  # noqa: E402
    FIGURE_DIR,
    METRICS_DIR,
    create_output_directories,
)


def save_scatter_map(
    dataframe: pd.DataFrame,
    value_column: str,
    title: str,
    colorbar_label: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(10, 9)
    )

    scatter = axis.scatter(
        dataframe["LAMBX"],
        dataframe["LAMBY"],
        c=dataframe[value_column],
        s=8,
        alpha=0.9,
    )

    axis.set_title(title)
    axis.set_xlabel("LAMBX")
    axis.set_ylabel("LAMBY")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)

    colorbar = figure.colorbar(
        scatter,
        ax=axis,
        shrink=0.8,
    )

    colorbar.set_label(colorbar_label)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    create_output_directories()

    metrics_path = (
        METRICS_DIR
        / "convlstm_many_centers_9892_by_point_metrics.csv"
    )

    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {metrics_path}"
        )

    dataframe = pd.read_csv(metrics_path)

    print("=" * 80)
    print("CARTOGRAPHIE DES ERREURS — 9 892 POINTS")
    print("=" * 80)

    print(f"Points : {len(dataframe):,}")

    figures = [
        (
            "convlstm_MAE_C",
            "ConvLSTM global — MAE par point SAFRAN",
            "MAE en °C",
            "map_9892_convlstm_mae.png",
        ),
        (
            "convlstm_RMSE_C",
            "ConvLSTM global — RMSE par point SAFRAN",
            "RMSE en °C",
            "map_9892_convlstm_rmse.png",
        ),
        (
            "convlstm_BIAS_C",
            "ConvLSTM global — biais par point SAFRAN",
            "Biais en °C",
            "map_9892_convlstm_bias.png",
        ),
        (
            "MAE_improvement_percent",
            "Gain MAE du ConvLSTM par rapport à la persistance",
            "Gain MAE en %",
            "map_9892_mae_improvement.png",
        ),
        (
            "RMSE_improvement_percent",
            "Gain RMSE du ConvLSTM par rapport à la persistance",
            "Gain RMSE en %",
            "map_9892_rmse_improvement.png",
        ),
        (
            "convlstm_R2",
            "ConvLSTM global — R² par point SAFRAN",
            "R²",
            "map_9892_convlstm_r2.png",
        ),
    ]

    for column, title, label, filename in figures:
        output_path = FIGURE_DIR / filename

        save_scatter_map(
            dataframe=dataframe,
            value_column=column,
            title=title,
            colorbar_label=label,
            output_path=output_path,
        )

        print(f"Figure créée : {output_path}")

    difficult_points = (
        dataframe
        .sort_values(
            "convlstm_MAE_C",
            ascending=False,
        )
        .head(100)
        .copy()
    )

    figure, axis = plt.subplots(
        figsize=(10, 9)
    )

    axis.scatter(
        dataframe["LAMBX"],
        dataframe["LAMBY"],
        s=5,
        alpha=0.25,
        label="Tous les points",
    )

    axis.scatter(
        difficult_points["LAMBX"],
        difficult_points["LAMBY"],
        s=18,
        alpha=0.95,
        label="100 points les plus difficiles",
    )

    axis.set_title(
        "Localisation des 100 points les plus difficiles"
    )

    axis.set_xlabel("LAMBX")
    axis.set_ylabel("LAMBY")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()

    difficult_path = (
        FIGURE_DIR
        / "map_9892_top100_difficult_points.png"
    )

    figure.savefig(
        difficult_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Figure créée : {difficult_path}")

    print("\nRésumé rapide :")
    print(
        dataframe[
            [
                "convlstm_MAE_C",
                "convlstm_RMSE_C",
                "convlstm_BIAS_C",
                "convlstm_R2",
                "MAE_improvement_percent",
                "RMSE_improvement_percent",
            ]
        ].describe().to_string()
    )


if __name__ == "__main__":
    main()
