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


def main() -> None:
    create_output_directories()

    path = (
        METRICS_DIR
        / "convlstm_many_centers_9892_by_point_metrics.csv"
    )

    df = pd.read_csv(path)

    not_improved = df[df["MAE_improvement_percent"] <= 0].copy()

    print("=" * 80)
    print("POINTS NON AMÉLIORÉS PAR LE MODÈLE")
    print("=" * 80)

    print("Nombre :", len(not_improved))

    print(
        not_improved[
            [
                "center_index",
                "LAMBX",
                "LAMBY",
                "persistence_MAE_C",
                "convlstm_MAE_C",
                "convlstm_BIAS_C",
                "MAE_improvement_percent",
                "RMSE_improvement_percent",
            ]
        ]
        .sort_values("MAE_improvement_percent")
        .to_string(index=False)
    )

    output_csv = (
        METRICS_DIR
        / "convlstm_many_centers_9892_not_improved_points.csv"
    )

    not_improved.to_csv(output_csv, index=False)

    figure, axis = plt.subplots(figsize=(10, 9))

    axis.scatter(
        df["LAMBX"],
        df["LAMBY"],
        s=5,
        alpha=0.18,
        label="Points améliorés ou neutres",
    )

    axis.scatter(
        not_improved["LAMBX"],
        not_improved["LAMBY"],
        s=45,
        marker="x",
        label="Points non améliorés en MAE",
    )

    for _, row in not_improved.iterrows():
        axis.text(
            row["LAMBX"],
            row["LAMBY"],
            str(int(row["center_index"])),
            fontsize=7,
        )

    axis.set_title(
        "Localisation des points non améliorés par le ConvLSTM"
    )

    axis.set_xlabel("LAMBX")
    axis.set_ylabel("LAMBY")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()

    output_figure = (
        FIGURE_DIR
        / "map_9892_not_improved_points.png"
    )

    figure.savefig(
        output_figure,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("\nFichiers créés :")
    print(f"- CSV   : {output_csv}")
    print(f"- Carte : {output_figure}")


if __name__ == "__main__":
    main()
