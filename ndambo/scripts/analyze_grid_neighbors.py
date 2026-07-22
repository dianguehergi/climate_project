from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from config import (
    METRICS_DIR,
    PERSONAL_PROCESSED_DATA_DIR,
    create_output_directories,
)


GRID_POINTS_PATH = (
    PERSONAL_PROCESSED_DATA_DIR
    / "safran_grid_points.csv"
)

GRID_STEP = 80
PATCH_SIZE = 5


def main() -> None:
    create_output_directories()

    grid = pd.read_csv(GRID_POINTS_PATH)

    grid = (
        grid[["LAMBX", "LAMBY"]]
        .drop_duplicates()
        .sort_values(["LAMBX", "LAMBY"])
        .reset_index(drop=True)
    )

    grid["LAMBX"] = grid["LAMBX"].astype(int)
    grid["LAMBY"] = grid["LAMBY"].astype(int)

    point_set = set(
        zip(
            grid["LAMBX"],
            grid["LAMBY"],
        )
    )

    radius = PATCH_SIZE // 2

    rows = []

    for index, row in grid.iterrows():
        center_x = int(row["LAMBX"])
        center_y = int(row["LAMBY"])

        existing_cells = 0
        missing_cells = 0

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                neighbor_x = center_x + dx * GRID_STEP
                neighbor_y = center_y + dy * GRID_STEP

                if (neighbor_x, neighbor_y) in point_set:
                    existing_cells += 1
                else:
                    missing_cells += 1

        rows.append(
            {
                "center_index": index,
                "LAMBX": center_x,
                "LAMBY": center_y,
                "existing_cells_5x5": existing_cells,
                "missing_cells_5x5": missing_cells,
                "complete_5x5": existing_cells == 25,
            }
        )

    dataframe = pd.DataFrame(rows)

    detail_path = (
        METRICS_DIR
        / "grid_neighbors_5x5_analysis.csv"
    )

    dataframe.to_csv(detail_path, index=False)

    summary = {
        "number_of_points": int(len(dataframe)),
        "complete_5x5_points": int(dataframe["complete_5x5"].sum()),
        "incomplete_5x5_points": int((~dataframe["complete_5x5"]).sum()),
        "min_existing_cells": int(dataframe["existing_cells_5x5"].min()),
        "max_existing_cells": int(dataframe["existing_cells_5x5"].max()),
        "mean_existing_cells": float(dataframe["existing_cells_5x5"].mean()),
    }

    summary_path = (
        METRICS_DIR
        / "grid_neighbors_5x5_summary.json"
    )

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("=" * 80)
    print("ANALYSE DES VOISINS 5×5")
    print("=" * 80)

    print(json.dumps(summary, ensure_ascii=False, indent=4))

    print("\nDistribution du nombre de cellules présentes dans le patch 5×5 :")
    print(
        dataframe["existing_cells_5x5"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nFichiers créés :")
    print(f"- Détail : {detail_path}")
    print(f"- Résumé : {summary_path}")


if __name__ == "__main__":
    main()
