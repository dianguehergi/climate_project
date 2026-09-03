from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# IMPORTS DU PROJET
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from src.utils.config import (  # noqa: E402
    FIGURE_DIR,
    LOG_DIR,
    MONTHLY_DATA_PATH,
    PERSONAL_PROCESSED_DATA_DIR,
    create_output_directories,
)


# ============================================================
# PARAMÈTRES
# ============================================================

CHUNK_SIZE = 500_000
PATCH_SIZE = 5
PATCH_RADIUS = PATCH_SIZE // 2


# ============================================================
# CHARGEMENT DES COORDONNÉES
# ============================================================

def collect_unique_grid_points(
    file_path: Path,
) -> pd.DataFrame:
    """
    Récupère toutes les coordonnées SAFRAN distinctes.

    Le fichier mensuel est utilisé car il contient la même
    grille spatiale tout en étant beaucoup plus petit que
    le fichier quotidien.
    """

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {file_path}"
        )

    unique_points: set[tuple[int, int]] = set()

    total_rows = 0
    start_time = time.perf_counter()

    reader = pd.read_csv(
        file_path,
        usecols=["LAMBX", "LAMBY"],
        dtype={
            "LAMBX": "int32",
            "LAMBY": "int32",
        },
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):
        total_rows += len(chunk)

        chunk_points = (
            chunk[["LAMBX", "LAMBY"]]
            .drop_duplicates()
            .itertuples(
                index=False,
                name=None,
            )
        )

        unique_points.update(
            (int(x), int(y))
            for x, y in chunk_points
        )

        elapsed = time.perf_counter() - start_time

        print(
            f"Bloc {chunk_number:03d} | "
            f"lignes={total_rows:,} | "
            f"points distincts={len(unique_points):,} | "
            f"durée={elapsed:.1f}s",
            flush=True,
        )

    dataframe = pd.DataFrame(
        sorted(unique_points),
        columns=["LAMBX", "LAMBY"],
    )

    if dataframe.empty:
        raise ValueError(
            "Aucun point géographique n'a été détecté."
        )

    return dataframe


# ============================================================
# PAS DE LA GRILLE
# ============================================================

def determine_grid_step(
    values: pd.Series,
) -> int:
    """
    Détermine le pas spatial le plus fréquent.
    """

    unique_values = np.sort(
        values.unique()
    )

    differences = np.diff(
        unique_values
    )

    positive_differences = differences[
        differences > 0
    ]

    if len(positive_differences) == 0:
        raise ValueError(
            "Impossible de déterminer le pas de la grille."
        )

    modes = pd.Series(
        positive_differences
    ).mode()

    return int(modes.iloc[0])


# ============================================================
# RECHERCHE DES MEILLEURS PATCHS
# ============================================================

def find_best_patch_centers(
    grid_dataframe: pd.DataFrame,
    step_x: int,
    step_y: int,
) -> pd.DataFrame:
    """
    Compte les voisins disponibles dans une fenêtre 5 × 5
    autour de chaque point.
    """

    point_set = set(
        grid_dataframe[
            ["LAMBX", "LAMBY"]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    offsets = list(
        product(
            range(
                -PATCH_RADIUS,
                PATCH_RADIUS + 1,
            ),
            repeat=2,
        )
    )

    median_x = float(
        grid_dataframe["LAMBX"].median()
    )

    median_y = float(
        grid_dataframe["LAMBY"].median()
    )

    candidates = []

    for lambx, lamby in point_set:
        available_points = 0

        for offset_x, offset_y in offsets:
            neighbor = (
                lambx + offset_x * step_x,
                lamby + offset_y * step_y,
            )

            if neighbor in point_set:
                available_points += 1

        distance_to_center = float(
            np.sqrt(
                (lambx - median_x) ** 2
                + (lamby - median_y) ** 2
            )
        )

        candidates.append(
            {
                "LAMBX": int(lambx),
                "LAMBY": int(lamby),
                "AVAILABLE_POINTS": available_points,
                "EXPECTED_POINTS": PATCH_SIZE ** 2,
                "PATCH_COMPLETENESS": (
                    available_points
                    / (PATCH_SIZE ** 2)
                ),
                "DISTANCE_TO_GRID_CENTER": (
                    distance_to_center
                ),
            }
        )

    candidates_dataframe = pd.DataFrame(
        candidates
    )

    candidates_dataframe = (
        candidates_dataframe
        .sort_values(
            by=[
                "AVAILABLE_POINTS",
                "DISTANCE_TO_GRID_CENTER",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    return candidates_dataframe


# ============================================================
# GRAPHIQUE
# ============================================================

def save_grid_plot(
    grid_dataframe: pd.DataFrame,
    best_center: pd.Series,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(10, 10)
    )

    axis.scatter(
        grid_dataframe["LAMBX"],
        grid_dataframe["LAMBY"],
        s=4,
        alpha=0.5,
        label="Points SAFRAN",
    )

    axis.scatter(
        [best_center["LAMBX"]],
        [best_center["LAMBY"]],
        s=100,
        marker="x",
        label="Centre proposé",
    )

    axis.set_title(
        "Grille SAFRAN et centre spatial proposé"
    )

    axis.set_xlabel(
        "Coordonnée Lambert X"
    )

    axis.set_ylabel(
        "Coordonnée Lambert Y"
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    axis.legend()
    axis.grid(alpha=0.2)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main() -> None:
    create_output_directories()

    print("=" * 80)
    print("INSPECTION DE LA GRILLE SPATIALE SAFRAN")
    print("=" * 80)

    grid_dataframe = collect_unique_grid_points(
        MONTHLY_DATA_PATH
    )

    step_x = determine_grid_step(
        grid_dataframe["LAMBX"]
    )

    step_y = determine_grid_step(
        grid_dataframe["LAMBY"]
    )

    candidates_dataframe = find_best_patch_centers(
        grid_dataframe=grid_dataframe,
        step_x=step_x,
        step_y=step_y,
    )

    best_center = candidates_dataframe.iloc[0]

    grid_output_path = (
        PERSONAL_PROCESSED_DATA_DIR
        / "safran_grid_points.csv"
    )

    candidates_output_path = (
        PERSONAL_PROCESSED_DATA_DIR
        / "safran_best_5x5_centers.csv"
    )

    summary_output_path = (
        LOG_DIR
        / "safran_grid_summary.json"
    )

    figure_output_path = (
        FIGURE_DIR
        / "safran_spatial_grid.png"
    )

    grid_dataframe.to_csv(
        grid_output_path,
        index=False,
    )

    candidates_dataframe.head(100).to_csv(
        candidates_output_path,
        index=False,
    )

    summary = {
        "number_of_grid_points": int(
            len(grid_dataframe)
        ),
        "number_of_unique_x": int(
            grid_dataframe["LAMBX"].nunique()
        ),
        "number_of_unique_y": int(
            grid_dataframe["LAMBY"].nunique()
        ),
        "minimum_x": int(
            grid_dataframe["LAMBX"].min()
        ),
        "maximum_x": int(
            grid_dataframe["LAMBX"].max()
        ),
        "minimum_y": int(
            grid_dataframe["LAMBY"].min()
        ),
        "maximum_y": int(
            grid_dataframe["LAMBY"].max()
        ),
        "grid_step_x": step_x,
        "grid_step_y": step_y,
        "patch_size": PATCH_SIZE,
        "best_center_lambx": int(
            best_center["LAMBX"]
        ),
        "best_center_lamby": int(
            best_center["LAMBY"]
        ),
        "available_points_in_patch": int(
            best_center["AVAILABLE_POINTS"]
        ),
        "expected_points_in_patch": int(
            best_center["EXPECTED_POINTS"]
        ),
        "patch_completeness": float(
            best_center["PATCH_COMPLETENESS"]
        ),
    }

    with summary_output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4,
        )

    save_grid_plot(
        grid_dataframe=grid_dataframe,
        best_center=best_center,
        output_path=figure_output_path,
    )

    print("\n" + "=" * 80)
    print("RÉSULTATS")
    print("=" * 80)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nMeilleurs centres 5 × 5 :")

    print(
        candidates_dataframe.head(10).to_string(
            index=False
        )
    )

    print("\nFichiers créés :")
    print(f"- Grille       : {grid_output_path}")
    print(f"- Centres      : {candidates_output_path}")
    print(f"- Résumé       : {summary_output_path}")
    print(f"- Visualisation: {figure_output_path}")


if __name__ == "__main__":
    main()