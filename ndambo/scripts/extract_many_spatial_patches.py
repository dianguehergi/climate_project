from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from config import (  # noqa: E402
    PERSONAL_PROCESSED_DATA_DIR,
    SPATIAL_CENTER_SERIES_PATH,
    create_output_directories,
)


SOURCE_PATH = (
    PROJECT_DIR.parent
    / "data"
    / "processed"
    / "safran_quot_clean.csv"
)

GRID_POINTS_PATH = (
    PERSONAL_PROCESSED_DATA_DIR
    / "safran_grid_points.csv"
)

GRID_STEP = 80

BASE_CHANNELS = [
    "T",
    "TINF_H",
    "TSUP_H",
]

CALENDAR_CHANNELS = [
    "DOY_SIN",
    "DOY_COS",
]


def make_point_key(
    lambx: np.ndarray | pd.Series,
    lamby: np.ndarray | pd.Series,
) -> np.ndarray | pd.Series:
    return lambx.astype(np.int64) * 100_000 + lamby.astype(np.int64)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extraction de patchs spatiaux SAFRAN pour "
            "plusieurs centres de grille."
        )
    )

    parser.add_argument(
        "--max-centers",
        type=int,
        default=100,
        help=(
            "Nombre maximal de centres à extraire. "
            "Utiliser une valeur plus grande après validation."
        ),
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=5,
        choices=[1, 3, 5],
        help="Taille du patch spatial centré.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Nombre de lignes lues par bloc.",
    )

    return parser.parse_args()


def load_dates() -> pd.DatetimeIndex:
    if not SPATIAL_CENTER_SERIES_PATH.is_file():
        raise FileNotFoundError(
            "La série du centre spatial est introuvable : "
            f"{SPATIAL_CENTER_SERIES_PATH}"
        )

    dataframe = pd.read_csv(
        SPATIAL_CENTER_SERIES_PATH,
        usecols=["DATE"],
    )

    dates = pd.to_datetime(
        dataframe["DATE"].astype(str),
        errors="raise",
    )

    return pd.DatetimeIndex(dates)


def load_grid_points() -> pd.DataFrame:
    if not GRID_POINTS_PATH.is_file():
        raise FileNotFoundError(
            f"Fichier de grille introuvable : {GRID_POINTS_PATH}"
        )

    dataframe = pd.read_csv(GRID_POINTS_PATH)

    required_columns = {"LAMBX", "LAMBY"}

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes dans la grille : {missing_columns}"
        )

    dataframe = dataframe[["LAMBX", "LAMBY"]].drop_duplicates()

    dataframe["LAMBX"] = dataframe["LAMBX"].astype(int)
    dataframe["LAMBY"] = dataframe["LAMBY"].astype(int)

    dataframe = dataframe.sort_values(
        ["LAMBX", "LAMBY"]
    ).reset_index(drop=True)

    return dataframe


def find_complete_centers(
    grid_points: pd.DataFrame,
    patch_size: int,
) -> pd.DataFrame:
    point_set = set(
        zip(
            grid_points["LAMBX"].tolist(),
            grid_points["LAMBY"].tolist(),
        )
    )

    radius = patch_size // 2

    centers = []

    for row in grid_points.itertuples(index=False):
        center_x = int(row.LAMBX)
        center_y = int(row.LAMBY)

        complete = True

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                point = (
                    center_x + dx * GRID_STEP,
                    center_y + dy * GRID_STEP,
                )

                if point not in point_set:
                    complete = False
                    break

            if not complete:
                break

        if complete:
            centers.append(
                {
                    "center_lambx": center_x,
                    "center_lamby": center_y,
                }
            )

    return pd.DataFrame(centers)


def select_centers(
    centers: pd.DataFrame,
    max_centers: int,
) -> pd.DataFrame:
    centers = centers.sort_values(
        ["center_lambx", "center_lamby"]
    ).reset_index(drop=True)

    if max_centers >= len(centers):
        return centers

    indices = np.linspace(
        0,
        len(centers) - 1,
        max_centers,
        dtype=int,
    )

    selected = centers.iloc[indices].copy()
    selected = selected.reset_index(drop=True)

    return selected


def build_required_points_mapping(
    centers: pd.DataFrame,
    patch_size: int,
) -> tuple[dict[int, list[tuple[int, int, int]]], np.ndarray]:
    radius = patch_size // 2

    point_to_positions: dict[
        int,
        list[tuple[int, int, int]],
    ] = defaultdict(list)

    for center_index, row in centers.iterrows():
        center_x = int(row["center_lambx"])
        center_y = int(row["center_lamby"])

        for row_offset, dy in enumerate(range(radius, -radius - 1, -1)):
            for column_offset, dx in enumerate(range(-radius, radius + 1)):
                lambx = center_x + dx * GRID_STEP
                lamby = center_y + dy * GRID_STEP

                key = int(lambx) * 100_000 + int(lamby)

                point_to_positions[key].append(
                    (
                        center_index,
                        row_offset,
                        column_offset,
                    )
                )

    required_keys = np.array(
        list(point_to_positions.keys()),
        dtype=np.int64,
    )

    return point_to_positions, required_keys


def main() -> None:
    args = parse_arguments()

    create_output_directories()

    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(
            f"Fichier SAFRAN quotidien introuvable : {SOURCE_PATH}"
        )

    if args.patch_size % 2 == 0:
        raise ValueError(
            "La taille du patch doit être impaire."
        )

    print("=" * 80)
    print("EXTRACTION MULTI-POINTS SAFRAN")
    print("=" * 80)

    print(f"Source       : {SOURCE_PATH}")
    print(f"Patch        : {args.patch_size}×{args.patch_size}")
    print(f"Max centres  : {args.max_centers}")

    dates = load_dates()

    date_values = pd.Series(dates).dt.strftime("%Y%m%d").astype(np.int64).to_numpy()

    date_to_index = {
        int(date_value): index
        for index, date_value in enumerate(date_values)
    }

    grid_points = load_grid_points()

    complete_centers = find_complete_centers(
        grid_points,
        patch_size=args.patch_size,
    )

    selected_centers = select_centers(
        complete_centers,
        max_centers=args.max_centers,
    )

    number_of_days = len(dates)
    number_of_centers = len(selected_centers)
    number_of_channels = len(BASE_CHANNELS) + len(CALENDAR_CHANNELS)

    print(f"Dates        : {number_of_days:,}")
    print(f"Points grille: {len(grid_points):,}")
    print(f"Centres complets disponibles : {len(complete_centers):,}")
    print(f"Centres sélectionnés         : {number_of_centers:,}")

    point_to_positions, required_keys = build_required_points_mapping(
        selected_centers,
        patch_size=args.patch_size,
    )

    print(f"Points requis uniques        : {len(required_keys):,}")

    data = np.full(
        (
            number_of_days,
            number_of_centers,
            number_of_channels,
            args.patch_size,
            args.patch_size,
        ),
        np.nan,
        dtype=np.float32,
    )

    day_of_year = pd.Series(dates).dt.dayofyear.to_numpy(dtype=np.float32)

    doy_sin = np.sin(2.0 * np.pi * day_of_year / 365.25).astype(np.float32)
    doy_cos = np.cos(2.0 * np.pi * day_of_year / 365.25).astype(np.float32)

    data[:, :, 3, :, :] = doy_sin[:, None, None, None]
    data[:, :, 4, :, :] = doy_cos[:, None, None, None]

    use_columns = [
        "LAMBX",
        "LAMBY",
        "DATE",
        *BASE_CHANNELS,
    ]

    total_rows = 0
    retained_rows = 0
    start_time = time.perf_counter()

    reader = pd.read_csv(
        SOURCE_PATH,
        usecols=use_columns,
        chunksize=args.chunk_size,
    )

    for chunk_index, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)

        keys = make_point_key(
            chunk["LAMBX"],
            chunk["LAMBY"],
        )

        mask = keys.isin(required_keys)

        if not mask.any():
            if chunk_index % 20 == 0:
                print(
                    f"Bloc {chunk_index:04d} | "
                    f"lignes lues={total_rows:,} | "
                    f"retenues={retained_rows:,}"
                )
            continue

        filtered = chunk.loc[mask].copy()
        filtered["KEY"] = keys.loc[mask].to_numpy()

        filtered["DATE_INT"] = (
            pd.to_datetime(
                filtered["DATE"].astype(str),
                errors="raise",
            )
            .dt.strftime("%Y%m%d")
            .astype(np.int64)
        )

        retained_rows += len(filtered)

        for record in filtered.itertuples(index=False):
            date_index = date_to_index.get(int(record.DATE_INT))

            if date_index is None:
                continue

            positions = point_to_positions[int(record.KEY)]

            values = [
                float(record.T),
                float(record.TINF_H),
                float(record.TSUP_H),
            ]

            for center_index, row_index, column_index in positions:
                data[
                    date_index,
                    center_index,
                    0:3,
                    row_index,
                    column_index,
                ] = values

        if chunk_index % 10 == 0:
            elapsed = time.perf_counter() - start_time

            print(
                f"Bloc {chunk_index:04d} | "
                f"lignes lues={total_rows:,} | "
                f"retenues={retained_rows:,} | "
                f"durée={elapsed:.1f}s"
            )

    missing_count = int(
        np.isnan(data[:, :, 0:3, :, :]).sum()
    )

    if missing_count > 0:
        raise ValueError(
            f"Il reste {missing_count:,} valeurs manquantes "
            "dans les canaux météo."
        )

    output_stem = (
        f"safran_many_centers_"
        f"{args.patch_size}x{args.patch_size}_"
        f"{number_of_centers}centers_temperature_family"
    )

    output_npz_path = (
        PERSONAL_PROCESSED_DATA_DIR
        / f"{output_stem}.npz"
    )

    output_metadata_path = (
        PERSONAL_PROCESSED_DATA_DIR
        / f"{output_stem}_metadata.json"
    )

    centers_path = (
        PERSONAL_PROCESSED_DATA_DIR
        / f"{output_stem}_centers.csv"
    )

    np.savez_compressed(
        output_npz_path,
        data=data,
        dates=date_values,
        centers=selected_centers.to_numpy(dtype=np.int64),
        channels=np.array(BASE_CHANNELS + CALENDAR_CHANNELS),
    )

    selected_centers.to_csv(
        centers_path,
        index=False,
    )

    metadata = {
        "source": str(SOURCE_PATH),
        "patch_size": args.patch_size,
        "grid_step": GRID_STEP,
        "number_of_days": number_of_days,
        "number_of_centers": number_of_centers,
        "number_of_complete_centers_available": int(len(complete_centers)),
        "number_of_unique_required_points": int(len(required_keys)),
        "channels": BASE_CHANNELS + CALENDAR_CHANNELS,
        "data_shape": list(data.shape),
        "date_start": str(dates.min().date()),
        "date_end": str(dates.max().date()),
        "target": "T au centre du patch à J+1",
        "center_row": args.patch_size // 2,
        "center_column": args.patch_size // 2,
        "retained_rows": int(retained_rows),
        "total_rows_scanned": int(total_rows),
    }

    with output_metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=4,
        )

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 80)
    print("EXTRACTION TERMINÉE")
    print("=" * 80)
    print(f"Durée      : {elapsed:.1f}s")
    print(f"Tenseur    : {data.shape}")
    print(f"NPZ        : {output_npz_path}")
    print(f"Métadonnées: {output_metadata_path}")
    print(f"Centres    : {centers_path}")


if __name__ == "__main__":
    main()
