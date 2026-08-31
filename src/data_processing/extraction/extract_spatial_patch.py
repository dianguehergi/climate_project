from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from config import (  # noqa: E402
    DAILY_DATA_PATH,
    SPATIAL_CENTER_SERIES_PATH,
    SPATIAL_PATCH_METADATA_PATH,
    SPATIAL_PATCH_PATH,
    create_output_directories,
)


# ============================================================
# PARAMÈTRES DU PATCH
# ============================================================

CENTER_X = 6440
CENTER_Y = 22010

GRID_STEP = 80
PATCH_SIZE = 5
PATCH_RADIUS = PATCH_SIZE // 2

CHUNK_SIZE = 1_000_000

BASE_VARIABLES = [
    "T",
    "TINF_H",
    "TSUP_H",
]

OUTPUT_CHANNELS = [
    "T",
    "TINF_H",
    "TSUP_H",
    "DOY_SIN",
    "DOY_COS",
]

USE_COLUMNS = [
    "LAMBX",
    "LAMBY",
    "DATE",
    *BASE_VARIABLES,
]

DTYPES = {
    "LAMBX": "int32",
    "LAMBY": "int32",
    "DATE": "int32",
    "T": "float32",
    "TINF_H": "float32",
    "TSUP_H": "float32",
}


def create_patch_coordinates() -> tuple[list[int], list[int]]:
    """Crée les cinq coordonnées X et Y du patch."""

    x_values = [
        CENTER_X + offset * GRID_STEP
        for offset in range(
            -PATCH_RADIUS,
            PATCH_RADIUS + 1,
        )
    ]

    y_values = [
        CENTER_Y + offset * GRID_STEP
        for offset in range(
            -PATCH_RADIUS,
            PATCH_RADIUS + 1,
        )
    ]

    return x_values, y_values


def extract_patch_rows(
    source_path: Path,
    x_values: list[int],
    y_values: list[int],
) -> pd.DataFrame:
    """Extrait les 25 points du fichier quotidien."""

    selected_chunks: list[pd.DataFrame] = []

    total_rows_read = 0
    total_rows_selected = 0

    start_time = time.perf_counter()

    reader = pd.read_csv(
        source_path,
        usecols=USE_COLUMNS,
        dtype=DTYPES,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):
        total_rows_read += len(chunk)

        mask = (
            chunk["LAMBX"].isin(x_values)
            & chunk["LAMBY"].isin(y_values)
        )

        selected = chunk.loc[
            mask,
            USE_COLUMNS,
        ].copy()

        if not selected.empty:
            selected_chunks.append(selected)
            total_rows_selected += len(selected)

        elapsed = time.perf_counter() - start_time

        print(
            f"Bloc {chunk_number:04d} | "
            f"lignes lues={total_rows_read:,} | "
            f"lignes retenues={total_rows_selected:,} | "
            f"durée={elapsed:.1f}s",
            flush=True,
        )

    if not selected_chunks:
        raise ValueError(
            "Aucune ligne n'a été trouvée pour le patch."
        )

    return pd.concat(
        selected_chunks,
        ignore_index=True,
    )


def validate_long_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Nettoie et contrôle les données extraites."""

    dataframe = dataframe.copy()

    dataframe["DATE"] = pd.to_datetime(
        dataframe["DATE"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    dataframe = (
        dataframe
        .dropna(
            subset=[
                "DATE",
                "LAMBX",
                "LAMBY",
                *BASE_VARIABLES,
            ]
        )
        .sort_values(
            [
                "DATE",
                "LAMBY",
                "LAMBX",
            ]
        )
        .reset_index(drop=True)
    )

    duplicate_count = int(
        dataframe.duplicated(
            subset=[
                "DATE",
                "LAMBX",
                "LAMBY",
            ]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"{duplicate_count} doublon(s) spatio-temporel(s)."
        )

    counts_per_date = dataframe.groupby(
        "DATE"
    ).size()

    invalid_dates = counts_per_date[
        counts_per_date != PATCH_SIZE**2
    ]

    if not invalid_dates.empty:
        raise ValueError(
            f"{len(invalid_dates)} date(s) ne contiennent "
            f"pas exactement {PATCH_SIZE**2} points."
        )

    expected_rows = (
        dataframe["DATE"].nunique()
        * PATCH_SIZE**2
    )

    if len(dataframe) != expected_rows:
        raise ValueError(
            f"Nombre de lignes incohérent : "
            f"{len(dataframe):,} au lieu de "
            f"{expected_rows:,}."
        )

    return dataframe


def build_spatial_array(
    dataframe: pd.DataFrame,
    x_values: list[int],
    y_values: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construit un tenseur :

    dates × canaux × hauteur × largeur
    """

    dates = pd.DatetimeIndex(
        dataframe["DATE"]
        .drop_duplicates()
        .sort_values()
    )

    # Y décroissant : valeur Y la plus élevée en haut.
    ordered_y = sorted(
        y_values,
        reverse=True,
    )

    ordered_x = sorted(
        x_values
    )

    ordered_columns = pd.MultiIndex.from_tuples(
        [
            (y_value, x_value)
            for y_value in ordered_y
            for x_value in ordered_x
        ],
        names=[
            "LAMBY",
            "LAMBX",
        ],
    )

    number_of_dates = len(dates)

    spatial_data = np.empty(
        (
            number_of_dates,
            len(OUTPUT_CHANNELS),
            PATCH_SIZE,
            PATCH_SIZE,
        ),
        dtype=np.float32,
    )

    for channel_index, variable in enumerate(
        BASE_VARIABLES
    ):
        pivot = dataframe.pivot(
            index="DATE",
            columns=[
                "LAMBY",
                "LAMBX",
            ],
            values=variable,
        )

        pivot = pivot.reindex(
            index=dates,
            columns=ordered_columns,
        )

        if pivot.isna().any().any():
            raise ValueError(
                f"Valeurs manquantes après pivot pour {variable}."
            )

        spatial_data[
            :,
            channel_index,
            :,
            :,
        ] = pivot.to_numpy(
            dtype=np.float32
        ).reshape(
            number_of_dates,
            PATCH_SIZE,
            PATCH_SIZE,
        )

    day_of_year = dates.dayofyear.to_numpy(
        dtype=np.float32
    )

    angle = (
        2.0
        * np.pi
        * (day_of_year - 1.0)
        / 365.25
    )

    day_sin = np.sin(angle).astype(
        np.float32
    )

    day_cos = np.cos(angle).astype(
        np.float32
    )

    spatial_data[:, 3, :, :] = day_sin[
        :,
        None,
        None,
    ]

    spatial_data[:, 4, :, :] = day_cos[
        :,
        None,
        None,
    ]

    dates_integer = np.asarray(
        [
            int(date.strftime("%Y%m%d"))
            for date in dates
        ],
        dtype=np.int32,
    )

    coordinates = np.asarray(
        [
            [x_value, y_value]
            for y_value in ordered_y
            for x_value in ordered_x
        ],
        dtype=np.int32,
    ).reshape(
        PATCH_SIZE,
        PATCH_SIZE,
        2,
    )

    return (
        spatial_data,
        dates_integer,
        coordinates,
    )


def save_center_series(
    dataframe: pd.DataFrame,
) -> None:
    """Enregistre la série correspondant au centre du patch."""

    center_dataframe = dataframe.loc[
        (
            dataframe["LAMBX"] == CENTER_X
        )
        & (
            dataframe["LAMBY"] == CENTER_Y
        ),
        [
            "DATE",
            "LAMBX",
            "LAMBY",
            *BASE_VARIABLES,
        ],
    ].copy()

    center_dataframe = (
        center_dataframe
        .sort_values("DATE")
        .reset_index(drop=True)
    )

    center_dataframe["DAY_OF_YEAR"] = (
        center_dataframe["DATE"].dt.dayofyear
    )

    angle = (
        2.0
        * np.pi
        * (
            center_dataframe["DAY_OF_YEAR"]
            - 1.0
        )
        / 365.25
    )

    center_dataframe["DOY_SIN"] = np.sin(
        angle
    ).astype("float32")

    center_dataframe["DOY_COS"] = np.cos(
        angle
    ).astype("float32")

    SPATIAL_CENTER_SERIES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    center_dataframe.to_csv(
        SPATIAL_CENTER_SERIES_PATH,
        index=False,
    )


def main() -> None:
    create_output_directories()

    if not DAILY_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Source introuvable : {DAILY_DATA_PATH}"
        )

    x_values, y_values = create_patch_coordinates()

    print("=" * 80)
    print("EXTRACTION DU PATCH SPATIAL SAFRAN 5 × 5")
    print("=" * 80)

    print(
        f"Centre : LAMBX={CENTER_X}, LAMBY={CENTER_Y}"
    )
    print(f"Coordonnées X : {x_values}")
    print(f"Coordonnées Y : {y_values}")
    print(f"Source : {DAILY_DATA_PATH}")

    dataframe = extract_patch_rows(
        source_path=DAILY_DATA_PATH,
        x_values=x_values,
        y_values=y_values,
    )

    dataframe = validate_long_dataframe(
        dataframe
    )

    (
        spatial_data,
        dates_integer,
        coordinates,
    ) = build_spatial_array(
        dataframe=dataframe,
        x_values=x_values,
        y_values=y_values,
    )

    SPATIAL_PATCH_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        SPATIAL_PATCH_PATH,
        data=spatial_data,
        dates=dates_integer,
        coordinates=coordinates,
        channels=np.asarray(
            OUTPUT_CHANNELS
        ),
    )

    save_center_series(
        dataframe
    )

    metadata = {
        "center_lambx": CENTER_X,
        "center_lamby": CENTER_Y,
        "grid_step": GRID_STEP,
        "patch_size": PATCH_SIZE,
        "number_of_points": PATCH_SIZE**2,
        "number_of_dates": int(
            spatial_data.shape[0]
        ),
        "channels": OUTPUT_CHANNELS,
        "data_shape": list(
            spatial_data.shape
        ),
        "date_start": str(
            pd.to_datetime(
                str(dates_integer[0]),
                format="%Y%m%d",
            ).date()
        ),
        "date_end": str(
            pd.to_datetime(
                str(dates_integer[-1]),
                format="%Y%m%d",
            ).date()
        ),
        "target": "T at center on next day",
        "center_row": PATCH_RADIUS,
        "center_column": PATCH_RADIUS,
        "source": str(DAILY_DATA_PATH),
    }

    with SPATIAL_PATCH_METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\n" + "=" * 80)
    print("EXTRACTION TERMINÉE")
    print("=" * 80)

    print(
        f"Lignes longues : {len(dataframe):,}"
    )
    print(
        f"Nombre de dates : "
        f"{dataframe['DATE'].nunique():,}"
    )
    print(
        f"Forme du tenseur : {spatial_data.shape}"
    )
    print(
        f"Valeurs non finies : "
        f"{np.size(spatial_data) - np.isfinite(spatial_data).sum():,}"
    )

    print(f"\nPatch NPZ : {SPATIAL_PATCH_PATH}")
    print(
        f"Métadonnées : {SPATIAL_PATCH_METADATA_PATH}"
    )
    print(
        f"Série centrale : {SPATIAL_CENTER_SERIES_PATH}"
    )


if __name__ == "__main__":
    main()