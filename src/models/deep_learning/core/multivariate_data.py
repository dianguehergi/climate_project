from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.utils.config import FIRST_POINT_MULTIVARIATE_PATH


TARGET_COLUMN = "T"

DEFAULT_FEATURE_COLUMNS = (
    "T",
    "PRETOT",
    "FF",
    "Q",
    "SSI",
    "HU",
    "ETP",
    "SWI",
    "SSWI_10J",
    "DRAINC",
    "TINF_H",
    "TSUP_H",
    "DOY_SIN",
    "DOY_COS",
)


def load_multivariate_point(
    file_path: Path = FIRST_POINT_MULTIVARIATE_PATH,
) -> pd.DataFrame:
    """Charge et contrôle le point SAFRAN multivarié."""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Fichier multivarié introuvable : {file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        low_memory=False,
    )

    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.strip()
        .str.upper()
    )

    required_columns = {
        "DATE",
        TARGET_COLUMN,
        *DEFAULT_FEATURE_COLUMNS[:-2],
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Colonnes nécessaires absentes : "
            f"{sorted(missing_columns)}"
        )

    dataframe["DATE"] = pd.to_datetime(
        dataframe["DATE"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    numeric_columns = [
        column
        for column in DEFAULT_FEATURE_COLUMNS
        if column not in {"DOY_SIN", "DOY_COS"}
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    # Représentation cyclique du jour de l'année.
    day_of_year = (
        dataframe["DATE"]
        .dt.dayofyear
        .astype("float32")
    )

    angle = (
        2.0
        * np.pi
        * (day_of_year - 1.0)
        / 365.25
    )

    dataframe["DOY_SIN"] = np.sin(
        angle
    ).astype("float32")

    dataframe["DOY_COS"] = np.cos(
        angle
    ).astype("float32")

    dataframe = (
        dataframe
        .dropna(
            subset=[
                "DATE",
                *DEFAULT_FEATURE_COLUMNS,
            ]
        )
        .sort_values("DATE")
        .drop_duplicates(
            subset=["DATE"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if dataframe.empty:
        raise ValueError(
            "Le jeu de données multivarié est vide "
            "après nettoyage."
        )

    date_differences = (
        dataframe["DATE"]
        .diff()
        .dropna()
    )

    invalid_gaps = date_differences[
        date_differences != pd.Timedelta(days=1)
    ]

    if not invalid_gaps.empty:
        raise ValueError(
            f"{len(invalid_gaps)} rupture(s) temporelle(s) "
            "ont été détectées."
        )

    return dataframe


def fit_feature_standardizer(
    train_dataframe: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Calcule moyenne et écart-type uniquement sur le train.
    """

    means = (
        train_dataframe[
            list(feature_columns)
        ]
        .mean()
    )

    standard_deviations = (
        train_dataframe[
            list(feature_columns)
        ]
        .std(ddof=0)
    )

    standard_deviations = standard_deviations.replace(
        0.0,
        1.0,
    )

    means_dictionary = {
        column: float(means[column])
        for column in feature_columns
    }

    standard_deviations_dictionary = {
        column: float(
            standard_deviations[column]
        )
        for column in feature_columns
    }

    return (
        means_dictionary,
        standard_deviations_dictionary,
    )


def transform_features(
    dataframe: pd.DataFrame,
    feature_columns: Sequence[str],
    means: dict[str, float],
    standard_deviations: dict[str, float],
) -> pd.DataFrame:
    """Applique la standardisation apprise sur le train."""

    transformed = dataframe.copy()

    for column in feature_columns:
        transformed[column] = (
            transformed[column]
            - means[column]
        ) / standard_deviations[column]

    return transformed


def create_multivariate_sequences(
    dataframe: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str = TARGET_COLUMN,
    input_length: int = 30,
    forecast_horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crée les tenseurs :

    X : séquences × variables × longueur
    y : séquences
    """

    if target_column not in feature_columns:
        raise ValueError(
            f"La cible {target_column} doit être incluse "
            "dans les variables standardisées."
        )

    feature_values = dataframe[
        list(feature_columns)
    ].to_numpy(
        dtype=np.float32,
    )

    target_values = dataframe[
        target_column
    ].to_numpy(
        dtype=np.float32,
    )

    number_of_sequences = (
        len(dataframe)
        - input_length
        - forecast_horizon
        + 1
    )

    if number_of_sequences <= 0:
        raise ValueError(
            "Pas assez de lignes pour créer les séquences."
        )

    inputs = np.empty(
        (
            number_of_sequences,
            len(feature_columns),
            input_length,
        ),
        dtype=np.float32,
    )

    targets = np.empty(
        number_of_sequences,
        dtype=np.float32,
    )

    for start_index in range(
        number_of_sequences
    ):
        input_end = (
            start_index
            + input_length
        )

        target_index = (
            input_end
            + forecast_horizon
            - 1
        )

        inputs[start_index] = (
            feature_values[
                start_index:input_end
            ].T
        )

        targets[start_index] = (
            target_values[target_index]
        )

    return inputs, targets
