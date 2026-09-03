from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from src.utils.config import (
    DAILY_DATA_PATH,
    FIRST_POINT_DAILY_PATH,
    MONTHLY_DATA_PATH,
)


# ============================================================
# CONFIGURATION DES SÉQUENCES
# ============================================================

@dataclass
class SequenceConfig:
    """Paramètres de création des séquences temporelles."""

    input_length: int = 30
    forecast_horizon: int = 1
    target_column: str = "T"


# ============================================================
# CONVERSION DES DATES
# ============================================================

def parse_daily_date(series: pd.Series) -> pd.Series:
    """
    Convertit les principaux formats de dates quotidiennes SAFRAN :

    - 19900101
    - 19900101.0
    - 1990-01-01
    - 01/01/1990
    """

    text_dates = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    parsed_dates = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]",
    )

    # --------------------------------------------------------
    # Format compact SAFRAN : YYYYMMDD
    # Exemple : 19900101
    # --------------------------------------------------------

    mask_compact = text_dates.str.fullmatch(
        r"\d{8}",
        na=False,
    )

    parsed_dates.loc[mask_compact] = pd.to_datetime(
        text_dates.loc[mask_compact],
        format="%Y%m%d",
        errors="coerce",
    )

    # --------------------------------------------------------
    # Format ISO : YYYY-MM-DD
    # Exemple : 1990-01-01
    # --------------------------------------------------------

    mask_iso = text_dates.str.fullmatch(
        r"\d{4}-\d{2}-\d{2}",
        na=False,
    )

    parsed_dates.loc[mask_iso] = pd.to_datetime(
        text_dates.loc[mask_iso],
        format="%Y-%m-%d",
        errors="coerce",
    )

    # --------------------------------------------------------
    # Format français : DD/MM/YYYY
    # Exemple : 01/01/1990
    # --------------------------------------------------------

    mask_french = text_dates.str.fullmatch(
        r"\d{2}/\d{2}/\d{4}",
        na=False,
    )

    parsed_dates.loc[mask_french] = pd.to_datetime(
        text_dates.loc[mask_french],
        format="%d/%m/%Y",
        errors="coerce",
    )

    # --------------------------------------------------------
    # Formats restants éventuels
    # --------------------------------------------------------

    known_formats = (
        mask_compact
        | mask_iso
        | mask_french
    )

    mask_other = ~known_formats

    if mask_other.any():
        parsed_dates.loc[mask_other] = pd.to_datetime(
            text_dates.loc[mask_other],
            errors="coerce",
        )

    return parsed_dates


def parse_monthly_date(series: pd.Series) -> pd.Series:
    """Convertit une date SAFRAN mensuelle YYYYMM."""

    return pd.to_datetime(
        series.astype(str).str.strip(),
        format="%Y%m",
        errors="coerce",
    )


# ============================================================
# CHARGEMENT DU PREMIER POINT
# ============================================================
def detect_csv_separator(file_path: Path) -> str:
    """Détecte le séparateur principal d'un fichier CSV."""

    with file_path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:
        first_line = file.readline()

    separators = {
        ",": first_line.count(","),
        ";": first_line.count(";"),
        "\t": first_line.count("\t"),
        "|": first_line.count("|"),
    }

    separator = max(
        separators,
        key=separators.get,
    )

    return separator


def load_first_point_temperature(
    file_path: Path = FIRST_POINT_DAILY_PATH,
) -> pd.DataFrame:
    """
    Charge et nettoie la série quotidienne de température
    correspondant au premier point SAFRAN.
    """

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {file_path}"
        )

    separator = detect_csv_separator(file_path)

    dataframe = pd.read_csv(
        file_path,
        sep=separator,
        low_memory=False,
    )

    # Nettoyage des noms de colonnes :
    # espaces, minuscules, caractères BOM éventuels
    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.upper()
    )

    print(
        f"Fichier chargé : {file_path}"
    )
    print(
        f"Séparateur détecté : {repr(separator)}"
    )
    print(
        f"Dimensions initiales : {dataframe.shape}"
    )
    print(
        f"Colonnes détectées : {dataframe.columns.tolist()}"
    )

    required_columns = {
        "DATE",
        "T",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Colonnes obligatoires absentes : "
            f"{sorted(missing_columns)}\n"
            "Colonnes disponibles : "
            f"{dataframe.columns.tolist()}"
        )

    dataframe = dataframe.copy()

    raw_dates = dataframe["DATE"].copy()
    raw_temperatures = dataframe["T"].copy()

    dataframe["DATE"] = parse_daily_date(
        dataframe["DATE"]
    )

    # Compatible avec les nombres utilisant un point ou une virgule
    dataframe["T"] = pd.to_numeric(
        dataframe["T"]
        .astype("string")
        .str.strip()
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )

    invalid_dates = int(
        dataframe["DATE"].isna().sum()
    )

    invalid_temperatures = int(
        dataframe["T"].isna().sum()
    )

    print(
        f"Dates invalides : "
        f"{invalid_dates:,}/{len(dataframe):,}"
    )

    print(
        f"Températures invalides : "
        f"{invalid_temperatures:,}/{len(dataframe):,}"
    )

    dataframe = (
        dataframe
        .dropna(
            subset=[
                "DATE",
                "T",
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
        diagnostics = pd.DataFrame(
            {
                "DATE_BRUTE": raw_dates.head(10),
                "T_BRUTE": raw_temperatures.head(10),
            }
        )

        raise ValueError(
            "Le DataFrame est vide après conversion.\n\n"
            "Exemples de valeurs brutes :\n"
            f"{diagnostics.to_string(index=False)}\n\n"
            "Le format réel du fichier doit être vérifié."
        )

    print(
        f"Dimensions après nettoyage : {dataframe.shape}"
    )

    print(
        f"Période : "
        f"{dataframe['DATE'].min().date()} → "
        f"{dataframe['DATE'].max().date()}"
    )

    return dataframe


# ============================================================
# CHARGEMENT PAR MORCEAUX DU FICHIER COMPLET
# ============================================================

def iter_daily_chunks(
    file_path: Path = DAILY_DATA_PATH,
    chunk_size: int = 250_000,
    usecols: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """
    Lit le fichier quotidien de 20 Go par morceaux.

    Le fichier n'est jamais chargé entièrement en mémoire.
    """

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {file_path}"
        )

    for chunk in pd.read_csv(
        file_path,
        chunksize=chunk_size,
        usecols=usecols,
        low_memory=False,
    ):
        if "DATE" in chunk.columns:
            chunk["DATE"] = parse_daily_date(
                chunk["DATE"]
            )

        yield chunk


def iter_monthly_chunks(
    file_path: Path = MONTHLY_DATA_PATH,
    chunk_size: int = 250_000,
    usecols: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Lit le fichier mensuel par morceaux."""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {file_path}"
        )

    for chunk in pd.read_csv(
        file_path,
        chunksize=chunk_size,
        usecols=usecols,
        low_memory=False,
    ):
        if "DATE" in chunk.columns:
            chunk["DATE"] = parse_monthly_date(
                chunk["DATE"]
            )

        yield chunk


# ============================================================
# CONTRÔLE DE LA SÉRIE TEMPORELLE
# ============================================================

def summarize_time_series(
    dataframe: pd.DataFrame,
) -> dict[str, object]:
    """Retourne les principales informations sur la série."""

    if dataframe.empty:
        raise ValueError(
            "Le DataFrame est vide."
        )

    complete_calendar = pd.date_range(
        start=dataframe["DATE"].min(),
        end=dataframe["DATE"].max(),
        freq="D",
    )

    observed_dates = pd.DatetimeIndex(
        dataframe["DATE"].dropna().unique()
    )

    missing_dates = complete_calendar.difference(
        observed_dates
    )

    summary = {
        "start_date": dataframe["DATE"].min(),
        "end_date": dataframe["DATE"].max(),
        "number_of_rows": len(dataframe),
        "expected_number_of_days": len(complete_calendar),
        "missing_days": len(missing_dates),
        "minimum_temperature": float(
            dataframe["T"].min()
        ),
        "maximum_temperature": float(
            dataframe["T"].max()
        ),
        "mean_temperature": float(
            dataframe["T"].mean()
        ),
        "standard_deviation": float(
            dataframe["T"].std()
        ),
    }

    return summary


# ============================================================
# DÉCOUPAGE CHRONOLOGIQUE
# ============================================================

def chronological_split(
    dataframe: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Découpe chronologiquement les données.

    Aucun mélange aléatoire n'est effectué afin d'éviter
    une fuite d'information provenant du futur.
    """

    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError(
            "Les ratios doivent être strictement positifs."
        )

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "La somme train_ratio + validation_ratio "
            "doit être inférieure à 1."
        )

    dataframe = (
        dataframe
        .sort_values("DATE")
        .reset_index(drop=True)
    )

    number_of_rows = len(dataframe)

    train_end = int(
        number_of_rows * train_ratio
    )

    validation_end = int(
        number_of_rows
        * (train_ratio + validation_ratio)
    )

    train_dataframe = dataframe.iloc[
        :train_end
    ].copy()

    validation_dataframe = dataframe.iloc[
        train_end:validation_end
    ].copy()

    test_dataframe = dataframe.iloc[
        validation_end:
    ].copy()

    return (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    )


# ============================================================
# NORMALISATION
# ============================================================

def standardize_splits(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    target_column: str = "T",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    float,
    float,
]:
    """
    Normalise les données avec les statistiques
    du jeu d'entraînement uniquement.
    """

    train_mean = float(
        train_dataframe[target_column].mean()
    )

    train_std = float(
        train_dataframe[target_column].std()
    )

    if train_std == 0:
        raise ValueError(
            "L'écart-type du jeu d'entraînement est nul."
        )

    normalized_splits = []

    for dataframe in (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ):
        normalized = dataframe.copy()

        normalized[target_column] = (
            normalized[target_column] - train_mean
        ) / train_std

        normalized_splits.append(normalized)

    return (
        normalized_splits[0],
        normalized_splits[1],
        normalized_splits[2],
        train_mean,
        train_std,
    )


# ============================================================
# CRÉATION DES SÉQUENCES
# ============================================================

def create_sequences(
    dataframe: pd.DataFrame,
    config: SequenceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crée les fenêtres temporelles X et les cibles y.

    Exemple :
    30 températures passées → température suivante.
    """

    values = dataframe[
        config.target_column
    ].to_numpy(dtype=np.float32)

    inputs = []
    targets = []

    last_start_index = (
        len(values)
        - config.input_length
        - config.forecast_horizon
        + 1
    )

    for start_index in range(last_start_index):
        input_end = (
            start_index
            + config.input_length
        )

        target_index = (
            input_end
            + config.forecast_horizon
            - 1
        )

        inputs.append(
            values[start_index:input_end]
        )

        targets.append(
            values[target_index]
        )

    x_array = np.asarray(
        inputs,
        dtype=np.float32,
    )

    y_array = np.asarray(
        targets,
        dtype=np.float32,
    )

    # Forme attendue pour PyTorch :
    # nombre de séquences × nombre de variables × longueur
    x_array = np.expand_dims(
        x_array,
        axis=1,
    )

    return x_array, y_array