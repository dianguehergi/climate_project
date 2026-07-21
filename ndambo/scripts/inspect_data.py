from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# IMPORT DU DOSSIER SRC
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    DAILY_DATA_PATH,
    MONTHLY_DATA_PATH,
    LOG_DIR,
    create_output_directories,
    validate_data_paths,
)


# ============================================================
# PARAMÈTRES
# ============================================================

MAX_ROWS = 100_000

DATE_KEYWORDS = (
    "date",
    "time",
    "jour",
    "day",
    "mois",
    "month",
    "annee",
    "année",
    "year",
)

SPATIAL_KEYWORDS = (
    "lat",
    "latitude",
    "lon",
    "longitude",
    "x",
    "y",
    "maille",
    "grid",
    "point",
    "station",
    "altitude",
)

CLIMATE_KEYWORDS = (
    "temperature",
    "temp",
    "tmin",
    "tmax",
    "tmoy",
    "precipitation",
    "precip",
    "rain",
    "pluie",
    "humidity",
    "humidite",
    "humidité",
    "wind",
    "vent",
    "radiation",
)


# ============================================================
# CHARGEMENT DES CSV
# ============================================================

def detect_separator(file_path: Path) -> str:
    """Détecte le séparateur utilisé dans le fichier CSV."""

    with file_path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:
        first_line = file.readline()

    candidates = {
        ";": first_line.count(";"),
        ",": first_line.count(","),
        "\t": first_line.count("\t"),
        "|": first_line.count("|"),
    }

    return max(candidates, key=candidates.get)


def load_csv_sample(
    file_path: Path,
    max_rows: int = MAX_ROWS,
) -> tuple[pd.DataFrame, str]:
    """Charge un échantillon du CSV."""

    separator = detect_separator(file_path)

    read_options = {
        "sep": separator,
        "nrows": max_rows,
        "low_memory": False,
    }

    try:
        dataframe = pd.read_csv(
            file_path,
            encoding="utf-8",
            **read_options,
        )
    except UnicodeDecodeError:
        dataframe = pd.read_csv(
            file_path,
            encoding="latin-1",
            **read_options,
        )

    return dataframe, separator


# ============================================================
# ANALYSE DES COLONNES
# ============================================================

def find_columns(
    dataframe: pd.DataFrame,
    keywords: tuple[str, ...],
) -> list[str]:
    """Recherche les colonnes correspondant à des mots-clés."""

    detected_columns = []

    for column in dataframe.columns:
        normalized_name = str(column).strip().lower()

        if any(
            keyword in normalized_name
            for keyword in keywords
        ):
            detected_columns.append(str(column))

    return detected_columns


def analyze_date_column(
    dataframe: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    """Essaie de convertir une colonne en date."""

    converted = pd.to_datetime(
        dataframe[column],
        errors="coerce",
    )

    valid_dates = converted.dropna()

    if valid_dates.empty:
        return {
            "column": column,
            "valid_dates": 0,
            "valid_percentage": 0.0,
            "minimum": None,
            "maximum": None,
        }

    return {
        "column": column,
        "valid_dates": int(valid_dates.shape[0]),
        "valid_percentage": round(
            valid_dates.shape[0] / dataframe.shape[0] * 100,
            2,
        ),
        "minimum": str(valid_dates.min()),
        "maximum": str(valid_dates.max()),
    }


def inspect_dataset(
    dataset_name: str,
    file_path: Path,
) -> dict[str, Any]:
    """Analyse un fichier de données climatiques."""

    dataframe, separator = load_csv_sample(file_path)

    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    date_columns = find_columns(
        dataframe,
        DATE_KEYWORDS,
    )

    spatial_columns = find_columns(
        dataframe,
        SPATIAL_KEYWORDS,
    )

    climate_columns = find_columns(
        dataframe,
        CLIMATE_KEYWORDS,
    )

    numeric_columns = dataframe.select_dtypes(
        include="number",
    ).columns.tolist()

    missing_values = (
        dataframe.isna()
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )

    unique_values = {
        str(column): int(dataframe[column].nunique(dropna=True))
        for column in dataframe.columns
    }

    date_analysis = [
        analyze_date_column(dataframe, column)
        for column in date_columns
    ]

    report = {
        "dataset": dataset_name,
        "file": str(file_path),
        "file_size_mb": round(file_size_mb, 2),
        "separator": separator,
        "sample_rows_loaded": int(dataframe.shape[0]),
        "number_of_columns": int(dataframe.shape[1]),
        "columns": [str(column) for column in dataframe.columns],
        "data_types": {
            str(column): str(dtype)
            for column, dtype in dataframe.dtypes.items()
        },
        "numeric_columns": [
            str(column)
            for column in numeric_columns
        ],
        "date_columns_detected": date_columns,
        "spatial_columns_detected": spatial_columns,
        "climate_columns_detected": climate_columns,
        "date_analysis": date_analysis,
        "missing_percentage": {
            str(column): float(value)
            for column, value in missing_values.items()
        },
        "unique_values": unique_values,
        "duplicates_in_sample": int(dataframe.duplicated().sum()),
        "first_rows": dataframe.head(5).astype(str).to_dict(
            orient="records"
        ),
    }

    print("\n" + "=" * 90)
    print(dataset_name.upper())
    print("=" * 90)

    print(f"Fichier : {file_path}")
    print(f"Taille : {file_size_mb:.2f} Mo")
    print(f"Séparateur : {repr(separator)}")
    print(f"Lignes chargées : {dataframe.shape[0]:,}")
    print(f"Nombre de colonnes : {dataframe.shape[1]}")

    print("\nCOLONNES")
    for column in dataframe.columns:
        print(f"- {column} [{dataframe[column].dtype}]")

    print("\nCOLONNES DE DATE POTENTIELLES")
    print(date_columns or "Aucune colonne détectée")

    print("\nCOLONNES SPATIALES POTENTIELLES")
    print(spatial_columns or "Aucune colonne détectée")

    print("\nVARIABLES CLIMATIQUES POTENTIELLES")
    print(climate_columns or "Aucune colonne détectée")

    print("\nANALYSE DES DATES")
    for result in date_analysis:
        print(
            f"- {result['column']} : "
            f"{result['minimum']} → {result['maximum']} "
            f"({result['valid_percentage']} % valides)"
        )

    print("\nVALEURS MANQUANTES LES PLUS IMPORTANTES")
    print(missing_values.head(15).to_string())

    print("\nNOMBRE DE VALEURS UNIQUES")
    for column, count in unique_values.items():
        print(f"- {column}: {count:,}")

    print("\nPREMIÈRES LIGNES")
    print(dataframe.head().to_string())

    return report


def main() -> None:
    create_output_directories()
    validate_data_paths()

    reports = {
        "daily": inspect_dataset(
            dataset_name="Données quotidiennes",
            file_path=DAILY_DATA_PATH,
        ),
        "monthly": inspect_dataset(
            dataset_name="Données mensuelles",
            file_path=MONTHLY_DATA_PATH,
        ),
    }

    output_file = LOG_DIR / "inspection_donnees.json"

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            reports,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\n" + "=" * 90)
    print("INSPECTION TERMINÉE")
    print("=" * 90)
    print(f"Rapport enregistré dans : {output_file}")


if __name__ == "__main__":
    main()