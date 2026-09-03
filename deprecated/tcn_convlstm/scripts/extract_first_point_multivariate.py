from __future__ import annotations

import sys
import time
from pathlib import Path

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
    FIRST_POINT_MULTIVARIATE_PATH,
    create_output_directories,
)


# ============================================================
# PARAMÈTRES
# ============================================================

CHUNK_SIZE = 1_000_000

SELECTED_COLUMNS = [
    "LAMBX",
    "LAMBY",
    "DATE",
    "T",
    "PRELIQ",
    "PRENEI",
    "FF",
    "Q",
    "DLI",
    "SSI",
    "HU",
    "EVAP",
    "ETP",
    "PE",
    "SWI",
    "SSWI_10J",
    "DRAINC",
    "RUNC",
    "WG_RACINE",
    "WGI_RACINE",
    "TINF_H",
    "TSUP_H",
]

DTYPES = {
    "LAMBX": "int32",
    "LAMBY": "int32",
    "DATE": "int32",
    "T": "float32",
    "PRELIQ": "float32",
    "PRENEI": "float32",
    "FF": "float32",
    "Q": "float32",
    "DLI": "float32",
    "SSI": "float32",
    "HU": "float32",
    "EVAP": "float32",
    "ETP": "float32",
    "PE": "float32",
    "SWI": "float32",
    "SSWI_10J": "float32",
    "DRAINC": "float32",
    "RUNC": "float32",
    "WG_RACINE": "float32",
    "WGI_RACINE": "float32",
    "TINF_H": "float32",
    "TSUP_H": "float32",
}


# ============================================================
# IDENTIFICATION DU PREMIER POINT
# ============================================================

def get_first_grid_point(
    file_path: Path,
) -> tuple[int, int]:
    """
    Lit uniquement la première ligne pour récupérer
    les coordonnées du premier point SAFRAN.
    """

    first_row = pd.read_csv(
        file_path,
        usecols=["LAMBX", "LAMBY"],
        nrows=1,
    )

    if first_row.empty:
        raise ValueError(
            "Le fichier quotidien ne contient aucune ligne."
        )

    lambx = int(first_row.iloc[0]["LAMBX"])
    lamby = int(first_row.iloc[0]["LAMBY"])

    return lambx, lamby


# ============================================================
# EXTRACTION PAR MORCEAUX
# ============================================================

def extract_grid_point(
    source_path: Path,
    output_path: Path,
    target_lambx: int,
    target_lamby: int,
) -> pd.DataFrame:
    """
    Parcourt le fichier de 20 Go par morceaux et conserve
    uniquement le point géographique demandé.
    """

    selected_chunks: list[pd.DataFrame] = []

    total_rows_read = 0
    total_rows_selected = 0

    start_time = time.perf_counter()

    reader = pd.read_csv(
        source_path,
        usecols=SELECTED_COLUMNS,
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
            (chunk["LAMBX"] == target_lambx)
            & (chunk["LAMBY"] == target_lamby)
        )

        selected = chunk.loc[mask].copy()

        if not selected.empty:
            selected_chunks.append(selected)
            total_rows_selected += len(selected)

        elapsed = time.perf_counter() - start_time

        print(
            f"Bloc {chunk_number:04d} | "
            f"lignes lues={total_rows_read:,} | "
            f"lignes trouvées={total_rows_selected:,} | "
            f"durée={elapsed:.1f}s",
            flush=True,
        )

    if not selected_chunks:
        raise ValueError(
            "Aucune donnée trouvée pour le point "
            f"LAMBX={target_lambx}, LAMBY={target_lamby}."
        )

    dataframe = pd.concat(
        selected_chunks,
        ignore_index=True,
    )

    dataframe["DATE"] = pd.to_datetime(
        dataframe["DATE"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    dataframe = (
        dataframe
        .dropna(subset=["DATE", "T"])
        .sort_values("DATE")
        .drop_duplicates(
            subset=["DATE"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # Variables temporelles cycliques utiles au modèle
    dataframe["DAY_OF_YEAR"] = (
        dataframe["DATE"].dt.dayofyear.astype("int16")
    )

    dataframe["MONTH"] = (
        dataframe["DATE"].dt.month.astype("int8")
    )

    # Précipitations totales du jour
    dataframe["PRETOT"] = (
        dataframe["PRELIQ"]
        + dataframe["PRENEI"]
    ).astype("float32")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return dataframe


# ============================================================
# CONTRÔLES
# ============================================================

def validate_extracted_data(
    dataframe: pd.DataFrame,
) -> None:
    if dataframe.empty:
        raise ValueError(
            "Le fichier extrait est vide."
        )

    expected_calendar = pd.date_range(
        dataframe["DATE"].min(),
        dataframe["DATE"].max(),
        freq="D",
    )

    observed_dates = pd.DatetimeIndex(
        dataframe["DATE"].unique()
    )

    missing_dates = expected_calendar.difference(
        observed_dates
    )

    print("\n" + "=" * 80)
    print("CONTRÔLE DU POINT MULTIVARIÉ")
    print("=" * 80)

    print(
        f"Coordonnées : "
        f"LAMBX={int(dataframe['LAMBX'].iloc[0])}, "
        f"LAMBY={int(dataframe['LAMBY'].iloc[0])}"
    )

    print(
        f"Période : "
        f"{dataframe['DATE'].min().date()} → "
        f"{dataframe['DATE'].max().date()}"
    )

    print(
        f"Nombre de lignes : {len(dataframe):,}"
    )

    print(
        f"Nombre de colonnes : {len(dataframe.columns)}"
    )

    print(
        f"Jours attendus : {len(expected_calendar):,}"
    )

    print(
        f"Jours manquants : {len(missing_dates):,}"
    )

    print(
        f"Doublons de dates : "
        f"{dataframe['DATE'].duplicated().sum():,}"
    )

    print("\nValeurs manquantes :")

    missing = (
        dataframe.isna()
        .sum()
        .sort_values(ascending=False)
    )

    print(
        missing[missing > 0].to_string()
        if (missing > 0).any()
        else "Aucune valeur manquante."
    )

    print("\nPremières lignes :")
    print(
        dataframe.head().to_string(index=False)
    )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main() -> None:
    create_output_directories()

    if not DAILY_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {DAILY_DATA_PATH}"
        )

    target_lambx, target_lamby = get_first_grid_point(
        DAILY_DATA_PATH
    )

    print("=" * 80)
    print("EXTRACTION DU PREMIER POINT SAFRAN")
    print("=" * 80)

    print(f"Source : {DAILY_DATA_PATH}")
    print(
        f"Point sélectionné : "
        f"LAMBX={target_lambx}, LAMBY={target_lamby}"
    )

    print(
        f"Destination : {FIRST_POINT_MULTIVARIATE_PATH}"
    )

    dataframe = extract_grid_point(
        source_path=DAILY_DATA_PATH,
        output_path=FIRST_POINT_MULTIVARIATE_PATH,
        target_lambx=target_lambx,
        target_lamby=target_lamby,
    )

    validate_extracted_data(
        dataframe
    )

    print("\nExtraction terminée.")
    print(
        f"Fichier créé : {FIRST_POINT_MULTIVARIATE_PATH}"
    )


if __name__ == "__main__":
    main()