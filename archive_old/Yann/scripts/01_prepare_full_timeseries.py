"""
============================================================
Préparation de la série temporelle mensuelle complète
============================================================

Projet :
    Prévision climatique SAFRAN / SIM avec XGBoost et ADAMONT

Entrées :
    - SAFRAN mensuel nettoyé : 1960-1999
    - SIM mensuel nettoyé    : 2000-2025

Sortie :
    - Série mensuelle continue : 1960-2025

Important :
    Ce script ne réalise pas l'entraînement du modèle.
    Il rassemble les deux périodes afin de permettre ensuite
    le calcul correct des lags et moyennes mobiles à la
    frontière entre décembre 1999 et janvier 2000.
"""

from pathlib import Path
import sys

import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path("/home/lab-c0de80861d/climate_project")

SAFRAN_FILE = (
    PROJECT_DIR
    / "archive_old"
    / "Yann"
    / "data"
    / "processed"
    / "safran_mens_clean.csv"
)

SIM_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "datasets"
    / "SIM_mensuelle_2000_2025_cleaned.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "merged"

OUTPUT_FILE = OUTPUT_DIR / "monthly_1960_2025.csv"

GRID_COLUMNS = ["LAMBX", "LAMBY"]
KEY_COLUMNS = ["LAMBX", "LAMBY", "DATE"]


# ============================================================
# Fonctions
# ============================================================

def print_separator(title: str) -> None:
    """Affiche un titre lisible dans le terminal."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_file_exists(file_path: Path, label: str) -> None:
    """Vérifie qu'un fichier d'entrée existe."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"{label} introuvable :\n{file_path}"
        )


def parse_monthly_dates(date_series: pd.Series) -> pd.Series:
    """
    Convertit une série de dates mensuelles en datetime.

    Formats acceptés, par exemple :
        196001
        196001.0
        '196001'
        '1960-01-01'
        '1960-01'
    """

    date_text = (
        date_series
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    # Premier essai : format AAAAMM.
    parsed_dates = pd.to_datetime(
        date_text,
        format="%Y%m",
        errors="coerce",
    )

    # Deuxième essai pour les dates déjà écrites avec des tirets.
    invalid_mask = parsed_dates.isna()

    if invalid_mask.any():
        parsed_dates.loc[invalid_mask] = pd.to_datetime(
            date_text.loc[invalid_mask],
            errors="coerce",
        )

    return parsed_dates


def load_dataset(file_path: Path, dataset_name: str) -> pd.DataFrame:
    """Charge un CSV et affiche les informations principales."""

    print(f"\nChargement de {dataset_name}...")
    print(f"Chemin : {file_path}")

    dataframe = pd.read_csv(
        file_path,
        low_memory=False,
    )

    print(f"Dimensions : {dataframe.shape}")
    print(
        "Mémoire utilisée : "
        f"{dataframe.memory_usage(deep=True).sum() / 1024**3:.2f} Go"
    )

    return dataframe


def validate_required_columns(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Vérifie la présence des colonnes essentielles."""

    required_columns = {"LAMBX", "LAMBY", "DATE", "T"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes obligatoires absentes dans {dataset_name} : "
            f"{sorted(missing_columns)}"
        )


def validate_columns(
    safran: pd.DataFrame,
    sim: pd.DataFrame,
) -> None:
    """Vérifie que les deux fichiers ont les mêmes colonnes."""

    safran_columns = list(safran.columns)
    sim_columns = list(sim.columns)

    missing_in_sim = sorted(
        set(safran_columns) - set(sim_columns)
    )

    missing_in_safran = sorted(
        set(sim_columns) - set(safran_columns)
    )

    if missing_in_sim or missing_in_safran:
        raise ValueError(
            "Les structures des deux datasets sont différentes.\n\n"
            f"Colonnes absentes dans SIM : {missing_in_sim}\n"
            f"Colonnes absentes dans SAFRAN : {missing_in_safran}"
        )

    # Même ordre de colonnes.
    if safran_columns != sim_columns:
        print(
            "\nLes colonnes sont identiques, mais leur ordre diffère."
        )
        print("Les colonnes SIM seront remises dans l'ordre SAFRAN.")

    print("\nCompatibilité des colonnes : validée")


def prepare_dates(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Convertit et contrôle la colonne DATE."""

    dataframe = dataframe.copy()

    dataframe["DATE"] = parse_monthly_dates(
        dataframe["DATE"]
    )

    invalid_dates = int(dataframe["DATE"].isna().sum())

    if invalid_dates > 0:
        raise ValueError(
            f"{invalid_dates:,} date(s) invalide(s) "
            f"dans {dataset_name}."
        )

    # Normalisation au premier jour du mois.
    dataframe["DATE"] = (
        dataframe["DATE"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    print(
        f"Période {dataset_name} : "
        f"{dataframe['DATE'].min():%Y-%m} à "
        f"{dataframe['DATE'].max():%Y-%m}"
    )

    return dataframe


def check_duplicates(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Recherche les doublons pour un point et un mois donnés."""

    duplicate_count = int(
        dataframe.duplicated(
            subset=KEY_COLUMNS,
            keep=False,
        ).sum()
    )

    print(
        f"Doublons point-date dans {dataset_name} : "
        f"{duplicate_count:,}"
    )

    if duplicate_count > 0:
        raise ValueError(
            f"{dataset_name} contient {duplicate_count:,} lignes "
            "appartenant à des doublons point-date. "
            "La fusion est interrompue pour éviter une suppression "
            "automatique non contrôlée."
        )


def check_expected_periods(
    safran: pd.DataFrame,
    sim: pd.DataFrame,
) -> None:
    """Contrôle les périodes attendues du protocole scientifique."""

    safran_max = safran["DATE"].max()
    sim_min = sim["DATE"].min()

    expected_safran_end = pd.Timestamp("1999-12-01")
    expected_sim_start = pd.Timestamp("2000-01-01")

    if safran_max != expected_safran_end:
        print(
            "\nAVERTISSEMENT : la dernière date SAFRAN "
            f"est {safran_max:%Y-%m}, alors que 1999-12 "
            "était attendu."
        )

    if sim_min != expected_sim_start:
        print(
            "\nAVERTISSEMENT : la première date SIM "
            f"est {sim_min:%Y-%m}, alors que 2000-01 "
            "était attendu."
        )

    if sim_min <= safran_max:
        raise ValueError(
            "Les périodes SAFRAN et SIM se chevauchent.\n"
            f"Dernière date SAFRAN : {safran_max:%Y-%m}\n"
            f"Première date SIM    : {sim_min:%Y-%m}\n"
            "Il faut examiner ce chevauchement avant la fusion."
        )

    expected_next_month = (
        safran_max.to_period("M") + 1
    ).to_timestamp()

    if sim_min != expected_next_month:
        raise ValueError(
            "Une rupture temporelle existe entre SAFRAN et SIM.\n"
            f"Dernière date SAFRAN : {safran_max:%Y-%m}\n"
            f"Première date SIM    : {sim_min:%Y-%m}\n"
            f"Date attendue        : {expected_next_month:%Y-%m}"
        )

    print(
        "\nContinuité temporelle globale : "
        f"{safran_max:%Y-%m} → {sim_min:%Y-%m} validée"
    )


def compare_grid_points(
    safran: pd.DataFrame,
    sim: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Compare les points de grille présents dans les deux datasets.

    Les différences sont sauvegardées pour inspection, mais ne
    bloquent pas nécessairement la fusion.
    """

    safran_points = (
        safran[GRID_COLUMNS]
        .drop_duplicates()
        .sort_values(GRID_COLUMNS)
        .reset_index(drop=True)
    )

    sim_points = (
        sim[GRID_COLUMNS]
        .drop_duplicates()
        .sort_values(GRID_COLUMNS)
        .reset_index(drop=True)
    )

    print(
        f"\nPoints de grille SAFRAN : "
        f"{len(safran_points):,}"
    )

    print(
        f"Points de grille SIM    : "
        f"{len(sim_points):,}"
    )

    comparison = safran_points.merge(
        sim_points,
        on=GRID_COLUMNS,
        how="outer",
        indicator=True,
    )

    only_safran = comparison[
        comparison["_merge"] == "left_only"
    ][GRID_COLUMNS]

    only_sim = comparison[
        comparison["_merge"] == "right_only"
    ][GRID_COLUMNS]

    print(
        "Points présents uniquement dans SAFRAN : "
        f"{len(only_safran):,}"
    )

    print(
        "Points présents uniquement dans SIM    : "
        f"{len(only_sim):,}"
    )

    if not only_safran.empty:
        output_file = (
            output_dir
            / "grid_points_only_in_safran.csv"
        )

        only_safran.to_csv(output_file, index=False)

        print(
            "Liste sauvegardée dans :\n"
            f"{output_file}"
        )

    if not only_sim.empty:
        output_file = (
            output_dir
            / "grid_points_only_in_sim.csv"
        )

        only_sim.to_csv(output_file, index=False)

        print(
            "Liste sauvegardée dans :\n"
            f"{output_file}"
        )


def check_temporal_gaps(
    dataframe: pd.DataFrame,
    output_dir: Path,
) -> int:
    """
    Recherche les ruptures mensuelles à l'intérieur de chaque
    point de grille.
    """

    print("\nRecherche des ruptures temporelles...")

    periods = dataframe["DATE"].dt.to_period("M")

    previous_periods = (
        periods.groupby(
            [
                dataframe["LAMBX"],
                dataframe["LAMBY"],
            ]
        )
        .shift(1)
    )

    month_difference = (
        periods.astype("int64")
        - previous_periods.astype("int64")
    )

    gap_mask = (
        previous_periods.notna()
        & (month_difference != 1)
    )

    gap_count = int(gap_mask.sum())

    print(
        f"Ruptures temporelles détectées : {gap_count:,}"
    )

    if gap_count > 0:
        gaps = dataframe.loc[
            gap_mask,
            KEY_COLUMNS,
        ].copy()

        gaps["PREVIOUS_DATE"] = (
            previous_periods.loc[gap_mask]
            .dt.to_timestamp()
            .values
        )

        gaps["MONTH_DIFFERENCE"] = (
            month_difference.loc[gap_mask]
            .astype(int)
            .values
        )

        gaps_file = (
            output_dir
            / "monthly_1960_2025_temporal_gaps.csv"
        )

        gaps.to_csv(
            gaps_file,
            index=False,
            date_format="%Y-%m-%d",
        )

        print(
            "Détails des ruptures enregistrés dans :\n"
            f"{gaps_file}"
        )

    return gap_count


# ============================================================
# Programme principal
# ============================================================

def main() -> None:
    print_separator(
        "PRÉPARATION DE LA SÉRIE MENSUELLE 1960-2025"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Vérification des chemins
    # --------------------------------------------------------

    check_file_exists(
        SAFRAN_FILE,
        "Fichier SAFRAN",
    )

    check_file_exists(
        SIM_FILE,
        "Fichier SIM",
    )

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    safran = load_dataset(
        SAFRAN_FILE,
        "SAFRAN 1960-1999",
    )

    sim = load_dataset(
        SIM_FILE,
        "SIM 2000-2025",
    )

    # --------------------------------------------------------
    # Validation des structures
    # --------------------------------------------------------

    validate_required_columns(
        safran,
        "SAFRAN",
    )

    validate_required_columns(
        sim,
        "SIM",
    )

    validate_columns(
        safran,
        sim,
    )

    # Le dataset SIM est remis dans l'ordre des colonnes SAFRAN.
    sim = sim[safran.columns]

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    print_separator("CONTRÔLE DES DATES")

    safran = prepare_dates(
        safran,
        "SAFRAN",
    )

    sim = prepare_dates(
        sim,
        "SIM",
    )

    check_duplicates(
        safran,
        "SAFRAN",
    )

    check_duplicates(
        sim,
        "SIM",
    )

    check_expected_periods(
        safran,
        sim,
    )

    # --------------------------------------------------------
    # Points de grille
    # --------------------------------------------------------

    print_separator("CONTRÔLE DES POINTS DE GRILLE")

    compare_grid_points(
        safran,
        sim,
        OUTPUT_DIR,
    )

    # --------------------------------------------------------
    # Concaténation temporaire des périodes
    # --------------------------------------------------------

    print_separator("CONCATÉNATION DES DEUX PÉRIODES")

    full_timeseries = pd.concat(
        [safran, sim],
        ignore_index=True,
        sort=False,
    )

    full_timeseries = (
        full_timeseries
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )

    print(
        "Dimensions après concaténation : "
        f"{full_timeseries.shape}"
    )

    print(
        "Mémoire utilisée après concaténation : "
        f"{full_timeseries.memory_usage(deep=True).sum() / 1024**3:.2f} Go"
    )

    # Libération des deux objets devenus inutiles.
    del safran
    del sim

    # --------------------------------------------------------
    # Contrôles finaux
    # --------------------------------------------------------

    final_duplicate_count = int(
        full_timeseries.duplicated(
            subset=KEY_COLUMNS,
            keep=False,
        ).sum()
    )

    if final_duplicate_count > 0:
        raise ValueError(
            "Des doublons ont été détectés après concaténation : "
            f"{final_duplicate_count:,} lignes."
        )

    min_date = full_timeseries["DATE"].min()
    max_date = full_timeseries["DATE"].max()

    number_of_points = (
        full_timeseries[GRID_COLUMNS]
        .drop_duplicates()
        .shape[0]
    )

    gap_count = check_temporal_gaps(
        full_timeseries,
        OUTPUT_DIR,
    )

    missing_values = (
        full_timeseries
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing_values = missing_values[
        missing_values > 0
    ]

    print_separator("RÉSUMÉ AVANT SAUVEGARDE")

    print(
        f"Période finale       : "
        f"{min_date:%Y-%m} à {max_date:%Y-%m}"
    )

    print(
        f"Nombre total de lignes : "
        f"{len(full_timeseries):,}"
    )

    print(
        f"Nombre de colonnes     : "
        f"{len(full_timeseries.columns):,}"
    )

    print(
        f"Nombre de points       : "
        f"{number_of_points:,}"
    )

    print(
        f"Ruptures temporelles   : "
        f"{gap_count:,}"
    )

    print("\nValeurs manquantes :")

    if missing_values.empty:
        print("Aucune valeur manquante détectée.")
    else:
        print(missing_values.to_string())

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    print_separator("SAUVEGARDE")

    print(
        "Écriture du fichier :\n"
        f"{OUTPUT_FILE}"
    )

    full_timeseries.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    print_separator("PRÉPARATION TERMINÉE AVEC SUCCÈS")

    print(
        f"Fichier produit :\n{OUTPUT_FILE}"
    )

    print(
        f"\nDimensions finales : "
        f"{full_timeseries.shape}"
    )

    print(
        f"Période finale : "
        f"{min_date:%Y-%m} à {max_date:%Y-%m}"
    )

    print(
        "\nCe fichier servira à construire les variables "
        "temporelles avant la séparation :"
    )

    print("Train : 1960-1999")
    print("Test  : 2000-2025")


if __name__ == "__main__":
    try:
        main()

    except (FileNotFoundError, ValueError) as error:
        print_separator("ERREUR")

        print(error)

        sys.exit(1)

    except MemoryError:
        print_separator("ERREUR MÉMOIRE")

        print(
            "La mémoire disponible est insuffisante pour charger "
            "et concaténer les deux fichiers CSV."
        )

        print(
            "Il faudra alors utiliser une préparation par blocs "
            "ou un format Parquet."
        )

        sys.exit(1)