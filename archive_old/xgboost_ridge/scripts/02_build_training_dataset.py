"""
============================================================
Construction des datasets d'entraînement et de test
============================================================

Entrée :
    data/merged/monthly_1960_2025.csv

Sorties :
    data/model/train_1960_1999.csv
    data/model/test_2000_2025.csv

Protocole :
    Train : janvier 1960 à décembre 1999
    Test  : janvier 2000 à décembre 2025

Les variables temporelles sont calculées sur toute la série
1960-2025 afin de conserver la continuité entre décembre 1999
et janvier 2000.

Aucune observation de 2000-2025 n'est utilisée pour entraîner
le modèle.
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path("/home/lab-c0de80861d/climate_project")

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "merged"
    / "monthly_1960_2025.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "model"

TRAIN_FILE = OUTPUT_DIR / "train_1960_1999.csv"
TEST_FILE = OUTPUT_DIR / "test_2000_2025.csv"

GRID_COLUMNS = ["LAMBX", "LAMBY"]
KEY_COLUMNS = ["LAMBX", "LAMBY", "DATE"]

TRAIN_END = pd.Timestamp("1999-12-01")
TEST_START = pd.Timestamp("2000-01-01")

TARGET_COLUMN = "T"

TEMPORAL_FEATURES = [
    "T_lag_1",
    "T_lag_2",
    "T_lag_12",
    "T_roll_3",
    "T_roll_12",
    "T_change_lag_1",
    "month_sin",
    "month_cos",
]


# ============================================================
# Fonctions utilitaires
# ============================================================

def print_separator(title: str) -> None:
    """Affiche un titre lisible dans le terminal."""

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def format_duration(seconds: float) -> str:
    """Formate une durée en minutes et secondes."""

    minutes, remaining_seconds = divmod(seconds, 60)

    return (
        f"{int(minutes)} min "
        f"{remaining_seconds:.1f} s"
    )


def check_input_file() -> None:
    """Vérifie que le fichier d'entrée existe."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Fichier d'entrée introuvable :\n"
            f"{INPUT_FILE}\n\n"
            "Exécute d'abord 01_prepare_full_timeseries.py."
        )


def load_dataset() -> pd.DataFrame:
    """Charge la série mensuelle complète."""

    print(f"Chargement :\n{INPUT_FILE}")

    start_time = time.perf_counter()

    dataframe = pd.read_csv(
        INPUT_FILE,
        parse_dates=["DATE"],
        low_memory=False,
    )

    elapsed_time = time.perf_counter() - start_time

    print(f"\nDimensions initiales : {dataframe.shape}")
    print(
        "Mémoire utilisée : "
        f"{dataframe.memory_usage(deep=True).sum() / 1024**3:.2f} Go"
    )
    print(
        "Temps de chargement : "
        f"{format_duration(elapsed_time)}"
    )

    return dataframe


def validate_dataset(dataframe: pd.DataFrame) -> None:
    """Effectue les contrôles indispensables avant calcul."""

    required_columns = {
        "LAMBX",
        "LAMBY",
        "DATE",
        TARGET_COLUMN,
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Colonnes obligatoires absentes : "
            f"{sorted(missing_columns)}"
        )

    invalid_dates = int(dataframe["DATE"].isna().sum())

    if invalid_dates > 0:
        raise ValueError(
            f"{invalid_dates:,} date(s) invalide(s) détectée(s)."
        )

    missing_target = int(
        dataframe[TARGET_COLUMN].isna().sum()
    )

    if missing_target > 0:
        raise ValueError(
            f"La cible {TARGET_COLUMN} contient "
            f"{missing_target:,} valeur(s) manquante(s)."
        )

    duplicate_count = int(
        dataframe.duplicated(
            subset=KEY_COLUMNS,
            keep=False,
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"{duplicate_count:,} ligne(s) appartiennent à "
            "des doublons point-date."
        )

    print(
        f"Période détectée : "
        f"{dataframe['DATE'].min():%Y-%m} à "
        f"{dataframe['DATE'].max():%Y-%m}"
    )

    print(
        "Nombre de points de grille : "
        f"{dataframe[GRID_COLUMNS].drop_duplicates().shape[0]:,}"
    )

    print("Contrôles initiaux : validés")


def sort_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Trie les observations par point puis par date."""

    print("\nTri par LAMBX, LAMBY et DATE...")

    dataframe = (
        dataframe
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )

    return dataframe


def check_temporal_continuity(
    dataframe: pd.DataFrame,
) -> None:
    """
    Vérifie qu'il n'existe pas de rupture mensuelle.

    Cette vérification garantit que shift(1) correspond bien
    au mois précédent et shift(12) au même mois de l'année
    précédente.
    """

    periods = dataframe["DATE"].dt.to_period("M")

    previous_periods = (
        periods.groupby(
            [
                dataframe["LAMBX"],
                dataframe["LAMBY"],
            ],
            sort=False,
        )
        .shift(1)
    )

    differences = (
        periods.astype("int64")
        - previous_periods.astype("int64")
    )

    invalid_gaps = (
        previous_periods.notna()
        & differences.ne(1)
    )

    gap_count = int(invalid_gaps.sum())

    if gap_count > 0:
        raise ValueError(
            f"{gap_count:,} rupture(s) temporelle(s) détectée(s). "
            "Les lags ne peuvent pas être construits en sécurité."
        )

    print("Continuité mensuelle par point : validée")


def build_temporal_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construit les variables temporelles sans utiliser la
    température du mois courant dans les prédicteurs.
    """

    print_separator("CRÉATION DES VARIABLES TEMPORELLES")

    start_time = time.perf_counter()

    group_keys = [
        dataframe["LAMBX"],
        dataframe["LAMBY"],
    ]

    temperature_grouped = dataframe.groupby(
        GRID_COLUMNS,
        sort=False,
        observed=True,
    )[TARGET_COLUMN]

    # --------------------------------------------------------
    # Retards temporels
    # --------------------------------------------------------

    print("Calcul de T_lag_1...")

    dataframe["T_lag_1"] = (
        temperature_grouped
        .shift(1)
        .astype("float32")
    )

    print("Calcul de T_lag_2...")

    dataframe["T_lag_2"] = (
        temperature_grouped
        .shift(2)
        .astype("float32")
    )

    print("Calcul de T_lag_12...")

    dataframe["T_lag_12"] = (
        temperature_grouped
        .shift(12)
        .astype("float32")
    )

    # --------------------------------------------------------
    # Moyennes mobiles utilisant uniquement le passé
    # --------------------------------------------------------

    print("Calcul de T_roll_3...")

    lagged_temperature = dataframe["T_lag_1"]

    dataframe["T_roll_3"] = (
        lagged_temperature
        .groupby(
            group_keys,
            sort=False,
            observed=True,
        )
        .rolling(
            window=3,
            min_periods=3,
        )
        .mean()
        .reset_index(
            level=[0, 1],
            drop=True,
        )
        .astype("float32")
    )

    print("Calcul de T_roll_12...")

    dataframe["T_roll_12"] = (
        lagged_temperature
        .groupby(
            group_keys,
            sort=False,
            observed=True,
        )
        .rolling(
            window=12,
            min_periods=12,
        )
        .mean()
        .reset_index(
            level=[0, 1],
            drop=True,
        )
        .astype("float32")
    )

    # --------------------------------------------------------
    # Variation observée avant le mois à prédire
    # --------------------------------------------------------

    dataframe["T_change_lag_1"] = (
        dataframe["T_lag_1"]
        - dataframe["T_lag_2"]
    ).astype("float32")

    # --------------------------------------------------------
    # Encodage cyclique du mois
    # --------------------------------------------------------

    month = dataframe["DATE"].dt.month

    dataframe["month_sin"] = np.sin(
        2.0 * np.pi * month / 12.0
    ).astype("float32")

    dataframe["month_cos"] = np.cos(
        2.0 * np.pi * month / 12.0
    ).astype("float32")

    # Variables pratiques pour les analyses futures.
    dataframe["year"] = (
        dataframe["DATE"]
        .dt.year
        .astype("int16")
    )

    dataframe["month"] = (
        month.astype("int8")
    )

    elapsed_time = time.perf_counter() - start_time

    print(
        "\nVariables temporelles créées : "
        f"{TEMPORAL_FEATURES}"
    )

    print(
        "Temps de feature engineering : "
        f"{format_duration(elapsed_time)}"
    )

    return dataframe


def inspect_missing_temporal_features(
    dataframe: pd.DataFrame,
) -> None:
    """Affiche les NaN créés naturellement par les lags."""

    print_separator("VALEURS MANQUANTES DES FEATURES")

    missing_values = (
        dataframe[TEMPORAL_FEATURES]
        .isna()
        .sum()
    )

    print(missing_values.to_string())

    expected_points = (
        dataframe[GRID_COLUMNS]
        .drop_duplicates()
        .shape[0]
    )

    expected_missing_lag_12 = expected_points * 12
    actual_missing_lag_12 = int(
        dataframe["T_lag_12"].isna().sum()
    )

    print(
        "\nNaN attendus pour T_lag_12 : "
        f"{expected_missing_lag_12:,}"
    )

    print(
        "NaN réellement détectés      : "
        f"{actual_missing_lag_12:,}"
    )

    if actual_missing_lag_12 != expected_missing_lag_12:
        print(
            "\nAVERTISSEMENT : le nombre de NaN de T_lag_12 "
            "ne correspond pas exactement aux douze premiers "
            "mois de chaque point."
        )


def remove_unusable_rows(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Supprime uniquement les lignes dont les variables
    temporelles nécessaires ne peuvent pas être calculées.

    Les NaN des colonnes SPI/SSWI ne sont pas supprimés ici.
    """

    print_separator("SUPPRESSION DES LIGNES NON UTILISABLES")

    initial_count = len(dataframe)

    dataframe = (
        dataframe
        .dropna(
            subset=TEMPORAL_FEATURES
        )
        .reset_index(drop=True)
    )

    removed_count = initial_count - len(dataframe)

    print(
        f"Lignes initiales  : {initial_count:,}"
    )

    print(
        f"Lignes supprimées : {removed_count:,}"
    )

    print(
        f"Lignes conservées : {len(dataframe):,}"
    )

    print(
        "\nLes suppressions correspondent principalement aux "
        "douze premiers mois de chaque point de grille."
    )

    return dataframe


def split_train_test(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les périodes d'apprentissage et de test."""

    print_separator("SÉPARATION TRAIN / TEST")

    train = dataframe.loc[
        dataframe["DATE"] <= TRAIN_END
    ].copy()

    test = dataframe.loc[
        dataframe["DATE"] >= TEST_START
    ].copy()

    if train.empty:
        raise ValueError(
            "Le dataset d'entraînement est vide."
        )

    if test.empty:
        raise ValueError(
            "Le dataset de test est vide."
        )

    overlap_count = len(
        set(train.index).intersection(test.index)
    )

    if overlap_count > 0:
        raise ValueError(
            "Des lignes appartiennent simultanément "
            "au train et au test."
        )

    print(
        f"Train : {train.shape}"
    )

    print(
        f"Période train : "
        f"{train['DATE'].min():%Y-%m} à "
        f"{train['DATE'].max():%Y-%m}"
    )

    print(
        f"\nTest : {test.shape}"
    )

    print(
        f"Période test : "
        f"{test['DATE'].min():%Y-%m} à "
        f"{test['DATE'].max():%Y-%m}"
    )

    return train, test


def validate_boundary(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Vérifie que janvier 2000 dispose bien des informations
    provenant de 1999.
    """

    january_2000 = test.loc[
        test["DATE"] == pd.Timestamp("2000-01-01")
    ]

    if january_2000.empty:
        raise ValueError(
            "Aucune observation trouvée pour janvier 2000."
        )

    missing_boundary_features = (
        january_2000[
            [
                "T_lag_1",
                "T_lag_12",
                "T_roll_12",
            ]
        ]
        .isna()
        .sum()
    )

    if missing_boundary_features.sum() > 0:
        raise ValueError(
            "Certaines variables de janvier 2000 sont "
            "manquantes malgré la concaténation :\n"
            f"{missing_boundary_features.to_string()}"
        )

    if train["DATE"].max() != TRAIN_END:
        raise ValueError(
            "Le train ne se termine pas en décembre 1999."
        )

    if test["DATE"].min() != TEST_START:
        raise ValueError(
            "Le test ne commence pas en janvier 2000."
        )

    print(
        "\nFrontière temporelle validée :"
    )

    print(
        "Décembre 1999 appartient uniquement au train."
    )

    print(
        "Janvier 2000 appartient uniquement au test."
    )

    print(
        "Les lags de janvier 2000 utilisent correctement "
        "l'historique de 1999."
    )


def save_dataset(
    dataframe: pd.DataFrame,
    output_file: Path,
    dataset_name: str,
) -> None:
    """Enregistre un dataset au format CSV."""

    print(f"\nSauvegarde de {dataset_name} :")
    print(output_file)

    start_time = time.perf_counter()

    dataframe.to_csv(
        output_file,
        index=False,
        date_format="%Y-%m-%d",
    )

    elapsed_time = time.perf_counter() - start_time

    file_size_gb = output_file.stat().st_size / 1024**3

    print(
        f"Taille du fichier : {file_size_gb:.2f} Go"
    )

    print(
        "Temps d'écriture : "
        f"{format_duration(elapsed_time)}"
    )


# ============================================================
# Programme principal
# ============================================================

def main() -> None:
    total_start_time = time.perf_counter()

    print_separator(
        "CONSTRUCTION DES DATASETS TRAIN ET TEST"
    )

    check_input_file()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_dataset()

    validate_dataset(dataframe)

    dataframe = sort_dataset(dataframe)

    check_temporal_continuity(dataframe)

    dataframe = build_temporal_features(dataframe)

    inspect_missing_temporal_features(dataframe)

    dataframe = remove_unusable_rows(dataframe)

    train, test = split_train_test(dataframe)

    validate_boundary(train, test)

    # L'objet complet n'est plus nécessaire.
    del dataframe

    print_separator("SAUVEGARDE DES DATASETS")

    save_dataset(
        train,
        TRAIN_FILE,
        "TRAIN",
    )

    save_dataset(
        test,
        TEST_FILE,
        "TEST",
    )

    total_elapsed_time = (
        time.perf_counter() - total_start_time
    )

    print_separator("ÉTAPE 02 TERMINÉE AVEC SUCCÈS")

    print("Fichiers produits :")

    print(f"\nTrain :\n{TRAIN_FILE}")
    print(f"\nTest :\n{TEST_FILE}")

    print(
        f"\nDimensions du train : {train.shape}"
    )

    print(
        f"Dimensions du test  : {test.shape}"
    )

    print(
        "\nVariables temporelles :"
    )

    for feature in TEMPORAL_FEATURES:
        print(f"  - {feature}")

    print(
        "\nDurée totale : "
        f"{format_duration(total_elapsed_time)}"
    )


if __name__ == "__main__":
    try:
        main()

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
    ) as error:
        print_separator("ERREUR")

        print(error)

        sys.exit(1)

    except MemoryError:
        print_separator("ERREUR MÉMOIRE")

        print(
            "La mémoire disponible est insuffisante pour "
            "construire toutes les variables temporelles."
        )

        print(
            "Il faudra utiliser un traitement par points de "
            "grille ou convertir les données au format Parquet."
        )

        sys.exit(1)