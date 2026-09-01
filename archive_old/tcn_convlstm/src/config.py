from pathlib import Path


# ============================================================
# DOSSIERS PRINCIPAUX
# ============================================================

# /home/lab-c0de80861d/climate_project/ndambo
PROJECT_DIR = Path(__file__).resolve().parents[1]
# Données dérivées propres au travail de Ndambo
PERSONAL_DATA_DIR = PROJECT_DIR / "data"
PERSONAL_PROCESSED_DATA_DIR = PERSONAL_DATA_DIR / "processed"

FIRST_POINT_MULTIVARIATE_PATH = (
    PERSONAL_PROCESSED_DATA_DIR
    / "first_point_multivariate_daily.csv"
)
SPATIAL_PATCH_PATH = (
    PERSONAL_PROCESSED_DATA_DIR
    / "safran_patch_5x5_temperature_family.npz"
)

SPATIAL_PATCH_METADATA_PATH = (
    PERSONAL_PROCESSED_DATA_DIR
    / "safran_patch_5x5_temperature_family_metadata.json"
)

SPATIAL_CENTER_SERIES_PATH = (
    PERSONAL_PROCESSED_DATA_DIR
    / "safran_center_6440_22010_daily.csv"
)

# /home/lab-c0de80861d/climate_project
ROOT_DIR = PROJECT_DIR.parent

# /home/lab-c0de80861d/climate_project/data
GLOBAL_DATA_DIR = ROOT_DIR / "data"

# Données intermédiaires communes
INTERIM_DATA_DIR = GLOBAL_DATA_DIR / "interim"

# Données nettoyées et prêtes pour la modélisation
PROCESSED_DATA_DIR = GLOBAL_DATA_DIR / "processed"

# Alias principal utilisé par les scripts de modélisation
DATA_DIR = PROCESSED_DATA_DIR


# ============================================================
# FICHIERS DE DONNÉES
# ============================================================

DAILY_DATA_PATH = (
    PROCESSED_DATA_DIR / "safran_quot_clean.csv"
)

MONTHLY_DATA_PATH = (
    PROCESSED_DATA_DIR / "safran_mens_clean.csv"
)

FIRST_POINT_DAILY_PATH = (
    PROCESSED_DATA_DIR / "first_point_T_daily.csv"
)

DAILY_REPORT_PATH = (
    PROCESSED_DATA_DIR / "profiling_quotidien_report.html"
)

MONTHLY_REPORT_PATH = (
    PROCESSED_DATA_DIR / "profiling_mensuel_report.html"
)


# ============================================================
# DOSSIERS PERSONNELS DE SORTIE
# ============================================================

OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
METRICS_DIR = OUTPUT_DIR / "metrics"
LOG_DIR = OUTPUT_DIR / "logs"


def create_output_directories() -> None:
    """Crée les dossiers destinés aux résultats personnels."""

    directories = [
        OUTPUT_DIR,
        MODEL_DIR,
        FIGURE_DIR,
        PREDICTION_DIR,
        METRICS_DIR,
        LOG_DIR,
        PERSONAL_DATA_DIR,
        PERSONAL_PROCESSED_DATA_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def validate_data_paths() -> None:
    """Vérifie la présence des fichiers nécessaires."""

    required_files = [
        DAILY_DATA_PATH,
        MONTHLY_DATA_PATH,
    ]

    missing_files = [
        file_path
        for file_path in required_files
        if not file_path.is_file()
    ]

    if missing_files:
        missing_text = "\n".join(
            f"- {file_path}"
            for file_path in missing_files
        )

        raise FileNotFoundError(
            "\nLes fichiers suivants sont introuvables :\n"
            f"{missing_text}\n\n"
            "Dossier actuellement vérifié :\n"
            f"{PROCESSED_DATA_DIR}\n"
        )


def display_configuration() -> None:
    """Affiche les chemins utilisés par le projet."""

    print("Configuration valide.")
    print(f"Racine du projet      : {ROOT_DIR}")
    print(f"Dossier personnel     : {PROJECT_DIR}")
    print(f"Données globales      : {GLOBAL_DATA_DIR}")
    print(f"Données intermédiaires: {INTERIM_DATA_DIR}")
    print(f"Données traitées      : {PROCESSED_DATA_DIR}")
    print(f"Données quotidiennes  : {DAILY_DATA_PATH}")
    print(f"Données mensuelles    : {MONTHLY_DATA_PATH}")
    print(f"Premier point journalier : {FIRST_POINT_DAILY_PATH}")
    print(f"Résultats personnels  : {OUTPUT_DIR}")


if __name__ == "__main__":
    create_output_directories()
    validate_data_paths()
    display_configuration()