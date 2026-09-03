"""Configuration centrale et portable du projet."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
LOG_DIR = RESULTS_DIR / "logs"
PREDICTION_DIR = RESULTS_DIR / "predictions"
PERSONAL_PROCESSED_DATA_DIR = PROCESSED_DATA_DIR


def create_output_directories() -> None:
    """Crée les dossiers de sortie communs sans toucher aux données."""
    for directory in (RESULTS_DIR, MODEL_DIR, FIGURE_DIR, METRICS_DIR, LOG_DIR, PREDICTION_DIR):
        directory.mkdir(parents=True, exist_ok=True)
