from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# IMPORT DU DOSSIER SRC
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from src.utils.config import (  # noqa: E402
    FIGURE_DIR,
    METRICS_DIR,
    PREDICTION_DIR,
    create_output_directories,
)

from data_loader import (  # noqa: E402
    SequenceConfig,
    chronological_split,
    create_sequences,
    load_first_point_temperature,
)

from models.baseline import (  # noqa: E402
    calculate_regression_metrics,
    persistence_forecast,
)


# ============================================================
# PARAMÈTRES
# ============================================================

SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TARGET_COLUMN = "T"

PLOT_LENGTH = 365


# ============================================================
# GRAPHIQUE
# ============================================================

def save_prediction_plot(
    predictions_dataframe: pd.DataFrame,
    output_path: Path,
    maximum_points: int = PLOT_LENGTH,
) -> None:
    """
    Enregistre un graphique comparant les observations
    et les prédictions de la baseline.
    """

    plot_dataframe = predictions_dataframe.head(
        maximum_points
    )

    figure, axis = plt.subplots(
        figsize=(14, 6),
    )

    axis.plot(
        plot_dataframe["DATE"],
        plot_dataframe["TEMPERATURE_REELLE"],
        label="Température réelle",
        linewidth=1.5,
    )

    axis.plot(
        plot_dataframe["DATE"],
        plot_dataframe["TEMPERATURE_PREDITE"],
        label="Baseline de persistance",
        linewidth=1.2,
    )

    axis.set_title(
        "Baseline naïve : température réelle et prédite"
    )

    axis.set_xlabel(
        "Date"
    )

    axis.set_ylabel(
        "Température en °C"
    )

    axis.legend()
    axis.grid(alpha=0.3)

    figure.autofmt_xdate()
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# ÉVALUATION
# ============================================================

def main() -> None:
    create_output_directories()

    print("=" * 80)
    print("ÉVALUATION DE LA BASELINE NAÏVE")
    print("=" * 80)

    # --------------------------------------------------------
    # Chargement de la série complète
    # --------------------------------------------------------

    dataframe = load_first_point_temperature()

    _, _, test_dataframe = chronological_split(
        dataframe=dataframe,
        train_ratio=0.70,
        validation_ratio=0.15,
    )

    print(
        f"Nombre de lignes du jeu de test : "
        f"{len(test_dataframe):,}"
    )

    # --------------------------------------------------------
    # Création des séquences non normalisées
    # --------------------------------------------------------

    sequence_config = SequenceConfig(
        input_length=SEQUENCE_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
        target_column=TARGET_COLUMN,
    )

    x_test, y_test = create_sequences(
        dataframe=test_dataframe,
        config=sequence_config,
    )

    print(
        f"Forme de X test : {x_test.shape}"
    )

    print(
        f"Forme de y test : {y_test.shape}"
    )

    # --------------------------------------------------------
    # Prédiction naïve
    # --------------------------------------------------------

    y_prediction = persistence_forecast(
        x_test
    )

    # --------------------------------------------------------
    # Calcul des métriques
    # --------------------------------------------------------

    metrics = calculate_regression_metrics(
        y_true=y_test,
        y_pred=y_prediction,
    )

    metrics.update(
        {
            "model": "persistence_baseline",
            "target": TARGET_COLUMN,
            "unit": "degrees_celsius",
            "input_length_days": SEQUENCE_LENGTH,
            "forecast_horizon_days": FORECAST_HORIZON,
            "number_of_test_sequences": int(
                len(y_test)
            ),
        }
    )

    # --------------------------------------------------------
    # Dates correspondant aux cibles
    # --------------------------------------------------------

    first_target_position = (
        SEQUENCE_LENGTH
        + FORECAST_HORIZON
        - 1
    )

    target_dates = (
        test_dataframe["DATE"]
        .iloc[
            first_target_position:
            first_target_position + len(y_test)
        ]
        .reset_index(drop=True)
    )

    if len(target_dates) != len(y_test):
        raise RuntimeError(
            "Le nombre de dates ne correspond pas au "
            "nombre de prédictions."
        )

    predictions_dataframe = pd.DataFrame(
        {
            "DATE": target_dates,
            "TEMPERATURE_REELLE": y_test,
            "TEMPERATURE_PREDITE": y_prediction,
        }
    )

    predictions_dataframe["ERREUR"] = (
        predictions_dataframe["TEMPERATURE_PREDITE"]
        - predictions_dataframe["TEMPERATURE_REELLE"]
    )

    predictions_dataframe["ERREUR_ABSOLUE"] = (
        predictions_dataframe["ERREUR"].abs()
    )

    # --------------------------------------------------------
    # Enregistrement des résultats
    # --------------------------------------------------------

    metrics_path = (
        METRICS_DIR
        / "baseline_persistence_metrics.json"
    )

    predictions_path = (
        PREDICTION_DIR
        / "baseline_persistence_predictions.csv"
    )

    figure_path = (
        FIGURE_DIR
        / "baseline_persistence_predictions.png"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=4,
        )

    predictions_dataframe.to_csv(
        predictions_path,
        index=False,
    )

    save_prediction_plot(
        predictions_dataframe=predictions_dataframe,
        output_path=figure_path,
    )

    # --------------------------------------------------------
    # Affichage
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("RÉSULTATS DE LA BASELINE")
    print("=" * 80)

    print(
        f"MAE  : {metrics['MAE_C']:.4f} °C"
    )

    print(
        f"RMSE : {metrics['RMSE_C']:.4f} °C"
    )

    print(
        f"Biais : {metrics['BIAS_C']:.4f} °C"
    )

    print(
        f"R²   : {metrics['R2']:.4f}"
    )

    print("\nFichiers créés :")

    print(
        f"- Métriques   : {metrics_path}"
    )

    print(
        f"- Prédictions : {predictions_path}"
    )

    print(
        f"- Graphique   : {figure_path}"
    )


if __name__ == "__main__":
    main()