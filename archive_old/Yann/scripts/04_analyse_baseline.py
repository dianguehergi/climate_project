#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
04_analyse_baseline.py
===============================================================================

Analyse scientifique du modèle XGBoost baseline.

Ce script :
    - NE RÉENTRAÎNE PAS le modèle ;
    - NE FAIT PAS appel à XGBoost ;
    - analyse uniquement les sorties produites par 03_train_xgboost.py.

Analyses réalisées :
    1. Statistiques globales des résidus.
    2. Observé vs prédit.
    3. Histogramme des résidus.
    4. Résidus vs prédictions.
    5. Importance des variables.
    6. MAE annuelle.
    7. R² annuel.
    8. Distribution des erreurs absolues.
    9. Analyse spatiale des 9 892 points de grille.
   10. MAE, RMSE, R² et biais par point de grille.
   11. Température moyenne observée par point.
   12. Température moyenne prédite par point.
   13. Génération de cartes spatiales / heatmaps.

Entrées attendues :
    results/xgboost_baseline/
        predictions_full.parquet
        metrics_global.csv
        metrics_by_year.csv
        feature_importance.csv
        model_configuration.json

Sorties :
    results/xgboost_baseline/
        figures/
        tables/
        reports/

===============================================================================
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score


# ==============================================================================
# CHEMINS
# ==============================================================================

PROJECT_DIR = Path(
    "/home/lab-c0de80861d/climate_project"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "xgboost_baseline"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

TABLES_DIR = (
    RESULTS_DIR
    / "tables"
)

REPORTS_DIR = (
    RESULTS_DIR
    / "reports"
)


# ==============================================================================
# FICHIERS D'ENTRÉE
# ==============================================================================

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "predictions_full.parquet"
)

METRICS_GLOBAL_FILE = (
    RESULTS_DIR
    / "metrics_global.csv"
)

METRICS_BY_YEAR_FILE = (
    RESULTS_DIR
    / "metrics_by_year.csv"
)

FEATURE_IMPORTANCE_FILE = (
    RESULTS_DIR
    / "feature_importance.csv"
)

MODEL_CONFIGURATION_FILE = (
    RESULTS_DIR
    / "model_configuration.json"
)


# ==============================================================================
# FICHIERS DE SORTIE
# ==============================================================================

RESIDUAL_STATISTICS_FILE = (
    TABLES_DIR
    / "residual_statistics.csv"
)

RESIDUAL_SUMMARY_FILE = (
    TABLES_DIR
    / "residual_summary.csv"
)

METRICS_BY_GRIDPOINT_FILE = (
    TABLES_DIR
    / "metrics_by_gridpoint.csv"
)

SPATIAL_SUMMARY_FILE = (
    TABLES_DIR
    / "spatial_summary.csv"
)

ANALYSIS_JSON_FILE = (
    REPORTS_DIR
    / "analysis_results.json"
)

ANALYSIS_MARKDOWN_FILE = (
    REPORTS_DIR
    / "analysis_summary.md"
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

FIGSIZE = (8, 6)

DPI = 300

FONT_SIZE = 12

TITLE_SIZE = 14

SCATTER_SAMPLE_SIZE = 100_000

RANDOM_STATE = 42

EXPECTED_GRID_POINTS = 9892


# ==============================================================================
# OUTILS
# ==============================================================================


def print_separator(
    title: str,
) -> None:

    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def format_duration(
    seconds: float,
) -> str:

    minutes = int(
        seconds // 60
    )

    remaining_seconds = (
        seconds % 60
    )

    return (
        f"{minutes} min "
        f"{remaining_seconds:.1f} s"
    )


def save_figure(
    fig: plt.Figure,
    filename: str,
) -> None:

    png_file = (
        FIGURES_DIR
        / f"{filename}.png"
    )

    pdf_file = (
        FIGURES_DIR
        / f"{filename}.pdf"
    )

    fig.savefig(
        png_file,
        dpi=DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_file,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==============================================================================
# INITIALISATION
# ==============================================================================


def create_output_directories() -> None:

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def check_inputs() -> None:

    files = [
        PREDICTIONS_FILE,
        METRICS_GLOBAL_FILE,
        METRICS_BY_YEAR_FILE,
        FEATURE_IMPORTANCE_FILE,
        MODEL_CONFIGURATION_FILE,
    ]

    missing = [
        str(file_path)
        for file_path in files
        if not file_path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Fichier(s) manquant(s) :\n"
            + "\n".join(missing)
            + "\n\n"
            + "Exécute d'abord 03_train_xgboost.py."
        )


# ==============================================================================
# CHARGEMENT DES DONNÉES
# ==============================================================================


def load_predictions() -> pd.DataFrame:

    print_separator(
        "CHARGEMENT DES PRÉDICTIONS"
    )

    start = time.time()

    predictions = pd.read_parquet(
        PREDICTIONS_FILE
    )

    predictions["DATE"] = pd.to_datetime(
        predictions["DATE"]
    )

    duration = (
        time.time()
        - start
    )

    memory_gb = (
        predictions
        .memory_usage(
            deep=True
        )
        .sum()
        / (1024 ** 3)
    )

    print(
        f"Dimensions : "
        f"{predictions.shape}"
    )

    print(
        f"Temps : "
        f"{format_duration(duration)}"
    )

    print(
        f"Mémoire : "
        f"{memory_gb:.2f} Go"
    )

    return predictions


def load_metrics():

    print_separator(
        "CHARGEMENT DES MÉTRIQUES"
    )

    global_metrics = pd.read_csv(
        METRICS_GLOBAL_FILE
    )

    yearly_metrics = pd.read_csv(
        METRICS_BY_YEAR_FILE
    )

    print(
        f"Métriques globales : "
        f"{len(global_metrics)} lignes"
    )

    print(
        f"Métriques annuelles : "
        f"{len(yearly_metrics)} lignes"
    )

    return (
        global_metrics,
        yearly_metrics,
    )


def load_feature_importance() -> pd.DataFrame:

    print_separator(
        "IMPORTANCE DES VARIABLES"
    )

    feature_importance = (
        pd.read_csv(
            FEATURE_IMPORTANCE_FILE
        )
    )

    print(
        f"{len(feature_importance)} variables"
    )

    return feature_importance


def load_configuration() -> dict:

    print_separator(
        "CONFIGURATION DU MODÈLE"
    )

    with open(
        MODEL_CONFIGURATION_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        configuration = json.load(
            file
        )

    print(
        f"Nombre de paramètres : "
        f"{len(configuration)}"
    )

    return configuration


# ==============================================================================
# VÉRIFICATIONS
# ==============================================================================


def check_prediction_columns(
    predictions: pd.DataFrame,
) -> None:

    required_columns = [
        "DATE",
        "LAMBX",
        "LAMBY",
        "T_observed",
        "T_predicted",
        "residual",
        "absolute_error",
    ]

    missing = [
        column
        for column in required_columns
        if column not in predictions.columns
    ]

    if missing:

        raise ValueError(
            "Colonnes manquantes : "
            f"{missing}"
        )

    print(
        "Colonnes vérifiées."
    )


def print_dataset_summary(
    predictions: pd.DataFrame,
) -> None:

    print_separator(
        "RÉSUMÉ DU JEU DE DONNÉES"
    )

    print(
        f"Nombre de lignes : "
        f"{len(predictions):,}"
    )

    print(
        f"Nombre de colonnes : "
        f"{predictions.shape[1]}"
    )

    print()

    print(
        predictions.dtypes
    )

    print()

    print(
        predictions.head()
    )


# ==============================================================================
# STATISTIQUES DES RÉSIDUS
# ==============================================================================


def compute_residual_statistics(
    predictions: pd.DataFrame,
) -> dict:

    residuals = (
        predictions[
            "residual"
        ]
        .to_numpy()
    )

    abs_errors = np.abs(
        residuals
    )

    statistics = {

        "n":
            int(
                len(residuals)
            ),

        "mean":
            float(
                np.mean(residuals)
            ),

        "median":
            float(
                np.median(residuals)
            ),

        "variance":
            float(
                np.var(residuals)
            ),

        "std":
            float(
                np.std(residuals)
            ),

        "min":
            float(
                np.min(residuals)
            ),

        "max":
            float(
                np.max(residuals)
            ),

        "MAE":
            float(
                np.mean(
                    abs_errors
                )
            ),

        "median_absolute_error":
            float(
                np.median(
                    abs_errors
                )
            ),

        "RMSE":
            float(
                np.sqrt(
                    np.mean(
                        residuals ** 2
                    )
                )
            ),

        "bias":
            float(
                np.mean(
                    residuals
                )
            ),

        "q01":
            float(
                np.quantile(
                    residuals,
                    0.01,
                )
            ),

        "q05":
            float(
                np.quantile(
                    residuals,
                    0.05,
                )
            ),

        "q25":
            float(
                np.quantile(
                    residuals,
                    0.25,
                )
            ),

        "q50":
            float(
                np.quantile(
                    residuals,
                    0.50,
                )
            ),

        "q75":
            float(
                np.quantile(
                    residuals,
                    0.75,
                )
            ),

        "q95":
            float(
                np.quantile(
                    residuals,
                    0.95,
                )
            ),

        "q99":
            float(
                np.quantile(
                    residuals,
                    0.99,
                )
            ),
    }

    return statistics


def save_residual_statistics(
    statistics: dict,
) -> None:

    df = pd.DataFrame(
        [
            statistics
        ]
    )

    df.to_csv(
        RESIDUAL_STATISTICS_FILE,
        index=False,
    )

    print(
        "Statistiques sauvegardées :"
    )

    print(
        RESIDUAL_STATISTICS_FILE
    )


def save_residual_summary(
    statistics: dict,
) -> None:

    summary = pd.DataFrame(
        {
            "metric":
                list(
                    statistics.keys()
                ),

            "value":
                list(
                    statistics.values()
                ),
        }
    )

    summary.to_csv(
        RESIDUAL_SUMMARY_FILE,
        index=False,
    )

    print(
        "Résumé sauvegardé :"
    )

    print(
        RESIDUAL_SUMMARY_FILE
    )


# ==============================================================================
# ÉCHANTILLON POUR LES SCATTERS
# ==============================================================================


def sample_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    sample_size = min(
        SCATTER_SAMPLE_SIZE,
        len(predictions),
    )

    return (
        predictions
        .sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        )
    )


# ==============================================================================
# FIGURES GLOBALES
# ==============================================================================


def plot_observed_vs_predicted(
    predictions: pd.DataFrame,
) -> None:

    print_separator(
        "FIGURE : OBSERVED VS PREDICTED"
    )

    sample = sample_predictions(
        predictions
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.scatter(
        sample["T_observed"],
        sample["T_predicted"],
        s=6,
        alpha=0.25,
    )

    min_value = min(
        sample["T_observed"].min(),
        sample["T_predicted"].min(),
    )

    max_value = max(
        sample["T_observed"].max(),
        sample["T_predicted"].max(),
    )

    ax.plot(
        [
            min_value,
            max_value,
        ],
        [
            min_value,
            max_value,
        ],
        linestyle="--",
    )

    ax.set_xlabel(
        "Température observée (°C)",
        fontsize=FONT_SIZE,
    )

    ax.set_ylabel(
        "Température prédite (°C)",
        fontsize=FONT_SIZE,
    )

    ax.set_title(
        "Températures observées vs prédites",
        fontsize=TITLE_SIZE,
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    save_figure(
        fig,
        "observed_vs_predicted",
    )


def plot_residual_histogram(
    predictions: pd.DataFrame,
) -> None:

    print_separator(
        "FIGURE : HISTOGRAMME DES RÉSIDUS"
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.hist(
        predictions["residual"],
        bins=100,
        alpha=0.8,
    )

    ax.axvline(
        0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Résidu : observé - prédit (°C)",
        fontsize=FONT_SIZE,
    )

    ax.set_ylabel(
        "Fréquence",
        fontsize=FONT_SIZE,
    )

    ax.set_title(
        "Distribution des résidus",
        fontsize=TITLE_SIZE,
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    save_figure(
        fig,
        "residual_histogram",
    )


def plot_residuals_vs_predictions(
    predictions: pd.DataFrame,
) -> None:

    print_separator(
        "FIGURE : RESIDUALS VS PREDICTIONS"
    )

    sample = sample_predictions(
        predictions
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.scatter(
        sample["T_predicted"],
        sample["residual"],
        s=6,
        alpha=0.25,
    )

    ax.axhline(
        0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Température prédite (°C)",
        fontsize=FONT_SIZE,
    )

    ax.set_ylabel(
        "Résidu (°C)",
        fontsize=FONT_SIZE,
    )

    ax.set_title(
        "Résidus en fonction des prédictions",
        fontsize=TITLE_SIZE,
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_03_residuals_vs_predictions",
    )


def plot_feature_importance(
    feature_importance: pd.DataFrame,
) -> None:

    print_separator(
        "FIGURE : FEATURE IMPORTANCE"
    )

    df = (
        feature_importance
        .sort_values(
            "importance",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.barh(
        df["feature"],
        df["importance"],
    )

    ax.set_xlabel(
        "Importance",
        fontsize=FONT_SIZE,
    )

    ax.set_ylabel(
        "Variable",
        fontsize=FONT_SIZE,
    )

    ax.set_title(
        "Importance des variables XGBoost",
        fontsize=TITLE_SIZE,
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_04_feature_importance",
    )


def plot_yearly_mae(
    yearly_metrics: pd.DataFrame,
) -> None:

    print_separator(
        "FIGURE : YEARLY MAE"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        yearly_metrics["year"],
        yearly_metrics["MAE"],
        marker="o",
    )

    ax.set_xlabel(
        "Année",
        fontsize=FONT_SIZE,
    )

    ax.set_ylabel(
        "MAE (°C)",
        fontsize=FONT_SIZE,
    )

    ax.set_title(
        "Évolution annuelle de la MAE",
        fontsize=TITLE_SIZE,
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_05_yearly_mae",
    )


def plot_yearly_r2(
    yearly_metrics: pd.DataFrame,
) -> None:

    print_separator(
        "FIGURE : YEARLY R2"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        yearly_metrics["year"],
        yearly_metrics["R2"],
        marker="o",
    )

    ax.set_xlabel(
        "Année",
        fontsize=FONT_SIZE,
    )

    ax.set_ylabel(
        "R²",
        fontsize=FONT_SIZE,
    )

    ax.set_title(
        "Évolution annuelle du R²",
        fontsize=TITLE_SIZE,
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_06_yearly_r2",
    )


def plot_absolute_error_distribution(
    predictions: pd.DataFrame,
) -> None:

    print_separator(
        "FIGURE : ABSOLUTE ERROR"
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.hist(
        predictions["absolute_error"],
        bins=100,
        alpha=0.8,
    )

    ax.set_xlabel(
        "Erreur absolue (°C)",
        fontsize=FONT_SIZE,
    )

    ax.set_ylabel(
        "Fréquence",
        fontsize=FONT_SIZE,
    )

    ax.set_title(
        "Distribution des erreurs absolues",
        fontsize=TITLE_SIZE,
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_07_absolute_error_distribution",
    )


# ==============================================================================
# ANALYSE PAR POINT DE GRILLE
# ==============================================================================


def compute_metrics_by_gridpoint(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    print_separator(
        "MÉTRIQUES PAR POINT DE GRILLE"
    )

    grouped = predictions.groupby(
        [
            "LAMBX",
            "LAMBY",
        ],
        sort=False,
    )

    print(
        f"Nombre de points détectés : "
        f"{grouped.ngroups:,}"
    )

    rows = []

    for (
        lambx,
        lamby,
    ), group in grouped:

        observed = (
            group[
                "T_observed"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        predicted = (
            group[
                "T_predicted"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        residual = (
            observed
            - predicted
        )

        mae = float(
            np.mean(
                np.abs(
                    residual
                )
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    residual ** 2
                )
            )
        )

        bias = float(
            np.mean(
                residual
            )
        )

        if (
            len(observed) > 1
            and np.var(observed) > 0
        ):

            r2 = float(
                r2_score(
                    observed,
                    predicted,
                )
            )

        else:

            r2 = np.nan

        rows.append(
            {
                "LAMBX":
                    lambx,

                "LAMBY":
                    lamby,

                "n_observations":
                    int(
                        len(group)
                    ),

                "MAE":
                    mae,

                "RMSE":
                    rmse,

                "R2":
                    r2,

                "Bias":
                    bias,

                "T_observed_mean":
                    float(
                        np.mean(
                            observed
                        )
                    ),

                "T_predicted_mean":
                    float(
                        np.mean(
                            predicted
                        )
                    ),

                "mean_absolute_error":
                    mae,
            }
        )

    metrics = pd.DataFrame(
        rows
    )

    print(
        f"Points analysés : "
        f"{len(metrics):,}"
    )

    return metrics


def validate_gridpoint_evaluation(
    metrics_by_gridpoint: pd.DataFrame,
) -> None:

    print_separator(
        "VALIDATION DE LA GRILLE"
    )

    actual_points = len(
        metrics_by_gridpoint
    )

    print(
        f"Points attendus : "
        f"{EXPECTED_GRID_POINTS:,}"
    )

    print(
        f"Points analysés : "
        f"{actual_points:,}"
    )

    if actual_points != EXPECTED_GRID_POINTS:

        raise ValueError(
            "Nombre incorrect de points de grille. "
            f"Attendu : {EXPECTED_GRID_POINTS}, "
            f"obtenu : {actual_points}."
        )

    n_values = (
        metrics_by_gridpoint[
            "n_observations"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nNombre d'observations par point :"
    )

    print(
        n_values
    )

    print(
        "\nOK : tous les points de grille "
        "ont été évalués."
    )


def save_metrics_by_gridpoint(
    metrics_by_gridpoint: pd.DataFrame,
) -> None:

    metrics_by_gridpoint.to_csv(
        METRICS_BY_GRIDPOINT_FILE,
        index=False,
    )

    print(
        f"\nMétriques spatiales sauvegardées :\n"
        f"{METRICS_BY_GRIDPOINT_FILE}"
    )


# ==============================================================================
# RÉSUMÉ SPATIAL
# ==============================================================================


def compute_spatial_summary(
    metrics_by_gridpoint: pd.DataFrame,
) -> pd.DataFrame:

    summary = pd.DataFrame(
        [
            {
                "n_grid_points":
                    int(
                        len(
                            metrics_by_gridpoint
                        )
                    ),

                "mean_MAE":
                    float(
                        metrics_by_gridpoint[
                            "MAE"
                        ]
                        .mean()
                    ),

                "median_MAE":
                    float(
                        metrics_by_gridpoint[
                            "MAE"
                        ]
                        .median()
                    ),

                "min_MAE":
                    float(
                        metrics_by_gridpoint[
                            "MAE"
                        ]
                        .min()
                    ),

                "max_MAE":
                    float(
                        metrics_by_gridpoint[
                            "MAE"
                        ]
                        .max()
                    ),

                "mean_RMSE":
                    float(
                        metrics_by_gridpoint[
                            "RMSE"
                        ]
                        .mean()
                    ),

                "median_RMSE":
                    float(
                        metrics_by_gridpoint[
                            "RMSE"
                        ]
                        .median()
                    ),

                "mean_R2":
                    float(
                        metrics_by_gridpoint[
                            "R2"
                        ]
                        .mean()
                    ),

                "median_R2":
                    float(
                        metrics_by_gridpoint[
                            "R2"
                        ]
                        .median()
                    ),

                "mean_Bias":
                    float(
                        metrics_by_gridpoint[
                            "Bias"
                        ]
                        .mean()
                    ),

                "median_Bias":
                    float(
                        metrics_by_gridpoint[
                            "Bias"
                        ]
                        .median()
                    ),
            }
        ]
    )

    summary.to_csv(
        SPATIAL_SUMMARY_FILE,
        index=False,
    )

    return summary


# ==============================================================================
# CARTES SPATIALES
# ==============================================================================


def plot_spatial_metric(
    metrics_by_gridpoint: pd.DataFrame,
    column: str,
    title: str,
    colorbar_label: str,
    filename: str,
    cmap: str = "viridis",
) -> None:

    print_separator(
        f"CARTE : {title.upper()}"
    )

    fig, ax = plt.subplots(
        figsize=(8, 9)
    )

    scatter = ax.scatter(
        metrics_by_gridpoint[
            "LAMBX"
        ],
        metrics_by_gridpoint[
            "LAMBY"
        ],
        c=metrics_by_gridpoint[
            column
        ],
        s=12,
        cmap=cmap,
        marker="s",
        linewidths=0,
    )

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        shrink=0.8,
    )

    colorbar.set_label(
        colorbar_label,
        fontsize=FONT_SIZE,
    )

    ax.set_xlabel(
        "LAMBX",
        fontsize=FONT_SIZE,
    )

    ax.set_ylabel(
        "LAMBY",
        fontsize=FONT_SIZE,
    )

    ax.set_title(
        title,
        fontsize=TITLE_SIZE,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        alpha=0.1
    )

    fig.tight_layout()

    save_figure(
        fig,
        filename,
    )


def plot_observed_temperature_map(
    metrics_by_gridpoint: pd.DataFrame,
) -> None:

    plot_spatial_metric(
        metrics_by_gridpoint=
            metrics_by_gridpoint,

        column=
            "T_observed_mean",

        title=
            "Température moyenne observée par point de grille",

        colorbar_label=
            "Température moyenne observée (°C)",

        filename=
            "Figure_08_temperature_observed_map",

        cmap=
            "coolwarm",
    )


def plot_predicted_temperature_map(
    metrics_by_gridpoint: pd.DataFrame,
) -> None:

    plot_spatial_metric(
        metrics_by_gridpoint=
            metrics_by_gridpoint,

        column=
            "T_predicted_mean",

        title=
            "Température moyenne prédite par point de grille",

        colorbar_label=
            "Température moyenne prédite (°C)",

        filename=
            "Figure_09_temperature_predicted_map",

        cmap=
            "coolwarm",
    )


def plot_mae_map(
    metrics_by_gridpoint: pd.DataFrame,
) -> None:

    plot_spatial_metric(
        metrics_by_gridpoint=
            metrics_by_gridpoint,

        column=
            "MAE",

        title=
            "MAE spatiale du modèle XGBoost",

        colorbar_label=
            "MAE (°C)",

        filename=
            "Figure_10_mae_map",

        cmap=
            "viridis",
    )


def plot_rmse_map(
    metrics_by_gridpoint: pd.DataFrame,
) -> None:

    plot_spatial_metric(
        metrics_by_gridpoint=
            metrics_by_gridpoint,

        column=
            "RMSE",

        title=
            "RMSE spatiale du modèle XGBoost",

        colorbar_label=
            "RMSE (°C)",

        filename=
            "Figure_11_rmse_map",

        cmap=
            "viridis",
    )


def plot_bias_map(
    metrics_by_gridpoint: pd.DataFrame,
) -> None:

    max_abs_bias = (
        metrics_by_gridpoint[
            "Bias"
        ]
        .abs()
        .max()
    )

    print_separator(
        "CARTE : BIAIS SPATIAL"
    )

    fig, ax = plt.subplots(
        figsize=(8, 9)
    )

    scatter = ax.scatter(
        metrics_by_gridpoint[
            "LAMBX"
        ],
        metrics_by_gridpoint[
            "LAMBY"
        ],
        c=metrics_by_gridpoint[
            "Bias"
        ],
        s=12,
        cmap="coolwarm",
        vmin=-max_abs_bias,
        vmax=max_abs_bias,
        marker="s",
        linewidths=0,
    )

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        shrink=0.8,
    )

    colorbar.set_label(
        "Biais : observé - prédit (°C)",
        fontsize=FONT_SIZE,
    )

    ax.set_xlabel(
        "LAMBX",
        fontsize=FONT_SIZE,
    )

    ax.set_ylabel(
        "LAMBY",
        fontsize=FONT_SIZE,
    )

    ax.set_title(
        "Biais spatial du modèle XGBoost",
        fontsize=TITLE_SIZE,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        alpha=0.1
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_12_bias_map",
    )


def plot_r2_map(
    metrics_by_gridpoint: pd.DataFrame,
) -> None:

    print_separator(
        "CARTE : R2 SPATIAL"
    )

    fig, ax = plt.subplots(
        figsize=(8, 9)
    )

    scatter = ax.scatter(
        metrics_by_gridpoint[
            "LAMBX"
        ],
        metrics_by_gridpoint[
            "LAMBY"
        ],
        c=metrics_by_gridpoint[
            "R2"
        ],
        s=12,
        cmap="viridis",
        marker="s",
        linewidths=0,
    )

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        shrink=0.8,
    )

    colorbar.set_label(
        "R²",
        fontsize=FONT_SIZE,
    )

    ax.set_xlabel(
        "LAMBX",
        fontsize=FONT_SIZE,
    )

    ax.set_ylabel(
        "LAMBY",
        fontsize=FONT_SIZE,
    )

    ax.set_title(
        "R² spatial du modèle XGBoost",
        fontsize=TITLE_SIZE,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        alpha=0.1
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_13_r2_map",
    )


# ==============================================================================
# DIFFÉRENCE TEMPÉRATURE MOYENNE OBSERVÉE / PRÉDITE
# ==============================================================================


def add_temperature_difference(
    metrics_by_gridpoint: pd.DataFrame,
) -> pd.DataFrame:

    metrics_by_gridpoint = (
        metrics_by_gridpoint.copy()
    )

    metrics_by_gridpoint[
        "temperature_mean_difference"
    ] = (
        metrics_by_gridpoint[
            "T_observed_mean"
        ]
        -
        metrics_by_gridpoint[
            "T_predicted_mean"
        ]
    )

    return metrics_by_gridpoint


def plot_temperature_difference_map(
    metrics_by_gridpoint: pd.DataFrame,
) -> None:

    max_abs = (
        metrics_by_gridpoint[
            "temperature_mean_difference"
        ]
        .abs()
        .max()
    )

    print_separator(
        "CARTE : DIFFÉRENCE TEMPÉRATURE MOYENNE"
    )

    fig, ax = plt.subplots(
        figsize=(8, 9)
    )

    scatter = ax.scatter(
        metrics_by_gridpoint[
            "LAMBX"
        ],
        metrics_by_gridpoint[
            "LAMBY"
        ],
        c=metrics_by_gridpoint[
            "temperature_mean_difference"
        ],
        s=12,
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
        marker="s",
        linewidths=0,
    )

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        shrink=0.8,
    )

    colorbar.set_label(
        "Observé - prédit (°C)",
        fontsize=FONT_SIZE,
    )

    ax.set_xlabel(
        "LAMBX",
        fontsize=FONT_SIZE,
    )

    ax.set_ylabel(
        "LAMBY",
        fontsize=FONT_SIZE,
    )

    ax.set_title(
        "Différence entre température moyenne observée et prédite",
        fontsize=TITLE_SIZE,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        alpha=0.1
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_14_temperature_difference_map",
    )


# ==============================================================================
# EXPORT JSON
# ==============================================================================


def export_analysis_json(
    residual_statistics: dict,
    spatial_summary: pd.DataFrame,
) -> None:

    print_separator(
        "EXPORT JSON"
    )

    spatial_dict = (
        spatial_summary
        .iloc[0]
        .to_dict()
    )

    spatial_dict = {
        key:
            (
                value.item()
                if hasattr(
                    value,
                    "item",
                )
                else value
            )
        for key, value in spatial_dict.items()
    }

    output = {
        "residual_statistics":
            residual_statistics,

        "spatial_statistics":
            spatial_dict,
    }

    with open(
        ANALYSIS_JSON_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        ANALYSIS_JSON_FILE
    )


# ==============================================================================
# RAPPORT MARKDOWN
# ==============================================================================


def write_markdown_report(
    metrics_global: pd.DataFrame,
    yearly_metrics: pd.DataFrame,
    residual_statistics: dict,
    spatial_summary: pd.DataFrame,
) -> None:

    print_separator(
        "RAPPORT MARKDOWN"
    )

    xgb_rows = (
        metrics_global[
            metrics_global[
                "Model"
            ]
            == "XGBoost"
        ]
    )

    if len(xgb_rows) == 0:

        raise ValueError(
            "Impossible de trouver "
            "la ligne XGBoost dans "
            "metrics_global.csv."
        )

    global_metrics = (
        xgb_rows
        .iloc[0]
    )

    best_year = (
        yearly_metrics
        .sort_values(
            "MAE"
        )
        .iloc[0]
    )

    worst_year = (
        yearly_metrics
        .sort_values(
            "MAE",
            ascending=False,
        )
        .iloc[0]
    )

    spatial = (
        spatial_summary
        .iloc[0]
    )

    report = (
        "# Analyse du modèle XGBoost baseline\n\n"

        "## 1. Performance globale\n\n"

        f"- MAE : **{global_metrics['MAE']:.3f} °C**\n"
        f"- RMSE : **{global_metrics['RMSE']:.3f} °C**\n"
        f"- R² : **{global_metrics['R2']:.3f}**\n"
        f"- Biais : **{global_metrics['Bias']:.3f} °C**\n\n"

        "## 2. Résidus\n\n"

        f"- Nombre de prédictions : "
        f"**{residual_statistics['n']:,}**\n"

        f"- Résidu moyen : "
        f"**{residual_statistics['mean']:.3f} °C**\n"

        f"- Écart-type des résidus : "
        f"**{residual_statistics['std']:.3f} °C**\n"

        f"- Médiane des résidus : "
        f"**{residual_statistics['median']:.3f} °C**\n\n"

        "## 3. Performance temporelle\n\n"

        f"- Meilleure année selon la MAE : "
        f"**{int(best_year['year'])}** "
        f"avec **{best_year['MAE']:.3f} °C**\n"

        f"- Année la plus difficile selon la MAE : "
        f"**{int(worst_year['year'])}** "
        f"avec **{worst_year['MAE']:.3f} °C**\n\n"

        "## 4. Analyse spatiale\n\n"

        f"- Nombre de points de grille évalués : "
        f"**{int(spatial['n_grid_points']):,}**\n"

        f"- MAE spatiale moyenne : "
        f"**{spatial['mean_MAE']:.3f} °C**\n"

        f"- MAE spatiale médiane : "
        f"**{spatial['median_MAE']:.3f} °C**\n"

        f"- MAE minimale : "
        f"**{spatial['min_MAE']:.3f} °C**\n"

        f"- MAE maximale : "
        f"**{spatial['max_MAE']:.3f} °C**\n"

        f"- R² spatial moyen : "
        f"**{spatial['mean_R2']:.3f}**\n"

        f"- Biais spatial moyen : "
        f"**{spatial['mean_Bias']:.3f} °C**\n\n"

        "## 5. Cartes générées\n\n"

        "- Température moyenne observée.\n"
        "- Température moyenne prédite.\n"
        "- Différence observé - prédit.\n"
        "- MAE spatiale.\n"
        "- RMSE spatiale.\n"
        "- Biais spatial.\n"
        "- R² spatial.\n\n"

        "## 6. Note méthodologique\n\n"

        "Les performances sont évaluées sur la période "
        "2000–2025. Les variables temporelles du jeu de test "
        "utilisent les observations historiques disponibles "
        "pour calculer les retards. Il s'agit donc d'une "
        "évaluation mensuelle one-step-ahead et non d'une "
        "prévision récursive complète sur 26 ans.\n"
    )

    with open(
        ANALYSIS_MARKDOWN_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            report
        )

    print(
        ANALYSIS_MARKDOWN_FILE
    )


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:

    start_total = time.time()

    print_separator(
        "ANALYSE DU MODÈLE XGBOOST"
    )

    create_output_directories()

    check_inputs()


    # --------------------------------------------------------------------------
    # Chargement
    # --------------------------------------------------------------------------

    predictions = (
        load_predictions()
    )

    (
        metrics_global,
        yearly_metrics,
    ) = load_metrics()

    feature_importance = (
        load_feature_importance()
    )

    configuration = (
        load_configuration()
    )


    # --------------------------------------------------------------------------
    # Vérifications
    # --------------------------------------------------------------------------

    check_prediction_columns(
        predictions
    )

    print_dataset_summary(
        predictions
    )


    # --------------------------------------------------------------------------
    # Résidus
    # --------------------------------------------------------------------------

    print_separator(
        "STATISTIQUES DES RÉSIDUS"
    )

    residual_statistics = (
        compute_residual_statistics(
            predictions
        )
    )

    save_residual_statistics(
        residual_statistics
    )

    save_residual_summary(
        residual_statistics
    )


    # --------------------------------------------------------------------------
    # Figures globales
    # --------------------------------------------------------------------------

    plot_observed_vs_predicted(
        predictions
    )

    plot_residual_histogram(
        predictions
    )

    plot_residuals_vs_predictions(
        predictions
    )

    plot_feature_importance(
        feature_importance
    )

    plot_yearly_mae(
        yearly_metrics
    )

    plot_yearly_r2(
        yearly_metrics
    )

    plot_absolute_error_distribution(
        predictions
    )


    # --------------------------------------------------------------------------
    # Analyse spatiale
    # --------------------------------------------------------------------------

    metrics_by_gridpoint = (
        compute_metrics_by_gridpoint(
            predictions
        )
    )

    metrics_by_gridpoint = (
        add_temperature_difference(
            metrics_by_gridpoint
        )
    )

    validate_gridpoint_evaluation(
        metrics_by_gridpoint
    )

    save_metrics_by_gridpoint(
        metrics_by_gridpoint
    )

    spatial_summary = (
        compute_spatial_summary(
            metrics_by_gridpoint
        )
    )


    # --------------------------------------------------------------------------
    # Cartes spatiales
    # --------------------------------------------------------------------------

    plot_observed_temperature_map(
        metrics_by_gridpoint
    )

    plot_predicted_temperature_map(
        metrics_by_gridpoint
    )

    plot_temperature_difference_map(
        metrics_by_gridpoint
    )

    plot_mae_map(
        metrics_by_gridpoint
    )

    plot_rmse_map(
        metrics_by_gridpoint
    )

    plot_bias_map(
        metrics_by_gridpoint
    )

    plot_r2_map(
        metrics_by_gridpoint
    )


    # --------------------------------------------------------------------------
    # Exports
    # --------------------------------------------------------------------------

    export_analysis_json(
        residual_statistics,
        spatial_summary,
    )

    write_markdown_report(
        metrics_global,
        yearly_metrics,
        residual_statistics,
        spatial_summary,
    )


    # --------------------------------------------------------------------------
    # Résumé final
    # --------------------------------------------------------------------------

    duration = (
        time.time()
        - start_total
    )

    print_separator(
        "ANALYSE TERMINÉE"
    )

    print(
        f"Observations : "
        f"{len(predictions):,}"
    )

    print(
        f"Points de grille : "
        f"{len(metrics_by_gridpoint):,}"
    )

    print(
        f"MAE : "
        f"{residual_statistics['MAE']:.3f}"
    )

    print(
        f"RMSE : "
        f"{residual_statistics['RMSE']:.3f}"
    )

    print(
        f"R² : "
        f"{metrics_global.loc[
            metrics_global['Model'] == 'XGBoost',
            'R2'
        ].iloc[0]:.3f}"
    )

    print(
        f"Temps total : "
        f"{format_duration(duration)}"
    )

    print(
        "\nFichier spatial :"
    )

    print(
        METRICS_BY_GRIDPOINT_FILE
    )

    print(
        "\nCartes générées :"
    )

    print(
        "  - Figure_08_temperature_observed_map"
    )

    print(
        "  - Figure_09_temperature_predicted_map"
    )

    print(
        "  - Figure_14_temperature_difference_map"
    )

    print(
        "  - Figure_10_mae_map"
    )

    print(
        "  - Figure_11_rmse_map"
    )

    print(
        "  - Figure_12_bias_map"
    )

    print(
        "  - Figure_13_r2_map"
    )


# ==============================================================================
# EXÉCUTION
# ==============================================================================


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nExécution interrompue."
        )

        sys.exit(1)

    except Exception as exc:

        print(
            "\nERREUR :"
        )

        print(
            str(exc)
        )

        raise