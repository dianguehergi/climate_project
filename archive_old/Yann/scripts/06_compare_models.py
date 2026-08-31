#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
06_compare_models.py
===============================================================================

Comparaison scientifique du modèle XGBoost baseline et du modèle
XGBoost avancé.

Pipeline :
    01_prepare_full_timeseries.py
        ↓
    02_build_training_dataset.py
        ↓
    03_train_xgboost.py
        ↓
    04_analyse_baseline.py
        ↓
    05_train_xgboost_advanced.py
        ↓
    06_compare_models.py

OBJECTIFS
---------
Ce script compare directement :

    - XGBoost baseline
    - XGBoost avancé

sur exactement la même période de test :

    2000-01 → 2025-12

et sur les mêmes :

    3 086 304 observations
    9 892 points de grille

COMPARAISONS
------------
Le script calcule et produit :

    1. Comparaison des métriques globales.
    2. Gains absolus et relatifs du modèle avancé.
    3. Comparaison année par année.
    4. Comparaison des erreurs prédiction par prédiction.
    5. Nombre de cas où chaque modèle est meilleur.
    6. Comparaison spatiale par point de grille.
    7. Cartes des gains spatiaux.
    8. Comparaison des importances de variables.
    9. Figures globales.
   10. Rapport Markdown.
   11. Export JSON synthétique.

IMPORTANT
---------
Ce script :

    - ne réentraîne aucun modèle ;
    - ne modifie aucune prédiction ;
    - utilise uniquement les sorties des étapes 03 et 05.

CONVENTION DU BIAIS
-------------------
    Bias = observed - predicted

Donc :

    Bias > 0
        → sous-estimation moyenne du modèle

    Bias < 0
        → surestimation moyenne du modèle

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

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==============================================================================
# CHEMINS
# ==============================================================================

PROJECT_DIR = Path(
    "/home/lab-c0de80861d/climate_project"
)

BASELINE_DIR = (
    PROJECT_DIR
    / "results"
    / "xgboost_baseline"
)

ADVANCED_DIR = (
    PROJECT_DIR
    / "results"
    / "xgboost_advanced"
)

COMPARISON_DIR = (
    PROJECT_DIR
    / "results"
    / "model_comparison"
)

FIGURES_DIR = (
    COMPARISON_DIR
    / "figures"
)

TABLES_DIR = (
    COMPARISON_DIR
    / "tables"
)

REPORTS_DIR = (
    COMPARISON_DIR
    / "reports"
)


# ==============================================================================
# FICHIERS D'ENTRÉE BASELINE
# ==============================================================================

BASELINE_METRICS_FILE = (
    BASELINE_DIR
    / "metrics_global.csv"
)

BASELINE_YEARLY_FILE = (
    BASELINE_DIR
    / "metrics_by_year.csv"
)

BASELINE_IMPORTANCE_FILE = (
    BASELINE_DIR
    / "feature_importance.csv"
)

BASELINE_PREDICTIONS_FILE = (
    BASELINE_DIR
    / "predictions_full.parquet"
)

BASELINE_CONFIGURATION_FILE = (
    BASELINE_DIR
    / "model_configuration.json"
)


# ==============================================================================
# FICHIERS D'ENTRÉE AVANCÉ
# ==============================================================================

ADVANCED_METRICS_FILE = (
    ADVANCED_DIR
    / "metrics_global.csv"
)

ADVANCED_YEARLY_FILE = (
    ADVANCED_DIR
    / "metrics_by_year.csv"
)

ADVANCED_IMPORTANCE_FILE = (
    ADVANCED_DIR
    / "feature_importance.csv"
)

ADVANCED_PREDICTIONS_FILE = (
    ADVANCED_DIR
    / "predictions_full.parquet"
)

ADVANCED_CONFIGURATION_FILE = (
    ADVANCED_DIR
    / "model_configuration.json"
)


# ==============================================================================
# FICHIERS DE SORTIE
# ==============================================================================

GLOBAL_COMPARISON_FILE = (
    TABLES_DIR
    / "global_model_comparison.csv"
)

YEARLY_COMPARISON_FILE = (
    TABLES_DIR
    / "yearly_model_comparison.csv"
)

PREDICTION_COMPARISON_FILE = (
    TABLES_DIR
    / "prediction_level_comparison.parquet"
)

PREDICTION_SUMMARY_FILE = (
    TABLES_DIR
    / "prediction_comparison_summary.csv"
)

GRIDPOINT_COMPARISON_FILE = (
    TABLES_DIR
    / "gridpoint_model_comparison.csv"
)

FEATURE_IMPORTANCE_COMPARISON_FILE = (
    TABLES_DIR
    / "feature_importance_comparison.csv"
)

COMPARISON_JSON_FILE = (
    REPORTS_DIR
    / "comparison_results.json"
)

COMPARISON_REPORT_FILE = (
    REPORTS_DIR
    / "comparison_summary.md"
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

EXPECTED_TEST_ROWS = 3_086_304

EXPECTED_GRID_POINTS = 9_892

EXPECTED_MONTHS_PER_GRIDPOINT = 312

RANDOM_STATE = 42

SCATTER_SAMPLE_SIZE = 100_000

FIGSIZE = (8, 6)

DPI = 300

FONT_SIZE = 12

TITLE_SIZE = 14


# ==============================================================================
# OUTILS
# ==============================================================================


def print_separator(
    title: str,
) -> None:

    print(
        "\n"
        + "=" * 76
    )

    print(
        title
    )

    print(
        "=" * 76
    )


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

    plt.close(
        fig
    )


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


def check_input_files() -> None:

    files = [
        BASELINE_METRICS_FILE,
        BASELINE_YEARLY_FILE,
        BASELINE_IMPORTANCE_FILE,
        BASELINE_PREDICTIONS_FILE,
        BASELINE_CONFIGURATION_FILE,
        ADVANCED_METRICS_FILE,
        ADVANCED_YEARLY_FILE,
        ADVANCED_IMPORTANCE_FILE,
        ADVANCED_PREDICTIONS_FILE,
        ADVANCED_CONFIGURATION_FILE,
    ]

    missing = [
        str(
            file_path
        )
        for file_path in files
        if not file_path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Fichier(s) manquant(s) :\n"
            + "\n".join(
                missing
            )
            + "\n\n"
            + "Vérifie que les étapes 03, 04 et 05 "
            + "ont été exécutées avec succès."
        )


# ==============================================================================
# CHARGEMENT
# ==============================================================================


def load_csv(
    file_path: Path,
    name: str,
) -> pd.DataFrame:

    print(
        f"Chargement {name} :\n"
        f"{file_path}"
    )

    df = pd.read_csv(
        file_path
    )

    print(
        f"Dimensions : "
        f"{df.shape}"
    )

    return df


def load_predictions(
    file_path: Path,
    name: str,
) -> pd.DataFrame:

    print(
        f"Chargement {name} :\n"
        f"{file_path}"
    )

    start = time.time()

    df = pd.read_parquet(
        file_path
    )

    df["DATE"] = pd.to_datetime(
        df["DATE"]
    )

    duration = (
        time.time()
        - start
    )

    memory_gb = (
        df
        .memory_usage(
            deep=True
        )
        .sum()
        / (1024 ** 3)
    )

    print(
        f"Dimensions : "
        f"{df.shape}"
    )

    print(
        f"Mémoire : "
        f"{memory_gb:.2f} Go"
    )

    print(
        f"Temps : "
        f"{format_duration(duration)}"
    )

    return df


def load_json(
    file_path: Path,
    name: str,
) -> dict:

    print(
        f"Chargement {name} :\n"
        f"{file_path}"
    )

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ==============================================================================
# NORMALISATION DES MÉTRIQUES
# ==============================================================================


def normalize_metric_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    rename_map = {}

    for column in df.columns:

        clean = (
            str(
                column
            )
            .strip()
        )

        upper = (
            clean
            .upper()
            .replace(
                "²",
                "2",
            )
        )

        if upper == "R2":

            rename_map[
                column
            ] = "R2"

        elif upper == "BIAS":

            rename_map[
                column
            ] = "Bias"

        elif upper == "MAE":

            rename_map[
                column
            ] = "MAE"

        elif upper == "RMSE":

            rename_map[
                column
            ] = "RMSE"

        elif upper == "MODEL":

            rename_map[
                column
            ] = "Model"

        elif upper == "YEAR":

            rename_map[
                column
            ] = "year"

    df.rename(
        columns=rename_map,
        inplace=True,
    )

    return df


# ==============================================================================
# EXTRACTION DES LIGNES XGBOOST
# ==============================================================================


def extract_baseline_metrics(
    metrics: pd.DataFrame,
) -> pd.Series:

    metrics = normalize_metric_columns(
        metrics
    )

    if "Model" not in metrics.columns:

        raise ValueError(
            "La colonne Model est absente "
            "des métriques baseline."
        )

    rows = (
        metrics[
            metrics[
                "Model"
            ]
            .astype(
                str
            )
            .str.lower()
            .eq(
                "xgboost"
            )
        ]
    )

    if len(
        rows
    ) == 0:

        raise ValueError(
            "Impossible de trouver la ligne XGBoost "
            "dans metrics_global.csv du baseline."
        )

    return (
        rows
        .iloc[0]
    )


def extract_advanced_metrics(
    metrics: pd.DataFrame,
) -> pd.Series:

    metrics = normalize_metric_columns(
        metrics
    )

    if "Model" not in metrics.columns:

        raise ValueError(
            "La colonne Model est absente "
            "des métriques avancées."
        )

    model_names = (
        metrics[
            "Model"
        ]
        .astype(
            str
        )
        .str.lower()
    )

    mask = (
        model_names.str.contains(
            "advanced"
        )
    )

    rows = (
        metrics[
            mask
        ]
    )

    if len(
        rows
    ) == 0:

        raise ValueError(
            "Impossible de trouver la ligne "
            "XGBoost_Advanced dans "
            "metrics_global.csv du modèle avancé."
        )

    return (
        rows
        .iloc[0]
    )


# ==============================================================================
# VALIDATION DES PRÉDICTIONS
# ==============================================================================


def validate_prediction_schema(
    df: pd.DataFrame,
    name: str,
) -> None:

    required = [
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
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{name} : "
            f"colonnes manquantes : "
            f"{missing}"
        )


def validate_test_dimensions(
    baseline: pd.DataFrame,
    advanced: pd.DataFrame,
) -> None:

    print_separator(
        "VALIDATION DES JEUX DE TEST"
    )

    print(
        f"Baseline : "
        f"{len(baseline):,} lignes"
    )

    print(
        f"Avancé   : "
        f"{len(advanced):,} lignes"
    )

    if (
        len(
            baseline
        )
        != len(
            advanced
        )
    ):

        raise ValueError(
            "Le baseline et le modèle avancé "
            "n'ont pas le même nombre de prédictions."
        )

    if (
        len(
            baseline
        )
        != EXPECTED_TEST_ROWS
    ):

        raise ValueError(
            "Le nombre de lignes du test "
            "ne correspond pas à la valeur attendue. "
            f"Attendu={EXPECTED_TEST_ROWS:,}, "
            f"obtenu={len(baseline):,}."
        )

    baseline_points = (
        baseline[
            [
                "LAMBX",
                "LAMBY",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    advanced_points = (
        advanced[
            [
                "LAMBX",
                "LAMBY",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        f"\nPoints baseline : "
        f"{baseline_points:,}"
    )

    print(
        f"Points avancé   : "
        f"{advanced_points:,}"
    )

    if (
        baseline_points
        != EXPECTED_GRID_POINTS
    ):

        raise ValueError(
            "Nombre incorrect de points "
            "dans le baseline."
        )

    if (
        advanced_points
        != EXPECTED_GRID_POINTS
    ):

        raise ValueError(
            "Nombre incorrect de points "
            "dans le modèle avancé."
        )

    print(
        "\nValidation des dimensions : OK"
    )


# ==============================================================================
# ALIGNEMENT DES PRÉDICTIONS
# ==============================================================================


def align_predictions(
    baseline: pd.DataFrame,
    advanced: pd.DataFrame,
) -> pd.DataFrame:

    print_separator(
        "ALIGNEMENT DES PRÉDICTIONS"
    )

    keys = [
        "DATE",
        "LAMBX",
        "LAMBY",
    ]

    baseline_tmp = (
        baseline[
            keys
            + [
                "T_observed",
                "T_predicted",
                "residual",
                "absolute_error",
            ]
        ]
        .copy()
    )

    advanced_tmp = (
        advanced[
            keys
            + [
                "T_observed",
                "T_predicted",
                "residual",
                "absolute_error",
            ]
        ]
        .copy()
    )

    baseline_tmp.rename(
        columns={
            "T_observed":
                "T_observed_baseline",

            "T_predicted":
                "T_predicted_baseline",

            "residual":
                "residual_baseline",

            "absolute_error":
                "absolute_error_baseline",
        },
        inplace=True,
    )

    advanced_tmp.rename(
        columns={
            "T_observed":
                "T_observed_advanced",

            "T_predicted":
                "T_predicted_advanced",

            "residual":
                "residual_advanced",

            "absolute_error":
                "absolute_error_advanced",
        },
        inplace=True,
    )

    merged = baseline_tmp.merge(
        advanced_tmp,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    print(
        f"Lignes alignées : "
        f"{len(merged):,}"
    )

    if (
        len(
            merged
        )
        != EXPECTED_TEST_ROWS
    ):

        raise ValueError(
            "L'alignement a perdu des observations."
        )


    # --------------------------------------------------------------------------
    # Vérification que la cible observée est identique
    # --------------------------------------------------------------------------

    observed_diff = np.abs(
        merged[
            "T_observed_baseline"
        ]
        -
        merged[
            "T_observed_advanced"
        ]
    )

    max_observed_diff = float(
        observed_diff.max()
    )

    print(
        f"Différence max sur T_observed : "
        f"{max_observed_diff:.10f}"
    )

    if (
        max_observed_diff
        > 1e-8
    ):

        raise ValueError(
            "Les observations utilisées par les deux "
            "modèles ne sont pas identiques."
        )

    merged[
        "T_observed"
    ] = (
        merged[
            "T_observed_baseline"
        ]
    )

    merged.drop(
        columns=[
            "T_observed_baseline",
            "T_observed_advanced",
        ],
        inplace=True,
    )


    # --------------------------------------------------------------------------
    # Gains au niveau prédiction
    #
    # gain_absolute_error > 0
    #     → le modèle avancé est meilleur
    # --------------------------------------------------------------------------

    merged[
        "gain_absolute_error"
    ] = (
        merged[
            "absolute_error_baseline"
        ]
        -
        merged[
            "absolute_error_advanced"
        ]
    )

    merged[
        "advanced_better"
    ] = (
        merged[
            "absolute_error_advanced"
        ]
        <
        merged[
            "absolute_error_baseline"
        ]
    )

    merged[
        "baseline_better"
    ] = (
        merged[
            "absolute_error_baseline"
        ]
        <
        merged[
            "absolute_error_advanced"
        ]
    )

    merged[
        "same_error"
    ] = np.isclose(
        merged[
            "absolute_error_baseline"
        ],
        merged[
            "absolute_error_advanced"
        ],
        atol=1e-10,
    )

    merged[
        "year"
    ] = (
        merged[
            "DATE"
        ]
        .dt.year
    )

    return merged


# ==============================================================================
# COMPARAISON GLOBALE
# ==============================================================================


def build_global_comparison(
    baseline_metrics: pd.DataFrame,
    advanced_metrics: pd.DataFrame,
) -> pd.DataFrame:

    print_separator(
        "COMPARAISON GLOBALE"
    )

    baseline = (
        extract_baseline_metrics(
            baseline_metrics
        )
    )

    advanced = (
        extract_advanced_metrics(
            advanced_metrics
        )
    )

    metrics = [
        "MAE",
        "RMSE",
        "R2",
        "Bias",
    ]

    rows = []

    for metric in metrics:

        baseline_value = float(
            baseline[
                metric
            ]
        )

        advanced_value = float(
            advanced[
                metric
            ]
        )

        if metric in [
            "MAE",
            "RMSE",
        ]:

            improvement_absolute = (
                baseline_value
                -
                advanced_value
            )

            improvement_percent = (
                improvement_absolute
                /
                baseline_value
                * 100
            )

        elif metric == "R2":

            improvement_absolute = (
                advanced_value
                -
                baseline_value
            )

            improvement_percent = (
                improvement_absolute
                /
                abs(
                    baseline_value
                )
                * 100
            )

        elif metric == "Bias":

            baseline_abs = abs(
                baseline_value
            )

            advanced_abs = abs(
                advanced_value
            )

            improvement_absolute = (
                baseline_abs
                -
                advanced_abs
            )

            if (
                baseline_abs
                > 0
            ):

                improvement_percent = (
                    improvement_absolute
                    /
                    baseline_abs
                    * 100
                )

            else:

                improvement_percent = np.nan

        else:

            improvement_absolute = np.nan
            improvement_percent = np.nan

        rows.append(
            {
                "metric":
                    metric,

                "baseline":
                    baseline_value,

                "advanced":
                    advanced_value,

                "improvement_absolute":
                    improvement_absolute,

                "improvement_percent":
                    improvement_percent,
            }
        )

    comparison = pd.DataFrame(
        rows
    )

    comparison.to_csv(
        GLOBAL_COMPARISON_FILE,
        index=False,
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        f"\nFichier :\n"
        f"{GLOBAL_COMPARISON_FILE}"
    )

    return comparison


# ==============================================================================
# COMPARAISON ANNUELLE
# ==============================================================================


def build_yearly_comparison(
    baseline_yearly: pd.DataFrame,
    advanced_yearly: pd.DataFrame,
) -> pd.DataFrame:

    print_separator(
        "COMPARAISON PAR ANNÉE"
    )

    baseline_yearly = (
        normalize_metric_columns(
            baseline_yearly
        )
    )

    advanced_yearly = (
        normalize_metric_columns(
            advanced_yearly
        )
    )

    required = [
        "year",
        "MAE",
        "RMSE",
        "R2",
        "Bias",
    ]

    for name, df in [
        (
            "baseline",
            baseline_yearly,
        ),
        (
            "advanced",
            advanced_yearly,
        ),
    ]:

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Métriques annuelles {name} : "
                f"colonnes manquantes : {missing}"
            )

    baseline_tmp = (
        baseline_yearly[
            required
        ]
        .copy()
    )

    advanced_tmp = (
        advanced_yearly[
            required
        ]
        .copy()
    )

    baseline_tmp.rename(
        columns={
            "MAE":
                "MAE_baseline",

            "RMSE":
                "RMSE_baseline",

            "R2":
                "R2_baseline",

            "Bias":
                "Bias_baseline",
        },
        inplace=True,
    )

    advanced_tmp.rename(
        columns={
            "MAE":
                "MAE_advanced",

            "RMSE":
                "RMSE_advanced",

            "R2":
                "R2_advanced",

            "Bias":
                "Bias_advanced",
        },
        inplace=True,
    )

    comparison = baseline_tmp.merge(
        advanced_tmp,
        on="year",
        how="inner",
        validate="one_to_one",
    )

    comparison[
        "MAE_gain"
    ] = (
        comparison[
            "MAE_baseline"
        ]
        -
        comparison[
            "MAE_advanced"
        ]
    )

    comparison[
        "MAE_gain_percent"
    ] = (
        comparison[
            "MAE_gain"
        ]
        /
        comparison[
            "MAE_baseline"
        ]
        * 100
    )

    comparison[
        "RMSE_gain"
    ] = (
        comparison[
            "RMSE_baseline"
        ]
        -
        comparison[
            "RMSE_advanced"
        ]
    )

    comparison[
        "R2_gain"
    ] = (
        comparison[
            "R2_advanced"
        ]
        -
        comparison[
            "R2_baseline"
        ]
    )

    comparison[
        "Bias_abs_gain"
    ] = (
        comparison[
            "Bias_baseline"
        ]
        .abs()
        -
        comparison[
            "Bias_advanced"
        ]
        .abs()
    )

    comparison[
        "advanced_better_MAE"
    ] = (
        comparison[
            "MAE_advanced"
        ]
        <
        comparison[
            "MAE_baseline"
        ]
    )

    comparison.to_csv(
        YEARLY_COMPARISON_FILE,
        index=False,
    )

    better_years = int(
        comparison[
            "advanced_better_MAE"
        ]
        .sum()
    )

    print(
        f"Années comparées : "
        f"{len(comparison)}"
    )

    print(
        f"Années avec meilleure MAE "
        f"pour le modèle avancé : "
        f"{better_years}/{len(comparison)}"
    )

    print(
        f"Fichier :\n"
        f"{YEARLY_COMPARISON_FILE}"
    )

    return comparison


# ==============================================================================
# COMPARAISON PRÉDICTION PAR PRÉDICTION
# ==============================================================================


def summarize_prediction_comparison(
    merged: pd.DataFrame,
) -> pd.DataFrame:

    print_separator(
        "COMPARAISON PRÉDICTION PAR PRÉDICTION"
    )

    advanced_better = int(
        merged[
            "advanced_better"
        ]
        .sum()
    )

    baseline_better = int(
        merged[
            "baseline_better"
        ]
        .sum()
    )

    same_error = int(
        merged[
            "same_error"
        ]
        .sum()
    )

    total = len(
        merged
    )

    mean_gain = float(
        merged[
            "gain_absolute_error"
        ]
        .mean()
    )

    median_gain = float(
        merged[
            "gain_absolute_error"
        ]
        .median()
    )

    summary = pd.DataFrame(
        [
            {
                "n_predictions":
                    total,

                "advanced_better_count":
                    advanced_better,

                "advanced_better_percent":
                    (
                        advanced_better
                        /
                        total
                        * 100
                    ),

                "baseline_better_count":
                    baseline_better,

                "baseline_better_percent":
                    (
                        baseline_better
                        /
                        total
                        * 100
                    ),

                "same_error_count":
                    same_error,

                "same_error_percent":
                    (
                        same_error
                        /
                        total
                        * 100
                    ),

                "mean_absolute_error_gain":
                    mean_gain,

                "median_absolute_error_gain":
                    median_gain,
            }
        ]
    )

    summary.to_csv(
        PREDICTION_SUMMARY_FILE,
        index=False,
    )

    merged.to_parquet(
        PREDICTION_COMPARISON_FILE,
        index=False,
        engine="pyarrow",
    )

    print(
        f"Modèle avancé meilleur : "
        f"{advanced_better:,} "
        f"({advanced_better / total * 100:.2f} %)"
    )

    print(
        f"Baseline meilleur : "
        f"{baseline_better:,} "
        f"({baseline_better / total * 100:.2f} %)"
    )

    print(
        f"Erreurs identiques : "
        f"{same_error:,} "
        f"({same_error / total * 100:.2f} %)"
    )

    print(
        f"\nGain absolu moyen sur l'erreur : "
        f"{mean_gain:.4f} °C"
    )

    return summary


# ==============================================================================
# MÉTRIQUES PAR POINT DE GRILLE
# ==============================================================================


def compute_gridpoint_metrics(
    merged: pd.DataFrame,
) -> pd.DataFrame:

    print_separator(
        "COMPARAISON SPATIALE"
    )

    rows = []

    grouped = merged.groupby(
        [
            "LAMBX",
            "LAMBY",
        ],
        sort=False,
    )

    print(
        f"Points détectés : "
        f"{grouped.ngroups:,}"
    )

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

        baseline_pred = (
            group[
                "T_predicted_baseline"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        advanced_pred = (
            group[
                "T_predicted_advanced"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        baseline_residual = (
            observed
            -
            baseline_pred
        )

        advanced_residual = (
            observed
            -
            advanced_pred
        )

        baseline_mae = float(
            np.mean(
                np.abs(
                    baseline_residual
                )
            )
        )

        advanced_mae = float(
            np.mean(
                np.abs(
                    advanced_residual
                )
            )
        )

        baseline_rmse = float(
            np.sqrt(
                np.mean(
                    baseline_residual ** 2
                )
            )
        )

        advanced_rmse = float(
            np.sqrt(
                np.mean(
                    advanced_residual ** 2
                )
            )
        )

        baseline_bias = float(
            np.mean(
                baseline_residual
            )
        )

        advanced_bias = float(
            np.mean(
                advanced_residual
            )
        )

        if (
            len(
                observed
            )
            > 1
            and np.var(
                observed
            )
            > 0
        ):

            baseline_r2 = float(
                r2_score(
                    observed,
                    baseline_pred,
                )
            )

            advanced_r2 = float(
                r2_score(
                    observed,
                    advanced_pred,
                )
            )

        else:

            baseline_r2 = np.nan
            advanced_r2 = np.nan

        mae_gain = (
            baseline_mae
            -
            advanced_mae
        )

        rmse_gain = (
            baseline_rmse
            -
            advanced_rmse
        )

        r2_gain = (
            advanced_r2
            -
            baseline_r2
        )

        bias_abs_gain = (
            abs(
                baseline_bias
            )
            -
            abs(
                advanced_bias
            )
        )

        rows.append(
            {
                "LAMBX":
                    lambx,

                "LAMBY":
                    lamby,

                "n_observations":
                    int(
                        len(
                            group
                        )
                    ),

                "MAE_baseline":
                    baseline_mae,

                "MAE_advanced":
                    advanced_mae,

                "MAE_gain":
                    mae_gain,

                "MAE_gain_percent":
                    (
                        mae_gain
                        /
                        baseline_mae
                        * 100
                        if baseline_mae > 0
                        else np.nan
                    ),

                "RMSE_baseline":
                    baseline_rmse,

                "RMSE_advanced":
                    advanced_rmse,

                "RMSE_gain":
                    rmse_gain,

                "R2_baseline":
                    baseline_r2,

                "R2_advanced":
                    advanced_r2,

                "R2_gain":
                    r2_gain,

                "Bias_baseline":
                    baseline_bias,

                "Bias_advanced":
                    advanced_bias,

                "Bias_abs_gain":
                    bias_abs_gain,

                "advanced_better_MAE":
                    advanced_mae
                    <
                    baseline_mae,
            }
        )

    comparison = pd.DataFrame(
        rows
    )

    if (
        len(
            comparison
        )
        != EXPECTED_GRID_POINTS
    ):

        raise ValueError(
            "Le nombre de points de grille comparés "
            "est incorrect. "
            f"Attendu={EXPECTED_GRID_POINTS:,}, "
            f"obtenu={len(comparison):,}."
        )

    observations_per_point = (
        comparison[
            "n_observations"
        ]
        .unique()
    )

    if (
        len(
            observations_per_point
        )
        != 1
        or observations_per_point[
            0
        ]
        != EXPECTED_MONTHS_PER_GRIDPOINT
    ):

        raise ValueError(
            "Le nombre d'observations par point "
            "n'est pas égal à 312."
        )

    comparison.to_csv(
        GRIDPOINT_COMPARISON_FILE,
        index=False,
    )

    better_points = int(
        comparison[
            "advanced_better_MAE"
        ]
        .sum()
    )

    print(
        f"Points analysés : "
        f"{len(comparison):,}"
    )

    print(
        f"Points où le modèle avancé "
        f"a une meilleure MAE : "
        f"{better_points:,} "
        f"({better_points / len(comparison) * 100:.2f} %)"
    )

    print(
        f"Fichier :\n"
        f"{GRIDPOINT_COMPARISON_FILE}"
    )

    return comparison


# ==============================================================================
# IMPORTANCE DES VARIABLES
# ==============================================================================


def build_feature_importance_comparison(
    baseline_importance: pd.DataFrame,
    advanced_importance: pd.DataFrame,
) -> pd.DataFrame:

    print_separator(
        "COMPARAISON DES IMPORTANCES"
    )

    required = [
        "feature",
        "importance",
    ]

    for name, df in [
        (
            "baseline",
            baseline_importance,
        ),
        (
            "advanced",
            advanced_importance,
        ),
    ]:

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Importance {name} : "
                f"colonnes manquantes : {missing}"
            )

    baseline_tmp = (
        baseline_importance[
            required
        ]
        .copy()
    )

    advanced_tmp = (
        advanced_importance[
            required
        ]
        .copy()
    )

    baseline_tmp.rename(
        columns={
            "importance":
                "importance_baseline",
        },
        inplace=True,
    )

    advanced_tmp.rename(
        columns={
            "importance":
                "importance_advanced",
        },
        inplace=True,
    )

    comparison = baseline_tmp.merge(
        advanced_tmp,
        on="feature",
        how="outer",
    )

    comparison[
        "importance_baseline"
    ] = (
        comparison[
            "importance_baseline"
        ]
        .fillna(
            0
        )
    )

    comparison[
        "importance_advanced"
    ] = (
        comparison[
            "importance_advanced"
        ]
        .fillna(
            0
        )
    )

    comparison[
        "importance_change"
    ] = (
        comparison[
            "importance_advanced"
        ]
        -
        comparison[
            "importance_baseline"
        ]
    )

    comparison.sort_values(
        "importance_advanced",
        ascending=False,
        inplace=True,
    )

    comparison.reset_index(
        drop=True,
        inplace=True,
    )

    comparison.to_csv(
        FEATURE_IMPORTANCE_COMPARISON_FILE,
        index=False,
    )

    print(
        comparison
        .head(
            31
        )
        .to_string(
            index=False
        )
    )

    return comparison


# ==============================================================================
# FIGURE : MÉTRIQUES GLOBALES
# ==============================================================================


def plot_global_metrics(
    global_comparison: pd.DataFrame,
) -> None:

    print_separator(
        "FIGURE : COMPARAISON DES MÉTRIQUES GLOBALES"
    )

    metrics_to_plot = [
        "MAE",
        "RMSE",
    ]

    subset = (
        global_comparison[
            global_comparison[
                "metric"
            ]
            .isin(
                metrics_to_plot
            )
        ]
        .copy()
    )

    x = np.arange(
        len(
            subset
        )
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.bar(
        x
        -
        width / 2,
        subset[
            "baseline"
        ],
        width=width,
        label="Baseline",
    )

    ax.bar(
        x
        +
        width / 2,
        subset[
            "advanced"
        ],
        width=width,
        label="Avancé",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        subset[
            "metric"
        ]
    )

    ax.set_ylabel(
        "Erreur (°C)",
        fontsize=FONT_SIZE,
    )

    ax.set_title(
        "Comparaison globale des erreurs",
        fontsize=TITLE_SIZE,
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_01_global_error_comparison",
    )


# ==============================================================================
# FIGURE : R2 GLOBAL
# ==============================================================================


def plot_global_r2(
    global_comparison: pd.DataFrame,
) -> None:

    print_separator(
        "FIGURE : COMPARAISON DU R2 GLOBAL"
    )

    row = (
        global_comparison[
            global_comparison[
                "metric"
            ]
            == "R2"
        ]
        .iloc[0]
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.bar(
        [
            "Baseline",
            "Avancé",
        ],
        [
            row[
                "baseline"
            ],
            row[
                "advanced"
            ],
        ],
    )

    ax.set_ylabel(
        "R²",
        fontsize=FONT_SIZE,
    )

    ax.set_title(
        "Comparaison du R² global",
        fontsize=TITLE_SIZE,
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_02_global_r2_comparison",
    )


# ==============================================================================
# FIGURE : MAE ANNUELLE
# ==============================================================================


def plot_yearly_mae_comparison(
    yearly_comparison: pd.DataFrame,
) -> None:

    print_separator(
        "FIGURE : MAE ANNUELLE"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        yearly_comparison[
            "year"
        ],
        yearly_comparison[
            "MAE_baseline"
        ],
        marker="o",
        label="Baseline",
    )

    ax.plot(
        yearly_comparison[
            "year"
        ],
        yearly_comparison[
            "MAE_advanced"
        ],
        marker="o",
        label="Avancé",
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
        "Comparaison annuelle de la MAE",
        fontsize=TITLE_SIZE,
    )

    ax.legend()

    ax.grid(
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_03_yearly_mae_comparison",
    )


# ==============================================================================
# FIGURE : GAIN MAE ANNUEL
# ==============================================================================


def plot_yearly_mae_gain(
    yearly_comparison: pd.DataFrame,
) -> None:

    print_separator(
        "FIGURE : GAIN ANNUEL DE MAE"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        yearly_comparison[
            "year"
        ],
        yearly_comparison[
            "MAE_gain"
        ],
    )

    ax.axhline(
        0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Année",
        fontsize=FONT_SIZE,
    )

    ax.set_ylabel(
        "MAE baseline - MAE avancée (°C)",
        fontsize=FONT_SIZE,
    )

    ax.set_title(
        "Gain annuel de MAE du modèle avancé",
        fontsize=TITLE_SIZE,
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_04_yearly_mae_gain",
    )


# ==============================================================================
# FIGURE : R2 ANNUEL
# ==============================================================================


def plot_yearly_r2_comparison(
    yearly_comparison: pd.DataFrame,
) -> None:

    print_separator(
        "FIGURE : R2 ANNUEL"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        yearly_comparison[
            "year"
        ],
        yearly_comparison[
            "R2_baseline"
        ],
        marker="o",
        label="Baseline",
    )

    ax.plot(
        yearly_comparison[
            "year"
        ],
        yearly_comparison[
            "R2_advanced"
        ],
        marker="o",
        label="Avancé",
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
        "Comparaison annuelle du R²",
        fontsize=TITLE_SIZE,
    )

    ax.legend()

    ax.grid(
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_05_yearly_r2_comparison",
    )


# ==============================================================================
# FIGURE : DISTRIBUTION DU GAIN D'ERREUR
# ==============================================================================


def plot_prediction_gain_distribution(
    merged: pd.DataFrame,
) -> None:

    print_separator(
        "FIGURE : DISTRIBUTION DU GAIN D'ERREUR"
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.hist(
        merged[
            "gain_absolute_error"
        ],
        bins=150,
        alpha=0.8,
    )

    ax.axvline(
        0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Erreur baseline - erreur avancée (°C)",
        fontsize=FONT_SIZE,
    )

    ax.set_ylabel(
        "Fréquence",
        fontsize=FONT_SIZE,
    )

    ax.set_title(
        "Distribution du gain prédiction par prédiction",
        fontsize=TITLE_SIZE,
    )

    ax.grid(
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_06_prediction_error_gain_distribution",
    )


# ==============================================================================
# FIGURE : ERREURS ABSOLUES BASELINE VS AVANCÉ
# ==============================================================================


def plot_absolute_error_scatter(
    merged: pd.DataFrame,
) -> None:

    print_separator(
        "FIGURE : ERREURS ABSOLUES BASELINE VS AVANCÉ"
    )

    sample_size = min(
        SCATTER_SAMPLE_SIZE,
        len(
            merged
        ),
    )

    sample = (
        merged
        .sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        )
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.scatter(
        sample[
            "absolute_error_baseline"
        ],
        sample[
            "absolute_error_advanced"
        ],
        s=6,
        alpha=0.25,
    )

    max_value = max(
        sample[
            "absolute_error_baseline"
        ]
        .max(),
        sample[
            "absolute_error_advanced"
        ]
        .max(),
    )

    ax.plot(
        [
            0,
            max_value,
        ],
        [
            0,
            max_value,
        ],
        linestyle="--",
    )

    ax.set_xlabel(
        "Erreur absolue baseline (°C)",
        fontsize=FONT_SIZE,
    )

    ax.set_ylabel(
        "Erreur absolue avancée (°C)",
        fontsize=FONT_SIZE,
    )

    ax.set_title(
        "Erreur absolue : baseline vs modèle avancé",
        fontsize=TITLE_SIZE,
    )

    ax.grid(
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_07_absolute_error_scatter",
    )


# ==============================================================================
# CARTES SPATIALES
# ==============================================================================


def plot_grid_metric(
    grid_comparison: pd.DataFrame,
    column: str,
    title: str,
    label: str,
    filename: str,
    cmap: str = "viridis",
    symmetric: bool = False,
) -> None:

    print_separator(
        f"CARTE : {title.upper()}"
    )

    fig, ax = plt.subplots(
        figsize=(8, 9)
    )

    kwargs = {}

    if symmetric:

        max_abs = float(
            grid_comparison[
                column
            ]
            .abs()
            .max()
        )

        kwargs[
            "vmin"
        ] = -max_abs

        kwargs[
            "vmax"
        ] = max_abs

    scatter = ax.scatter(
        grid_comparison[
            "LAMBX"
        ],
        grid_comparison[
            "LAMBY"
        ],
        c=grid_comparison[
            column
        ],
        s=12,
        marker="s",
        linewidths=0,
        cmap=cmap,
        **kwargs,
    )

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        shrink=0.8,
    )

    colorbar.set_label(
        label,
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
        alpha=0.1,
    )

    fig.tight_layout()

    save_figure(
        fig,
        filename,
    )


def plot_mae_gain_map(
    grid_comparison: pd.DataFrame,
) -> None:

    plot_grid_metric(
        grid_comparison=
            grid_comparison,

        column=
            "MAE_gain",

        title=
            "Gain spatial de MAE du modèle avancé",

        label=
            "MAE baseline - MAE avancée (°C)",

        filename=
            "Figure_08_spatial_mae_gain",

        cmap=
            "coolwarm",

        symmetric=
            True,
    )


def plot_rmse_gain_map(
    grid_comparison: pd.DataFrame,
) -> None:

    plot_grid_metric(
        grid_comparison=
            grid_comparison,

        column=
            "RMSE_gain",

        title=
            "Gain spatial de RMSE du modèle avancé",

        label=
            "RMSE baseline - RMSE avancée (°C)",

        filename=
            "Figure_09_spatial_rmse_gain",

        cmap=
            "coolwarm",

        symmetric=
            True,
    )


def plot_r2_gain_map(
    grid_comparison: pd.DataFrame,
) -> None:

    plot_grid_metric(
        grid_comparison=
            grid_comparison,

        column=
            "R2_gain",

        title=
            "Gain spatial de R² du modèle avancé",

        label=
            "R² avancé - R² baseline",

        filename=
            "Figure_10_spatial_r2_gain",

        cmap=
            "coolwarm",

        symmetric=
            True,
    )


def plot_bias_gain_map(
    grid_comparison: pd.DataFrame,
) -> None:

    plot_grid_metric(
        grid_comparison=
            grid_comparison,

        column=
            "Bias_abs_gain",

        title=
            "Gain spatial sur le biais absolu",

        label=
            "|Bias baseline| - |Bias avancé| (°C)",

        filename=
            "Figure_11_spatial_bias_gain",

        cmap=
            "coolwarm",

        symmetric=
            True,
    )


# ==============================================================================
# FIGURE : IMPORTANCE DES VARIABLES AVANCÉES
# ==============================================================================


def plot_advanced_feature_importance(
    comparison: pd.DataFrame,
) -> None:

    print_separator(
        "FIGURE : IMPORTANCE DES VARIABLES AVANCÉES"
    )

    df = (
        comparison
        .sort_values(
            "importance_advanced",
            ascending=False,
        )
        .head(
            20
        )
        .sort_values(
            "importance_advanced",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.barh(
        df[
            "feature"
        ],
        df[
            "importance_advanced"
        ],
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
        "Top 20 des variables du modèle avancé",
        fontsize=TITLE_SIZE,
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_12_advanced_feature_importance",
    )


# ==============================================================================
# EXPORT JSON
# ==============================================================================


def export_comparison_json(
    global_comparison: pd.DataFrame,
    yearly_comparison: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    grid_comparison: pd.DataFrame,
    baseline_configuration: dict,
    advanced_configuration: dict,
) -> None:

    print_separator(
        "EXPORT JSON"
    )

    global_dict = {}

    for _, row in (
        global_comparison
        .iterrows()
    ):

        global_dict[
            row[
                "metric"
            ]
        ] = {
            "baseline":
                float(
                    row[
                        "baseline"
                    ]
                ),

            "advanced":
                float(
                    row[
                        "advanced"
                    ]
                ),

            "improvement_absolute":
                float(
                    row[
                        "improvement_absolute"
                    ]
                ),

            "improvement_percent":
                (
                    float(
                        row[
                            "improvement_percent"
                        ]
                    )
                    if pd.notna(
                        row[
                            "improvement_percent"
                        ]
                    )
                    else None
                ),
        }

    prediction_row = (
        prediction_summary
        .iloc[0]
    )

    better_years = int(
        yearly_comparison[
            "advanced_better_MAE"
        ]
        .sum()
    )

    better_points = int(
        grid_comparison[
            "advanced_better_MAE"
        ]
        .sum()
    )

    output = {

        "global_comparison":
            global_dict,

        "temporal_comparison": {

            "n_years":
                int(
                    len(
                        yearly_comparison
                    )
                ),

            "advanced_better_mae_years":
                better_years,

            "advanced_better_mae_percent":
                (
                    better_years
                    /
                    len(
                        yearly_comparison
                    )
                    * 100
                ),
        },

        "prediction_level_comparison": {

            "n_predictions":
                int(
                    prediction_row[
                        "n_predictions"
                    ]
                ),

            "advanced_better_count":
                int(
                    prediction_row[
                        "advanced_better_count"
                    ]
                ),

            "advanced_better_percent":
                float(
                    prediction_row[
                        "advanced_better_percent"
                    ]
                ),

            "baseline_better_count":
                int(
                    prediction_row[
                        "baseline_better_count"
                    ]
                ),

            "baseline_better_percent":
                float(
                    prediction_row[
                        "baseline_better_percent"
                    ]
                ),

            "mean_absolute_error_gain":
                float(
                    prediction_row[
                        "mean_absolute_error_gain"
                    ]
                ),
        },

        "spatial_comparison": {

            "n_grid_points":
                int(
                    len(
                        grid_comparison
                    )
                ),

            "advanced_better_mae_points":
                better_points,

            "advanced_better_mae_percent":
                (
                    better_points
                    /
                    len(
                        grid_comparison
                    )
                    * 100
                ),

            "mean_mae_gain":
                float(
                    grid_comparison[
                        "MAE_gain"
                    ]
                    .mean()
                ),

            "median_mae_gain":
                float(
                    grid_comparison[
                        "MAE_gain"
                    ]
                    .median()
                ),

            "mean_r2_gain":
                float(
                    grid_comparison[
                        "R2_gain"
                    ]
                    .mean()
                ),
        },

        "baseline_model": {

            "n_features":
                len(
                    baseline_configuration.get(
                        "features",
                        [],
                    )
                ),

            "features":
                baseline_configuration.get(
                    "features",
                    [],
                ),
        },

        "advanced_model": {

            "n_features":
                advanced_configuration.get(
                    "n_features",
                    len(
                        advanced_configuration.get(
                            "features",
                            [],
                        )
                    ),
                ),

            "features":
                advanced_configuration.get(
                    "features",
                    [],
                ),

            "best_n_estimators":
                advanced_configuration.get(
                    "best_n_estimators"
                ),
        },
    }

    with open(
        COMPARISON_JSON_FILE,
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
        COMPARISON_JSON_FILE
    )


# ==============================================================================
# RAPPORT MARKDOWN
# ==============================================================================


def write_markdown_report(
    global_comparison: pd.DataFrame,
    yearly_comparison: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    grid_comparison: pd.DataFrame,
    feature_comparison: pd.DataFrame,
) -> None:

    print_separator(
        "RAPPORT MARKDOWN"
    )

    def get_metric(
        metric_name: str,
        column: str,
    ) -> float:

        return float(
            global_comparison.loc[
                global_comparison[
                    "metric"
                ]
                == metric_name,
                column,
            ]
            .iloc[0]
        )

    baseline_mae = get_metric(
        "MAE",
        "baseline",
    )

    advanced_mae = get_metric(
        "MAE",
        "advanced",
    )

    mae_gain = get_metric(
        "MAE",
        "improvement_absolute",
    )

    mae_gain_percent = get_metric(
        "MAE",
        "improvement_percent",
    )

    baseline_rmse = get_metric(
        "RMSE",
        "baseline",
    )

    advanced_rmse = get_metric(
        "RMSE",
        "advanced",
    )

    rmse_gain_percent = get_metric(
        "RMSE",
        "improvement_percent",
    )

    baseline_r2 = get_metric(
        "R2",
        "baseline",
    )

    advanced_r2 = get_metric(
        "R2",
        "advanced",
    )

    r2_gain = get_metric(
        "R2",
        "improvement_absolute",
    )

    baseline_bias = get_metric(
        "Bias",
        "baseline",
    )

    advanced_bias = get_metric(
        "Bias",
        "advanced",
    )

    prediction_row = (
        prediction_summary
        .iloc[0]
    )

    better_years = int(
        yearly_comparison[
            "advanced_better_MAE"
        ]
        .sum()
    )

    total_years = len(
        yearly_comparison
    )

    better_points = int(
        grid_comparison[
            "advanced_better_MAE"
        ]
        .sum()
    )

    total_points = len(
        grid_comparison
    )

    best_year_row = (
        yearly_comparison
        .sort_values(
            "MAE_gain",
            ascending=False,
        )
        .iloc[0]
    )

    worst_year_row = (
        yearly_comparison
        .sort_values(
            "MAE_gain",
            ascending=True,
        )
        .iloc[0]
    )

    best_grid_row = (
        grid_comparison
        .sort_values(
            "MAE_gain",
            ascending=False,
        )
        .iloc[0]
    )

    worst_grid_row = (
        grid_comparison
        .sort_values(
            "MAE_gain",
            ascending=True,
        )
        .iloc[0]
    )

    top_advanced_features = (
        feature_comparison
        .sort_values(
            "importance_advanced",
            ascending=False,
        )
        .head(
            10
        )
    )

    feature_lines = ""

    for _, row in (
        top_advanced_features
        .iterrows()
    ):

        feature_lines += (
            f"- {row['feature']} : "
            f"{row['importance_advanced']:.4f}\n"
        )

    report = (
        "# Comparaison XGBoost baseline vs XGBoost avancé\n\n"

        "## 1. Résumé\n\n"

        "Les deux modèles sont évalués sur le même jeu de test, "
        "composé de **3 086 304 observations mensuelles**, "
        "réparties sur **9 892 points de grille**, "
        "pour la période **2000–2025**.\n\n"

        "Le modèle avancé utilise davantage de variables historiques, "
        "notamment des retards thermiques supplémentaires et des "
        "variables physiques retardées.\n\n"

        "## 2. Performance globale\n\n"

        "| Métrique | Baseline | Avancé | Gain |\n"
        "|---|---:|---:|---:|\n"

        f"| MAE | {baseline_mae:.4f} °C | "
        f"{advanced_mae:.4f} °C | "
        f"{mae_gain:.4f} °C "
        f"({mae_gain_percent:.2f} %) |\n"

        f"| RMSE | {baseline_rmse:.4f} °C | "
        f"{advanced_rmse:.4f} °C | "
        f"{rmse_gain_percent:.2f} % |\n"

        f"| R² | {baseline_r2:.4f} | "
        f"{advanced_r2:.4f} | "
        f"+{r2_gain:.4f} |\n"

        f"| Bias | {baseline_bias:.4f} °C | "
        f"{advanced_bias:.4f} °C | "
        "réduction évaluée en valeur absolue |\n\n"

        "## 3. Comparaison temporelle\n\n"

        f"- Nombre d'années comparées : **{total_years}**\n"

        f"- Années où le modèle avancé présente une meilleure MAE : "
        f"**{better_years}/{total_years} "
        f"({better_years / total_years * 100:.2f} %)**\n"

        f"- Plus forte amélioration annuelle : "
        f"**{int(best_year_row['year'])}**, "
        f"gain de MAE = "
        f"**{best_year_row['MAE_gain']:.4f} °C**\n"

        f"- Année la moins favorable au modèle avancé : "
        f"**{int(worst_year_row['year'])}**, "
        f"gain de MAE = "
        f"**{worst_year_row['MAE_gain']:.4f} °C**\n\n"

        "## 4. Comparaison prédiction par prédiction\n\n"

        f"- Nombre total de prédictions : "
        f"**{int(prediction_row['n_predictions']):,}**\n"

        f"- Modèle avancé meilleur : "
        f"**{int(prediction_row['advanced_better_count']):,} "
        f"({prediction_row['advanced_better_percent']:.2f} %)**\n"

        f"- Baseline meilleur : "
        f"**{int(prediction_row['baseline_better_count']):,} "
        f"({prediction_row['baseline_better_percent']:.2f} %)**\n"

        f"- Gain moyen d'erreur absolue : "
        f"**{prediction_row['mean_absolute_error_gain']:.4f} °C**\n\n"

        "## 5. Comparaison spatiale\n\n"

        f"- Points de grille comparés : "
        f"**{total_points:,}**\n"

        f"- Points où le modèle avancé a une meilleure MAE : "
        f"**{better_points:,}/{total_points:,} "
        f"({better_points / total_points * 100:.2f} %)**\n"

        f"- Gain spatial moyen de MAE : "
        f"**{grid_comparison['MAE_gain'].mean():.4f} °C**\n"

        f"- Gain spatial médian de MAE : "
        f"**{grid_comparison['MAE_gain'].median():.4f} °C**\n"

        f"- Plus forte amélioration spatiale : "
        f"point **({int(best_grid_row['LAMBX'])}, "
        f"{int(best_grid_row['LAMBY'])})**, "
        f"gain de MAE = "
        f"**{best_grid_row['MAE_gain']:.4f} °C**\n"

        f"- Point le moins favorable au modèle avancé : "
        f"**({int(worst_grid_row['LAMBX'])}, "
        f"{int(worst_grid_row['LAMBY'])})**, "
        f"gain de MAE = "
        f"**{worst_grid_row['MAE_gain']:.4f} °C**\n\n"

        "## 6. Variables les plus importantes du modèle avancé\n\n"

        f"{feature_lines}\n"

        "## 7. Interprétation\n\n"

        "Un gain de MAE positif signifie que le modèle avancé "
        "réduit l'erreur par rapport au modèle baseline. "
        "De même, un gain positif de RMSE indique une amélioration, "
        "alors qu'un gain positif de R² correspond à une augmentation "
        "du pouvoir explicatif du modèle avancé.\n\n"

        "Les comparaisons spatiales permettent d'identifier les régions "
        "où l'ajout des nouvelles variables améliore réellement les "
        "prédictions et celles où la baseline reste compétitive.\n\n"

        "## 8. Note méthodologique\n\n"

        "Les deux modèles sont comparés dans le même cadre "
        "one-step-ahead. Les variables retardées du test utilisent "
        "les observations historiques précédentes disponibles. "
        "Cette comparaison mesure donc l'amélioration de la prédiction "
        "mensuelle conditionnelle à l'historique observé, et non une "
        "simulation récursive autonome sur 2000–2025.\n"
    )

    with open(
        COMPARISON_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            report
        )

    print(
        COMPARISON_REPORT_FILE
    )


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:

    global_start = time.time()

    print_separator(
        "COMPARAISON DES MODÈLES XGBOOST"
    )


    # --------------------------------------------------------------------------
    # 1. Initialisation
    # --------------------------------------------------------------------------

    create_output_directories()

    check_input_files()


    # --------------------------------------------------------------------------
    # 2. Chargement des métriques
    # --------------------------------------------------------------------------

    print_separator(
        "CHARGEMENT DES MÉTRIQUES"
    )

    baseline_metrics = load_csv(
        BASELINE_METRICS_FILE,
        "métriques baseline",
    )

    advanced_metrics = load_csv(
        ADVANCED_METRICS_FILE,
        "métriques avancées",
    )

    baseline_yearly = load_csv(
        BASELINE_YEARLY_FILE,
        "métriques annuelles baseline",
    )

    advanced_yearly = load_csv(
        ADVANCED_YEARLY_FILE,
        "métriques annuelles avancées",
    )


    # --------------------------------------------------------------------------
    # 3. Importances
    # --------------------------------------------------------------------------

    print_separator(
        "CHARGEMENT DES IMPORTANCES"
    )

    baseline_importance = load_csv(
        BASELINE_IMPORTANCE_FILE,
        "importance baseline",
    )

    advanced_importance = load_csv(
        ADVANCED_IMPORTANCE_FILE,
        "importance avancée",
    )


    # --------------------------------------------------------------------------
    # 4. Configurations
    # --------------------------------------------------------------------------

    print_separator(
        "CHARGEMENT DES CONFIGURATIONS"
    )

    baseline_configuration = load_json(
        BASELINE_CONFIGURATION_FILE,
        "configuration baseline",
    )

    advanced_configuration = load_json(
        ADVANCED_CONFIGURATION_FILE,
        "configuration avancée",
    )


    # --------------------------------------------------------------------------
    # 5. Prédictions
    # --------------------------------------------------------------------------

    print_separator(
        "CHARGEMENT DES PRÉDICTIONS"
    )

    baseline_predictions = load_predictions(
        BASELINE_PREDICTIONS_FILE,
        "prédictions baseline",
    )

    advanced_predictions = load_predictions(
        ADVANCED_PREDICTIONS_FILE,
        "prédictions avancées",
    )


    # --------------------------------------------------------------------------
    # 6. Contrôles
    # --------------------------------------------------------------------------

    validate_prediction_schema(
        baseline_predictions,
        "Baseline",
    )

    validate_prediction_schema(
        advanced_predictions,
        "Avancé",
    )

    validate_test_dimensions(
        baseline_predictions,
        advanced_predictions,
    )


    # --------------------------------------------------------------------------
    # 7. Comparaison globale
    # --------------------------------------------------------------------------

    global_comparison = (
        build_global_comparison(
            baseline_metrics,
            advanced_metrics,
        )
    )


    # --------------------------------------------------------------------------
    # 8. Comparaison annuelle
    # --------------------------------------------------------------------------

    yearly_comparison = (
        build_yearly_comparison(
            baseline_yearly,
            advanced_yearly,
        )
    )


    # --------------------------------------------------------------------------
    # 9. Alignement prédictions
    # --------------------------------------------------------------------------

    merged_predictions = (
        align_predictions(
            baseline_predictions,
            advanced_predictions,
        )
    )


    # --------------------------------------------------------------------------
    # Libération mémoire
    # --------------------------------------------------------------------------

    del baseline_predictions
    del advanced_predictions


    # --------------------------------------------------------------------------
    # 10. Comparaison prédiction par prédiction
    # --------------------------------------------------------------------------

    prediction_summary = (
        summarize_prediction_comparison(
            merged_predictions
        )
    )


    # --------------------------------------------------------------------------
    # 11. Comparaison spatiale
    # --------------------------------------------------------------------------

    grid_comparison = (
        compute_gridpoint_metrics(
            merged_predictions
        )
    )


    # --------------------------------------------------------------------------
    # 12. Comparaison importance des variables
    # --------------------------------------------------------------------------

    feature_comparison = (
        build_feature_importance_comparison(
            baseline_importance,
            advanced_importance,
        )
    )


    # --------------------------------------------------------------------------
    # 13. Figures globales
    # --------------------------------------------------------------------------

    plot_global_metrics(
        global_comparison
    )

    plot_global_r2(
        global_comparison
    )

    plot_yearly_mae_comparison(
        yearly_comparison
    )

    plot_yearly_mae_gain(
        yearly_comparison
    )

    plot_yearly_r2_comparison(
        yearly_comparison
    )

    plot_prediction_gain_distribution(
        merged_predictions
    )

    plot_absolute_error_scatter(
        merged_predictions
    )


    # --------------------------------------------------------------------------
    # 14. Cartes
    # --------------------------------------------------------------------------

    plot_mae_gain_map(
        grid_comparison
    )

    plot_rmse_gain_map(
        grid_comparison
    )

    plot_r2_gain_map(
        grid_comparison
    )

    plot_bias_gain_map(
        grid_comparison
    )


    # --------------------------------------------------------------------------
    # 15. Importance avancée
    # --------------------------------------------------------------------------

    plot_advanced_feature_importance(
        feature_comparison
    )


    # --------------------------------------------------------------------------
    # 16. Exports
    # --------------------------------------------------------------------------

    export_comparison_json(
        global_comparison=
            global_comparison,

        yearly_comparison=
            yearly_comparison,

        prediction_summary=
            prediction_summary,

        grid_comparison=
            grid_comparison,

        baseline_configuration=
            baseline_configuration,

        advanced_configuration=
            advanced_configuration,
    )

    write_markdown_report(
        global_comparison=
            global_comparison,

        yearly_comparison=
            yearly_comparison,

        prediction_summary=
            prediction_summary,

        grid_comparison=
            grid_comparison,

        feature_comparison=
            feature_comparison,
    )


    # --------------------------------------------------------------------------
    # 17. Résumé final
    # --------------------------------------------------------------------------

    total_duration = (
        time.time()
        - global_start
    )

    mae_row = (
        global_comparison[
            global_comparison[
                "metric"
            ]
            == "MAE"
        ]
        .iloc[0]
    )

    rmse_row = (
        global_comparison[
            global_comparison[
                "metric"
            ]
            == "RMSE"
        ]
        .iloc[0]
    )

    r2_row = (
        global_comparison[
            global_comparison[
                "metric"
            ]
            == "R2"
        ]
        .iloc[0]
    )

    prediction_row = (
        prediction_summary
        .iloc[0]
    )

    better_years = int(
        yearly_comparison[
            "advanced_better_MAE"
        ]
        .sum()
    )

    better_points = int(
        grid_comparison[
            "advanced_better_MAE"
        ]
        .sum()
    )

    print_separator(
        "COMPARAISON TERMINÉE"
    )

    print(
        "MAE"
    )

    print(
        f"  Baseline : "
        f"{mae_row['baseline']:.4f} °C"
    )

    print(
        f"  Avancé   : "
        f"{mae_row['advanced']:.4f} °C"
    )

    print(
        f"  Gain     : "
        f"{mae_row['improvement_absolute']:.4f} °C "
        f"({mae_row['improvement_percent']:.2f} %)"
    )

    print(
        "\nRMSE"
    )

    print(
        f"  Baseline : "
        f"{rmse_row['baseline']:.4f} °C"
    )

    print(
        f"  Avancé   : "
        f"{rmse_row['advanced']:.4f} °C"
    )

    print(
        f"  Gain     : "
        f"{rmse_row['improvement_absolute']:.4f} °C "
        f"({rmse_row['improvement_percent']:.2f} %)"
    )

    print(
        "\nR²"
    )

    print(
        f"  Baseline : "
        f"{r2_row['baseline']:.4f}"
    )

    print(
        f"  Avancé   : "
        f"{r2_row['advanced']:.4f}"
    )

    print(
        f"  Gain     : "
        f"{r2_row['improvement_absolute']:+.4f}"
    )

    print(
        "\nComparaison temporelle"
    )

    print(
        f"  Années améliorées : "
        f"{better_years}/"
        f"{len(yearly_comparison)}"
    )

    print(
        "\nComparaison spatiale"
    )

    print(
        f"  Points améliorés : "
        f"{better_points:,}/"
        f"{len(grid_comparison):,} "
        f"({better_points / len(grid_comparison) * 100:.2f} %)"
    )

    print(
        "\nComparaison prédiction par prédiction"
    )

    print(
        f"  Avancé meilleur : "
        f"{int(prediction_row['advanced_better_count']):,} "
        f"({prediction_row['advanced_better_percent']:.2f} %)"
    )

    print(
        f"  Baseline meilleur : "
        f"{int(prediction_row['baseline_better_count']):,} "
        f"({prediction_row['baseline_better_percent']:.2f} %)"
    )

    print(
        f"\nDurée totale : "
        f"{format_duration(total_duration)}"
    )

    print(
        "\nFichiers principaux créés :"
    )

    output_files = [
        GLOBAL_COMPARISON_FILE,
        YEARLY_COMPARISON_FILE,
        PREDICTION_COMPARISON_FILE,
        PREDICTION_SUMMARY_FILE,
        GRIDPOINT_COMPARISON_FILE,
        FEATURE_IMPORTANCE_COMPARISON_FILE,
        COMPARISON_JSON_FILE,
        COMPARISON_REPORT_FILE,
    ]

    for file_path in output_files:

        status = (
            "[OK]"
            if file_path.exists()
            else "[MANQUANT]"
        )

        print(
            f"  {status} "
            f"{file_path}"
        )

    print(
        "\nÉtape suivante : "
        "07_generate_final_report.py"
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

        sys.exit(
            1
        )

    except Exception as exc:

        print_separator(
            "ERREUR"
        )

        print(
            str(
                exc
            )
        )

        raise