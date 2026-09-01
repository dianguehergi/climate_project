#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
07_generate_final_report.py
===============================================================================

Génère la synthèse finale du pipeline XGBoost climat à partir des sorties déjà
produites par les étapes précédentes.

Aucun modèle n'est entraîné dans ce script.

Pipeline :
    01_prepare_full_timeseries.py
    02_build_training_dataset.py
    03_train_xgboost.py
    04_analyse_baseline.py
    05_train_xgboost_advanced.py
    06_compare_models.py
    07_generate_final_report.py

Sorties :
    results/final_report/
        tables/
            final_global_metrics.csv
            final_key_results.csv
            final_yearly_summary.csv
            final_spatial_summary.csv
            final_feature_summary.csv

        reports/
            final_report.md
            final_report.txt
            final_report_summary.json

===============================================================================
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ==============================================================================
# CONFIGURATION
# ==============================================================================

PROJECT_DIR = Path("/home/lab-c0de80861d/climate_project")

BASELINE_DIR = PROJECT_DIR / "results" / "xgboost_baseline"
ADVANCED_DIR = PROJECT_DIR / "results" / "xgboost_advanced"
COMPARISON_DIR = PROJECT_DIR / "results" / "model_comparison"

FINAL_DIR = PROJECT_DIR / "results" / "final_report"
TABLES_DIR = FINAL_DIR / "tables"
REPORTS_DIR = FINAL_DIR / "reports"


# ==============================================================================
# CONSTANTES
# ==============================================================================

EXPECTED_TEST_ROWS = 3_086_304
EXPECTED_GRID_POINTS = 9_892
EXPECTED_MONTHS_PER_POINT = 312

TEST_START_YEAR = 2000
TEST_END_YEAR = 2025

EXPECTED_TEST_YEARS = (
    TEST_END_YEAR
    - TEST_START_YEAR
    + 1
)


# ==============================================================================
# FICHIERS D'ENTRÉE
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

BASELINE_CONFIG_FILE = (
    BASELINE_DIR
    / "model_configuration.json"
)


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

ADVANCED_CONFIG_FILE = (
    ADVANCED_DIR
    / "model_configuration.json"
)

ADVANCED_VALIDATION_FILE = (
    ADVANCED_DIR
    / "validation_results.csv"
)

ADVANCED_FEATURES_FILE = (
    ADVANCED_DIR
    / "selected_features.csv"
)


GLOBAL_COMPARISON_FILE = (
    COMPARISON_DIR
    / "tables"
    / "global_model_comparison.csv"
)

YEARLY_COMPARISON_FILE = (
    COMPARISON_DIR
    / "tables"
    / "yearly_model_comparison.csv"
)

PREDICTION_SUMMARY_FILE = (
    COMPARISON_DIR
    / "tables"
    / "prediction_comparison_summary.csv"
)

GRIDPOINT_COMPARISON_FILE = (
    COMPARISON_DIR
    / "tables"
    / "gridpoint_model_comparison.csv"
)

FEATURE_COMPARISON_FILE = (
    COMPARISON_DIR
    / "tables"
    / "feature_importance_comparison.csv"
)

COMPARISON_JSON_FILE = (
    COMPARISON_DIR
    / "reports"
    / "comparison_results.json"
)


# ==============================================================================
# FICHIERS DE SORTIE
# ==============================================================================

FINAL_GLOBAL_METRICS_FILE = (
    TABLES_DIR
    / "final_global_metrics.csv"
)

FINAL_KEY_RESULTS_FILE = (
    TABLES_DIR
    / "final_key_results.csv"
)

FINAL_YEARLY_SUMMARY_FILE = (
    TABLES_DIR
    / "final_yearly_summary.csv"
)

FINAL_SPATIAL_SUMMARY_FILE = (
    TABLES_DIR
    / "final_spatial_summary.csv"
)

FINAL_FEATURE_SUMMARY_FILE = (
    TABLES_DIR
    / "final_feature_summary.csv"
)

FINAL_REPORT_MD_FILE = (
    REPORTS_DIR
    / "final_report.md"
)

FINAL_REPORT_TXT_FILE = (
    REPORTS_DIR
    / "final_report.txt"
)

FINAL_REPORT_JSON_FILE = (
    REPORTS_DIR
    / "final_report_summary.json"
)


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
        seconds
        % 60
    )

    return (
        f"{minutes} min "
        f"{remaining_seconds:.1f} s"
    )


def create_output_directories() -> None:

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_csv(
    path: Path,
    label: str,
) -> pd.DataFrame:

    print(
        f"Chargement {label} :"
    )

    print(
        path
    )

    df = pd.read_csv(
        path
    )

    print(
        f"Dimensions : "
        f"{df.shape}"
    )

    return df


def load_json(
    path: Path,
    label: str,
) -> dict:

    print(
        f"Chargement {label} :"
    )

    print(
        path
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    return data


def safe_float(
    value,
    default=np.nan,
) -> float:

    try:

        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return float(
            default
        )


def json_safe(
    value,
):

    if isinstance(
        value,
        dict,
    ):

        return {
            key:
                json_safe(
                    item
                )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        list,
    ):

        return [
            json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):

        return [
            json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        np.integer,
    ):

        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):

        if np.isnan(
            value
        ):

            return None

        return float(
            value
        )

    if isinstance(
        value,
        pd.Timestamp,
    ):

        return str(
            value
        )

    try:

        if pd.isna(
            value
        ):

            return None

    except Exception:

        pass

    return value


# ==============================================================================
# VÉRIFICATION DES FICHIERS
# ==============================================================================


def check_required_files() -> None:

    print_separator(
        "VÉRIFICATION DES FICHIERS"
    )

    required_files = [
        BASELINE_METRICS_FILE,
        BASELINE_YEARLY_FILE,
        BASELINE_IMPORTANCE_FILE,
        BASELINE_CONFIG_FILE,

        ADVANCED_METRICS_FILE,
        ADVANCED_YEARLY_FILE,
        ADVANCED_IMPORTANCE_FILE,
        ADVANCED_CONFIG_FILE,
        ADVANCED_VALIDATION_FILE,
        ADVANCED_FEATURES_FILE,

        GLOBAL_COMPARISON_FILE,
        YEARLY_COMPARISON_FILE,
        PREDICTION_SUMMARY_FILE,
        GRIDPOINT_COMPARISON_FILE,
        FEATURE_COMPARISON_FILE,
        COMPARISON_JSON_FILE,
    ]

    missing_files = []

    for file_path in required_files:

        if file_path.exists():

            print(
                f"[OK] "
                f"{file_path}"
            )

        else:

            print(
                f"[MANQUANT] "
                f"{file_path}"
            )

            missing_files.append(
                file_path
            )

    if missing_files:

        raise FileNotFoundError(
            "Fichiers requis manquants :\n"
            + "\n".join(
                str(
                    file_path
                )
                for file_path
                in missing_files
            )
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

        normalized = (
            str(
                column
            )
            .strip()
            .upper()
            .replace(
                "²",
                "2",
            )
        )

        if normalized == "MAE":

            rename_map[
                column
            ] = "MAE"

        elif normalized == "RMSE":

            rename_map[
                column
            ] = "RMSE"

        elif normalized == "R2":

            rename_map[
                column
            ] = "R2"

        elif normalized == "BIAS":

            rename_map[
                column
            ] = "Bias"

        elif normalized == "MODEL":

            rename_map[
                column
            ] = "Model"

        elif normalized == "YEAR":

            rename_map[
                column
            ] = "year"

    df.rename(
        columns=rename_map,
        inplace=True,
    )

    return df


def extract_model_row(
    df: pd.DataFrame,
    advanced: bool,
) -> pd.Series:

    df = normalize_metric_columns(
        df
    )

    required_metrics = {
        "MAE",
        "RMSE",
        "R2",
        "Bias",
    }

    missing_metrics = (
        required_metrics
        - set(
            df.columns
        )
    )

    if missing_metrics:

        raise ValueError(
            "Colonnes métriques manquantes : "
            f"{sorted(missing_metrics)}"
        )

    if "Model" not in df.columns:

        if len(
            df
        ) == 1:

            return (
                df
                .iloc[0]
            )

        raise ValueError(
            "Colonne 'Model' absente."
        )

    model_names = (
        df[
            "Model"
        ]
        .astype(
            str
        )
        .str.lower()
    )

    if advanced:

        mask = (
            model_names
            .str.contains(
                "advanced",
                na=False,
            )
            |
            model_names
            .str.contains(
                "avanc",
                na=False,
            )
        )

        if mask.any():

            return (
                df.loc[
                    mask
                ]
                .iloc[0]
            )

        mask = (
            model_names
            .str.contains(
                "xgboost",
                na=False,
            )
            &
            ~model_names
            .str.contains(
                "naive",
                na=False,
            )
            &
            ~model_names
            .str.contains(
                "lag",
                na=False,
            )
        )

        if mask.any():

            return (
                df.loc[
                    mask
                ]
                .iloc[-1]
            )

    else:

        exact_mask = (
            model_names
            .eq(
                "xgboost"
            )
        )

        if exact_mask.any():

            return (
                df.loc[
                    exact_mask
                ]
                .iloc[0]
            )

        mask = (
            model_names
            .str.contains(
                "xgboost",
                na=False,
            )
            &
            ~model_names
            .str.contains(
                "advanced",
                na=False,
            )
            &
            ~model_names
            .str.contains(
                "avanc",
                na=False,
            )
        )

        if mask.any():

            return (
                df.loc[
                    mask
                ]
                .iloc[0]
            )

    raise ValueError(
        "Impossible d'identifier automatiquement "
        f"le modèle {'avancé' if advanced else 'baseline'}."
    )


# ==============================================================================
# VALIDATION
# ==============================================================================


def validate_inputs(
    baseline_metrics: pd.DataFrame,
    advanced_metrics: pd.DataFrame,
    yearly_comparison: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    grid_comparison: pd.DataFrame,
) -> None:

    print_separator(
        "VALIDATION DES RÉSULTATS"
    )

    extract_model_row(
        baseline_metrics,
        advanced=False,
    )

    extract_model_row(
        advanced_metrics,
        advanced=True,
    )

    if (
        len(
            yearly_comparison
        )
        != EXPECTED_TEST_YEARS
    ):

        raise ValueError(
            "Nombre d'années incorrect : "
            f"{len(yearly_comparison)} "
            f"au lieu de "
            f"{EXPECTED_TEST_YEARS}."
        )

    if (
        len(
            grid_comparison
        )
        != EXPECTED_GRID_POINTS
    ):

        raise ValueError(
            "Nombre de points de grille incorrect : "
            f"{len(grid_comparison):,} "
            f"au lieu de "
            f"{EXPECTED_GRID_POINTS:,}."
        )

    if (
        len(
            prediction_summary
        )
        != 1
    ):

        raise ValueError(
            "prediction_comparison_summary.csv "
            "doit contenir exactement une ligne."
        )

    if (
        "n_predictions"
        not in prediction_summary.columns
    ):

        raise ValueError(
            "Colonne 'n_predictions' absente."
        )

    n_predictions = int(
        prediction_summary[
            "n_predictions"
        ]
        .iloc[0]
    )

    if (
        n_predictions
        != EXPECTED_TEST_ROWS
    ):

        raise ValueError(
            "Nombre de prédictions incorrect : "
            f"{n_predictions:,} "
            f"au lieu de "
            f"{EXPECTED_TEST_ROWS:,}."
        )

    print(
        f"Observations test : "
        f"{n_predictions:,}"
    )

    print(
        f"Années comparées : "
        f"{len(yearly_comparison)}"
    )

    print(
        f"Points de grille : "
        f"{len(grid_comparison):,}"
    )

    print(
        "Validation : OK"
    )


# ==============================================================================
# MÉTRIQUES GLOBALES
# ==============================================================================


def get_global_metric_row(
    global_comparison: pd.DataFrame,
    metric_name: str,
) -> pd.Series:

    if (
        "metric"
        not in global_comparison.columns
    ):

        raise ValueError(
            "Colonne 'metric' absente "
            "de global_model_comparison.csv."
        )

    mask = (
        global_comparison[
            "metric"
        ]
        .astype(
            str
        )
        .str.lower()
        .eq(
            metric_name.lower()
        )
    )

    if not mask.any():

        raise ValueError(
            f"Métrique globale introuvable : "
            f"{metric_name}"
        )

    return (
        global_comparison.loc[
            mask
        ]
        .iloc[0]
    )


def build_final_global_metrics(
    baseline_metrics: pd.DataFrame,
    advanced_metrics: pd.DataFrame,
) -> pd.DataFrame:

    print_separator(
        "TABLE DES MÉTRIQUES GLOBALES"
    )

    baseline = extract_model_row(
        baseline_metrics,
        advanced=False,
    )

    advanced = extract_model_row(
        advanced_metrics,
        advanced=True,
    )

    result = pd.DataFrame(
        [
            {
                "Model":
                    "XGBoost_Baseline",

                "MAE":
                    safe_float(
                        baseline[
                            "MAE"
                        ]
                    ),

                "RMSE":
                    safe_float(
                        baseline[
                            "RMSE"
                        ]
                    ),

                "R2":
                    safe_float(
                        baseline[
                            "R2"
                        ]
                    ),

                "Bias":
                    safe_float(
                        baseline[
                            "Bias"
                        ]
                    ),
            },
            {
                "Model":
                    "XGBoost_Advanced",

                "MAE":
                    safe_float(
                        advanced[
                            "MAE"
                        ]
                    ),

                "RMSE":
                    safe_float(
                        advanced[
                            "RMSE"
                        ]
                    ),

                "R2":
                    safe_float(
                        advanced[
                            "R2"
                        ]
                    ),

                "Bias":
                    safe_float(
                        advanced[
                            "Bias"
                        ]
                    ),
            },
        ]
    )

    result.to_csv(
        FINAL_GLOBAL_METRICS_FILE,
        index=False,
    )

    print(
        result.to_string(
            index=False
        )
    )

    print(
        f"\nFichier :\n"
        f"{FINAL_GLOBAL_METRICS_FILE}"
    )

    return result


# ==============================================================================
# INDICATEURS CLÉS
# ==============================================================================


def build_key_results(
    global_comparison: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    yearly_comparison: pd.DataFrame,
    grid_comparison: pd.DataFrame,
) -> pd.DataFrame:

    print_separator(
        "INDICATEURS CLÉS"
    )

    mae = get_global_metric_row(
        global_comparison,
        "MAE",
    )

    rmse = get_global_metric_row(
        global_comparison,
        "RMSE",
    )

    r2 = get_global_metric_row(
        global_comparison,
        "R2",
    )

    bias = get_global_metric_row(
        global_comparison,
        "Bias",
    )

    prediction_row = (
        prediction_summary
        .iloc[0]
    )

    better_years = int(
        yearly_comparison[
            "advanced_better_MAE"
        ]
        .astype(
            bool
        )
        .sum()
    )

    better_points = int(
        grid_comparison[
            "advanced_better_MAE"
        ]
        .astype(
            bool
        )
        .sum()
    )

    rows = [
        (
            "Baseline_MAE_C",
            mae[
                "baseline"
            ],
        ),
        (
            "Advanced_MAE_C",
            mae[
                "advanced"
            ],
        ),
        (
            "MAE_improvement_C",
            mae[
                "improvement_absolute"
            ],
        ),
        (
            "MAE_improvement_percent",
            mae[
                "improvement_percent"
            ],
        ),
        (
            "Baseline_RMSE_C",
            rmse[
                "baseline"
            ],
        ),
        (
            "Advanced_RMSE_C",
            rmse[
                "advanced"
            ],
        ),
        (
            "RMSE_improvement_C",
            rmse[
                "improvement_absolute"
            ],
        ),
        (
            "RMSE_improvement_percent",
            rmse[
                "improvement_percent"
            ],
        ),
        (
            "Baseline_R2",
            r2[
                "baseline"
            ],
        ),
        (
            "Advanced_R2",
            r2[
                "advanced"
            ],
        ),
        (
            "R2_gain",
            r2[
                "improvement_absolute"
            ],
        ),
        (
            "Baseline_Bias_C",
            bias[
                "baseline"
            ],
        ),
        (
            "Advanced_Bias_C",
            bias[
                "advanced"
            ],
        ),
        (
            "Bias_improvement_percent",
            bias[
                "improvement_percent"
            ],
        ),
        (
            "Advanced_better_prediction_percent",
            prediction_row[
                "advanced_better_percent"
            ],
        ),
        (
            "Advanced_better_years",
            better_years,
        ),
        (
            "Total_test_years",
            len(
                yearly_comparison
            ),
        ),
        (
            "Advanced_better_gridpoints",
            better_points,
        ),
        (
            "Total_gridpoints",
            len(
                grid_comparison
            ),
        ),
        (
            "Advanced_better_gridpoints_percent",
            (
                100
                * better_points
                / len(
                    grid_comparison
                )
            ),
        ),
    ]

    result = pd.DataFrame(
        rows,
        columns=[
            "indicator",
            "value",
        ],
    )

    result.to_csv(
        FINAL_KEY_RESULTS_FILE,
        index=False,
    )

    print(
        result.to_string(
            index=False
        )
    )

    print(
        f"\nFichier :\n"
        f"{FINAL_KEY_RESULTS_FILE}"
    )

    return result


# ==============================================================================
# SYNTHÈSE TEMPORELLE
# ==============================================================================


def build_yearly_summary(
    yearly_comparison: pd.DataFrame,
) -> pd.DataFrame:

    print_separator(
        "SYNTHÈSE TEMPORELLE"
    )

    required_columns = [
        "year",
        "MAE_baseline",
        "MAE_advanced",
        "MAE_gain",
        "RMSE_baseline",
        "RMSE_advanced",
        "R2_baseline",
        "R2_advanced",
        "R2_gain",
        "Bias_baseline",
        "Bias_advanced",
        "advanced_better_MAE",
    ]

    missing_columns = [
        column
        for column
        in required_columns
        if column
        not in yearly_comparison.columns
    ]

    if missing_columns:

        raise ValueError(
            "Colonnes annuelles manquantes : "
            f"{missing_columns}"
        )

    result = (
        yearly_comparison
        .copy()
    )

    result.to_csv(
        FINAL_YEARLY_SUMMARY_FILE,
        index=False,
    )

    best_year = (
        result
        .sort_values(
            "MAE_gain",
            ascending=False,
        )
        .iloc[0]
    )

    worst_year = (
        result
        .sort_values(
            "MAE_gain",
            ascending=True,
        )
        .iloc[0]
    )

    better_years = int(
        result[
            "advanced_better_MAE"
        ]
        .astype(
            bool
        )
        .sum()
    )

    print(
        f"Années analysées : "
        f"{len(result)}"
    )

    print(
        f"Années améliorées : "
        f"{better_years}/"
        f"{len(result)}"
    )

    print(
        f"Meilleure année : "
        f"{int(best_year['year'])} "
        f"(gain MAE = "
        f"{best_year['MAE_gain']:.4f} °C)"
    )

    print(
        f"Année la moins favorable : "
        f"{int(worst_year['year'])} "
        f"(gain MAE = "
        f"{worst_year['MAE_gain']:.4f} °C)"
    )

    print(
        f"\nFichier :\n"
        f"{FINAL_YEARLY_SUMMARY_FILE}"
    )

    return result


# ==============================================================================
# SYNTHÈSE SPATIALE
# ==============================================================================


def build_spatial_summary(
    grid_comparison: pd.DataFrame,
) -> pd.DataFrame:

    print_separator(
        "SYNTHÈSE SPATIALE"
    )

    required_columns = [
        "LAMBX",
        "LAMBY",
        "MAE_baseline",
        "MAE_advanced",
        "MAE_gain",
        "RMSE_gain",
        "R2_gain",
        "Bias_abs_gain",
        "advanced_better_MAE",
    ]

    missing_columns = [
        column
        for column
        in required_columns
        if column
        not in grid_comparison.columns
    ]

    if missing_columns:

        raise ValueError(
            "Colonnes spatiales manquantes : "
            f"{missing_columns}"
        )

    better = (
        grid_comparison[
            "advanced_better_MAE"
        ]
        .astype(
            bool
        )
    )

    rows = [
        (
            "n_grid_points",
            len(
                grid_comparison
            ),
        ),
        (
            "advanced_better_points",
            int(
                better.sum()
            ),
        ),
        (
            "advanced_better_percent",
            float(
                100
                * better.mean()
            ),
        ),
        (
            "mean_MAE_baseline",
            float(
                grid_comparison[
                    "MAE_baseline"
                ]
                .mean()
            ),
        ),
        (
            "mean_MAE_advanced",
            float(
                grid_comparison[
                    "MAE_advanced"
                ]
                .mean()
            ),
        ),
        (
            "mean_MAE_gain",
            float(
                grid_comparison[
                    "MAE_gain"
                ]
                .mean()
            ),
        ),
        (
            "median_MAE_gain",
            float(
                grid_comparison[
                    "MAE_gain"
                ]
                .median()
            ),
        ),
        (
            "mean_RMSE_gain",
            float(
                grid_comparison[
                    "RMSE_gain"
                ]
                .mean()
            ),
        ),
        (
            "mean_R2_gain",
            float(
                grid_comparison[
                    "R2_gain"
                ]
                .mean()
            ),
        ),
        (
            "mean_Bias_abs_gain",
            float(
                grid_comparison[
                    "Bias_abs_gain"
                ]
                .mean()
            ),
        ),
        (
            "MAE_gain_q10",
            float(
                grid_comparison[
                    "MAE_gain"
                ]
                .quantile(
                    0.10
                )
            ),
        ),
        (
            "MAE_gain_q25",
            float(
                grid_comparison[
                    "MAE_gain"
                ]
                .quantile(
                    0.25
                )
            ),
        ),
        (
            "MAE_gain_q50",
            float(
                grid_comparison[
                    "MAE_gain"
                ]
                .quantile(
                    0.50
                )
            ),
        ),
        (
            "MAE_gain_q75",
            float(
                grid_comparison[
                    "MAE_gain"
                ]
                .quantile(
                    0.75
                )
            ),
        ),
        (
            "MAE_gain_q90",
            float(
                grid_comparison[
                    "MAE_gain"
                ]
                .quantile(
                    0.90
                )
            ),
        ),
    ]

    result = pd.DataFrame(
        rows,
        columns=[
            "metric",
            "value",
        ],
    )

    result.to_csv(
        FINAL_SPATIAL_SUMMARY_FILE,
        index=False,
    )

    print(
        result.to_string(
            index=False
        )
    )

    print(
        f"\nFichier :\n"
        f"{FINAL_SPATIAL_SUMMARY_FILE}"
    )

    return result


# ==============================================================================
# SYNTHÈSE DES VARIABLES
# ==============================================================================


def build_feature_summary(
    feature_comparison: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:

    print_separator(
        "SYNTHÈSE DES VARIABLES"
    )

    required_columns = [
        "feature",
        "importance_baseline",
        "importance_advanced",
        "importance_change",
    ]

    missing_columns = [
        column
        for column
        in required_columns
        if column
        not in feature_comparison.columns
    ]

    if missing_columns:

        raise ValueError(
            "Colonnes d'importance manquantes : "
            f"{missing_columns}"
        )

    result = (
        feature_comparison
        .sort_values(
            "importance_advanced",
            ascending=False,
        )
        .head(
            top_n
        )
        .copy()
        .reset_index(
            drop=True
        )
    )

    result.insert(
        0,
        "rank_advanced",
        np.arange(
            1,
            len(
                result
            )
            + 1,
        ),
    )

    result.to_csv(
        FINAL_FEATURE_SUMMARY_FILE,
        index=False,
    )

    print(
        result.to_string(
            index=False
        )
    )

    print(
        f"\nFichier :\n"
        f"{FINAL_FEATURE_SUMMARY_FILE}"
    )

    return result


# ==============================================================================
# VALIDATION AVANCÉE
# ==============================================================================


def extract_validation_stats(
    validation_df: pd.DataFrame,
) -> dict:

    df = normalize_metric_columns(
        validation_df
    )

    if len(
        df
    ) == 0:

        return {}

    row = (
        df
        .iloc[0]
    )

    def first_existing(
        *names,
    ):

        for name in names:

            if name in row.index:

                return row[
                    name
                ]

        return None

    return {
        "start_year":
            first_existing(
                "validation_start_year",
                "start_year",
                "VALIDATION_START_YEAR",
            ),

        "end_year":
            first_existing(
                "validation_end_year",
                "end_year",
                "VALIDATION_END_YEAR",
            ),

        "MAE":
            first_existing(
                "MAE"
            ),

        "RMSE":
            first_existing(
                "RMSE"
            ),

        "R2":
            first_existing(
                "R2"
            ),

        "Bias":
            first_existing(
                "Bias"
            ),
    }


def extract_advanced_feature_count(
    selected_features: pd.DataFrame,
    advanced_config: dict,
) -> int:

    if (
        len(
            selected_features
        )
        > 0
    ):

        return int(
            len(
                selected_features
            )
        )

    config_features = (
        advanced_config
        .get(
            "features"
        )
    )

    if isinstance(
        config_features,
        list,
    ):

        return len(
            config_features
        )

    n_features = (
        advanced_config
        .get(
            "n_features"
        )
    )

    if (
        n_features
        is not None
    ):

        return int(
            n_features
        )

    return 0


def extract_best_n_estimators(
    advanced_config: dict,
):

    candidate_keys = [
        "best_n_estimators",
        "optimal_n_estimators",
        "n_estimators_final",
        "best_iteration_plus_one",
    ]

    for key in candidate_keys:

        value = (
            advanced_config
            .get(
                key
            )
        )

        if (
            value
            is not None
        ):

            try:

                return int(
                    value
                )

            except Exception:

                pass

    best_iteration = (
        advanced_config
        .get(
            "best_iteration"
        )
    )

    if (
        best_iteration
        is not None
    ):

        try:

            return (
                int(
                    best_iteration
                )
                + 1
            )

        except Exception:

            pass

    model_params = (
        advanced_config
        .get(
            "model_params"
        )
    )

    if isinstance(
        model_params,
        dict,
    ):

        value = (
            model_params
            .get(
                "n_estimators"
            )
        )

        if (
            value
            is not None
        ):

            try:

                return int(
                    value
                )

            except Exception:

                pass

    return None


def extract_physical_variables(
    advanced_config: dict,
) -> list[str]:

    candidate_keys = [
        "physical_variables_available",
        "physical_variables",
        "available_physical_variables",
        "retained_physical_variables",
    ]

    for key in candidate_keys:

        value = (
            advanced_config
            .get(
                key
            )
        )

        if isinstance(
            value,
            list,
        ):

            return [
                str(
                    item
                )
                for item
                in value
            ]

    return [
        "SWI",
        "ETP",
        "EVAP",
        "PE",
        "PRENEI",
        "DRAINC",
    ]


# ==============================================================================
# STATISTIQUES PRINCIPALES
# ==============================================================================


def build_statistics(
    global_comparison: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    yearly_comparison: pd.DataFrame,
    grid_comparison: pd.DataFrame,
    validation_df: pd.DataFrame,
    advanced_config: dict,
    selected_features: pd.DataFrame,
) -> dict:

    mae = get_global_metric_row(
        global_comparison,
        "MAE",
    )

    rmse = get_global_metric_row(
        global_comparison,
        "RMSE",
    )

    r2 = get_global_metric_row(
        global_comparison,
        "R2",
    )

    bias = get_global_metric_row(
        global_comparison,
        "Bias",
    )

    prediction_row = (
        prediction_summary
        .iloc[0]
    )

    better_years = int(
        yearly_comparison[
            "advanced_better_MAE"
        ]
        .astype(
            bool
        )
        .sum()
    )

    better_points = int(
        grid_comparison[
            "advanced_better_MAE"
        ]
        .astype(
            bool
        )
        .sum()
    )

    best_year = (
        yearly_comparison
        .sort_values(
            "MAE_gain",
            ascending=False,
        )
        .iloc[0]
    )

    worst_year = (
        yearly_comparison
        .sort_values(
            "MAE_gain",
            ascending=True,
        )
        .iloc[0]
    )

    best_point = (
        grid_comparison
        .sort_values(
            "MAE_gain",
            ascending=False,
        )
        .iloc[0]
    )

    worst_point = (
        grid_comparison
        .sort_values(
            "MAE_gain",
            ascending=True,
        )
        .iloc[0]
    )

    validation = extract_validation_stats(
        validation_df
    )

    return {
        "baseline": {
            "MAE":
                safe_float(
                    mae[
                        "baseline"
                    ]
                ),

            "RMSE":
                safe_float(
                    rmse[
                        "baseline"
                    ]
                ),

            "R2":
                safe_float(
                    r2[
                        "baseline"
                    ]
                ),

            "Bias":
                safe_float(
                    bias[
                        "baseline"
                    ]
                ),
        },

        "advanced": {
            "MAE":
                safe_float(
                    mae[
                        "advanced"
                    ]
                ),

            "RMSE":
                safe_float(
                    rmse[
                        "advanced"
                    ]
                ),

            "R2":
                safe_float(
                    r2[
                        "advanced"
                    ]
                ),

            "Bias":
                safe_float(
                    bias[
                        "advanced"
                    ]
                ),

            "n_features":
                extract_advanced_feature_count(
                    selected_features,
                    advanced_config,
                ),

            "best_n_estimators":
                extract_best_n_estimators(
                    advanced_config
                ),
        },

        "improvements": {
            "MAE_absolute":
                safe_float(
                    mae[
                        "improvement_absolute"
                    ]
                ),

            "MAE_percent":
                safe_float(
                    mae[
                        "improvement_percent"
                    ]
                ),

            "RMSE_absolute":
                safe_float(
                    rmse[
                        "improvement_absolute"
                    ]
                ),

            "RMSE_percent":
                safe_float(
                    rmse[
                        "improvement_percent"
                    ]
                ),

            "R2_absolute":
                safe_float(
                    r2[
                        "improvement_absolute"
                    ]
                ),

            "Bias_absolute":
                safe_float(
                    bias[
                        "improvement_absolute"
                    ]
                ),

            "Bias_percent":
                safe_float(
                    bias[
                        "improvement_percent"
                    ]
                ),
        },

        "validation": {
            "start_year":
                validation.get(
                    "start_year"
                ),

            "end_year":
                validation.get(
                    "end_year"
                ),

            "MAE":
                safe_float(
                    validation.get(
                        "MAE"
                    )
                ),

            "RMSE":
                safe_float(
                    validation.get(
                        "RMSE"
                    )
                ),

            "R2":
                safe_float(
                    validation.get(
                        "R2"
                    )
                ),

            "Bias":
                safe_float(
                    validation.get(
                        "Bias"
                    )
                ),
        },

        "prediction_level": {
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
                safe_float(
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
                safe_float(
                    prediction_row[
                        "baseline_better_percent"
                    ]
                ),

            "same_error_count":
                int(
                    prediction_row.get(
                        "same_error_count",
                        0,
                    )
                ),

            "mean_absolute_error_gain":
                safe_float(
                    prediction_row[
                        "mean_absolute_error_gain"
                    ]
                ),
        },

        "temporal": {
            "n_years":
                len(
                    yearly_comparison
                ),

            "advanced_better_years":
                better_years,

            "advanced_better_percent":
                (
                    100
                    * better_years
                    / len(
                        yearly_comparison
                    )
                ),

            "best_year":
                int(
                    best_year[
                        "year"
                    ]
                ),

            "best_year_mae_gain":
                safe_float(
                    best_year[
                        "MAE_gain"
                    ]
                ),

            "worst_year":
                int(
                    worst_year[
                        "year"
                    ]
                ),

            "worst_year_mae_gain":
                safe_float(
                    worst_year[
                        "MAE_gain"
                    ]
                ),
        },

        "spatial": {
            "n_grid_points":
                len(
                    grid_comparison
                ),

            "advanced_better_points":
                better_points,

            "advanced_better_percent":
                (
                    100
                    * better_points
                    / len(
                        grid_comparison
                    )
                ),

            "mean_MAE_gain":
                float(
                    grid_comparison[
                        "MAE_gain"
                    ]
                    .mean()
                ),

            "median_MAE_gain":
                float(
                    grid_comparison[
                        "MAE_gain"
                    ]
                    .median()
                ),

            "best_point": {
                "LAMBX":
                    int(
                        best_point[
                            "LAMBX"
                        ]
                    ),

                "LAMBY":
                    int(
                        best_point[
                            "LAMBY"
                        ]
                    ),

                "MAE_gain":
                    safe_float(
                        best_point[
                            "MAE_gain"
                        ]
                    ),
            },

            "worst_point": {
                "LAMBX":
                    int(
                        worst_point[
                            "LAMBX"
                        ]
                    ),

                "LAMBY":
                    int(
                        worst_point[
                            "LAMBY"
                        ]
                    ),

                "MAE_gain":
                    safe_float(
                        worst_point[
                            "MAE_gain"
                        ]
                    ),
            },
        },
    }


# ==============================================================================
# INVENTAIRE DES FIGURES
# ==============================================================================


def collect_png_figures() -> dict[str, list[str]]:

    groups = {}

    directories = {
        "baseline":
            BASELINE_DIR
            / "figures",

        "comparison":
            COMPARISON_DIR
            / "figures",
    }

    for name, directory in directories.items():

        if directory.exists():

            groups[
                name
            ] = sorted(
                file_path.name
                for file_path
                in directory.glob(
                    "*.png"
                )
                if file_path.is_file()
            )

        else:

            groups[
                name
            ] = []

    return groups


def collect_project_outputs() -> dict[str, list[str]]:

    result = {}

    directories = {
        "baseline":
            BASELINE_DIR,

        "advanced":
            ADVANCED_DIR,

        "comparison":
            COMPARISON_DIR,
    }

    for name, directory in directories.items():

        files = []

        if directory.exists():

            for file_path in (
                directory
                .rglob(
                    "*"
                )
            ):

                if file_path.is_file():

                    files.append(
                        str(
                            file_path.relative_to(
                                PROJECT_DIR
                            )
                        )
                    )

        result[
            name
        ] = sorted(
            files
        )

    return result


# ==============================================================================
# FORMATAGE
# ==============================================================================


def format_metric(
    value,
    digits: int = 4,
    suffix: str = "",
) -> str:

    try:

        numeric_value = float(
            value
        )

        if np.isnan(
            numeric_value
        ):

            return "n.d."

        return (
            f"{numeric_value:.{digits}f}"
            f"{suffix}"
        )

    except Exception:

        return "n.d."


# ==============================================================================
# RAPPORT MARKDOWN
# ==============================================================================


def build_markdown_report(
    statistics: dict,
    feature_summary: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    grid_comparison: pd.DataFrame,
    advanced_config: dict,
    figures: dict[str, list[str]],
) -> str:

    baseline = (
        statistics[
            "baseline"
        ]
    )

    advanced = (
        statistics[
            "advanced"
        ]
    )

    improvements = (
        statistics[
            "improvements"
        ]
    )

    validation = (
        statistics[
            "validation"
        ]
    )

    prediction = (
        statistics[
            "prediction_level"
        ]
    )

    temporal = (
        statistics[
            "temporal"
        ]
    )

    spatial = (
        statistics[
            "spatial"
        ]
    )

    physical_variables = (
        extract_physical_variables(
            advanced_config
        )
    )

    physical_text = (
        ", ".join(
            physical_variables
        )
    )

    improved_years = (
        yearly_summary.loc[
            yearly_summary[
                "advanced_better_MAE"
            ]
            .astype(
                bool
            ),
            "year",
        ]
        .astype(
            int
        )
        .tolist()
    )

    degraded_years = (
        yearly_summary.loc[
            ~yearly_summary[
                "advanced_better_MAE"
            ]
            .astype(
                bool
            ),
            "year",
        ]
        .astype(
            int
        )
        .tolist()
    )

    q10 = float(
        grid_comparison[
            "MAE_gain"
        ]
        .quantile(
            0.10
        )
    )

    q25 = float(
        grid_comparison[
            "MAE_gain"
        ]
        .quantile(
            0.25
        )
    )

    q50 = float(
        grid_comparison[
            "MAE_gain"
        ]
        .quantile(
            0.50
        )
    )

    q75 = float(
        grid_comparison[
            "MAE_gain"
        ]
        .quantile(
            0.75
        )
    )

    q90 = float(
        grid_comparison[
            "MAE_gain"
        ]
        .quantile(
            0.90
        )
    )

    lines = []

    def add(
        text: str = "",
    ) -> None:

        lines.append(
            text
        )

    add(
        "# Rapport final — Prédiction mensuelle de température par XGBoost"
    )

    add()

    add(
        "## 1. Résumé exécutif"
    )

    add()

    add(
        "Ce projet étudie la prédiction mensuelle de la température "
        "sur une grille couvrant la France à partir d'une série "
        "climatique SAFRAN–SIM."
    )

    add()

    add(
        f"Le jeu de test couvre **{TEST_START_YEAR}–{TEST_END_YEAR}**, "
        f"soit **{EXPECTED_MONTHS_PER_POINT} mois**, "
        f"sur **{EXPECTED_GRID_POINTS:,} points de grille**, "
        f"correspondant à **{EXPECTED_TEST_ROWS:,} prédictions**."
    )

    add()

    add(
        "Deux modèles XGBoost ont été comparés :"
    )

    add()

    add(
        "- une **baseline** reposant sur les coordonnées, "
        "la saisonnalité et l'historique thermique ;"
    )

    add(
        "- un **modèle avancé** ajoutant une mémoire thermique "
        "plus longue et des variables physiques retardées."
    )

    add()

    add(
        "Le modèle avancé atteint :"
    )

    add()

    add(
        f"- MAE = **{advanced['MAE']:.4f} °C** ;"
    )

    add(
        f"- RMSE = **{advanced['RMSE']:.4f} °C** ;"
    )

    add(
        f"- R² = **{advanced['R2']:.4f}** ;"
    )

    add(
        f"- Bias = **{advanced['Bias']:.4f} °C**."
    )

    add()

    add(
        f"Par rapport à la baseline, la MAE diminue de "
        f"**{improvements['MAE_absolute']:.4f} °C "
        f"({improvements['MAE_percent']:.2f} %)** "
        f"et le RMSE de "
        f"**{improvements['RMSE_absolute']:.4f} °C "
        f"({improvements['RMSE_percent']:.2f} %)**."
    )

    add()

    add(
        f"Spatialement, le modèle avancé améliore la MAE sur "
        f"**{spatial['advanced_better_points']:,}/"
        f"{spatial['n_grid_points']:,} points "
        f"({spatial['advanced_better_percent']:.2f} %)**."
    )

    add()

    add(
        "---"
    )

    add()

    add(
        "## 2. Données et protocole expérimental"
    )

    add()

    add(
        "La série combine les données SAFRAN historiques et SIM. "
        "Les variables temporelles ont été construites par point de "
        "grille avant la séparation train/test afin de préserver "
        "correctement l'historique disponible lors du passage "
        "de 1999 à 2000."
    )

    add()

    add(
        f"Le test couvre **{TEST_START_YEAR}–{TEST_END_YEAR}**, "
        f"avec **{EXPECTED_MONTHS_PER_POINT} observations mensuelles "
        f"par point de grille**."
    )

    add()

    add(
        "Les coordonnées spatiales sont représentées par "
        "`LAMBX` et `LAMBY`."
    )

    add()

    add(
        "Les cartes du pipeline sont interprétées dans cet espace "
        "de coordonnées sans supposer un système de projection "
        "géographique non documenté."
    )

    add()

    add(
        "---"
    )

    add()

    add(
        "## 3. Variables explicatives"
    )

    add()

    add(
        "### 3.1 Variables baseline"
    )

    add()

    baseline_features = [
        "LAMBX",
        "LAMBY",
        "month_sin",
        "month_cos",
        "T_lag_1",
        "T_lag_2",
        "T_lag_12",
        "T_roll_3",
        "T_roll_12",
        "T_change_lag_1",
    ]

    for feature in baseline_features:

        add(
            f"- `{feature}`"
        )

    add()

    add(
        "### 3.2 Variables avancées"
    )

    add()

    advanced_temperature_features = [
        "T_lag_24",
        "T_roll_24",
        "T_roll_36",
    ]

    for feature in advanced_temperature_features:

        add(
            f"- `{feature}`"
        )

    add()

    add(
        f"Variables physiques utilisées : "
        f"**{physical_text}**."
    )

    add()

    add(
        "Pour ces variables physiques, seules des informations "
        "historiques sont utilisées : retards et moyennes mobiles. "
        "Les valeurs contemporaines du mois cible ne sont pas "
        "utilisées directement."
    )

    add()

    add(
        "---"
    )

    add()

    add(
        "## 4. Validation temporelle du modèle avancé"
    )

    add()

    if (
        validation[
            "start_year"
        ]
        is not None
        and
        validation[
            "end_year"
        ]
        is not None
    ):

        add(
            f"La fenêtre de validation couvre "
            f"**{int(validation['start_year'])}–"
            f"{int(validation['end_year'])}**."
        )

        add()

    add(
        "| Métrique | Validation |"
    )

    add(
        "|---|---:|"
    )

    add(
        f"| MAE | "
        f"{format_metric(validation['MAE'], 4, ' °C')} |"
    )

    add(
        f"| RMSE | "
        f"{format_metric(validation['RMSE'], 4, ' °C')} |"
    )

    add(
        f"| R² | "
        f"{format_metric(validation['R2'], 4)} |"
    )

    add(
        f"| Bias | "
        f"{format_metric(validation['Bias'], 4, ' °C')} |"
    )

    add()

    if (
        advanced[
            "best_n_estimators"
        ]
        is not None
    ):

        add(
            f"Nombre d'arbres retenu après early stopping : "
            f"**{advanced['best_n_estimators']}**."
        )

        add()

    add(
        "---"
    )

    add()

    add(
        "## 5. Performances globales"
    )

    add()

    add(
        "| Métrique | Baseline | Avancé | Gain |"
    )

    add(
        "|---|---:|---:|---:|"
    )

    add(
        f"| MAE | "
        f"{baseline['MAE']:.4f} °C | "
        f"{advanced['MAE']:.4f} °C | "
        f"{improvements['MAE_absolute']:.4f} °C "
        f"({improvements['MAE_percent']:.2f} %) |"
    )

    add(
        f"| RMSE | "
        f"{baseline['RMSE']:.4f} °C | "
        f"{advanced['RMSE']:.4f} °C | "
        f"{improvements['RMSE_absolute']:.4f} °C "
        f"({improvements['RMSE_percent']:.2f} %) |"
    )

    add(
        f"| R² | "
        f"{baseline['R2']:.4f} | "
        f"{advanced['R2']:.4f} | "
        f"+{improvements['R2_absolute']:.4f} |"
    )

    add(
        f"| Bias | "
        f"{baseline['Bias']:.4f} °C | "
        f"{advanced['Bias']:.4f} °C | "
        f"{improvements['Bias_absolute']:.4f} °C "
        f"({improvements['Bias_percent']:.2f} %) |"
    )

    add()

    add(
        "Le modèle avancé améliore simultanément la MAE, le RMSE, "
        "le R² et la valeur absolue du biais."
    )

    add()

    add(
        "---"
    )

    add()

    add(
        "## 6. Comparaison prédiction par prédiction"
    )

    add()

    add(
        f"- Avancé meilleur : "
        f"**{prediction['advanced_better_count']:,} "
        f"({prediction['advanced_better_percent']:.2f} %)**."
    )

    add(
        f"- Baseline meilleure : "
        f"**{prediction['baseline_better_count']:,} "
        f"({prediction['baseline_better_percent']:.2f} %)**."
    )

    add(
        f"- Gain moyen d'erreur absolue : "
        f"**{prediction['mean_absolute_error_gain']:.4f} °C**."
    )

    add()

    add(
        "Le taux de victoires proche de 50 % indique que le gain "
        "global ne vient pas d'une domination sur chaque observation. "
        "Le modèle avancé réduit davantage certaines erreurs importantes."
    )

    add()

    add(
        "---"
    )

    add()

    add(
        "## 7. Analyse temporelle"
    )

    add()

    add(
        f"Le modèle avancé améliore la MAE sur "
        f"**{temporal['advanced_better_years']}/"
        f"{temporal['n_years']} années "
        f"({temporal['advanced_better_percent']:.2f} %)**."
    )

    add()

    add(
        "Années améliorées : "
        + ", ".join(
            map(
                str,
                improved_years,
            )
        )
        + "."
    )

    add()

    add(
        "Années où la baseline reste meilleure : "
        + ", ".join(
            map(
                str,
                degraded_years,
            )
        )
        + "."
    )

    add()

    add(
        f"Meilleure année pour le modèle avancé : "
        f"**{temporal['best_year']}**, "
        f"gain MAE = "
        f"**{temporal['best_year_mae_gain']:.4f} °C**."
    )

    add()

    add(
        f"Année la moins favorable : "
        f"**{temporal['worst_year']}**, "
        f"gain MAE = "
        f"**{temporal['worst_year_mae_gain']:.4f} °C**."
    )

    add()

    add(
        "---"
    )

    add()

    add(
        "## 8. Analyse spatiale"
    )

    add()

    add(
        f"Le modèle avancé améliore la MAE sur "
        f"**{spatial['advanced_better_points']:,}/"
        f"{spatial['n_grid_points']:,} points "
        f"({spatial['advanced_better_percent']:.2f} %)**."
    )

    add()

    add(
        f"Gain MAE moyen : "
        f"**{spatial['mean_MAE_gain']:.4f} °C**."
    )

    add(
        f"Gain MAE médian : "
        f"**{spatial['median_MAE_gain']:.4f} °C**."
    )

    add()

    add(
        "| Quantile | Gain MAE |"
    )

    add(
        "|---|---:|"
    )

    add(
        f"| 10 % | "
        f"{q10:.4f} °C |"
    )

    add(
        f"| 25 % | "
        f"{q25:.4f} °C |"
    )

    add(
        f"| 50 % | "
        f"{q50:.4f} °C |"
    )

    add(
        f"| 75 % | "
        f"{q75:.4f} °C |"
    )

    add(
        f"| 90 % | "
        f"{q90:.4f} °C |"
    )

    add()

    add(
        f"Point avec la plus forte amélioration : "
        f"`LAMBX={spatial['best_point']['LAMBX']}`, "
        f"`LAMBY={spatial['best_point']['LAMBY']}`, "
        f"gain MAE = "
        f"**{spatial['best_point']['MAE_gain']:.4f} °C**."
    )

    add()

    add(
        f"Point le moins favorable : "
        f"`LAMBX={spatial['worst_point']['LAMBX']}`, "
        f"`LAMBY={spatial['worst_point']['LAMBY']}`, "
        f"gain MAE = "
        f"**{spatial['worst_point']['MAE_gain']:.4f} °C**."
    )

    add()

    add(
        "---"
    )

    add()

    add(
        "## 9. Importance des variables"
    )

    add()

    add(
        f"Le modèle avancé utilise "
        f"**{advanced['n_features']} variables explicatives**."
    )

    add()

    add(
        "Top 10 des variables du modèle avancé :"
    )

    add()

    for _, row in (
        feature_summary
        .head(
            10
        )
        .iterrows()
    ):

        add(
            f"{int(row['rank_advanced'])}. "
            f"`{row['feature']}` : "
            f"{row['importance_advanced']:.6f}"
        )

    add()

    add(
        "La forte importance des retards thermiques, notamment "
        "`T_lag_12` et `T_lag_24`, confirme le rôle majeur de la "
        "persistance saisonnière et interannuelle."
    )

    add()

    add(
        "Les variables physiques apportent une information "
        "complémentaire mais restent individuellement moins dominantes."
    )

    add()

    add(
        "Ces importances décrivent l'utilisation interne des variables "
        "par XGBoost et ne doivent pas être interprétées comme des "
        "relations causales."
    )

    add()

    add(
        "---"
    )

    add()

    add(
        "## 10. Interprétation scientifique"
    )

    add()

    add(
        "Les résultats mettent en évidence une forte mémoire thermique "
        "annuelle et bisannuelle."
    )

    add()

    add(
        "L'ajout de variables physiques historiques améliore les "
        "performances sans remplacer le rôle dominant de l'historique "
        "de température."
    )

    add()

    add(
        f"La réduction globale de MAE de "
        f"**{improvements['MAE_percent']:.2f} %** reste modérée, "
        f"mais elle est cohérente avec la baisse du RMSE, "
        f"l'augmentation du R², la réduction du biais absolu et "
        f"l'amélioration observée sur la majorité des points de grille."
    )

    add()

    add(
        "---"
    )

    add()

    add(
        "## 11. Limites méthodologiques"
    )

    add()

    add(
        "### 11.1 Évaluation one-step-ahead"
    )

    add()

    add(
        "Les variables retardées du test utilisent les observations "
        "des mois précédents. L'évaluation correspond donc à une "
        "prédiction mensuelle **one-step-ahead**, et non à une "
        "prévision récursive autonome sur l'ensemble de la période "
        "2000–2025."
    )

    add()

    add(
        "### 11.2 Dépendance aux observations historiques"
    )

    add()

    add(
        "Un usage opérationnel réellement récursif nécessiterait de "
        "remplacer progressivement certaines observations retardées "
        "par les prédictions du modèle."
    )

    add()

    add(
        "### 11.3 Importance des variables"
    )

    add()

    add(
        "Les importances XGBoost n'indiquent ni causalité ni signe "
        "d'effet. Une analyse SHAP pourrait compléter l'interprétation."
    )

    add()

    add(
        "### 11.4 Validation"
    )

    add()

    add(
        "Le modèle avancé utilise une fenêtre temporelle dédiée à "
        "l'early stopping. Une validation temporelle multi-fenêtres "
        "permettrait de tester davantage la robustesse."
    )

    add()

    add(
        "### 11.5 Cartographie"
    )

    add()

    add(
        "Les cartes utilisent `LAMBX` et `LAMBY`. Une cartographie "
        "dans un référentiel géographique standard nécessiterait "
        "d'identifier et de documenter explicitement le système "
        "de coordonnées."
    )

    add()

    add(
        "---"
    )

    add()

    add(
        "## 12. Pistes d'amélioration"
    )

    add()

    improvements_list = [
        "validation temporelle multi-fenêtres",
        "optimisation contrôlée des hyperparamètres",
        "analyse SHAP",
        "analyse des performances par mois de l'année",
        "analyse des événements extrêmes",
        "analyse régionale",
        "variables de climatologie locale et d'anomalie",
        "prévision récursive",
        "comparaison avec LightGBM ou CatBoost",
        "conversion cartographique documentée",
    ]

    for item in improvements_list:

        add(
            f"- {item}"
        )

    add()

    add(
        "---"
    )

    add()

    add(
        "## 13. Conclusion"
    )

    add()

    add(
        f"La baseline atteint une MAE de "
        f"**{baseline['MAE']:.4f} °C**, "
        f"un RMSE de **{baseline['RMSE']:.4f} °C** "
        f"et un R² de **{baseline['R2']:.4f}**."
    )

    add()

    add(
        f"Le modèle avancé atteint une MAE de "
        f"**{advanced['MAE']:.4f} °C**, "
        f"un RMSE de **{advanced['RMSE']:.4f} °C** "
        f"et un R² de **{advanced['R2']:.4f}**."
    )

    add()

    add(
        f"Il réduit la MAE de "
        f"**{improvements['MAE_percent']:.2f} %** "
        f"et le RMSE de "
        f"**{improvements['RMSE_percent']:.2f} %**, "
        f"tout en améliorant la MAE sur "
        f"**{spatial['advanced_better_percent']:.2f} % "
        f"des points de grille**."
    )

    add()

    add(
        "À ce stade du pipeline, le modèle avancé constitue "
        "le meilleur modèle testé."
    )

    add()

    add(
        "---"
    )

    add()

    add(
        "## 14. Figures disponibles"
    )

    add()

    add(
        "### Analyse baseline"
    )

    add()

    if figures.get(
        "baseline"
    ):

        for filename in figures[
            "baseline"
        ]:

            add(
                f"- `{filename}`"
            )

    else:

        add(
            "- Aucune figure PNG détectée."
        )

    add()

    add(
        "### Comparaison des modèles"
    )

    add()

    if figures.get(
        "comparison"
    ):

        for filename in figures[
            "comparison"
        ]:

            add(
                f"- `{filename}`"
            )

    else:

        add(
            "- Aucune figure PNG détectée."
        )

    add()

    add(
        "---"
    )

    add()

    add(
        "## 15. Reproductibilité"
    )

    add()

    add(
        "Ordre d'exécution :"
    )

    add()

    add(
        "```text"
    )

    add(
        "01_prepare_full_timeseries.py"
    )

    add(
        "02_build_training_dataset.py"
    )

    add(
        "03_train_xgboost.py"
    )

    add(
        "04_analyse_baseline.py"
    )

    add(
        "05_train_xgboost_advanced.py"
    )

    add(
        "06_compare_models.py"
    )

    add(
        "07_generate_final_report.py"
    )

    add(
        "```"
    )

    add()

    add(
        "Les métriques, prédictions et analyses intermédiaires "
        "sont sauvegardées afin de permettre la reproduction des "
        "étapes d'analyse sans réentraîner les modèles."
    )

    add()

    return (
        "\n".join(
            lines
        )
    )


# ==============================================================================
# RAPPORT TEXTE
# ==============================================================================


def markdown_to_plain_text(
    markdown: str,
) -> str:

    text = (
        markdown
        .replace(
            "### ",
            "",
        )
        .replace(
            "## ",
            "",
        )
        .replace(
            "# ",
            "",
        )
        .replace(
            "**",
            "",
        )
        .replace(
            "`",
            "",
        )
    )

    return text


# ==============================================================================
# EXPORT DES RAPPORTS
# ==============================================================================


def save_reports(
    markdown_report: str,
) -> None:

    print_separator(
        "SAUVEGARDE DU RAPPORT FINAL"
    )

    FINAL_REPORT_MD_FILE.write_text(
        markdown_report,
        encoding="utf-8",
    )

    FINAL_REPORT_TXT_FILE.write_text(
        markdown_to_plain_text(
            markdown_report
        ),
        encoding="utf-8",
    )

    print(
        f"Markdown :\n"
        f"{FINAL_REPORT_MD_FILE}"
    )

    print(
        f"\nTexte :\n"
        f"{FINAL_REPORT_TXT_FILE}"
    )


# ==============================================================================
# EXPORT JSON
# ==============================================================================


def export_json(
    statistics: dict,
    feature_summary: pd.DataFrame,
    figures: dict[str, list[str]],
    outputs: dict[str, list[str]],
) -> None:

    print_separator(
        "EXPORT JSON FINAL"
    )

    top_features = []

    for _, row in (
        feature_summary
        .head(
            10
        )
        .iterrows()
    ):

        top_features.append(
            {
                "rank":
                    int(
                        row[
                            "rank_advanced"
                        ]
                    ),

                "feature":
                    str(
                        row[
                            "feature"
                        ]
                    ),

                "importance":
                    float(
                        row[
                            "importance_advanced"
                        ]
                    ),
            }
        )

    payload = {
        "project": {
            "name":
                "Monthly temperature prediction with XGBoost",

            "test_period":
                f"{TEST_START_YEAR}-{TEST_END_YEAR}",

            "n_test_predictions":
                EXPECTED_TEST_ROWS,

            "n_grid_points":
                EXPECTED_GRID_POINTS,

            "months_per_gridpoint":
                EXPECTED_MONTHS_PER_POINT,
        },

        "results":
            statistics,

        "top_advanced_features":
            top_features,

        "figures":
            figures,

        "project_outputs":
            outputs,
    }

    with open(
        FINAL_REPORT_JSON_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_safe(
                payload
            ),
            file,
            ensure_ascii=False,
            indent=4,
        )

    print(
        FINAL_REPORT_JSON_FILE
    )


# ==============================================================================
# VALIDATION DES SORTIES
# ==============================================================================


def validate_final_outputs() -> None:

    print_separator(
        "VALIDATION DES SORTIES FINALES"
    )

    output_files = [
        FINAL_GLOBAL_METRICS_FILE,
        FINAL_KEY_RESULTS_FILE,
        FINAL_YEARLY_SUMMARY_FILE,
        FINAL_SPATIAL_SUMMARY_FILE,
        FINAL_FEATURE_SUMMARY_FILE,
        FINAL_REPORT_MD_FILE,
        FINAL_REPORT_TXT_FILE,
        FINAL_REPORT_JSON_FILE,
    ]

    missing_files = []

    for file_path in output_files:

        if file_path.exists():

            size_kb = (
                file_path
                .stat()
                .st_size
                / 1024
            )

            print(
                f"[OK] "
                f"{file_path} "
                f"({size_kb:.1f} KB)"
            )

        else:

            print(
                f"[MANQUANT] "
                f"{file_path}"
            )

            missing_files.append(
                file_path
            )

    if missing_files:

        raise FileNotFoundError(
            "Sorties finales manquantes :\n"
            + "\n".join(
                str(
                    file_path
                )
                for file_path
                in missing_files
            )
        )

    print(
        "\nToutes les sorties finales "
        "ont été générées."
    )


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:

    start_time = time.time()

    print_separator(
        "GÉNÉRATION DU RAPPORT FINAL"
    )

    create_output_directories()

    check_required_files()


    # --------------------------------------------------------------------------
    # CHARGEMENT
    # --------------------------------------------------------------------------

    print_separator(
        "CHARGEMENT DES RÉSULTATS"
    )

    baseline_metrics = load_csv(
        BASELINE_METRICS_FILE,
        "métriques baseline",
    )

    baseline_yearly = load_csv(
        BASELINE_YEARLY_FILE,
        "métriques annuelles baseline",
    )

    baseline_importance = load_csv(
        BASELINE_IMPORTANCE_FILE,
        "importance baseline",
    )

    baseline_config = load_json(
        BASELINE_CONFIG_FILE,
        "configuration baseline",
    )

    advanced_metrics = load_csv(
        ADVANCED_METRICS_FILE,
        "métriques avancées",
    )

    advanced_yearly = load_csv(
        ADVANCED_YEARLY_FILE,
        "métriques annuelles avancées",
    )

    advanced_importance = load_csv(
        ADVANCED_IMPORTANCE_FILE,
        "importance avancée",
    )

    advanced_config = load_json(
        ADVANCED_CONFIG_FILE,
        "configuration avancée",
    )

    validation_df = load_csv(
        ADVANCED_VALIDATION_FILE,
        "validation avancée",
    )

    selected_features = load_csv(
        ADVANCED_FEATURES_FILE,
        "features avancées",
    )

    global_comparison = load_csv(
        GLOBAL_COMPARISON_FILE,
        "comparaison globale",
    )

    yearly_comparison = load_csv(
        YEARLY_COMPARISON_FILE,
        "comparaison annuelle",
    )

    prediction_summary = load_csv(
        PREDICTION_SUMMARY_FILE,
        "résumé prédiction par prédiction",
    )

    grid_comparison = load_csv(
        GRIDPOINT_COMPARISON_FILE,
        "comparaison spatiale",
    )

    feature_comparison = load_csv(
        FEATURE_COMPARISON_FILE,
        "comparaison des importances",
    )

    comparison_json = load_json(
        COMPARISON_JSON_FILE,
        "résumé JSON de comparaison",
    )


    # --------------------------------------------------------------------------
    # VÉRIFICATION DE LECTURE
    # --------------------------------------------------------------------------

    _ = (
        baseline_yearly,
        baseline_importance,
        baseline_config,
        advanced_yearly,
        advanced_importance,
        comparison_json,
    )


    # --------------------------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------------------------

    validate_inputs(
        baseline_metrics=
            baseline_metrics,

        advanced_metrics=
            advanced_metrics,

        yearly_comparison=
            yearly_comparison,

        prediction_summary=
            prediction_summary,

        grid_comparison=
            grid_comparison,
    )


    # --------------------------------------------------------------------------
    # TABLES FINALES
    # --------------------------------------------------------------------------

    build_final_global_metrics(
        baseline_metrics=
            baseline_metrics,

        advanced_metrics=
            advanced_metrics,
    )

    build_key_results(
        global_comparison=
            global_comparison,

        prediction_summary=
            prediction_summary,

        yearly_comparison=
            yearly_comparison,

        grid_comparison=
            grid_comparison,
    )

    yearly_summary = (
        build_yearly_summary(
            yearly_comparison
        )
    )

    build_spatial_summary(
        grid_comparison
    )

    feature_summary = (
        build_feature_summary(
            feature_comparison
        )
    )


    # --------------------------------------------------------------------------
    # STATISTIQUES
    # --------------------------------------------------------------------------

    print_separator(
        "EXTRACTION DES RÉSULTATS PRINCIPAUX"
    )

    statistics = build_statistics(
        global_comparison=
            global_comparison,

        prediction_summary=
            prediction_summary,

        yearly_comparison=
            yearly_comparison,

        grid_comparison=
            grid_comparison,

        validation_df=
            validation_df,

        advanced_config=
            advanced_config,

        selected_features=
            selected_features,
    )


    # --------------------------------------------------------------------------
    # FIGURES ET SORTIES
    # --------------------------------------------------------------------------

    figures = (
        collect_png_figures()
    )

    project_outputs = (
        collect_project_outputs()
    )


    # --------------------------------------------------------------------------
    # RAPPORT
    # --------------------------------------------------------------------------

    print_separator(
        "CONSTRUCTION DU RAPPORT"
    )

    markdown_report = build_markdown_report(
        statistics=
            statistics,

        feature_summary=
            feature_summary,

        yearly_summary=
            yearly_summary,

        grid_comparison=
            grid_comparison,

        advanced_config=
            advanced_config,

        figures=
            figures,
    )


    # --------------------------------------------------------------------------
    # EXPORTS
    # --------------------------------------------------------------------------

    save_reports(
        markdown_report
    )

    export_json(
        statistics=
            statistics,

        feature_summary=
            feature_summary,

        figures=
            figures,

        outputs=
            project_outputs,
    )


    # --------------------------------------------------------------------------
    # VALIDATION FINALE
    # --------------------------------------------------------------------------

    validate_final_outputs()


    # --------------------------------------------------------------------------
    # RÉSUMÉ
    # --------------------------------------------------------------------------

    duration = (
        time.time()
        - start_time
    )

    baseline = (
        statistics[
            "baseline"
        ]
    )

    advanced = (
        statistics[
            "advanced"
        ]
    )

    improvements = (
        statistics[
            "improvements"
        ]
    )

    temporal = (
        statistics[
            "temporal"
        ]
    )

    spatial = (
        statistics[
            "spatial"
        ]
    )

    prediction = (
        statistics[
            "prediction_level"
        ]
    )

    print_separator(
        "PIPELINE TERMINÉ"
    )

    print(
        "MODÈLE BASELINE"
    )

    print(
        f"  MAE  : "
        f"{baseline['MAE']:.4f} °C"
    )

    print(
        f"  RMSE : "
        f"{baseline['RMSE']:.4f} °C"
    )

    print(
        f"  R²   : "
        f"{baseline['R2']:.4f}"
    )

    print(
        f"  Bias : "
        f"{baseline['Bias']:.4f} °C"
    )

    print(
        "\nMODÈLE AVANCÉ"
    )

    print(
        f"  MAE  : "
        f"{advanced['MAE']:.4f} °C"
    )

    print(
        f"  RMSE : "
        f"{advanced['RMSE']:.4f} °C"
    )

    print(
        f"  R²   : "
        f"{advanced['R2']:.4f}"
    )

    print(
        f"  Bias : "
        f"{advanced['Bias']:.4f} °C"
    )

    print(
        f"  Features : "
        f"{advanced['n_features']}"
    )

    print(
        f"  Arbres : "
        f"{advanced['best_n_estimators']}"
    )

    print(
        "\nAMÉLIORATION"
    )

    print(
        f"  MAE  : "
        f"{improvements['MAE_absolute']:.4f} °C "
        f"({improvements['MAE_percent']:.2f} %)"
    )

    print(
        f"  RMSE : "
        f"{improvements['RMSE_absolute']:.4f} °C "
        f"({improvements['RMSE_percent']:.2f} %)"
    )

    print(
        f"  R²   : "
        f"+{improvements['R2_absolute']:.4f}"
    )

    print(
        f"  Bias : "
        f"{improvements['Bias_absolute']:.4f} °C "
        f"({improvements['Bias_percent']:.2f} %)"
    )

    print(
        "\nROBUSTESSE TEMPORELLE"
    )

    print(
        f"  Années améliorées : "
        f"{temporal['advanced_better_years']}/"
        f"{temporal['n_years']} "
        f"({temporal['advanced_better_percent']:.2f} %)"
    )

    print(
        "\nROBUSTESSE SPATIALE"
    )

    print(
        f"  Points améliorés : "
        f"{spatial['advanced_better_points']:,}/"
        f"{spatial['n_grid_points']:,} "
        f"({spatial['advanced_better_percent']:.2f} %)"
    )

    print(
        "\nPRÉDICTION PAR PRÉDICTION"
    )

    print(
        f"  Avancé meilleur : "
        f"{prediction['advanced_better_count']:,} "
        f"({prediction['advanced_better_percent']:.2f} %)"
    )

    print(
        f"  Baseline meilleure : "
        f"{prediction['baseline_better_count']:,} "
        f"({prediction['baseline_better_percent']:.2f} %)"
    )

    print(
        f"\nDurée totale : "
        f"{format_duration(duration)}"
    )

    print(
        "\nFichiers principaux créés :"
    )

    output_files = [
        FINAL_GLOBAL_METRICS_FILE,
        FINAL_KEY_RESULTS_FILE,
        FINAL_YEARLY_SUMMARY_FILE,
        FINAL_SPATIAL_SUMMARY_FILE,
        FINAL_FEATURE_SUMMARY_FILE,
        FINAL_REPORT_MD_FILE,
        FINAL_REPORT_TXT_FILE,
        FINAL_REPORT_JSON_FILE,
    ]

    for file_path in output_files:

        print(
            f"  [OK] "
            f"{file_path}"
        )

    print(
        "\nLe pipeline expérimental "
        "01 → 07 est terminé."
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