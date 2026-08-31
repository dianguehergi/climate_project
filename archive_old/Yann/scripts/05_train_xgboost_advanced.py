#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
05_train_xgboost_advanced.py
===============================================================================

Entraînement d'un modèle XGBoost avancé pour la prédiction mensuelle
de la température.

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

OBJECTIFS
---------
Cette étape améliore le modèle baseline en ajoutant :

    - T_lag_24
    - T_roll_24
    - T_roll_36

ainsi que des variables physiques disponibles dans les données :

    - SWI
    - ETP
    - EVAP
    - PE
    - PRENEI
    - DRAINC

Pour limiter le risque de fuite d'information, les variables physiques
contemporaines ne sont PAS utilisées directement.

On construit uniquement des informations historiques :

    variable_lag_1
    variable_lag_12
    variable_roll_3

Exemple :
    SWI_lag_1
    SWI_lag_12
    SWI_roll_3

VALIDATION
----------
La validation respecte strictement la chronologie.

    Sous-train : données historiques avant 1990
    Validation : 1990-1999
    Test final : 2000-2025

Le jeu de test 2000-2025 n'est jamais utilisé pour sélectionner
le nombre d'arbres.

EARLY STOPPING
--------------
Le nombre optimal d'arbres est déterminé uniquement avec la validation
historique.

Une fois ce nombre obtenu, un modèle final est réentraîné sur tout
l'historique disponible jusqu'à décembre 1999.

SORTIES
-------
results/xgboost_advanced/

    metrics_global.csv
    metrics_by_year.csv
    feature_importance.csv
    model_configuration.json
    validation_results.csv
    predictions_sample.csv
    predictions_full.parquet
    selected_features.csv

NOTE MÉTHODOLOGIQUE
-------------------
L'évaluation finale reste une évaluation mensuelle one-step-ahead.

Les variables retardées du jeu 2000-2025 utilisent les observations
historiques disponibles des mois précédents.

Il ne s'agit donc pas d'une simulation récursive complète de 26 ans.

===============================================================================
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


# ==============================================================================
# CHEMINS
# ==============================================================================

PROJECT_DIR = Path(
    "/home/lab-c0de80861d/climate_project"
)

DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "model"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "xgboost_advanced"
)


# ==============================================================================
# ENTRÉES
# ==============================================================================

TRAIN_FILE = (
    DATA_DIR
    / "train_1960_1999.csv"
)

TEST_FILE = (
    DATA_DIR
    / "test_2000_2025.csv"
)


# ==============================================================================
# SORTIES
# ==============================================================================

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

VALIDATION_RESULTS_FILE = (
    RESULTS_DIR
    / "validation_results.csv"
)

PREDICTIONS_SAMPLE_FILE = (
    RESULTS_DIR
    / "predictions_sample.csv"
)

PREDICTIONS_FULL_FILE = (
    RESULTS_DIR
    / "predictions_full.parquet"
)

SELECTED_FEATURES_FILE = (
    RESULTS_DIR
    / "selected_features.csv"
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

TARGET = "T"

GRID_COLUMNS = [
    "LAMBX",
    "LAMBY",
]

DATE_COLUMN = "DATE"


# ==============================================================================
# FEATURES BASELINE
# ==============================================================================

BASE_FEATURES = [
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


# ==============================================================================
# VARIABLES PHYSIQUES CANDIDATES
# ==============================================================================

PHYSICAL_VARIABLES = [
    "SWI",
    "ETP",
    "EVAP",
    "PE",
    "PRENEI",
    "DRAINC",
]


# ==============================================================================
# VALIDATION TEMPORELLE
# ==============================================================================

VALIDATION_START_YEAR = 1990

VALIDATION_END_YEAR = 1999


# ==============================================================================
# PARAMÈTRES XGBOOST POUR LA SÉLECTION DU NOMBRE D'ARBRES
# ==============================================================================

EARLY_STOPPING_ROUNDS = 50

MAX_ESTIMATORS = 1500

MODEL_PARAMS = {

    "objective":
        "reg:squarederror",

    "n_estimators":
        MAX_ESTIMATORS,

    "max_depth":
        8,

    "learning_rate":
        0.03,

    "min_child_weight":
        5,

    "subsample":
        0.85,

    "colsample_bytree":
        0.85,

    "reg_alpha":
        0.05,

    "reg_lambda":
        1.0,

    "tree_method":
        "hist",

    "random_state":
        42,

    "n_jobs":
        -1,

    "eval_metric":
        "rmse",

    "early_stopping_rounds":
        EARLY_STOPPING_ROUNDS,
}


# ==============================================================================
# PARAMÈTRES GÉNÉRAUX
# ==============================================================================

RANDOM_STATE = 42

PREDICTIONS_SAMPLE_SIZE = 100_000

EXPECTED_GRID_POINTS = 9892


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


# ==============================================================================
# INITIALISATION
# ==============================================================================


def create_output_directory() -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def check_input_files() -> None:

    missing = []

    for file_path in [
        TRAIN_FILE,
        TEST_FILE,
    ]:

        if not file_path.exists():

            missing.append(
                str(
                    file_path
                )
            )

    if missing:

        raise FileNotFoundError(
            "Fichier(s) manquant(s) :\n"
            + "\n".join(
                missing
            )
            + "\n\n"
            + "Exécute d'abord "
            + "02_build_training_dataset.py."
        )


# ==============================================================================
# CHARGEMENT
# ==============================================================================


def load_dataset(
    file_path: Path,
    name: str,
) -> pd.DataFrame:

    print(
        f"Chargement de {name} :\n"
        f"{file_path}"
    )

    start = time.time()

    df = pd.read_csv(
        file_path,
        parse_dates=[
            DATE_COLUMN
        ],
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
        f"{name} : "
        f"{len(df):,} lignes "
        f"× {df.shape[1]} colonnes"
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


# ==============================================================================
# CONTRÔLES
# ==============================================================================


def check_basic_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:

    required = [
        DATE_COLUMN,
        "LAMBX",
        "LAMBY",
        TARGET,
    ] + BASE_FEATURES

    required = list(
        dict.fromkeys(
            required
        )
    )

    for dataset_name, df in [
        (
            "TRAIN",
            train,
        ),
        (
            "TEST",
            test,
        ),
    ]:

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                f"{dataset_name} : "
                f"colonnes manquantes : "
                f"{missing}"
            )


# ==============================================================================
# VARIABLES PHYSIQUES DISPONIBLES
# ==============================================================================


def detect_physical_variables(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> list[str]:

    print_separator(
        "VARIABLES PHYSIQUES DISPONIBLES"
    )

    available = []

    for variable in PHYSICAL_VARIABLES:

        train_has = (
            variable
            in train.columns
        )

        test_has = (
            variable
            in test.columns
        )

        if (
            train_has
            and test_has
        ):

            available.append(
                variable
            )

            print(
                f"[OK] {variable}"
            )

        else:

            print(
                f"[ABSENTE] {variable}"
            )

    print(
        f"\nVariables physiques retenues : "
        f"{len(available)}"
    )

    return available


# ==============================================================================
# COMBINAISON TEMPORAIRE TRAIN + TEST
# ==============================================================================


def combine_datasets(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:

    print_separator(
        "RECONSTRUCTION DE LA CONTINUITÉ TEMPORELLE"
    )

    train_tmp = (
        train.copy()
    )

    test_tmp = (
        test.copy()
    )

    train_tmp[
        "_dataset"
    ] = "train"

    test_tmp[
        "_dataset"
    ] = "test"

    full = pd.concat(
        [
            train_tmp,
            test_tmp,
        ],
        ignore_index=True,
        sort=False,
    )

    full.sort_values(
        [
            "LAMBX",
            "LAMBY",
            DATE_COLUMN,
        ],
        inplace=True,
    )

    full.reset_index(
        drop=True,
        inplace=True,
    )

    print(
        f"Dimensions combinées : "
        f"{full.shape}"
    )

    print(
        f"Période : "
        f"{full[DATE_COLUMN].min().date()} "
        f"→ "
        f"{full[DATE_COLUMN].max().date()}"
    )

    return full


# ==============================================================================
# FEATURE ENGINEERING AVANCÉ
# ==============================================================================


def add_temperature_features(
    full: pd.DataFrame,
) -> pd.DataFrame:

    print_separator(
        "FEATURE ENGINEERING TEMPÉRATURE"
    )

    grouped_t = (
        full.groupby(
            [
                "LAMBX",
                "LAMBY",
            ],
            sort=False,
        )[TARGET]
    )


    # --------------------------------------------------------------------------
    # Lag 24 mois
    # --------------------------------------------------------------------------

    print(
        "Calcul de T_lag_24..."
    )

    full[
        "T_lag_24"
    ] = grouped_t.shift(
        24
    )


    # --------------------------------------------------------------------------
    # Moyenne mobile 24 mois
    #
    # shift(1) avant rolling :
    # le mois cible n'entre jamais dans sa propre feature.
    # --------------------------------------------------------------------------

    print(
        "Calcul de T_roll_24..."
    )

    full[
        "T_roll_24"
    ] = (
        grouped_t
        .shift(1)
        .groupby(
            [
                full["LAMBX"],
                full["LAMBY"],
            ],
            sort=False,
        )
        .rolling(
            window=24,
            min_periods=24,
        )
        .mean()
        .reset_index(
            level=[
                0,
                1,
            ],
            drop=True,
        )
    )


    # --------------------------------------------------------------------------
    # Moyenne mobile 36 mois
    # --------------------------------------------------------------------------

    print(
        "Calcul de T_roll_36..."
    )

    full[
        "T_roll_36"
    ] = (
        grouped_t
        .shift(1)
        .groupby(
            [
                full["LAMBX"],
                full["LAMBY"],
            ],
            sort=False,
        )
        .rolling(
            window=36,
            min_periods=36,
        )
        .mean()
        .reset_index(
            level=[
                0,
                1,
            ],
            drop=True,
        )
    )

    return full


# ==============================================================================
# FEATURES PHYSIQUES HISTORIQUES
# ==============================================================================


def add_physical_features(
    full: pd.DataFrame,
    physical_variables: list[str],
) -> tuple[
    pd.DataFrame,
    list[str],
]:

    print_separator(
        "FEATURE ENGINEERING VARIABLES PHYSIQUES"
    )

    generated_features = []

    for variable in physical_variables:

        print(
            f"\n{variable}"
        )

        grouped = (
            full.groupby(
                [
                    "LAMBX",
                    "LAMBY",
                ],
                sort=False,
            )[variable]
        )


        # ----------------------------------------------------------------------
        # Lag 1
        # ----------------------------------------------------------------------

        lag1_name = (
            f"{variable}_lag_1"
        )

        print(
            f"  Calcul de {lag1_name}..."
        )

        full[
            lag1_name
        ] = grouped.shift(
            1
        )

        generated_features.append(
            lag1_name
        )


        # ----------------------------------------------------------------------
        # Lag 12
        # ----------------------------------------------------------------------

        lag12_name = (
            f"{variable}_lag_12"
        )

        print(
            f"  Calcul de {lag12_name}..."
        )

        full[
            lag12_name
        ] = grouped.shift(
            12
        )

        generated_features.append(
            lag12_name
        )


        # ----------------------------------------------------------------------
        # Rolling 3 mois, uniquement historique
        # ----------------------------------------------------------------------

        roll3_name = (
            f"{variable}_roll_3"
        )

        print(
            f"  Calcul de {roll3_name}..."
        )

        full[
            roll3_name
        ] = (
            grouped
            .shift(1)
            .groupby(
                [
                    full["LAMBX"],
                    full["LAMBY"],
                ],
                sort=False,
            )
            .rolling(
                window=3,
                min_periods=3,
            )
            .mean()
            .reset_index(
                level=[
                    0,
                    1,
                ],
                drop=True,
            )
        )

        generated_features.append(
            roll3_name
        )

    return (
        full,
        generated_features,
    )


# ==============================================================================
# CONSTRUCTION DE LA LISTE DES FEATURES
# ==============================================================================


def build_feature_list(
    physical_features: list[str],
) -> list[str]:

    advanced_temperature_features = [
        "T_lag_24",
        "T_roll_24",
        "T_roll_36",
    ]

    features = (
        BASE_FEATURES
        + advanced_temperature_features
        + physical_features
    )

    features = list(
        dict.fromkeys(
            features
        )
    )

    return features


# ==============================================================================
# NETTOYAGE DES NAN AVANCÉS
# ==============================================================================


def prepare_advanced_datasets(
    full: pd.DataFrame,
    features: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    print_separator(
        "NETTOYAGE DES FEATURES AVANCÉES"
    )

    before = len(
        full
    )

    missing_before = (
        full[
            features
        ]
        .isna()
        .sum()
    )

    missing_before = (
        missing_before[
            missing_before > 0
        ]
    )

    if not missing_before.empty:

        print(
            "Valeurs manquantes avant nettoyage :"
        )

        print(
            missing_before
        )


    # --------------------------------------------------------------------------
    # On supprime seulement les lignes ne disposant pas
    # de l'historique nécessaire aux nouvelles features.
    # --------------------------------------------------------------------------

    full_clean = (
        full
        .dropna(
            subset=features
            + [
                TARGET
            ]
        )
        .copy()
    )

    removed = (
        before
        - len(
            full_clean
        )
    )

    print(
        f"\nLignes avant : "
        f"{before:,}"
    )

    print(
        f"Lignes supprimées : "
        f"{removed:,}"
    )

    print(
        f"Lignes conservées : "
        f"{len(full_clean):,}"
    )


    # --------------------------------------------------------------------------
    # Reconstitution train / test
    # --------------------------------------------------------------------------

    train_advanced = (
        full_clean[
            full_clean[
                "_dataset"
            ]
            == "train"
        ]
        .copy()
    )

    test_advanced = (
        full_clean[
            full_clean[
                "_dataset"
            ]
            == "test"
        ]
        .copy()
    )


    # --------------------------------------------------------------------------
    # Validation essentielle :
    # aucune ligne du test ne doit être perdue.
    #
    # Comme le test commence en 2000 et bénéficie de tout
    # l'historique 1961-1999 chargé auparavant, toutes les
    # features avancées doivent être disponibles.
    # --------------------------------------------------------------------------

    expected_test_rows = (
        full[
            full[
                "_dataset"
            ]
            == "test"
        ]
        .shape[0]
    )

    actual_test_rows = len(
        test_advanced
    )

    print(
        f"\nTest attendu : "
        f"{expected_test_rows:,}"
    )

    print(
        f"Test obtenu  : "
        f"{actual_test_rows:,}"
    )

    if (
        actual_test_rows
        != expected_test_rows
    ):

        raise ValueError(
            "Des lignes du jeu de test ont été perdues "
            "pendant le feature engineering avancé. "
            f"Attendu={expected_test_rows:,}, "
            f"obtenu={actual_test_rows:,}."
        )

    return (
        train_advanced,
        test_advanced,
    )


# ==============================================================================
# VALIDATION DE LA GRILLE
# ==============================================================================


def validate_grid(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:

    train_points = (
        train[
            GRID_COLUMNS
        ]
        .drop_duplicates()
        .shape[0]
    )

    test_points = (
        test[
            GRID_COLUMNS
        ]
        .drop_duplicates()
        .shape[0]
    )

    print_separator(
        "VALIDATION DE LA GRILLE"
    )

    print(
        f"Points train : "
        f"{train_points:,}"
    )

    print(
        f"Points test  : "
        f"{test_points:,}"
    )

    if (
        test_points
        != EXPECTED_GRID_POINTS
    ):

        raise ValueError(
            "Le nombre de points du test "
            "ne correspond pas à la grille attendue. "
            f"Attendu={EXPECTED_GRID_POINTS:,}, "
            f"obtenu={test_points:,}."
        )


# ==============================================================================
# SÉPARATION SOUS-TRAIN / VALIDATION
# ==============================================================================


def temporal_validation_split(
    train: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    print_separator(
        "VALIDATION TEMPORELLE"
    )

    years = (
        train[
            DATE_COLUMN
        ]
        .dt.year
    )

    subtrain = (
        train[
            years
            < VALIDATION_START_YEAR
        ]
        .copy()
    )

    validation = (
        train[
            (
                years
                >= VALIDATION_START_YEAR
            )
            &
            (
                years
                <= VALIDATION_END_YEAR
            )
        ]
        .copy()
    )

    if (
        len(subtrain) == 0
        or len(validation) == 0
    ):

        raise ValueError(
            "Découpage temporel de validation impossible."
        )

    print(
        f"Sous-train : "
        f"{len(subtrain):,} lignes"
    )

    print(
        f"Période : "
        f"{subtrain[DATE_COLUMN].min().date()} "
        f"→ "
        f"{subtrain[DATE_COLUMN].max().date()}"
    )

    print(
        f"\nValidation : "
        f"{len(validation):,} lignes"
    )

    print(
        f"Période : "
        f"{validation[DATE_COLUMN].min().date()} "
        f"→ "
        f"{validation[DATE_COLUMN].max().date()}"
    )

    return (
        subtrain,
        validation,
    )


# ==============================================================================
# MÉTRIQUES
# ==============================================================================


def compute_metrics(
    y_true,
    y_pred,
) -> dict:

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    residual = (
        y_true
        - y_pred
    )

    return {

        "MAE":
            float(
                mean_absolute_error(
                    y_true,
                    y_pred,
                )
            ),

        "RMSE":
            float(
                np.sqrt(
                    mean_squared_error(
                        y_true,
                        y_pred,
                    )
                )
            ),

        "R2":
            float(
                r2_score(
                    y_true,
                    y_pred,
                )
            ),

        "Bias":
            float(
                np.mean(
                    residual
                )
            ),
    }


# ==============================================================================
# BASELINES NAÏVES
# ==============================================================================


def evaluate_naive_baselines(
    test: pd.DataFrame,
) -> pd.DataFrame:

    print_separator(
        "BASELINES NAÏVES"
    )

    y_true = (
        test[
            TARGET
        ]
        .to_numpy()
    )

    rows = []

    baselines = {

        "Naive_T_lag_1":
            "T_lag_1",

        "Naive_T_lag_12":
            "T_lag_12",

        "Naive_T_lag_24":
            "T_lag_24",
    }

    for model_name, column in baselines.items():

        metrics = (
            compute_metrics(
                y_true,
                test[
                    column
                ],
            )
        )

        rows.append(
            {
                "Model":
                    model_name,

                **metrics,
            }
        )

        print(
            f"{model_name}"
        )

        print(
            f"  MAE  : "
            f"{metrics['MAE']:.4f} °C"
        )

        print(
            f"  RMSE : "
            f"{metrics['RMSE']:.4f} °C"
        )

        print(
            f"  R²   : "
            f"{metrics['R2']:.4f}"
        )

        print(
            f"  Bias : "
            f"{metrics['Bias']:.4f} °C"
        )

        print()

    return pd.DataFrame(
        rows
    )


# ==============================================================================
# ENTRAÎNEMENT AVEC EARLY STOPPING
# ==============================================================================


def train_validation_model(
    subtrain: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> tuple[
    XGBRegressor,
    int,
    pd.DataFrame,
]:

    print_separator(
        "ENTRAÎNEMENT AVANCÉ AVEC EARLY STOPPING"
    )

    X_train = (
        subtrain[
            features
        ]
    )

    y_train = (
        subtrain[
            TARGET
        ]
    )

    X_validation = (
        validation[
            features
        ]
    )

    y_validation = (
        validation[
            TARGET
        ]
    )

    print(
        f"Features : "
        f"{len(features)}"
    )

    print(
        f"Sous-train : "
        f"{len(X_train):,}"
    )

    print(
        f"Validation : "
        f"{len(X_validation):,}"
    )

    model = XGBRegressor(
        **MODEL_PARAMS
    )

    start = time.time()

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        verbose=False,
    )

    duration = (
        time.time()
        - start
    )

    print(
        f"\nEntraînement terminé : "
        f"{format_duration(duration)}"
    )


    # --------------------------------------------------------------------------
    # Meilleure itération
    # --------------------------------------------------------------------------

    best_iteration = getattr(
        model,
        "best_iteration",
        None,
    )

    if best_iteration is None:

        best_n_estimators = (
            MAX_ESTIMATORS
        )

    else:

        best_n_estimators = (
            int(
                best_iteration
            )
            + 1
        )

    print(
        f"Nombre optimal d'arbres : "
        f"{best_n_estimators}"
    )


    # --------------------------------------------------------------------------
    # Métriques de validation
    # --------------------------------------------------------------------------

    validation_pred = model.predict(
        X_validation
    )

    validation_metrics = (
        compute_metrics(
            y_validation,
            validation_pred,
        )
    )

    print(
        "\nPerformances validation :"
    )

    print(
        f"MAE  : "
        f"{validation_metrics['MAE']:.4f} °C"
    )

    print(
        f"RMSE : "
        f"{validation_metrics['RMSE']:.4f} °C"
    )

    print(
        f"R²   : "
        f"{validation_metrics['R2']:.4f}"
    )

    print(
        f"Bias : "
        f"{validation_metrics['Bias']:.4f} °C"
    )

    validation_results = (
        pd.DataFrame(
            [
                {
                    "validation_start_year":
                        VALIDATION_START_YEAR,

                    "validation_end_year":
                        VALIDATION_END_YEAR,

                    "best_n_estimators":
                        best_n_estimators,

                    **validation_metrics,
                }
            ]
        )
    )

    validation_results.to_csv(
        VALIDATION_RESULTS_FILE,
        index=False,
    )

    return (
        model,
        best_n_estimators,
        validation_results,
    )


# ==============================================================================
# MODÈLE FINAL
# ==============================================================================


def train_final_model(
    train: pd.DataFrame,
    features: list[str],
    best_n_estimators: int,
) -> XGBRegressor:

    print_separator(
        "RÉENTRAÎNEMENT DU MODÈLE FINAL"
    )

    final_params = (
        MODEL_PARAMS.copy()
    )

    final_params.pop(
        "early_stopping_rounds",
        None,
    )

    final_params[
        "n_estimators"
    ] = best_n_estimators

    model = XGBRegressor(
        **final_params
    )

    X_train = (
        train[
            features
        ]
    )

    y_train = (
        train[
            TARGET
        ]
    )

    print(
        f"Observations historiques : "
        f"{len(train):,}"
    )

    print(
        f"Nombre d'arbres : "
        f"{best_n_estimators}"
    )

    start = time.time()

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    duration = (
        time.time()
        - start
    )

    print(
        f"Durée : "
        f"{format_duration(duration)}"
    )

    return model


# ==============================================================================
# PRÉDICTION TEST
# ==============================================================================


def predict_test(
    model: XGBRegressor,
    test: pd.DataFrame,
    features: list[str],
) -> np.ndarray:

    print_separator(
        "PRÉDICTION TEST 2000-2025"
    )

    X_test = (
        test[
            features
        ]
    )

    start = time.time()

    y_pred = model.predict(
        X_test
    )

    duration = (
        time.time()
        - start
    )

    print(
        f"Prédictions : "
        f"{len(y_pred):,}"
    )

    print(
        f"Durée : "
        f"{format_duration(duration)}"
    )

    return y_pred


# ==============================================================================
# ÉVALUATION TEST
# ==============================================================================


def evaluate_test(
    test: pd.DataFrame,
    y_pred: np.ndarray,
) -> dict:

    print_separator(
        "PERFORMANCES XGBOOST AVANCÉ"
    )

    metrics = (
        compute_metrics(
            test[
                TARGET
            ],
            y_pred,
        )
    )

    print(
        f"MAE  : "
        f"{metrics['MAE']:.4f} °C"
    )

    print(
        f"RMSE : "
        f"{metrics['RMSE']:.4f} °C"
    )

    print(
        f"R²   : "
        f"{metrics['R2']:.4f}"
    )

    print(
        f"Bias : "
        f"{metrics['Bias']:.4f} °C"
    )

    return metrics


# ==============================================================================
# MÉTRIQUES GLOBALES
# ==============================================================================


def save_global_metrics(
    baseline_metrics: pd.DataFrame,
    advanced_metrics: dict,
) -> pd.DataFrame:

    advanced_row = pd.DataFrame(
        [
            {
                "Model":
                    "XGBoost_Advanced",

                **advanced_metrics,
            }
        ]
    )

    metrics = pd.concat(
        [
            baseline_metrics,
            advanced_row,
        ],
        ignore_index=True,
    )

    metrics.to_csv(
        METRICS_GLOBAL_FILE,
        index=False,
    )

    print(
        f"\nMétriques globales :\n"
        f"{METRICS_GLOBAL_FILE}"
    )

    return metrics


# ==============================================================================
# MÉTRIQUES PAR ANNÉE
# ==============================================================================


def compute_metrics_by_year(
    test: pd.DataFrame,
    y_pred: np.ndarray,
) -> pd.DataFrame:

    print_separator(
        "MÉTRIQUES PAR ANNÉE"
    )

    evaluation = pd.DataFrame(
        {
            "DATE":
                test[
                    DATE_COLUMN
                ]
                .values,

            "T_observed":
                test[
                    TARGET
                ]
                .values,

            "T_predicted":
                y_pred,
        }
    )

    evaluation[
        "year"
    ] = (
        pd.to_datetime(
            evaluation[
                "DATE"
            ]
        )
        .dt.year
    )

    rows = []

    for year, group in (
        evaluation.groupby(
            "year"
        )
    ):

        metrics = (
            compute_metrics(
                group[
                    "T_observed"
                ],
                group[
                    "T_predicted"
                ],
            )
        )

        rows.append(
            {
                "year":
                    int(
                        year
                    ),

                **metrics,
            }
        )

    metrics_by_year = (
        pd.DataFrame(
            rows
        )
    )

    metrics_by_year.to_csv(
        METRICS_BY_YEAR_FILE,
        index=False,
    )

    print(
        f"Années évaluées : "
        f"{len(metrics_by_year)}"
    )

    print(
        f"Fichier :\n"
        f"{METRICS_BY_YEAR_FILE}"
    )

    return metrics_by_year


# ==============================================================================
# FEATURE IMPORTANCE
# ==============================================================================


def save_feature_importance(
    model: XGBRegressor,
    features: list[str],
) -> pd.DataFrame:

    print_separator(
        "IMPORTANCE DES VARIABLES"
    )

    importance = pd.DataFrame(
        {
            "feature":
                features,

            "importance":
                model.feature_importances_,
        }
    )

    importance.sort_values(
        "importance",
        ascending=False,
        inplace=True,
    )

    importance.reset_index(
        drop=True,
        inplace=True,
    )

    importance.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
    )

    print(
        importance
        .head(
            30
        )
        .to_string(
            index=False
        )
    )

    print(
        f"\nFichier :\n"
        f"{FEATURE_IMPORTANCE_FILE}"
    )

    return importance


# ==============================================================================
# SAUVEGARDE LISTE DES FEATURES
# ==============================================================================


def save_selected_features(
    features: list[str],
) -> None:

    df = pd.DataFrame(
        {
            "feature":
                features,
        }
    )

    df.to_csv(
        SELECTED_FEATURES_FILE,
        index=False,
    )


# ==============================================================================
# PRÉDICTIONS COMPLÈTES
# ==============================================================================


def build_full_predictions(
    test: pd.DataFrame,
    y_pred: np.ndarray,
) -> pd.DataFrame:

    print_separator(
        "CONSTRUCTION DES PRÉDICTIONS COMPLÈTES"
    )

    predictions = pd.DataFrame(
        {
            "DATE":
                test[
                    DATE_COLUMN
                ]
                .values,

            "LAMBX":
                test[
                    "LAMBX"
                ]
                .values,

            "LAMBY":
                test[
                    "LAMBY"
                ]
                .values,

            "T_observed":
                test[
                    TARGET
                ]
                .values,

            "T_predicted":
                y_pred,
        }
    )

    predictions[
        "residual"
    ] = (
        predictions[
            "T_observed"
        ]
        -
        predictions[
            "T_predicted"
        ]
    )

    predictions[
        "absolute_error"
    ] = (
        predictions[
            "residual"
        ]
        .abs()
    )

    print(
        f"Prédictions : "
        f"{len(predictions):,}"
    )

    grid_points = (
        predictions[
            GRID_COLUMNS
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        f"Points de grille : "
        f"{grid_points:,}"
    )

    if (
        grid_points
        != EXPECTED_GRID_POINTS
    ):

        raise ValueError(
            "Nombre incorrect de points de grille."
        )

    return predictions


# ==============================================================================
# SAUVEGARDE PARQUET
# ==============================================================================


def save_full_predictions(
    predictions: pd.DataFrame,
) -> None:

    print_separator(
        "SAUVEGARDE DES PRÉDICTIONS COMPLÈTES"
    )

    predictions.to_parquet(
        PREDICTIONS_FULL_FILE,
        index=False,
        engine="pyarrow",
    )

    size_mb = (
        PREDICTIONS_FULL_FILE
        .stat()
        .st_size
        / (
            1024 ** 2
        )
    )

    print(
        f"Fichier :\n"
        f"{PREDICTIONS_FULL_FILE}"
    )

    print(
        f"Taille : "
        f"{size_mb:.2f} MB"
    )


# ==============================================================================
# ÉCHANTILLON CSV
# ==============================================================================


def save_predictions_sample(
    predictions: pd.DataFrame,
) -> None:

    sample_size = min(
        PREDICTIONS_SAMPLE_SIZE,
        len(
            predictions
        ),
    )

    sample = (
        predictions
        .sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        )
        .sort_values(
            DATE_COLUMN
        )
    )

    sample.to_csv(
        PREDICTIONS_SAMPLE_FILE,
        index=False,
    )


# ==============================================================================
# CONFIGURATION JSON
# ==============================================================================


def save_configuration(
    train: pd.DataFrame,
    test: pd.DataFrame,
    physical_variables: list[str],
    features: list[str],
    best_n_estimators: int,
    validation_results: pd.DataFrame,
    test_metrics: dict,
) -> None:

    final_params = (
        MODEL_PARAMS.copy()
    )

    final_params[
        "n_estimators"
    ] = best_n_estimators

    final_params.pop(
        "early_stopping_rounds",
        None,
    )

    validation_row = (
        validation_results
        .iloc[0]
        .to_dict()
    )

    validation_row = {
        key:
            (
                value.item()
                if hasattr(
                    value,
                    "item",
                )
                else value
            )
        for key, value
        in validation_row.items()
    }

    configuration = {

        "model":
            "XGBRegressor",

        "model_version":
            "advanced",

        "target":
            TARGET,

        "features":
            features,

        "n_features":
            len(
                features
            ),

        "physical_variables_available":
            physical_variables,

        "physical_variable_strategy":
            (
                "Les variables physiques contemporaines "
                "ne sont pas utilisées directement. "
                "Seules leurs valeurs historiques lag_1, "
                "lag_12 et roll_3 sont utilisées."
            ),

        "advanced_temperature_features": [
            "T_lag_24",
            "T_roll_24",
            "T_roll_36",
        ],

        "validation_strategy": {
            "type":
                "chronological_holdout",

            "validation_start_year":
                VALIDATION_START_YEAR,

            "validation_end_year":
                VALIDATION_END_YEAR,

            "test_start_year":
                2000,

            "test_end_year":
                2025,
        },

        "validation_results":
            validation_row,

        "best_n_estimators":
            best_n_estimators,

        "final_model_parameters":
            final_params,

        "train": {

            "n_rows":
                int(
                    len(
                        train
                    )
                ),

            "start_date":
                str(
                    train[
                        DATE_COLUMN
                    ]
                    .min()
                    .date()
                ),

            "end_date":
                str(
                    train[
                        DATE_COLUMN
                    ]
                    .max()
                    .date()
                ),
        },

        "test": {

            "n_rows":
                int(
                    len(
                        test
                    )
                ),

            "start_date":
                str(
                    test[
                        DATE_COLUMN
                    ]
                    .min()
                    .date()
                ),

            "end_date":
                str(
                    test[
                        DATE_COLUMN
                    ]
                    .max()
                    .date()
                ),

            "n_grid_points":
                int(
                    test[
                        GRID_COLUMNS
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),
        },

        "test_metrics":
            test_metrics,

        "methodological_note":
            (
                "Le test 2000-2025 n'est pas utilisé pour "
                "l'early stopping ou la sélection du nombre "
                "d'arbres. L'évaluation reste one-step-ahead "
                "car les variables retardées utilisent les "
                "observations historiques disponibles."
            ),
    }

    with open(
        MODEL_CONFIGURATION_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            configuration,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ==============================================================================
# AFFICHAGE DES FEATURES
# ==============================================================================


def print_feature_summary(
    features: list[str],
) -> None:

    print_separator(
        "FEATURES DU MODÈLE AVANCÉ"
    )

    print(
        f"Nombre total : "
        f"{len(features)}"
    )

    for index, feature in enumerate(
        features,
        start=1,
    ):

        print(
            f"{index:02d}. {feature}"
        )


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:

    global_start = time.time()

    print_separator(
        "XGBOOST AVANCÉ"
    )


    # --------------------------------------------------------------------------
    # 1. Initialisation
    # --------------------------------------------------------------------------

    create_output_directory()

    check_input_files()


    # --------------------------------------------------------------------------
    # 2. Chargement
    # --------------------------------------------------------------------------

    print_separator(
        "CHARGEMENT DES DONNÉES"
    )

    train = load_dataset(
        TRAIN_FILE,
        "TRAIN",
    )

    test = load_dataset(
        TEST_FILE,
        "TEST",
    )


    # --------------------------------------------------------------------------
    # 3. Contrôles
    # --------------------------------------------------------------------------

    check_basic_columns(
        train,
        test,
    )


    # --------------------------------------------------------------------------
    # 4. Détection variables physiques
    # --------------------------------------------------------------------------

    physical_variables = (
        detect_physical_variables(
            train,
            test,
        )
    )


    # --------------------------------------------------------------------------
    # 5. Reconstruction temporelle continue
    # --------------------------------------------------------------------------

    full = combine_datasets(
        train,
        test,
    )


    # --------------------------------------------------------------------------
    # 6. Features température avancées
    # --------------------------------------------------------------------------

    full = add_temperature_features(
        full
    )


    # --------------------------------------------------------------------------
    # 7. Features physiques historiques
    # --------------------------------------------------------------------------

    (
        full,
        physical_features,
    ) = add_physical_features(
        full,
        physical_variables,
    )


    # --------------------------------------------------------------------------
    # 8. Liste finale des features
    # --------------------------------------------------------------------------

    features = build_feature_list(
        physical_features
    )

    print_feature_summary(
        features
    )

    save_selected_features(
        features
    )


    # --------------------------------------------------------------------------
    # 9. Nettoyage
    # --------------------------------------------------------------------------

    (
        train_advanced,
        test_advanced,
    ) = prepare_advanced_datasets(
        full,
        features,
    )


    # --------------------------------------------------------------------------
    # Libération mémoire
    # --------------------------------------------------------------------------

    del full
    del train
    del test


    # --------------------------------------------------------------------------
    # 10. Validation grille
    # --------------------------------------------------------------------------

    validate_grid(
        train_advanced,
        test_advanced,
    )


    # --------------------------------------------------------------------------
    # 11. Baselines
    # --------------------------------------------------------------------------

    baseline_metrics = (
        evaluate_naive_baselines(
            test_advanced
        )
    )


    # --------------------------------------------------------------------------
    # 12. Split temporel validation
    # --------------------------------------------------------------------------

    (
        subtrain,
        validation,
    ) = temporal_validation_split(
        train_advanced
    )


    # --------------------------------------------------------------------------
    # 13. Early stopping
    # --------------------------------------------------------------------------

    (
        validation_model,
        best_n_estimators,
        validation_results,
    ) = train_validation_model(
        subtrain,
        validation,
        features,
    )


    # --------------------------------------------------------------------------
    # Libération mémoire
    # --------------------------------------------------------------------------

    del validation_model
    del subtrain
    del validation


    # --------------------------------------------------------------------------
    # 14. Modèle final sur toute la période historique
    # --------------------------------------------------------------------------

    final_model = train_final_model(
        train_advanced,
        features,
        best_n_estimators,
    )


    # --------------------------------------------------------------------------
    # 15. Test 2000-2025
    # --------------------------------------------------------------------------

    y_pred = predict_test(
        final_model,
        test_advanced,
        features,
    )


    # --------------------------------------------------------------------------
    # 16. Évaluation
    # --------------------------------------------------------------------------

    advanced_metrics = (
        evaluate_test(
            test_advanced,
            y_pred,
        )
    )


    # --------------------------------------------------------------------------
    # 17. Métriques globales
    # --------------------------------------------------------------------------

    save_global_metrics(
        baseline_metrics,
        advanced_metrics,
    )


    # --------------------------------------------------------------------------
    # 18. Métriques par année
    # --------------------------------------------------------------------------

    compute_metrics_by_year(
        test_advanced,
        y_pred,
    )


    # --------------------------------------------------------------------------
    # 19. Importance des variables
    # --------------------------------------------------------------------------

    save_feature_importance(
        final_model,
        features,
    )


    # --------------------------------------------------------------------------
    # 20. Prédictions complètes
    # --------------------------------------------------------------------------

    predictions_full = (
        build_full_predictions(
            test_advanced,
            y_pred,
        )
    )

    save_full_predictions(
        predictions_full
    )

    save_predictions_sample(
        predictions_full
    )


    # --------------------------------------------------------------------------
    # 21. Configuration
    # --------------------------------------------------------------------------

    save_configuration(
        train=train_advanced,
        test=test_advanced,
        physical_variables=
            physical_variables,
        features=features,
        best_n_estimators=
            best_n_estimators,
        validation_results=
            validation_results,
        test_metrics=
            advanced_metrics,
    )


    # --------------------------------------------------------------------------
    # 22. Résumé
    # --------------------------------------------------------------------------

    total_duration = (
        time.time()
        - global_start
    )

    print_separator(
        "ÉTAPE 05 TERMINÉE"
    )

    print(
        f"Features : "
        f"{len(features)}"
    )

    print(
        f"Variables physiques : "
        f"{len(physical_variables)}"
    )

    print(
        f"Nombre optimal d'arbres : "
        f"{best_n_estimators}"
    )

    print(
        f"\nMAE  : "
        f"{advanced_metrics['MAE']:.4f} °C"
    )

    print(
        f"RMSE : "
        f"{advanced_metrics['RMSE']:.4f} °C"
    )

    print(
        f"R²   : "
        f"{advanced_metrics['R2']:.4f}"
    )

    print(
        f"Bias : "
        f"{advanced_metrics['Bias']:.4f} °C"
    )

    print(
        f"\nObservations test : "
        f"{len(test_advanced):,}"
    )

    print(
        f"Points de grille : "
        f"{test_advanced[GRID_COLUMNS].drop_duplicates().shape[0]:,}"
    )

    print(
        f"\nDurée totale : "
        f"{format_duration(total_duration)}"
    )

    print(
        "\nFichiers créés :"
    )

    output_files = [
        METRICS_GLOBAL_FILE,
        METRICS_BY_YEAR_FILE,
        FEATURE_IMPORTANCE_FILE,
        MODEL_CONFIGURATION_FILE,
        VALIDATION_RESULTS_FILE,
        PREDICTIONS_SAMPLE_FILE,
        PREDICTIONS_FULL_FILE,
        SELECTED_FEATURES_FILE,
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
        "06_compare_models.py"
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

        print_separator(
            "ERREUR"
        )

        print(
            str(
                exc
            )
        )

        raise