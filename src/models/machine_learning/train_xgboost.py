#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
03_train_xgboost.py
===============================================================================

Entraînement du modèle XGBoost baseline pour la prédiction mensuelle
de la température.

Pipeline :
    01_prepare_full_timeseries.py
        ↓
    02_build_training_dataset.py
        ↓
    03_train_xgboost.py
        ↓
    04_analyse_baseline.py

Ce script :
    1. Charge les données train et test.
    2. Vérifie les colonnes nécessaires.
    3. Entraîne un modèle XGBoost baseline.
    4. Calcule deux modèles naïfs :
         - température du mois précédent ;
         - température du même mois de l'année précédente.
    5. Évalue XGBoost sur 2000–2025.
    6. Calcule les métriques globales.
    7. Calcule les métriques année par année.
    8. Calcule l'importance des variables.
    9. Sauvegarde un échantillon des prédictions.
   10. Sauvegarde TOUTES les prédictions dans predictions_full.parquet.

IMPORTANT
---------
Le fichier predictions_full.parquet est indispensable pour :
    - 04_analyse_baseline.py ;
    - l'analyse des résidus ;
    - l'analyse spatiale ;
    - les métriques par point de grille ;
    - les cartes / heatmaps de France.

Auteur : Projet climat
===============================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

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
# CHEMINS DU PROJET
# ==============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_DIR / "data"

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "xgboost_baseline"
)



# ------------------------------------------------------------------------------
# Fichiers d'entrée
# ------------------------------------------------------------------------------

DATA_DIR = PROJECT_DIR / "data"
MODEL_DATA_DIR = DATA_DIR / "model"

TRAIN_FILE = (
    MODEL_DATA_DIR
    / "train_1960_1999.csv"
)

TEST_FILE = (
    MODEL_DATA_DIR
    / "test_2000_2025.csv"
)


# ------------------------------------------------------------------------------
# Fichiers de sortie
# ------------------------------------------------------------------------------

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

PREDICTIONS_SAMPLE_FILE = (
    RESULTS_DIR
    / "predictions_sample.csv"
)

PREDICTIONS_FULL_FILE = (
    RESULTS_DIR
    / "predictions_full.parquet"
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

TARGET = "T"

FEATURES = [
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


# ------------------------------------------------------------------------------
# Paramètres XGBoost baseline
# ------------------------------------------------------------------------------

MODEL_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 400,
    "max_depth": 8,
    "learning_rate": 0.05,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}


# ------------------------------------------------------------------------------
# Échantillon destiné à une inspection rapide
# ------------------------------------------------------------------------------

PREDICTIONS_SAMPLE_SIZE = 100_000

RANDOM_STATE = 42


# ==============================================================================
# OUTILS
# ==============================================================================


def print_separator(title: str) -> None:
    """
    Affiche un séparateur lisible dans le terminal.
    """

    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def format_duration(seconds: float) -> str:
    """
    Convertit une durée en secondes vers un format lisible.
    """

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    if minutes > 0:
        return (
            f"{minutes} min "
            f"{remaining_seconds:.1f} s"
        )

    return f"{remaining_seconds:.1f} s"


# ==============================================================================
# INITIALISATION
# ==============================================================================


def create_output_directory() -> None:
    """
    Crée le dossier de résultats s'il n'existe pas.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def check_input_files() -> None:
    """
    Vérifie que les datasets train et test existent.
    """

    missing_files = []

    for file_path in [
        TRAIN_FILE,
        TEST_FILE,
    ]:

        if not file_path.exists():
            missing_files.append(
                str(file_path)
            )

    if missing_files:

        message = (
            "Fichier(s) d'entrée manquant(s) :\n"
            + "\n".join(missing_files)
            + "\n\n"
            + "Exécute d'abord "
            + "02_build_training_dataset.py."
        )

        raise FileNotFoundError(
            message
        )


# ==============================================================================
# CHARGEMENT DES DONNÉES
# ==============================================================================


def load_dataset(
    file_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Charge un dataset CSV et convertit DATE
    au format datetime.
    """

    print(
        f"Chargement de {dataset_name} :\n"
        f"{file_path}"
    )

    start_time = time.time()

    df = pd.read_csv(
        file_path,
        parse_dates=["DATE"],
    )

    duration = time.time() - start_time

    print(
        f"{dataset_name} chargé : "
        f"{len(df):,} lignes "
        f"× {df.shape[1]} colonnes"
    )

    print(
        f"Durée : "
        f"{format_duration(duration)}"
    )

    return df


# ==============================================================================
# CONTRÔLES
# ==============================================================================


def check_required_columns(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Vérifie la présence des colonnes nécessaires.
    """

    required_columns = (
        ["DATE", TARGET]
        + FEATURES
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{dataset_name} : "
            f"colonnes manquantes : "
            f"{missing_columns}"
        )


def check_missing_values(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Vérifie les valeurs manquantes dans les
    variables utilisées pour l'apprentissage.
    """

    columns = FEATURES + [TARGET]

    missing = (
        df[columns]
        .isna()
        .sum()
    )

    missing = missing[
        missing > 0
    ]

    if not missing.empty:

        print(
            f"\nATTENTION : valeurs manquantes "
            f"détectées dans {dataset_name}:"
        )

        print(missing)

        raise ValueError(
            f"{dataset_name} contient des "
            "valeurs manquantes dans les "
            "variables utilisées par le modèle."
        )

    print(
        f"{dataset_name} : "
        "aucune valeur manquante "
        "dans les variables du modèle."
    )


def print_dataset_summary(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Affiche un résumé des datasets.
    """

    print_separator(
        "RÉSUMÉ DES DONNÉES"
    )

    print(
        f"Train : "
        f"{len(train):,} lignes"
    )

    print(
        f"Période train : "
        f"{train['DATE'].min().date()} "
        f"→ "
        f"{train['DATE'].max().date()}"
    )

    print(
        f"\nTest : "
        f"{len(test):,} lignes"
    )

    print(
        f"Période test : "
        f"{test['DATE'].min().date()} "
        f"→ "
        f"{test['DATE'].max().date()}"
    )

    n_grid_train = (
        train[
            ["LAMBX", "LAMBY"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    n_grid_test = (
        test[
            ["LAMBX", "LAMBY"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        f"\nPoints de grille train : "
        f"{n_grid_train:,}"
    )

    print(
        f"Points de grille test : "
        f"{n_grid_test:,}"
    )

    print(
        f"\nNombre de features : "
        f"{len(FEATURES)}"
    )

    print(
        "\nFeatures utilisées :"
    )

    for feature in FEATURES:
        print(
            f"  - {feature}"
        )


# ==============================================================================
# PRÉPARATION X / y
# ==============================================================================


def prepare_arrays(
    train: pd.DataFrame,
    test: pd.DataFrame,
):
    """
    Prépare les matrices d'apprentissage
    et d'évaluation.
    """

    X_train = train[
        FEATURES
    ]

    y_train = train[
        TARGET
    ]

    X_test = test[
        FEATURES
    ]

    y_test = test[
        TARGET
    ]

    return (
        X_train,
        y_train,
        X_test,
        y_test,
    )


# ==============================================================================
# MÉTRIQUES
# ==============================================================================


def compute_metrics(
    y_true,
    y_pred,
) -> dict:
    """
    Calcule les métriques principales.
    """

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    residuals = (
        y_true
        - y_pred
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    bias = np.mean(
        residuals
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Bias": float(bias),
    }


# ==============================================================================
# BASELINES NAÏVES
# ==============================================================================


def evaluate_naive_baselines(
    test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Évalue deux modèles de référence :

        1. T_lag_1
        2. T_lag_12
    """

    print_separator(
        "BASELINES NAÏVES"
    )

    y_true = test[
        TARGET
    ]

    baseline_results = []


    # --------------------------------------------------------------------------
    # Baseline lag 1
    # --------------------------------------------------------------------------

    metrics_lag1 = compute_metrics(
        y_true,
        test["T_lag_1"],
    )

    baseline_results.append(
        {
            "Model":
                "Naive_T_lag_1",
            **metrics_lag1,
        }
    )

    print(
        "Naive T_lag_1"
    )

    print(
        f"  MAE  : "
        f"{metrics_lag1['MAE']:.4f} °C"
    )

    print(
        f"  RMSE : "
        f"{metrics_lag1['RMSE']:.4f} °C"
    )

    print(
        f"  R²   : "
        f"{metrics_lag1['R2']:.4f}"
    )

    print(
        f"  Bias : "
        f"{metrics_lag1['Bias']:.4f} °C"
    )


    # --------------------------------------------------------------------------
    # Baseline lag 12
    # --------------------------------------------------------------------------

    metrics_lag12 = compute_metrics(
        y_true,
        test["T_lag_12"],
    )

    baseline_results.append(
        {
            "Model":
                "Naive_T_lag_12",
            **metrics_lag12,
        }
    )

    print(
        "\nNaive T_lag_12"
    )

    print(
        f"  MAE  : "
        f"{metrics_lag12['MAE']:.4f} °C"
    )

    print(
        f"  RMSE : "
        f"{metrics_lag12['RMSE']:.4f} °C"
    )

    print(
        f"  R²   : "
        f"{metrics_lag12['R2']:.4f}"
    )

    print(
        f"  Bias : "
        f"{metrics_lag12['Bias']:.4f} °C"
    )

    return pd.DataFrame(
        baseline_results
    )


# ==============================================================================
# MODÈLE XGBOOST
# ==============================================================================


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBRegressor:
    """
    Entraîne le modèle XGBoost baseline.
    """

    print_separator(
        "ENTRAÎNEMENT XGBOOST"
    )

    print(
        "Configuration :"
    )

    for key, value in MODEL_PARAMS.items():
        print(
            f"  {key}: {value}"
        )

    model = XGBRegressor(
        **MODEL_PARAMS
    )

    start_time = time.time()

    model.fit(
        X_train,
        y_train,
    )

    duration = (
        time.time()
        - start_time
    )

    print(
        "\nEntraînement terminé."
    )

    print(
        f"Durée : "
        f"{format_duration(duration)}"
    )

    return model


def predict_model(
    model: XGBRegressor,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Produit les prédictions du jeu de test.
    """

    print_separator(
        "PRÉDICTION SUR LE JEU DE TEST"
    )

    start_time = time.time()

    predictions = model.predict(
        X_test
    )

    duration = (
        time.time()
        - start_time
    )

    print(
        f"Prédictions : "
        f"{len(predictions):,}"
    )

    print(
        f"Durée : "
        f"{format_duration(duration)}"
    )

    return predictions


# ==============================================================================
# ÉVALUATION XGBOOST
# ==============================================================================


def evaluate_xgboost(
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> dict:
    """
    Calcule et affiche les performances globales
    de XGBoost.
    """

    print_separator(
        "PERFORMANCES XGBOOST"
    )

    metrics = compute_metrics(
        y_test,
        y_pred,
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
    xgboost_metrics: dict,
) -> None:
    """
    Sauvegarde les métriques globales des
    modèles naïfs et de XGBoost.
    """

    xgb_row = pd.DataFrame(
        [
            {
                "Model":
                    "XGBoost",
                **xgboost_metrics,
            }
        ]
    )

    metrics_global = pd.concat(
        [
            baseline_metrics,
            xgb_row,
        ],
        ignore_index=True,
    )

    metrics_global.to_csv(
        METRICS_GLOBAL_FILE,
        index=False,
    )

    print(
        f"\nMétriques globales sauvegardées :\n"
        f"{METRICS_GLOBAL_FILE}"
    )


# ==============================================================================
# MÉTRIQUES PAR ANNÉE
# ==============================================================================


def compute_metrics_by_year(
    test: pd.DataFrame,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Calcule les performances XGBoost
    année par année.
    """

    print_separator(
        "MÉTRIQUES PAR ANNÉE"
    )

    evaluation = pd.DataFrame(
        {
            "DATE":
                test["DATE"].values,

            "T_observed":
                test[TARGET].values,

            "T_predicted":
                y_pred,
        }
    )

    evaluation["year"] = (
        pd.to_datetime(
            evaluation["DATE"]
        )
        .dt.year
    )

    rows = []

    for year, group in (
        evaluation
        .groupby("year")
    ):

        metrics = compute_metrics(
            group["T_observed"],
            group["T_predicted"],
        )

        rows.append(
            {
                "year":
                    int(year),

                "MAE":
                    metrics["MAE"],

                "RMSE":
                    metrics["RMSE"],

                "R2":
                    metrics["R2"],

                "Bias":
                    metrics["Bias"],
            }
        )

    metrics_by_year = pd.DataFrame(
        rows
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
) -> pd.DataFrame:
    """
    Sauvegarde l'importance des variables.
    """

    print_separator(
        "IMPORTANCE DES VARIABLES"
    )

    feature_importance = pd.DataFrame(
        {
            "feature":
                FEATURES,

            "importance":
                model.feature_importances_,
        }
    )

    feature_importance = (
        feature_importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    feature_importance.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
    )

    print(
        feature_importance.to_string(
            index=False
        )
    )

    print(
        f"\nFichier :\n"
        f"{FEATURE_IMPORTANCE_FILE}"
    )

    return feature_importance


# ==============================================================================
# PRÉDICTIONS COMPLÈTES
# ==============================================================================


def build_full_predictions(
    test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Construit le dataframe contenant toutes
    les prédictions du jeu de test.

    Ce dataframe est indispensable pour
    l'analyse spatiale du modèle.
    """

    print_separator(
        "CONSTRUCTION DES PRÉDICTIONS COMPLÈTES"
    )

    predictions = pd.DataFrame(
        {
            "DATE":
                test["DATE"].values,

            "LAMBX":
                test["LAMBX"].values,

            "LAMBY":
                test["LAMBY"].values,

            "T_observed":
                y_test.to_numpy(),

            "T_predicted":
                y_pred,
        }
    )


    # --------------------------------------------------------------------------
    # Résidu
    #
    # Convention utilisée :
    #
    # residual = observation - prédiction
    #
    # residual positif :
    #     le modèle sous-estime la température.
    #
    # residual négatif :
    #     le modèle surestime la température.
    # --------------------------------------------------------------------------

    predictions["residual"] = (
        predictions["T_observed"]
        - predictions["T_predicted"]
    )


    # --------------------------------------------------------------------------
    # Erreur absolue
    # --------------------------------------------------------------------------

    predictions["absolute_error"] = (
        predictions["residual"]
        .abs()
    )


    print(
        f"Nombre de prédictions : "
        f"{len(predictions):,}"
    )


    # --------------------------------------------------------------------------
    # Vérification du nombre de points de grille
    # --------------------------------------------------------------------------

    n_grid_points = (
        predictions[
            ["LAMBX", "LAMBY"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        f"Nombre de points de grille : "
        f"{n_grid_points:,}"
    )


    # --------------------------------------------------------------------------
    # Vérification période
    # --------------------------------------------------------------------------

    print(
        f"Période : "
        f"{predictions['DATE'].min().date()} "
        f"→ "
        f"{predictions['DATE'].max().date()}"
    )


    # --------------------------------------------------------------------------
    # Vérification valeurs manquantes
    # --------------------------------------------------------------------------

    missing = (
        predictions
        .isna()
        .sum()
    )

    if missing.sum() > 0:

        print(
            "\nATTENTION : valeurs manquantes "
            "dans les prédictions :"
        )

        print(
            missing[
                missing > 0
            ]
        )

    else:

        print(
            "Aucune valeur manquante."
        )

    return predictions


# ==============================================================================
# SAUVEGARDE DU PARQUET COMPLET
# ==============================================================================


def save_full_predictions(
    predictions: pd.DataFrame,
) -> None:
    """
    Sauvegarde toutes les prédictions en Parquet.

    Format attendu par 04_analyse_baseline.py :

        DATE
        LAMBX
        LAMBY
        T_observed
        T_predicted
        residual
        absolute_error
    """

    print_separator(
        "SAUVEGARDE DES PRÉDICTIONS COMPLÈTES"
    )

    try:

        predictions.to_parquet(
            PREDICTIONS_FULL_FILE,
            index=False,
            engine="pyarrow",
        )

    except ImportError as exc:

        raise ImportError(
            "\nImpossible de sauvegarder le fichier Parquet.\n"
            "Le package pyarrow n'est probablement pas installé.\n\n"
            "Installe-le avec :\n\n"
            "pip install pyarrow\n"
        ) from exc


    print(
        f"Fichier créé :\n"
        f"{PREDICTIONS_FULL_FILE}"
    )


    # --------------------------------------------------------------------------
    # Vérification du fichier créé
    # --------------------------------------------------------------------------

    if not PREDICTIONS_FULL_FILE.exists():

        raise FileNotFoundError(
            "Le fichier predictions_full.parquet "
            "n'a pas été créé correctement."
        )


    file_size_mb = (
        PREDICTIONS_FULL_FILE
        .stat()
        .st_size
        / (1024 ** 2)
    )

    print(
        f"Taille du fichier : "
        f"{file_size_mb:.2f} MB"
    )

    print(
        f"Lignes sauvegardées : "
        f"{len(predictions):,}"
    )


# ==============================================================================
# ÉCHANTILLON DES PRÉDICTIONS
# ==============================================================================


def save_predictions_sample(
    predictions: pd.DataFrame,
) -> None:
    """
    Sauvegarde un sous-échantillon CSV permettant
    une inspection rapide.
    """

    print_separator(
        "ÉCHANTILLON DES PRÉDICTIONS"
    )

    sample_size = min(
        PREDICTIONS_SAMPLE_SIZE,
        len(predictions),
    )

    predictions_sample = (
        predictions
        .sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        )
        .sort_values("DATE")
        .reset_index(drop=True)
    )

    predictions_sample.to_csv(
        PREDICTIONS_SAMPLE_FILE,
        index=False,
    )

    print(
        f"Échantillon : "
        f"{len(predictions_sample):,} lignes"
    )

    print(
        f"Fichier :\n"
        f"{PREDICTIONS_SAMPLE_FILE}"
    )


# ==============================================================================
# CONFIGURATION DU MODÈLE
# ==============================================================================


def save_model_configuration(
    train: pd.DataFrame,
    test: pd.DataFrame,
    xgboost_metrics: dict,
) -> None:
    """
    Sauvegarde les paramètres du modèle
    et les informations principales
    de l'expérience.
    """

    configuration = {

        "model":
            "XGBRegressor",

        "target":
            TARGET,

        "features":
            FEATURES,

        "model_parameters":
            MODEL_PARAMS,

        "train": {
            "n_rows":
                int(len(train)),

            "start_date":
                str(
                    train["DATE"]
                    .min()
                    .date()
                ),

            "end_date":
                str(
                    train["DATE"]
                    .max()
                    .date()
                ),

            "n_grid_points":
                int(
                    train[
                        ["LAMBX", "LAMBY"]
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),
        },

        "test": {
            "n_rows":
                int(len(test)),

            "start_date":
                str(
                    test["DATE"]
                    .min()
                    .date()
                ),

            "end_date":
                str(
                    test["DATE"]
                    .max()
                    .date()
                ),

            "n_grid_points":
                int(
                    test[
                        ["LAMBX", "LAMBY"]
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),
        },

        "test_metrics":
            xgboost_metrics,

        "methodological_note": (
            "Evaluation one-step-ahead. "
            "Les variables de lag du jeu de test "
            "utilisent les températures observées "
            "des mois précédents. "
            "Il ne s'agit donc pas d'une prévision "
            "récursive complète sur 2000-2025."
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

    print(
        f"\nConfiguration sauvegardée :\n"
        f"{MODEL_CONFIGURATION_FILE}"
    )


# ==============================================================================
# VALIDATION DU FICHIER predictions_full.parquet
# ==============================================================================


def validate_full_predictions(
    predictions: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Vérifie que le fichier complet correspond bien
    au dataset de test.
    """

    print_separator(
        "VALIDATION DES PRÉDICTIONS COMPLÈTES"
    )

    expected_rows = len(
        test
    )

    actual_rows = len(
        predictions
    )

    expected_grid_points = (
        test[
            ["LAMBX", "LAMBY"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    actual_grid_points = (
        predictions[
            ["LAMBX", "LAMBY"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        f"Lignes test attendues : "
        f"{expected_rows:,}"
    )

    print(
        f"Lignes prédictions    : "
        f"{actual_rows:,}"
    )

    print(
        f"\nPoints grille attendus : "
        f"{expected_grid_points:,}"
    )

    print(
        f"Points grille obtenus   : "
        f"{actual_grid_points:,}"
    )


    if actual_rows != expected_rows:

        raise ValueError(
            "Le nombre de prédictions ne "
            "correspond pas au nombre de "
            "lignes du jeu de test."
        )


    if actual_grid_points != expected_grid_points:

        raise ValueError(
            "Le nombre de points de grille "
            "des prédictions ne correspond "
            "pas au jeu de test."
        )


    expected_columns = [
        "DATE",
        "LAMBX",
        "LAMBY",
        "T_observed",
        "T_predicted",
        "residual",
        "absolute_error",
    ]

    missing_columns = [
        column
        for column in expected_columns
        if column not in predictions.columns
    ]

    if missing_columns:

        raise ValueError(
            "Colonnes manquantes dans "
            "predictions_full : "
            f"{missing_columns}"
        )


    print(
        "\nOK : predictions_full est cohérent "
        "avec le jeu de test."
    )


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:
    """
    Pipeline principal d'entraînement
    et d'évaluation.
    """

    global_start_time = time.time()

    print_separator(
        "ENTRAÎNEMENT DU MODÈLE XGBOOST BASELINE"
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

    print_separator(
        "CONTRÔLES DES DONNÉES"
    )

    check_required_columns(
        train,
        "TRAIN",
    )

    check_required_columns(
        test,
        "TEST",
    )

    check_missing_values(
        train,
        "TRAIN",
    )

    check_missing_values(
        test,
        "TEST",
    )

    print_dataset_summary(
        train,
        test,
    )


    # --------------------------------------------------------------------------
    # 4. Préparation X / y
    # --------------------------------------------------------------------------

    (
        X_train,
        y_train,
        X_test,
        y_test,
    ) = prepare_arrays(
        train,
        test,
    )


    # --------------------------------------------------------------------------
    # 5. Baselines naïves
    # --------------------------------------------------------------------------

    baseline_metrics = (
        evaluate_naive_baselines(
            test
        )
    )


    # --------------------------------------------------------------------------
    # 6. Entraînement XGBoost
    # --------------------------------------------------------------------------

    model = train_model(
        X_train,
        y_train,
    )


    # --------------------------------------------------------------------------
    # 7. Prédictions
    # --------------------------------------------------------------------------

    y_pred = predict_model(
        model,
        X_test,
    )


    # --------------------------------------------------------------------------
    # 8. Métriques globales XGBoost
    # --------------------------------------------------------------------------

    xgboost_metrics = (
        evaluate_xgboost(
            y_test,
            y_pred,
        )
    )


    # --------------------------------------------------------------------------
    # 9. Sauvegarde métriques globales
    # --------------------------------------------------------------------------

    save_global_metrics(
        baseline_metrics,
        xgboost_metrics,
    )


    # --------------------------------------------------------------------------
    # 10. Métriques annuelles
    # --------------------------------------------------------------------------

    compute_metrics_by_year(
        test,
        y_pred,
    )


    # --------------------------------------------------------------------------
    # 11. Importance des variables
    # --------------------------------------------------------------------------

    save_feature_importance(
        model
    )


    # --------------------------------------------------------------------------
    # 12. Construction des prédictions complètes
    #
    # IMPORTANT :
    # C'est cette partie qui manquait dans la version précédente.
    # --------------------------------------------------------------------------

    predictions_full = (
        build_full_predictions(
            test=test,
            y_test=y_test,
            y_pred=y_pred,
        )
    )


    # --------------------------------------------------------------------------
    # 13. Validation
    # --------------------------------------------------------------------------

    validate_full_predictions(
        predictions_full,
        test,
    )


    # --------------------------------------------------------------------------
    # 14. Sauvegarde predictions_full.parquet
    #
    # Ce fichier sera ensuite lu par :
    # 04_analyse_baseline.py
    # --------------------------------------------------------------------------

    save_full_predictions(
        predictions_full
    )


    # --------------------------------------------------------------------------
    # 15. Sauvegarde d'un petit échantillon CSV
    # --------------------------------------------------------------------------

    save_predictions_sample(
        predictions_full
    )


    # --------------------------------------------------------------------------
    # 16. Configuration
    # --------------------------------------------------------------------------

    save_model_configuration(
        train,
        test,
        xgboost_metrics,
    )


    # --------------------------------------------------------------------------
    # 17. Résumé final
    # --------------------------------------------------------------------------

    total_duration = (
        time.time()
        - global_start_time
    )

    print_separator(
        "ENTRAÎNEMENT TERMINÉ"
    )

    print(
        f"MAE XGBoost  : "
        f"{xgboost_metrics['MAE']:.4f} °C"
    )

    print(
        f"RMSE XGBoost : "
        f"{xgboost_metrics['RMSE']:.4f} °C"
    )

    print(
        f"R² XGBoost   : "
        f"{xgboost_metrics['R2']:.4f}"
    )

    print(
        f"Bias XGBoost : "
        f"{xgboost_metrics['Bias']:.4f} °C"
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
        PREDICTIONS_SAMPLE_FILE,
        PREDICTIONS_FULL_FILE,
    ]

    for file_path in output_files:

        if file_path.exists():

            print(
                f"  [OK] {file_path}"
            )

        else:

            print(
                f"  [MANQUANT] {file_path}"
            )


    print(
        "\nÉtape suivante :\n"
        "04_analyse_baseline.py"
    )


# ==============================================================================
# EXÉCUTION
# ==============================================================================


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n\nExécution interrompue "
            "par l'utilisateur."
        )

        sys.exit(1)

    except Exception as exc:

        print_separator(
            "ERREUR"
        )

        print(
            str(exc)
        )

        raise