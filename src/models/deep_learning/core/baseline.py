from __future__ import annotations

import math

import numpy as np


def persistence_forecast(
    x_array: np.ndarray,
) -> np.ndarray:
    """
    Baseline de persistance.

    La température prédite pour demain est égale à la
    dernière température observée dans la séquence.

    Forme attendue de X :
    (nombre_sequences, nombre_variables, longueur_sequence)
    """

    if x_array.ndim != 3:
        raise ValueError(
            "X doit avoir trois dimensions : "
            "(séquences, variables, longueur). "
            f"Forme reçue : {x_array.shape}"
        )

    if x_array.shape[0] == 0:
        raise ValueError(
            "Aucune séquence n'a été fournie."
        )

    if x_array.shape[1] != 1:
        raise ValueError(
            "La baseline actuelle attend une seule variable. "
            f"Nombre reçu : {x_array.shape[1]}"
        )

    predictions = x_array[:, 0, -1]

    return predictions.astype(
        np.float32,
        copy=False,
    )


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calcule les principales métriques de régression.

    Les métriques sont calculées directement en degrés Celsius.
    """

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    ).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "Les observations et les prédictions doivent avoir "
            "la même forme. "
            f"y_true={y_true.shape}, y_pred={y_pred.shape}"
        )

    if y_true.size == 0:
        raise ValueError(
            "Les tableaux de données sont vides."
        )

    if not np.isfinite(y_true).all():
        raise ValueError(
            "y_true contient des valeurs non finies."
        )

    if not np.isfinite(y_pred).all():
        raise ValueError(
            "y_pred contient des valeurs non finies."
        )

    residuals = y_pred - y_true

    mae = float(
        np.mean(np.abs(residuals))
    )

    mse = float(
        np.mean(np.square(residuals))
    )

    rmse = float(
        math.sqrt(mse)
    )

    bias = float(
        np.mean(residuals)
    )

    denominator = float(
        np.sum(
            np.square(
                y_true - np.mean(y_true)
            )
        )
    )

    if denominator == 0:
        r2 = float("nan")
    else:
        numerator = float(
            np.sum(
                np.square(
                    y_true - y_pred
                )
            )
        )

        r2 = float(
            1.0 - numerator / denominator
        )

    return {
        "MAE_C": mae,
        "MSE_C2": mse,
        "RMSE_C": rmse,
        "BIAS_C": bias,
        "R2": r2,
    }