from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from config import (  # noqa: E402
    METRICS_DIR,
    PREDICTION_DIR,
    create_output_directories,
)


BOOTSTRAP_ITERATIONS = 5_000
BLOCK_LENGTH = 30
RANDOM_SEED = 42


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    errors = y_pred - y_true

    mae = float(
        np.mean(np.abs(errors))
    )

    mse = float(
        np.mean(errors**2)
    )

    rmse = float(
        np.sqrt(mse)
    )

    bias = float(
        np.mean(errors)
    )

    denominator = float(
        np.sum(
            (
                y_true
                - np.mean(y_true)
            )
            ** 2
        )
    )

    r2 = (
        float(
            1.0
            - np.sum(errors**2)
            / denominator
        )
        if denominator > 0
        else float("nan")
    )

    return {
        "MAE_C": mae,
        "RMSE_C": rmse,
        "BIAS_C": bias,
        "R2": r2,
    }


def identify_season(
    month: int,
) -> str:
    if month in (12, 1, 2):
        return "DJF_hiver"

    if month in (3, 4, 5):
        return "MAM_printemps"

    if month in (6, 7, 8):
        return "JJA_ete"

    return "SON_automne"


def moving_block_indices(
    number_of_observations: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    number_of_blocks = int(
        np.ceil(
            number_of_observations
            / block_length
        )
    )

    maximum_start = (
        number_of_observations
        - block_length
    )

    starts = rng.integers(
        low=0,
        high=maximum_start + 1,
        size=number_of_blocks,
    )

    indices = np.concatenate(
        [
            np.arange(
                start,
                start + block_length,
            )
            for start in starts
        ]
    )

    return indices[
        :number_of_observations
    ]


def block_bootstrap_comparison(
    y_true: np.ndarray,
    tcn_predictions: np.ndarray,
    convlstm_predictions: np.ndarray,
) -> dict[str, object]:
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    number_of_observations = len(
        y_true
    )

    mae_differences = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=np.float64,
    )

    rmse_differences = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=np.float64,
    )

    for iteration in range(
        BOOTSTRAP_ITERATIONS
    ):
        indices = moving_block_indices(
            number_of_observations,
            BLOCK_LENGTH,
            rng,
        )

        sampled_true = y_true[indices]

        sampled_tcn = (
            tcn_predictions[indices]
        )

        sampled_convlstm = (
            convlstm_predictions[indices]
        )

        tcn_errors = (
            sampled_tcn
            - sampled_true
        )

        convlstm_errors = (
            sampled_convlstm
            - sampled_true
        )

        tcn_mae = np.mean(
            np.abs(tcn_errors)
        )

        convlstm_mae = np.mean(
            np.abs(convlstm_errors)
        )

        tcn_rmse = np.sqrt(
            np.mean(tcn_errors**2)
        )

        convlstm_rmse = np.sqrt(
            np.mean(convlstm_errors**2)
        )

        # Une différence positive signifie que
        # le ConvLSTM est meilleur.
        mae_differences[iteration] = (
            tcn_mae
            - convlstm_mae
        )

        rmse_differences[iteration] = (
            tcn_rmse
            - convlstm_rmse
        )

    return {
        "iterations": BOOTSTRAP_ITERATIONS,
        "block_length_days": BLOCK_LENGTH,
        "MAE_difference_TCN_minus_ConvLSTM_C": {
            "mean": float(
                mae_differences.mean()
            ),
            "confidence_interval_95": [
                float(
                    np.percentile(
                        mae_differences,
                        2.5,
                    )
                ),
                float(
                    np.percentile(
                        mae_differences,
                        97.5,
                    )
                ),
            ],
            "probability_ConvLSTM_better": float(
                np.mean(
                    mae_differences > 0
                )
            ),
        },
        "RMSE_difference_TCN_minus_ConvLSTM_C": {
            "mean": float(
                rmse_differences.mean()
            ),
            "confidence_interval_95": [
                float(
                    np.percentile(
                        rmse_differences,
                        2.5,
                    )
                ),
                float(
                    np.percentile(
                        rmse_differences,
                        97.5,
                    )
                ),
            ],
            "probability_ConvLSTM_better": float(
                np.mean(
                    rmse_differences > 0
                )
            ),
        },
    }


def main() -> None:
    create_output_directories()

    convlstm_path = (
        PREDICTION_DIR
        / "convlstm_spatial_center_predictions.csv"
    )

    tcn_path = (
        PREDICTION_DIR
        / "tcn_spatial_center_predictions.csv"
    )

    if not convlstm_path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {convlstm_path}"
        )

    dataframe = pd.read_csv(
        convlstm_path
    )

    dataframe["DATE"] = pd.to_datetime(
        dataframe["DATE"],
        errors="coerce",
    )

    if "TEMPERATURE_TCN" not in dataframe.columns:
        if not tcn_path.is_file():
            raise FileNotFoundError(
                f"Fichier TCN introuvable : {tcn_path}"
            )

        tcn_dataframe = pd.read_csv(
            tcn_path
        )

        tcn_dataframe["DATE"] = pd.to_datetime(
            tcn_dataframe["DATE"],
            errors="coerce",
        )

        dataframe = dataframe.merge(
            tcn_dataframe[
                [
                    "DATE",
                    "TEMPERATURE_TCN",
                ]
            ],
            on="DATE",
            how="left",
            validate="one_to_one",
        )

    required_columns = [
        "DATE",
        "TEMPERATURE_REELLE",
        "TEMPERATURE_PERSISTANCE",
        "TEMPERATURE_TCN",
        "TEMPERATURE_CONVLSTM",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes absentes : {missing_columns}"
        )

    dataframe = (
        dataframe
        .dropna(
            subset=required_columns
        )
        .sort_values("DATE")
        .reset_index(drop=True)
    )

    dataframe["SAISON"] = (
        dataframe["DATE"]
        .dt.month
        .map(identify_season)
    )

    cold_threshold = float(
        dataframe[
            "TEMPERATURE_REELLE"
        ].quantile(0.10)
    )

    hot_threshold = float(
        dataframe[
            "TEMPERATURE_REELLE"
        ].quantile(0.90)
    )

    groups: dict[str, pd.DataFrame] = {
        "ensemble_test": dataframe,
        "DJF_hiver": dataframe.loc[
            dataframe["SAISON"]
            == "DJF_hiver"
        ],
        "MAM_printemps": dataframe.loc[
            dataframe["SAISON"]
            == "MAM_printemps"
        ],
        "JJA_ete": dataframe.loc[
            dataframe["SAISON"]
            == "JJA_ete"
        ],
        "SON_automne": dataframe.loc[
            dataframe["SAISON"]
            == "SON_automne"
        ],
        "extremes_froids_10pct":
            dataframe.loc[
                dataframe[
                    "TEMPERATURE_REELLE"
                ]
                <= cold_threshold
            ],
        "extremes_chauds_10pct":
            dataframe.loc[
                dataframe[
                    "TEMPERATURE_REELLE"
                ]
                >= hot_threshold
            ],
    }

    model_columns = {
        "Persistance":
            "TEMPERATURE_PERSISTANCE",
        "TCN":
            "TEMPERATURE_TCN",
        "ConvLSTM":
            "TEMPERATURE_CONVLSTM",
    }

    result_rows = []

    for group_name, group_dataframe in (
        groups.items()
    ):
        y_true = group_dataframe[
            "TEMPERATURE_REELLE"
        ].to_numpy(
            dtype=np.float64
        )

        for model_name, prediction_column in (
            model_columns.items()
        ):
            y_pred = group_dataframe[
                prediction_column
            ].to_numpy(
                dtype=np.float64
            )

            metrics = calculate_metrics(
                y_true,
                y_pred,
            )

            result_rows.append(
                {
                    "group": group_name,
                    "model": model_name,
                    "number_of_observations":
                        len(group_dataframe),
                    **metrics,
                }
            )

    results_dataframe = pd.DataFrame(
        result_rows
    )

    results_path = (
        METRICS_DIR
        / "spatial_models_by_season_and_extremes.csv"
    )

    results_dataframe.to_csv(
        results_path,
        index=False,
    )

    y_true = dataframe[
        "TEMPERATURE_REELLE"
    ].to_numpy(
        dtype=np.float64
    )

    tcn_predictions = dataframe[
        "TEMPERATURE_TCN"
    ].to_numpy(
        dtype=np.float64
    )

    convlstm_predictions = dataframe[
        "TEMPERATURE_CONVLSTM"
    ].to_numpy(
        dtype=np.float64
    )

    bootstrap_results = (
        block_bootstrap_comparison(
            y_true=y_true,
            tcn_predictions=tcn_predictions,
            convlstm_predictions=(
                convlstm_predictions
            ),
        )
    )

    bootstrap_results[
        "cold_threshold_C"
    ] = cold_threshold

    bootstrap_results[
        "hot_threshold_C"
    ] = hot_threshold

    bootstrap_path = (
        METRICS_DIR
        / "spatial_models_block_bootstrap.json"
    )

    with bootstrap_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            bootstrap_results,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("=" * 80)
    print("ANALYSE PAR SAISON ET PAR EXTRÊME")
    print("=" * 80)

    print(
        results_dataframe.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print("\n" + "=" * 80)
    print("BOOTSTRAP TEMPOREL PAR BLOCS")
    print("=" * 80)

    print(
        json.dumps(
            bootstrap_results,
            ensure_ascii=False,
            indent=4,
        )
    )

    print("\nFichiers créés :")
    print(f"- Groupes   : {results_path}")
    print(f"- Bootstrap : {bootstrap_path}")


if __name__ == "__main__":
    main()
