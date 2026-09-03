from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from data_loader import (  # noqa: E402
    SequenceConfig,
    chronological_split,
    create_sequences,
    load_first_point_temperature,
    standardize_splits,
    summarize_time_series,
)


def main() -> None:
    dataframe = load_first_point_temperature()

    summary = summarize_time_series(
        dataframe
    )

    print("=" * 80)
    print("SÉRIE DE TEMPÉRATURE DU PREMIER POINT")
    print("=" * 80)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nPremières lignes :")
    print(dataframe.head().to_string(index=False))

    print("\nDernières lignes :")
    print(dataframe.tail().to_string(index=False))

    train_df, validation_df, test_df = (
        chronological_split(dataframe)
    )

    (
        train_scaled,
        validation_scaled,
        test_scaled,
        train_mean,
        train_std,
    ) = standardize_splits(
        train_df,
        validation_df,
        test_df,
    )

    sequence_config = SequenceConfig(
        input_length=30,
        forecast_horizon=1,
        target_column="T",
    )

    x_train, y_train = create_sequences(
        train_scaled,
        sequence_config,
    )

    x_validation, y_validation = create_sequences(
        validation_scaled,
        sequence_config,
    )

    x_test, y_test = create_sequences(
        test_scaled,
        sequence_config,
    )

    print("\n" + "=" * 80)
    print("DÉCOUPAGE CHRONOLOGIQUE")
    print("=" * 80)

    print(f"Train      : {len(train_df):,} lignes")
    print(
        f"Validation : {len(validation_df):,} lignes"
    )
    print(f"Test       : {len(test_df):,} lignes")

    print("\n" + "=" * 80)
    print("NORMALISATION")
    print("=" * 80)

    print(f"Moyenne du train : {train_mean:.4f}")
    print(f"Écart-type train : {train_std:.4f}")

    print("\n" + "=" * 80)
    print("SÉQUENCES")
    print("=" * 80)

    print(f"X train      : {x_train.shape}")
    print(f"y train      : {y_train.shape}")

    print(
        f"X validation : {x_validation.shape}"
    )
    print(
        f"y validation : {y_validation.shape}"
    )

    print(f"X test       : {x_test.shape}")
    print(f"y test       : {y_test.shape}")


if __name__ == "__main__":
    main()