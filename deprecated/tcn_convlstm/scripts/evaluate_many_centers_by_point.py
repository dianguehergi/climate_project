from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
SCRIPTS_DIR = PROJECT_DIR / "scripts"

for directory in (SRC_DIR, SCRIPTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


from config import (  # noqa: E402
    METRICS_DIR,
    MODEL_DIR,
    PERSONAL_PROCESSED_DATA_DIR,
    create_output_directories,
)

from models.convlstm import ConvLSTMRegressor  # noqa: E402

from train_convlstm_many_centers import (  # noqa: E402
    FORECAST_HORIZON,
    SEQUENCE_LENGTH,
    ManyCentersSpatialDataset,
    chronological_split,
    create_targets,
    standardize,
)


BATCH_SIZE = 1024


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    errors = y_pred - y_true

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    denominator = float(
        np.sum((y_true - np.mean(y_true)) ** 2)
    )

    r2 = (
        float(1.0 - np.sum(errors ** 2) / denominator)
        if denominator > 0
        else float("nan")
    )

    return {
        "MAE_C": mae,
        "RMSE_C": rmse,
        "BIAS_C": bias,
        "R2": r2,
    }


@torch.no_grad()
def predict_all(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    predictions = []
    targets = []

    for batch_index, (inputs, batch_targets) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            batch_predictions = model(inputs)

        predictions.append(
            batch_predictions.cpu().numpy()
        )

        targets.append(
            batch_targets.numpy()
        )

        if batch_index % 1000 == 0:
            print(f"Batchs évalués : {batch_index:,}")

    return (
        np.concatenate(predictions),
        np.concatenate(targets),
    )


def main() -> None:
    create_output_directories()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError("CUDA est requis pour cette évaluation.")

    centers_count = 9892
    patch_size = 1

    data_path = (
        PERSONAL_PROCESSED_DATA_DIR
        / "safran_many_centers_1x1_9892centers_temperature_family.npz"
    )

    checkpoint_path = (
        MODEL_DIR
        / "convlstm_many_centers_9892_best.pt"
    )

    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    print("=" * 80)
    print("ÉVALUATION POINT PAR POINT — 9 892 POINTS SAFRAN")
    print("=" * 80)

    archive = np.load(data_path)

    data = archive["data"].astype(np.float32, copy=False)
    dates = archive["dates"]
    centers = archive["centers"]
    channels = [str(channel) for channel in archive["channels"].tolist()]

    print(f"Tenseur : {data.shape}")
    print(f"Centres : {centers.shape}")
    print(f"Canaux  : {channels}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    temperature_channel_index = channels.index("T")
    center_row = patch_size // 2
    center_column = patch_size // 2

    (
        train_raw,
        validation_raw,
        test_raw,
        _,
        _,
        _,
    ) = chronological_split(data, dates)

    means = np.array(
        checkpoint["feature_means"],
        dtype=np.float32,
    )

    stds = np.array(
        checkpoint["feature_stds"],
        dtype=np.float32,
    )

    target_mean = float(checkpoint["target_mean"])
    target_std = float(checkpoint["target_std"])

    test_features = standardize(
        test_raw,
        means,
        stds,
    )

    test_targets = create_targets(
        test_raw,
        temperature_channel_index,
        center_row,
        center_column,
        target_mean,
        target_std,
    )

    test_dataset = ManyCentersSpatialDataset(
        test_features,
        test_targets,
        SEQUENCE_LENGTH,
        FORECAST_HORIZON,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    model = ConvLSTMRegressor(
        input_channels=len(channels),
        hidden_channels=tuple(checkpoint["hidden_channels"]),
        kernel_size=int(checkpoint["kernel_size"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    normalized_predictions, normalized_targets = predict_all(
        model,
        test_loader,
        device,
    )

    true_celsius = (
        normalized_targets * target_std
        + target_mean
    )

    predicted_celsius = (
        normalized_predictions * target_std
        + target_mean
    )

    sequences_per_center = (
        test_dataset.number_of_sequences_per_center
    )

    true_by_center = true_celsius.reshape(
        centers_count,
        sequences_per_center,
    )

    pred_by_center = predicted_celsius.reshape(
        centers_count,
        sequences_per_center,
    )

    persistence_by_center = []

    for center_index in range(test_raw.shape[1]):
        center_temperature = test_raw[
            :,
            center_index,
            temperature_channel_index,
            center_row,
            center_column,
        ]

        persistence_by_center.append(
            center_temperature[
                SEQUENCE_LENGTH - 1:
                SEQUENCE_LENGTH - 1 + sequences_per_center
            ]
        )

    persistence_by_center = np.stack(
        persistence_by_center,
        axis=0,
    )

    rows = []

    for center_index in range(centers_count):
        y_true = true_by_center[center_index]
        y_pred = pred_by_center[center_index]
        y_persistence = persistence_by_center[center_index]

        model_metrics = calculate_metrics(
            y_true,
            y_pred,
        )

        persistence_metrics = calculate_metrics(
            y_true,
            y_persistence,
        )

        mae_improvement = (
            100.0
            * (
                persistence_metrics["MAE_C"]
                - model_metrics["MAE_C"]
            )
            / persistence_metrics["MAE_C"]
        )

        rmse_improvement = (
            100.0
            * (
                persistence_metrics["RMSE_C"]
                - model_metrics["RMSE_C"]
            )
            / persistence_metrics["RMSE_C"]
        )

        rows.append(
            {
                "center_index": center_index,
                "LAMBX": int(centers[center_index, 0]),
                "LAMBY": int(centers[center_index, 1]),
                "persistence_MAE_C": persistence_metrics["MAE_C"],
                "persistence_RMSE_C": persistence_metrics["RMSE_C"],
                "convlstm_MAE_C": model_metrics["MAE_C"],
                "convlstm_RMSE_C": model_metrics["RMSE_C"],
                "convlstm_BIAS_C": model_metrics["BIAS_C"],
                "convlstm_R2": model_metrics["R2"],
                "MAE_improvement_percent": mae_improvement,
                "RMSE_improvement_percent": rmse_improvement,
            }
        )

    dataframe = pd.DataFrame(rows)

    output_path = (
        METRICS_DIR
        / "convlstm_many_centers_9892_by_point_metrics.csv"
    )

    dataframe.to_csv(output_path, index=False)

    summary = {
        "number_of_points": int(len(dataframe)),
        "MAE_mean": float(dataframe["convlstm_MAE_C"].mean()),
        "MAE_std": float(dataframe["convlstm_MAE_C"].std()),
        "MAE_min": float(dataframe["convlstm_MAE_C"].min()),
        "MAE_max": float(dataframe["convlstm_MAE_C"].max()),
        "RMSE_mean": float(dataframe["convlstm_RMSE_C"].mean()),
        "RMSE_std": float(dataframe["convlstm_RMSE_C"].std()),
        "R2_mean": float(dataframe["convlstm_R2"].mean()),
        "MAE_improvement_mean_percent": float(
            dataframe["MAE_improvement_percent"].mean()
        ),
        "RMSE_improvement_mean_percent": float(
            dataframe["RMSE_improvement_percent"].mean()
        ),
        "points_with_positive_MAE_improvement": int(
            (dataframe["MAE_improvement_percent"] > 0).sum()
        ),
        "points_with_negative_MAE_improvement": int(
            (dataframe["MAE_improvement_percent"] <= 0).sum()
        ),
    }

    summary_path = (
        METRICS_DIR
        / "convlstm_many_centers_9892_by_point_summary.json"
    )

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\n" + "=" * 80)
    print("RÉSUMÉ POINT PAR POINT")
    print("=" * 80)

    print(json.dumps(summary, ensure_ascii=False, indent=4))

    print("\nTop 10 meilleurs points :")
    print(
        dataframe.sort_values("MAE_improvement_percent", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\nTop 10 points les plus difficiles :")
    print(
        dataframe.sort_values("convlstm_MAE_C", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\nFichiers créés :")
    print(f"- Détail par point : {output_path}")
    print(f"- Résumé          : {summary_path}")


if __name__ == "__main__":
    main()
