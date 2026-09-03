from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from src.utils.config import (
    METRICS_DIR,
    MODEL_DIR,
    PERSONAL_PROCESSED_DATA_DIR,
    create_output_directories,
)

from src.models.deep_learning.core.convlstm import ConvLSTMRegressor


PATCH_SIZE = 5
GRID_STEP = 80
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

BATCH_SIZE = 2048

DATA_PATH = (
    PERSONAL_PROCESSED_DATA_DIR
    / "safran_many_centers_1x1_9892centers_temperature_family.npz"
)

CHECKPOINT_PATH = (
    MODEL_DIR
    / "convlstm_all_points_masked_5x5_best.pt"
)


def calculate_metrics_from_sums(
    n: int,
    sum_abs_error: float,
    sum_squared_error: float,
    sum_error: float,
    sum_y: float,
    sum_y_squared: float,
) -> dict[str, float]:
    mae = sum_abs_error / n
    mse = sum_squared_error / n
    rmse = float(np.sqrt(mse))
    bias = sum_error / n

    denominator = sum_y_squared - (sum_y ** 2) / n

    if denominator <= 0:
        r2 = float("nan")
    else:
        r2 = 1.0 - sum_squared_error / denominator

    return {
        "MAE_C": float(mae),
        "RMSE_C": float(rmse),
        "BIAS_C": float(bias),
        "R2": float(r2),
    }


def build_neighbor_indices(
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    point_to_index = {
        (int(x), int(y)): index
        for index, (x, y) in enumerate(centers)
    }

    radius = PATCH_SIZE // 2

    neighbor_indices = np.full(
        (len(centers), PATCH_SIZE * PATCH_SIZE),
        fill_value=-1,
        dtype=np.int64,
    )

    neighbor_mask = np.zeros(
        (len(centers), PATCH_SIZE * PATCH_SIZE),
        dtype=np.float32,
    )

    for center_index, (center_x, center_y) in enumerate(centers):
        position = 0

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                neighbor = (
                    int(center_x) + dx * GRID_STEP,
                    int(center_y) + dy * GRID_STEP,
                )

                if neighbor in point_to_index:
                    neighbor_indices[center_index, position] = point_to_index[neighbor]
                    neighbor_mask[center_index, position] = 1.0

                position += 1

    return neighbor_indices, neighbor_mask


def create_batch_for_center(
    features: np.ndarray,
    raw_data: np.ndarray,
    center_index: int,
    start_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_mask: np.ndarray,
    target_mean: float,
    target_std: float,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    batch_size = len(start_indices)

    sequence_offsets = np.arange(SEQUENCE_LENGTH)
    day_matrix = start_indices[:, None] + sequence_offsets[None, :]

    neighbors = neighbor_indices[center_index]
    mask = neighbor_mask[center_index]

    safe_neighbors = np.where(neighbors < 0, 0, neighbors)

    gathered = features[
        day_matrix[:, :, None],
        safe_neighbors[None, None, :],
        :,
    ]

    gathered = gathered * mask[None, None, :, None]

    gathered = gathered.reshape(
        batch_size,
        SEQUENCE_LENGTH,
        PATCH_SIZE,
        PATCH_SIZE,
        features.shape[2],
    )

    gathered = np.transpose(
        gathered,
        (0, 1, 4, 2, 3),
    )

    mask_channel = mask.reshape(
        1,
        1,
        1,
        PATCH_SIZE,
        PATCH_SIZE,
    )

    mask_channel = np.repeat(
        mask_channel,
        batch_size,
        axis=0,
    )

    mask_channel = np.repeat(
        mask_channel,
        SEQUENCE_LENGTH,
        axis=1,
    )

    batch_inputs = np.concatenate(
        [gathered, mask_channel],
        axis=2,
    ).astype(np.float32)

    target_days = (
        start_indices
        + SEQUENCE_LENGTH
        + FORECAST_HORIZON
        - 1
    )

    persistence_days = (
        start_indices
        + SEQUENCE_LENGTH
        - 1
    )

    y_true = raw_data[
        target_days,
        center_index,
        0,
    ]

    y_persistence = raw_data[
        persistence_days,
        center_index,
        0,
    ]

    return (
        torch.from_numpy(batch_inputs).to(device),
        y_true.astype(np.float32),
        y_persistence.astype(np.float32),
    )


def main() -> None:
    create_output_directories()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError("CUDA est requis.")

    print("=" * 80)
    print("ÉVALUATION POINT PAR POINT — CONVLSTM 5×5 MASQUÉ")
    print("=" * 80)

    archive = np.load(DATA_PATH)

    raw_data = archive["data"].astype(np.float32, copy=False)
    raw_data = raw_data[:, :, :, 0, 0]

    dates = archive["dates"]
    centers = archive["centers"]
    channels = [str(x) for x in archive["channels"].tolist()]

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    print(f"Données : {raw_data.shape}")
    print(f"Centres : {centers.shape}")
    print(f"Canaux  : {channels}")
    print(f"Checkpoint époque : {checkpoint['epoch']}")

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

    train_end = int(raw_data.shape[0] * TRAIN_RATIO)
    validation_end = int(raw_data.shape[0] * (TRAIN_RATIO + VALIDATION_RATIO))

    features = (
        (raw_data - means[None, None, :])
        / stds[None, None, :]
    ).astype(np.float32)

    neighbor_indices, neighbor_mask = build_neighbor_indices(centers)

    model = ConvLSTMRegressor(
        input_channels=len(channels) + 1,
        hidden_channels=tuple(checkpoint["hidden_channels"]),
        kernel_size=int(checkpoint["kernel_size"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    first_start = validation_end
    last_start = (
        raw_data.shape[0]
        - SEQUENCE_LENGTH
        - FORECAST_HORIZON
    )

    start_indices_all = np.arange(
        first_start,
        last_start + 1,
        dtype=np.int64,
    )

    print(f"Séquences test par point : {len(start_indices_all):,}")
    print(f"Total évalué : {len(start_indices_all) * len(centers):,}")

    rows = []

    global_model = {
        "n": 0,
        "abs": 0.0,
        "sq": 0.0,
        "err": 0.0,
        "sum_y": 0.0,
        "sum_y2": 0.0,
    }

    global_persistence = {
        "n": 0,
        "abs": 0.0,
        "sq": 0.0,
        "err": 0.0,
        "sum_y": 0.0,
        "sum_y2": 0.0,
    }

    start_time = time.perf_counter()

    for center_index in range(len(centers)):
        model_abs = 0.0
        model_sq = 0.0
        model_err = 0.0

        persistence_abs = 0.0
        persistence_sq = 0.0
        persistence_err = 0.0

        sum_y = 0.0
        sum_y2 = 0.0
        n = 0

        for batch_start in range(0, len(start_indices_all), BATCH_SIZE):
            batch_starts = start_indices_all[
                batch_start:batch_start + BATCH_SIZE
            ]

            inputs, y_true, y_persistence = create_batch_for_center(
                features=features,
                raw_data=raw_data,
                center_index=center_index,
                start_indices=batch_starts,
                neighbor_indices=neighbor_indices,
                neighbor_mask=neighbor_mask,
                target_mean=target_mean,
                target_std=target_std,
                device=device,
            )

            with torch.no_grad():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    normalized_predictions = model(inputs)

            y_pred = (
                normalized_predictions.cpu().numpy()
                * target_std
                + target_mean
            )

            model_errors = y_pred - y_true
            persistence_errors = y_persistence - y_true

            model_abs += float(np.abs(model_errors).sum())
            model_sq += float((model_errors ** 2).sum())
            model_err += float(model_errors.sum())

            persistence_abs += float(np.abs(persistence_errors).sum())
            persistence_sq += float((persistence_errors ** 2).sum())
            persistence_err += float(persistence_errors.sum())

            sum_y += float(y_true.sum())
            sum_y2 += float((y_true ** 2).sum())
            n += len(y_true)

        model_metrics = calculate_metrics_from_sums(
            n=n,
            sum_abs_error=model_abs,
            sum_squared_error=model_sq,
            sum_error=model_err,
            sum_y=sum_y,
            sum_y_squared=sum_y2,
        )

        persistence_metrics = calculate_metrics_from_sums(
            n=n,
            sum_abs_error=persistence_abs,
            sum_squared_error=persistence_sq,
            sum_error=persistence_err,
            sum_y=sum_y,
            sum_y_squared=sum_y2,
        )

        mae_gain = (
            100.0
            * (
                persistence_metrics["MAE_C"]
                - model_metrics["MAE_C"]
            )
            / persistence_metrics["MAE_C"]
        )

        rmse_gain = (
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
                "existing_cells_5x5": int(neighbor_mask[center_index].sum()),
                "complete_5x5": bool(neighbor_mask[center_index].sum() == 25),
                "persistence_MAE_C": persistence_metrics["MAE_C"],
                "persistence_RMSE_C": persistence_metrics["RMSE_C"],
                "convlstm_MAE_C": model_metrics["MAE_C"],
                "convlstm_RMSE_C": model_metrics["RMSE_C"],
                "convlstm_BIAS_C": model_metrics["BIAS_C"],
                "convlstm_R2": model_metrics["R2"],
                "MAE_improvement_percent": mae_gain,
                "RMSE_improvement_percent": rmse_gain,
            }
        )

        for container, abs_value, sq_value, err_value in [
            (global_model, model_abs, model_sq, model_err),
            (global_persistence, persistence_abs, persistence_sq, persistence_err),
        ]:
            container["n"] += n
            container["abs"] += abs_value
            container["sq"] += sq_value
            container["err"] += err_value
            container["sum_y"] += sum_y
            container["sum_y2"] += sum_y2

        if (center_index + 1) % 250 == 0:
            elapsed = time.perf_counter() - start_time
            print(
                f"Points évalués : {center_index + 1:,}/{len(centers):,} "
                f"| durée={elapsed:.1f}s"
            )

    dataframe = pd.DataFrame(rows)

    output_csv = (
        METRICS_DIR
        / "convlstm_all_points_masked_5x5_by_point_metrics.csv"
    )

    dataframe.to_csv(output_csv, index=False)

    global_model_metrics = calculate_metrics_from_sums(
        n=global_model["n"],
        sum_abs_error=global_model["abs"],
        sum_squared_error=global_model["sq"],
        sum_error=global_model["err"],
        sum_y=global_model["sum_y"],
        sum_y_squared=global_model["sum_y2"],
    )

    global_persistence_metrics = calculate_metrics_from_sums(
        n=global_persistence["n"],
        sum_abs_error=global_persistence["abs"],
        sum_squared_error=global_persistence["sq"],
        sum_error=global_persistence["err"],
        sum_y=global_persistence["sum_y"],
        sum_y_squared=global_persistence["sum_y2"],
    )

    summary = {
        "number_of_points": int(len(dataframe)),
        "test_sequences_per_point": int(len(start_indices_all)),
        "total_test_sequences": int(len(start_indices_all) * len(centers)),
        "complete_5x5_points": int(dataframe["complete_5x5"].sum()),
        "incomplete_5x5_points": int((~dataframe["complete_5x5"]).sum()),
        "global_persistence_metrics": global_persistence_metrics,
        "global_convlstm_metrics": global_model_metrics,
        "global_MAE_improvement_percent": float(
            100.0
            * (
                global_persistence_metrics["MAE_C"]
                - global_model_metrics["MAE_C"]
            )
            / global_persistence_metrics["MAE_C"]
        ),
        "global_RMSE_improvement_percent": float(
            100.0
            * (
                global_persistence_metrics["RMSE_C"]
                - global_model_metrics["RMSE_C"]
            )
            / global_persistence_metrics["RMSE_C"]
        ),
        "MAE_mean_by_point": float(dataframe["convlstm_MAE_C"].mean()),
        "RMSE_mean_by_point": float(dataframe["convlstm_RMSE_C"].mean()),
        "R2_mean_by_point": float(dataframe["convlstm_R2"].mean()),
        "points_with_positive_MAE_improvement": int(
            (dataframe["MAE_improvement_percent"] > 0).sum()
        ),
        "points_with_negative_MAE_improvement": int(
            (dataframe["MAE_improvement_percent"] <= 0).sum()
        ),
    }

    output_summary = (
        METRICS_DIR
        / "convlstm_all_points_masked_5x5_by_point_summary.json"
    )

    with output_summary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=4)

    print("\n" + "=" * 80)
    print("RÉSUMÉ ÉVALUATION 5×5 MASQUÉ")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=4))

    print("\nFichiers créés :")
    print(f"- Détail : {output_csv}")
    print(f"- Résumé : {output_summary}")


if __name__ == "__main__":
    main()
