from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from config import (
    LOG_DIR,
    METRICS_DIR,
    MODEL_DIR,
    PERSONAL_PROCESSED_DATA_DIR,
    create_output_directories,
)

from models.baseline import calculate_regression_metrics
from models.convlstm import ConvLSTMRegressor


RANDOM_SEED = 42

PATCH_SIZE = 5
GRID_STEP = 80
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

BATCH_SIZE = 512
MAX_EPOCHS = 30
PATIENCE = 8

TRAIN_BATCHES_PER_EPOCH = 500
VALIDATION_BATCHES = 120
TEST_BATCHES = 500

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN_CHANNELS = (32, 64)
KERNEL_SIZE = 3
DROPOUT = 0.15

DATA_PATH = (
    PERSONAL_PROCESSED_DATA_DIR
    / "safran_many_centers_1x1_9892centers_temperature_family.npz"
)

GRID_POINTS_PATH = (
    PERSONAL_PROCESSED_DATA_DIR
    / "safran_grid_points.csv"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    archive = np.load(DATA_PATH)

    data = archive["data"].astype(np.float32, copy=False)
    dates = archive["dates"]
    centers = archive["centers"]
    channels = [str(x) for x in archive["channels"].tolist()]

    # data : jours, points, canaux, 1, 1
    data = data[:, :, :, 0, 0]

    return data, dates, centers, channels


def build_neighbor_indices(centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    mask = np.zeros(
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
                    mask[center_index, position] = 1.0

                position += 1

    return neighbor_indices, mask


def split_indices(number_of_days: int):
    train_end = int(number_of_days * TRAIN_RATIO)
    validation_end = int(number_of_days * (TRAIN_RATIO + VALIDATION_RATIO))

    return train_end, validation_end


def standardize_data(data: np.ndarray, train_end: int):
    train_data = data[:train_end]

    means = train_data.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    stds = train_data.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    stds = np.where(stds < 1e-8, 1.0, stds).astype(np.float32)

    standardized = ((data - means[None, None, :]) / stds[None, None, :]).astype(np.float32)

    target_mean = float(train_data[:, :, 0].mean(dtype=np.float64))
    target_std = float(train_data[:, :, 0].std(dtype=np.float64))

    targets = ((data[:, :, 0] - target_mean) / target_std).astype(np.float32)

    return standardized, targets, means, stds, target_mean, target_std


def sample_batch(
    features: np.ndarray,
    targets: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_mask: np.ndarray,
    start_day: int,
    end_day: int,
    device: torch.device,
):
    max_start = end_day - SEQUENCE_LENGTH - FORECAST_HORIZON

    if max_start <= start_day:
        raise ValueError("Intervalle temporel trop petit pour créer un batch.")

    number_of_points = features.shape[1]

    time_indices = np.random.randint(
        start_day,
        max_start + 1,
        size=BATCH_SIZE,
    )

    center_indices = np.random.randint(
        0,
        number_of_points,
        size=BATCH_SIZE,
    )

    sequence_offsets = np.arange(SEQUENCE_LENGTH)

    day_matrix = time_indices[:, None] + sequence_offsets[None, :]

    neighbors = neighbor_indices[center_indices]
    masks = neighbor_mask[center_indices]

    safe_neighbors = np.where(neighbors < 0, 0, neighbors)

    gathered = features[
        day_matrix[:, :, None],
        safe_neighbors[:, None, :],
        :,
    ]

    # B, seq, 25, channels
    gathered = gathered * masks[:, None, :, None]

    gathered = gathered.reshape(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        PATCH_SIZE,
        PATCH_SIZE,
        features.shape[2],
    )

    gathered = np.transpose(
        gathered,
        (0, 1, 4, 2, 3),
    )

    mask_channel = masks.reshape(
        BATCH_SIZE,
        1,
        1,
        PATCH_SIZE,
        PATCH_SIZE,
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

    target_days = time_indices + SEQUENCE_LENGTH + FORECAST_HORIZON - 1

    batch_targets = targets[
        target_days,
        center_indices,
    ]

    return (
        torch.from_numpy(batch_inputs).to(device),
        torch.from_numpy(batch_targets.astype(np.float32)).to(device),
    )


def run_epoch(
    model,
    features,
    targets,
    neighbor_indices,
    neighbor_mask,
    start_day,
    end_day,
    optimizer,
    criterion,
    scaler,
    device,
    train: bool,
    number_of_batches: int,
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    for _ in range(number_of_batches):
        inputs, batch_targets = sample_batch(
            features=features,
            targets=targets,
            neighbor_indices=neighbor_indices,
            neighbor_mask=neighbor_mask,
            start_day=start_day,
            end_day=end_day,
            device=device,
        )

        if train:
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predictions = model(inputs)
                loss = criterion(predictions, batch_targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

        else:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    predictions = model(inputs)
                    loss = criterion(predictions, batch_targets)

        total_loss += float(loss.item())

    return total_loss / number_of_batches


@torch.no_grad()
def sampled_test_metrics(
    model,
    features,
    raw_data,
    targets,
    neighbor_indices,
    neighbor_mask,
    validation_end,
    number_of_days,
    target_mean,
    target_std,
    device,
):
    model.eval()

    all_predictions = []
    all_targets = []
    all_persistence = []

    for _ in range(TEST_BATCHES):
        inputs, batch_targets = sample_batch(
            features=features,
            targets=targets,
            neighbor_indices=neighbor_indices,
            neighbor_mask=neighbor_mask,
            start_day=validation_end,
            end_day=number_of_days,
            device=device,
        )

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            predictions = model(inputs)

        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(batch_targets.cpu().numpy())

    normalized_predictions = np.concatenate(all_predictions)
    normalized_targets = np.concatenate(all_targets)

    predicted_celsius = normalized_predictions * target_std + target_mean
    true_celsius = normalized_targets * target_std + target_mean

    # Persistance calculée sur le même nombre d'exemples tirés aléatoirement
    for _ in range(TEST_BATCHES):
        max_start = number_of_days - SEQUENCE_LENGTH - FORECAST_HORIZON

        time_indices = np.random.randint(
            validation_end,
            max_start + 1,
            size=BATCH_SIZE,
        )

        center_indices = np.random.randint(
            0,
            raw_data.shape[1],
            size=BATCH_SIZE,
        )

        target_days = time_indices + SEQUENCE_LENGTH + FORECAST_HORIZON - 1
        persistence_days = time_indices + SEQUENCE_LENGTH - 1

        y_true = raw_data[target_days, center_indices, 0]
        y_persistence = raw_data[persistence_days, center_indices, 0]

        all_persistence.append(
            np.stack([y_true, y_persistence], axis=1)
        )

    persistence_array = np.concatenate(all_persistence)

    persistence_metrics = calculate_regression_metrics(
        y_true=persistence_array[:, 0],
        y_pred=persistence_array[:, 1],
    )

    model_metrics = calculate_regression_metrics(
        y_true=true_celsius,
        y_pred=predicted_celsius,
    )

    return model_metrics, persistence_metrics


def main() -> None:
    create_output_directories()
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        raise RuntimeError("CUDA est requis.")

    print("=" * 80)
    print("CONVLSTM 5×5 MASQUÉ — TOUS LES 9 892 POINTS SAFRAN")
    print("=" * 80)
    print(f"GPU : {torch.cuda.get_device_name(0)}")

    raw_data, dates, centers, channels = load_data()

    print(f"Données : {raw_data.shape}")
    print(f"Centres : {centers.shape}")
    print(f"Canaux initiaux : {channels}")

    neighbor_indices, neighbor_mask = build_neighbor_indices(centers)

    print(f"Patchs complets : {(neighbor_mask.sum(axis=1) == 25).sum():,}")
    print(f"Patchs incomplets : {(neighbor_mask.sum(axis=1) < 25).sum():,}")
    print(f"Cellules moyennes disponibles : {neighbor_mask.sum(axis=1).mean():.2f}/25")

    number_of_days = raw_data.shape[0]
    train_end, validation_end = split_indices(number_of_days)

    features, targets, means, stds, target_mean, target_std = standardize_data(
        raw_data,
        train_end,
    )

    model = ConvLSTMRegressor(
        input_channels=len(channels) + 1,
        hidden_channels=HIDDEN_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
    ).to(device)

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Paramètres entraînables : {number_of_parameters:,}")

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    scaler = torch.amp.GradScaler("cuda")

    best_validation_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0

    history = []

    checkpoint_path = MODEL_DIR / "convlstm_all_points_masked_5x5_best.pt"

    start_time = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(
            model=model,
            features=features,
            targets=targets,
            neighbor_indices=neighbor_indices,
            neighbor_mask=neighbor_mask,
            start_day=0,
            end_day=train_end,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            train=True,
            number_of_batches=TRAIN_BATCHES_PER_EPOCH,
        )

        validation_loss = run_epoch(
            model=model,
            features=features,
            targets=targets,
            neighbor_indices=neighbor_indices,
            neighbor_mask=neighbor_mask,
            start_day=train_end,
            end_day=validation_end,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            train=False,
            number_of_batches=VALIDATION_BATCHES,
        )

        scheduler.step(validation_loss)

        lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": lr,
            }
        )

        print(
            f"Époque {epoch:03d}/{MAX_EPOCHS} | "
            f"train={train_loss:.6f} | "
            f"val={validation_loss:.6f} | "
            f"lr={lr:.2e}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "channels": channels + ["MASK"],
                    "feature_means": means.tolist(),
                    "feature_stds": stds.tolist(),
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "patch_size": PATCH_SIZE,
                    "sequence_length": SEQUENCE_LENGTH,
                    "forecast_horizon": FORECAST_HORIZON,
                    "hidden_channels": HIDDEN_CHANNELS,
                    "kernel_size": KERNEL_SIZE,
                    "dropout": DROPOUT,
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print("\nArrêt anticipé.")
            break

    duration = time.perf_counter() - start_time

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model_metrics, persistence_metrics = sampled_test_metrics(
        model=model,
        features=features,
        raw_data=raw_data,
        targets=targets,
        neighbor_indices=neighbor_indices,
        neighbor_mask=neighbor_mask,
        validation_end=validation_end,
        number_of_days=number_of_days,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
    )

    results = {
        "model": "ConvLSTM_all_points_masked_5x5",
        "number_of_points": int(raw_data.shape[1]),
        "patch_size": PATCH_SIZE,
        "channels": channels + ["MASK"],
        "number_of_parameters": int(number_of_parameters),
        "train_batches_per_epoch": TRAIN_BATCHES_PER_EPOCH,
        "validation_batches": VALIDATION_BATCHES,
        "test_batches": TEST_BATCHES,
        "test_samples": TEST_BATCHES * BATCH_SIZE,
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_loss": float(checkpoint["validation_loss"]),
        "training_duration_seconds": float(duration),
        "persistence_metrics": persistence_metrics,
        "convlstm_metrics": model_metrics,
        "MAE_improvement_vs_persistence_percent": float(
            100.0
            * (
                persistence_metrics["MAE_C"]
                - model_metrics["MAE_C"]
            )
            / persistence_metrics["MAE_C"]
        ),
        "RMSE_improvement_vs_persistence_percent": float(
            100.0
            * (
                persistence_metrics["RMSE_C"]
                - model_metrics["RMSE_C"]
            )
            / persistence_metrics["RMSE_C"]
        ),
    }

    metrics_path = METRICS_DIR / "convlstm_all_points_masked_5x5_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

    history_path = LOG_DIR / "convlstm_all_points_masked_5x5_history.csv"

    pd.DataFrame(history).to_csv(history_path, index=False)

    print("\n" + "=" * 80)
    print("RÉSULTATS CONVLSTM 5×5 MASQUÉ")
    print("=" * 80)

    print(f"Meilleure époque : {checkpoint['epoch']}")
    print(f"Durée             : {duration:.1f}s")

    print("\nPersistance :")
    print(f"MAE  : {persistence_metrics['MAE_C']:.4f} °C")
    print(f"RMSE : {persistence_metrics['RMSE_C']:.4f} °C")

    print("\nConvLSTM 5×5 masqué :")
    print(f"MAE  : {model_metrics['MAE_C']:.4f} °C")
    print(f"RMSE : {model_metrics['RMSE_C']:.4f} °C")
    print(f"Biais: {model_metrics['BIAS_C']:.4f} °C")
    print(f"R²   : {model_metrics['R2']:.4f}")

    print("\nAmélioration :")
    print(f"MAE  : {results['MAE_improvement_vs_persistence_percent']:.2f} %")
    print(f"RMSE : {results['RMSE_improvement_vs_persistence_percent']:.2f} %")

    print("\nFichiers créés :")
    print(f"- Modèle    : {checkpoint_path}")
    print(f"- Métriques : {metrics_path}")
    print(f"- Historique: {history_path}")


if __name__ == "__main__":
    main()
