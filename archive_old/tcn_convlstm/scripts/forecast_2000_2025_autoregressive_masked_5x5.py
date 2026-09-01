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


from config import (
    METRICS_DIR,
    MODEL_DIR,
    PERSONAL_PROCESSED_DATA_DIR,
    PREDICTION_DIR,
    create_output_directories,
)

from models.convlstm import ConvLSTMRegressor


PATCH_SIZE = 5
GRID_STEP = 80
SEQUENCE_LENGTH = 30
BATCH_SIZE = 2048

FORECAST_START = "2000-01-01"
FORECAST_END = "2025-12-31"

HISTORY_PATH = (
    PERSONAL_PROCESSED_DATA_DIR
    / "safran_many_centers_1x1_9892centers_temperature_family.npz"
)

CHECKPOINT_PATH = (
    MODEL_DIR
    / "convlstm_all_points_masked_5x5_best.pt"
)

FORECAST_ARRAY_PATH = (
    PREDICTION_DIR
    / "forecast_2000_2025_autoregressive_masked_5x5.npy"
)

FORECAST_DATES_PATH = (
    PREDICTION_DIR
    / "forecast_dates_2000_2025.npy"
)

FORECAST_DAILY_SUMMARY_PATH = (
    PREDICTION_DIR
    / "forecast_2000_2025_daily_summary.csv"
)

FORECAST_SAMPLE_PATH = (
    PREDICTION_DIR
    / "forecast_2000_2025_sample_points.csv"
)

FORECAST_METADATA_PATH = (
    METRICS_DIR
    / "forecast_2000_2025_autoregressive_metadata.json"
)


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


def compute_climatological_offsets(
    historical_data: np.ndarray,
    historical_dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """
    On calcule, pour chaque jour de l'année et chaque point :
    TINF_H - T
    TSUP_H - T

    Ces écarts servent à reconstruire TINF_H et TSUP_H pendant la prévision.
    """

    number_of_points = historical_data.shape[1]

    offset_tinf = np.zeros((367, number_of_points), dtype=np.float32)
    offset_tsup = np.zeros((367, number_of_points), dtype=np.float32)

    doys = historical_dates.dayofyear.to_numpy()

    t = historical_data[:, :, 0]
    tinf = historical_data[:, :, 1]
    tsup = historical_data[:, :, 2]

    for doy in range(1, 367):
        mask = doys == doy

        if mask.any():
            offset_tinf[doy] = (tinf[mask] - t[mask]).mean(axis=0)
            offset_tsup[doy] = (tsup[mask] - t[mask]).mean(axis=0)
        else:
            # Sécurité au cas où le jour 366 serait absent.
            offset_tinf[doy] = offset_tinf[doy - 1]
            offset_tsup[doy] = offset_tsup[doy - 1]

    return offset_tinf, offset_tsup


def make_inputs_from_window(
    window_raw: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    center_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_mask: np.ndarray,
) -> np.ndarray:
    """
    window_raw : 30 jours × points × 5 canaux
    """

    window_features = (
        (window_raw - means[None, None, :])
        / stds[None, None, :]
    ).astype(np.float32)

    neighbors = neighbor_indices[center_indices]
    masks = neighbor_mask[center_indices]

    safe_neighbors = np.where(neighbors < 0, 0, neighbors)

    gathered = window_features[
        np.arange(SEQUENCE_LENGTH)[None, :, None],
        safe_neighbors[:, None, :],
        :,
    ]

    gathered = gathered * masks[:, None, :, None]

    batch_size = len(center_indices)

    gathered = gathered.reshape(
        batch_size,
        SEQUENCE_LENGTH,
        PATCH_SIZE,
        PATCH_SIZE,
        window_features.shape[2],
    )

    gathered = np.transpose(
        gathered,
        (0, 1, 4, 2, 3),
    )

    mask_channel = masks.reshape(
        batch_size,
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

    inputs = np.concatenate(
        [gathered, mask_channel],
        axis=2,
    ).astype(np.float32)

    return inputs


def main() -> None:
    create_output_directories()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError("CUDA est requis.")

    print("=" * 80)
    print("PRÉDICTION SIMULÉE 2000–2025 — CONVLSTM 5×5 MASQUÉ")
    print("=" * 80)

    print(f"GPU : {torch.cuda.get_device_name(0)}")

    archive = np.load(HISTORY_PATH)

    historical_data = archive["data"].astype(np.float32, copy=False)
    historical_data = historical_data[:, :, :, 0, 0]

    centers = archive["centers"]
    channels = [str(x) for x in archive["channels"].tolist()]
    historical_dates = pd.to_datetime(archive["dates"].astype(str))

    print(f"Historique : {historical_data.shape}")
    print(f"Dates historiques : {historical_dates.min().date()} → {historical_dates.max().date()}")
    print(f"Points : {centers.shape[0]:,}")
    print(f"Canaux : {channels}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    means = np.array(checkpoint["feature_means"], dtype=np.float32)
    stds = np.array(checkpoint["feature_stds"], dtype=np.float32)
    target_mean = float(checkpoint["target_mean"])
    target_std = float(checkpoint["target_std"])

    model = ConvLSTMRegressor(
        input_channels=6,
        hidden_channels=tuple(checkpoint["hidden_channels"]),
        kernel_size=int(checkpoint["kernel_size"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint chargé : {CHECKPOINT_PATH}")
    print(f"Meilleure époque : {checkpoint['epoch']}")

    neighbor_indices, neighbor_mask = build_neighbor_indices(centers)

    offset_tinf, offset_tsup = compute_climatological_offsets(
        historical_data=historical_data,
        historical_dates=historical_dates,
    )

    forecast_dates = pd.date_range(
        FORECAST_START,
        FORECAST_END,
        freq="D",
    )

    number_of_days = len(forecast_dates)
    number_of_points = centers.shape[0]

    print(f"Période prédite : {FORECAST_START} → {FORECAST_END}")
    print(f"Nombre de jours : {number_of_days:,}")
    print(f"Nombre total de prédictions : {number_of_days * number_of_points:,}")

    forecasts = np.lib.format.open_memmap(
        FORECAST_ARRAY_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(number_of_days, number_of_points),
    )

    # Fenêtre initiale = 30 derniers jours réels de 1999.
    rolling_window = historical_data[-SEQUENCE_LENGTH:].copy()

    center_indices_all = np.arange(number_of_points, dtype=np.int64)

    start_time = time.perf_counter()

    for day_index, forecast_date in enumerate(forecast_dates):
        day_predictions = []

        for batch_start in range(0, number_of_points, BATCH_SIZE):
            batch_centers = center_indices_all[
                batch_start:batch_start + BATCH_SIZE
            ]

            batch_inputs = make_inputs_from_window(
                window_raw=rolling_window,
                means=means,
                stds=stds,
                center_indices=batch_centers,
                neighbor_indices=neighbor_indices,
                neighbor_mask=neighbor_mask,
            )

            inputs_tensor = torch.from_numpy(batch_inputs).to(device)

            with torch.no_grad():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    normalized_predictions = model(inputs_tensor)

            batch_predictions = (
                normalized_predictions.cpu().numpy()
                * target_std
                + target_mean
            ).astype(np.float32)

            day_predictions.append(batch_predictions)

        predicted_t = np.concatenate(day_predictions)

        forecasts[day_index] = predicted_t

        doy = int(forecast_date.dayofyear)
        angle = 2.0 * np.pi * doy / 366.0

        next_row = np.zeros_like(rolling_window[0])
        next_row[:, 0] = predicted_t
        next_row[:, 1] = predicted_t + offset_tinf[doy]
        next_row[:, 2] = predicted_t + offset_tsup[doy]
        next_row[:, 3] = np.sin(angle)
        next_row[:, 4] = np.cos(angle)

        rolling_window = np.concatenate(
            [
                rolling_window[1:],
                next_row[None, :, :],
            ],
            axis=0,
        )

        if (day_index + 1) % 250 == 0:
            elapsed = time.perf_counter() - start_time
            print(
                f"Jours prédits : {day_index + 1:,}/{number_of_days:,} "
                f"| date={forecast_date.date()} "
                f"| durée={elapsed:.1f}s"
            )

    forecasts.flush()

    np.save(
        FORECAST_DATES_PATH,
        forecast_dates.strftime("%Y-%m-%d").to_numpy(),
    )

    forecast_array = np.asarray(forecasts)

    daily_summary = pd.DataFrame(
        {
            "DATE": forecast_dates.strftime("%Y-%m-%d"),
            "PRED_MEAN_C": forecast_array.mean(axis=1),
            "PRED_STD_C": forecast_array.std(axis=1),
            "PRED_MIN_C": forecast_array.min(axis=1),
            "PRED_MAX_C": forecast_array.max(axis=1),
            "PRED_Q05_C": np.quantile(forecast_array, 0.05, axis=1),
            "PRED_Q50_C": np.quantile(forecast_array, 0.50, axis=1),
            "PRED_Q95_C": np.quantile(forecast_array, 0.95, axis=1),
        }
    )

    daily_summary.to_csv(
        FORECAST_DAILY_SUMMARY_PATH,
        index=False,
    )

    sample_points = [
        0,
        number_of_points // 4,
        number_of_points // 2,
        3 * number_of_points // 4,
        number_of_points - 1,
    ]

    sample_rows = []

    for point_index in sample_points:
        for day_index, date in enumerate(forecast_dates):
            sample_rows.append(
                {
                    "DATE": date.strftime("%Y-%m-%d"),
                    "center_index": int(point_index),
                    "LAMBX": int(centers[point_index, 0]),
                    "LAMBY": int(centers[point_index, 1]),
                    "PRED_T_C": float(forecast_array[day_index, point_index]),
                }
            )

    pd.DataFrame(sample_rows).to_csv(
        FORECAST_SAMPLE_PATH,
        index=False,
    )

    metadata = {
        "type": "autoregressive_simulated_forecast",
        "warning": (
            "Prévision simulée sans données d'entrée SAFRAN réelles 2000–2025. "
            "Les températures prédites sont réinjectées dans la fenêtre temporelle. "
            "TINF_H et TSUP_H sont reconstruits à partir d'écarts climatologiques "
            "calculés sur 1960–1999."
        ),
        "model": "ConvLSTM_all_points_masked_5x5",
        "checkpoint": str(CHECKPOINT_PATH),
        "forecast_start": FORECAST_START,
        "forecast_end": FORECAST_END,
        "number_of_days": int(number_of_days),
        "number_of_points": int(number_of_points),
        "number_of_predictions": int(number_of_days * number_of_points),
        "input_seed": "last_30_days_of_1999",
        "forecast_array_path": str(FORECAST_ARRAY_PATH),
        "forecast_dates_path": str(FORECAST_DATES_PATH),
        "daily_summary_path": str(FORECAST_DAILY_SUMMARY_PATH),
        "sample_points_path": str(FORECAST_SAMPLE_PATH),
    }

    with FORECAST_METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=4,
        )

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 80)
    print("PRÉDICTION SIMULÉE TERMINÉE")
    print("=" * 80)

    print(f"Durée : {elapsed:.1f}s")
    print(f"Prévisions : {FORECAST_ARRAY_PATH}")
    print(f"Dates      : {FORECAST_DATES_PATH}")
    print(f"Résumé     : {FORECAST_DAILY_SUMMARY_PATH}")
    print(f"Exemples   : {FORECAST_SAMPLE_PATH}")
    print(f"Métadonnées: {FORECAST_METADATA_PATH}")

    print("\nRésumé global des températures prédites :")
    print(daily_summary.describe().to_string())


if __name__ == "__main__":
    main()
