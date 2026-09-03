from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from config import (  # noqa: E402
    FIGURE_DIR,
    LOG_DIR,
    METRICS_DIR,
    MODEL_DIR,
    PERSONAL_PROCESSED_DATA_DIR,
    PREDICTION_DIR,
    create_output_directories,
)

from models.baseline import (  # noqa: E402
    calculate_regression_metrics,
)

from models.convlstm import (  # noqa: E402
    ConvLSTMRegressor,
)


# ============================================================
# PARAMÈTRES
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1

BATCH_SIZE = 512
MAX_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 12

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0

HIDDEN_CHANNELS = (32, 64)
KERNEL_SIZE = 3
DROPOUT = 0.15

TRAIN_SAMPLES_PER_EPOCH = 250_000
VALIDATION_SAMPLES_PER_EPOCH = 60_000

REQUIRE_CUDA = True


# ============================================================
# DATASET MULTI-CENTRES
# ============================================================

class ManyCentersSpatialDataset(Dataset):
    """
    Dataset ConvLSTM multi-points.

    Le tenseur d'entrée a la forme :
        jours × centres × canaux × hauteur × largeur

    Chaque exemple correspond à :
        un centre spatial
        une fenêtre temporelle de 30 jours
        une cible T au centre à J+1
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        input_length: int,
        forecast_horizon: int,
    ) -> None:
        if features.ndim != 5:
            raise ValueError(
                "features doit avoir la forme "
                "(jours, centres, canaux, hauteur, largeur)."
            )

        if targets.ndim != 2:
            raise ValueError(
                "targets doit avoir la forme "
                "(jours, centres)."
            )

        if features.shape[:2] != targets.shape:
            raise ValueError(
                "features et targets ne correspondent pas "
                "sur les dimensions jours/centres."
            )

        self.features = torch.from_numpy(
            features.astype(np.float32, copy=False)
        )

        self.targets = torch.from_numpy(
            targets.astype(np.float32, copy=False)
        )

        self.input_length = input_length
        self.forecast_horizon = forecast_horizon

        self.number_of_days = features.shape[0]
        self.number_of_centers = features.shape[1]

        self.number_of_sequences_per_center = (
            self.number_of_days
            - input_length
            - forecast_horizon
            + 1
        )

        if self.number_of_sequences_per_center <= 0:
            raise ValueError(
                "Pas assez de jours pour créer les séquences."
            )

        self.total_sequences = (
            self.number_of_sequences_per_center
            * self.number_of_centers
        )

    def __len__(self) -> int:
        return self.total_sequences

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        center_index = (
            index
            // self.number_of_sequences_per_center
        )

        sequence_index = (
            index
            % self.number_of_sequences_per_center
        )

        input_end = sequence_index + self.input_length

        target_index = (
            input_end
            + self.forecast_horizon
            - 1
        )

        inputs = self.features[
            sequence_index:input_end,
            center_index,
            :,
            :,
            :,
        ]

        target = self.targets[
            target_index,
            center_index,
        ]

        return inputs, target


# ============================================================
# OUTILS
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entraînement ConvLSTM global sur plusieurs "
            "centres SAFRAN."
        )
    )

    parser.add_argument(
        "--centers",
        type=int,
        default=100,
        help="Nombre de centres dans le fichier extrait.",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=5,
        choices=[1, 3, 5],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=MAX_EPOCHS,
    )

    return parser.parse_args()


def load_many_centers_data(
    centers: int,
    patch_size: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    path = (
        PERSONAL_PROCESSED_DATA_DIR
        / (
            f"safran_many_centers_"
            f"{patch_size}x{patch_size}_"
            f"{centers}centers_temperature_family.npz"
        )
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier multi-centres introuvable : {path}"
        )

    archive = np.load(path)

    data = archive["data"].astype(
        np.float32,
        copy=False,
    )

    dates = archive["dates"]

    centers_array = archive["centers"]

    channels = [
        str(channel)
        for channel in archive["channels"].tolist()
    ]

    if not np.isfinite(data).all():
        raise ValueError(
            "Le tenseur contient des valeurs non finies."
        )

    return data, dates, centers_array, channels


def chronological_split(
    data: np.ndarray,
    dates: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    number_of_days = data.shape[0]

    train_end = int(number_of_days * TRAIN_RATIO)

    validation_end = int(
        number_of_days
        * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    train_data = data[:train_end]
    validation_data = data[train_end:validation_end]
    test_data = data[validation_end:]

    train_dates = dates[:train_end]
    validation_dates = dates[train_end:validation_end]
    test_dates = dates[validation_end:]

    return (
        train_data,
        validation_data,
        test_data,
        train_dates,
        validation_dates,
        test_dates,
    )


def calculate_feature_statistics(
    train_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Moyenne et écart-type par canal, calculés uniquement
    sur le train, sur tous les jours, centres et pixels.
    """

    means = train_data.mean(
        axis=(0, 1, 3, 4),
        dtype=np.float64,
    ).astype(np.float32)

    stds = train_data.std(
        axis=(0, 1, 3, 4),
        dtype=np.float64,
    ).astype(np.float32)

    stds = np.where(
        stds < 1e-8,
        1.0,
        stds,
    ).astype(np.float32)

    return means, stds


def standardize(
    data: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    return (
        (
            data
            - means[None, None, :, None, None]
        )
        / stds[None, None, :, None, None]
    ).astype(np.float32)


def create_targets(
    raw_data: np.ndarray,
    temperature_channel_index: int,
    center_row: int,
    center_column: int,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    center_temperatures = raw_data[
        :,
        :,
        temperature_channel_index,
        center_row,
        center_column,
    ]

    return (
        (
            center_temperatures
            - target_mean
        )
        / target_std
    ).astype(np.float32)


def create_sampled_loader(
    dataset: Dataset,
    sample_count: int | None,
    shuffle: bool,
) -> DataLoader:
    if sample_count is None or sample_count >= len(dataset):
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

    indices = np.random.choice(
        len(dataset),
        size=sample_count,
        replace=False,
    )

    sampler = SubsetRandomSampler(
        indices.tolist()
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def create_full_loader(
    dataset: Dataset,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


# ============================================================
# TRAIN / EVAL
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            predictions = model(inputs)
            loss = criterion(predictions, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP_NORM,
        )

        scaler.step(optimizer)
        scaler.update()

        batch_size = inputs.shape[0]

        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / total_examples


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()

    total_loss = 0.0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            predictions = model(inputs)
            loss = criterion(predictions, targets)

        batch_size = inputs.shape[0]

        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / total_examples


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    predictions = []
    targets = []

    for inputs, batch_targets in loader:
        inputs = inputs.to(
            device,
            non_blocking=True,
        )

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

    return (
        np.concatenate(predictions),
        np.concatenate(targets),
    )


# ============================================================
# PROGRAMME
# ============================================================

def main() -> None:
    args = parse_arguments()

    create_output_directories()
    set_seed(RANDOM_SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if REQUIRE_CUDA and device.type != "cuda":
        raise RuntimeError(
            "CUDA est requis pour cet entraînement."
        )

    print("=" * 80)
    print("CONVLSTM GLOBAL MULTI-POINTS SAFRAN")
    print("=" * 80)

    print(f"GPU : {torch.cuda.get_device_name(0)}")

    data, dates, centers_array, channels = (
        load_many_centers_data(
            centers=args.centers,
            patch_size=args.patch_size,
        )
    )

    print(f"Tenseur : {data.shape}")
    print(f"Centres : {centers_array.shape}")
    print(f"Canaux  : {channels}")

    temperature_channel_index = channels.index("T")
    center_row = args.patch_size // 2
    center_column = args.patch_size // 2

    (
        train_raw,
        validation_raw,
        test_raw,
        train_dates,
        validation_dates,
        test_dates,
    ) = chronological_split(data, dates)

    means, stds = calculate_feature_statistics(
        train_raw
    )

    train_features = standardize(
        train_raw,
        means,
        stds,
    )

    validation_features = standardize(
        validation_raw,
        means,
        stds,
    )

    test_features = standardize(
        test_raw,
        means,
        stds,
    )

    train_center_temperatures = train_raw[
        :,
        :,
        temperature_channel_index,
        center_row,
        center_column,
    ]

    target_mean = float(
        train_center_temperatures.mean(
            dtype=np.float64
        )
    )

    target_std = float(
        train_center_temperatures.std(
            dtype=np.float64
        )
    )

    train_targets = create_targets(
        train_raw,
        temperature_channel_index,
        center_row,
        center_column,
        target_mean,
        target_std,
    )

    validation_targets = create_targets(
        validation_raw,
        temperature_channel_index,
        center_row,
        center_column,
        target_mean,
        target_std,
    )

    test_targets = create_targets(
        test_raw,
        temperature_channel_index,
        center_row,
        center_column,
        target_mean,
        target_std,
    )

    train_dataset = ManyCentersSpatialDataset(
        train_features,
        train_targets,
        SEQUENCE_LENGTH,
        FORECAST_HORIZON,
    )

    validation_dataset = ManyCentersSpatialDataset(
        validation_features,
        validation_targets,
        SEQUENCE_LENGTH,
        FORECAST_HORIZON,
    )

    test_dataset = ManyCentersSpatialDataset(
        test_features,
        test_targets,
        SEQUENCE_LENGTH,
        FORECAST_HORIZON,
    )

    print(f"Séquences train      : {len(train_dataset):,}")
    print(f"Séquences validation : {len(validation_dataset):,}")
    print(f"Séquences test       : {len(test_dataset):,}")

    validation_loader_fixed = create_sampled_loader(
        validation_dataset,
        VALIDATION_SAMPLES_PER_EPOCH,
        shuffle=False,
    )

    model = ConvLSTMRegressor(
        input_channels=len(channels),
        hidden_channels=HIDDEN_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
    ).to(device)

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Paramètres entraînables : "
        f"{number_of_parameters:,}"
    )

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
        patience=4,
        min_lr=1e-6,
    )

    scaler = torch.amp.GradScaler("cuda")

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    checkpoint_path = (
        MODEL_DIR
        / f"convlstm_many_centers_{args.centers}_best.pt"
    )

    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loader = create_sampled_loader(
            train_dataset,
            TRAIN_SAMPLES_PER_EPOCH,
            shuffle=True,
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
        )

        validation_loss = evaluate_loss(
            model,
            validation_loader_fixed,
            criterion,
            device,
        )

        scheduler.step(validation_loss)

        learning_rate = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": learning_rate,
            }
        )

        print(
            f"Époque {epoch:03d}/{args.epochs} | "
            f"train={train_loss:.6f} | "
            f"val={validation_loss:.6f} | "
            f"lr={learning_rate:.2e}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "channels": channels,
                    "centers": centers_array.tolist(),
                    "feature_means": means.tolist(),
                    "feature_stds": stds.tolist(),
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "sequence_length": SEQUENCE_LENGTH,
                    "forecast_horizon": FORECAST_HORIZON,
                    "patch_size": args.patch_size,
                    "hidden_channels": HIDDEN_CHANNELS,
                    "kernel_size": KERNEL_SIZE,
                    "dropout": DROPOUT,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nArrêt anticipé : aucune amélioration "
                f"depuis {EARLY_STOPPING_PATIENCE} époques."
            )
            break

    training_duration = time.perf_counter() - start_time

    history_dataframe = pd.DataFrame(history)

    history_path = (
        LOG_DIR
        / f"convlstm_many_centers_{args.centers}_history.csv"
    )

    history_dataframe.to_csv(
        history_path,
        index=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_loader = create_full_loader(
        test_dataset
    )

    normalized_predictions, normalized_targets = predict(
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

    metrics = calculate_regression_metrics(
        y_true=true_celsius,
        y_pred=predicted_celsius,
    )

    number_of_sequences_per_center = (
        test_dataset.number_of_sequences_per_center
    )

    persistence_values = []

    for center_index in range(test_raw.shape[1]):
        center_temperature = test_raw[
            :,
            center_index,
            temperature_channel_index,
            center_row,
            center_column,
        ]

        persistence_values.append(
            center_temperature[
                SEQUENCE_LENGTH - 1:
                SEQUENCE_LENGTH - 1
                + number_of_sequences_per_center
            ]
        )

    persistence_celsius = np.concatenate(
        persistence_values
    )

    persistence_metrics = calculate_regression_metrics(
        y_true=true_celsius,
        y_pred=persistence_celsius,
    )

    predictions_path = (
        PREDICTION_DIR
        / f"convlstm_many_centers_{args.centers}_predictions_sample.csv"
    )

    sample_size = min(
        20_000,
        len(true_celsius),
    )

    sample_dataframe = pd.DataFrame(
        {
            "TEMPERATURE_REELLE": true_celsius[:sample_size],
            "TEMPERATURE_CONVLSTM": predicted_celsius[:sample_size],
            "TEMPERATURE_PERSISTANCE": persistence_celsius[:sample_size],
        }
    )

    sample_dataframe.to_csv(
        predictions_path,
        index=False,
    )

    metrics_path = (
        METRICS_DIR
        / f"convlstm_many_centers_{args.centers}_metrics.json"
    )

    results = {
        "model": "ConvLSTM_many_centers",
        "number_of_centers": int(args.centers),
        "patch_size": int(args.patch_size),
        "channels": channels,
        "number_of_parameters": int(number_of_parameters),
        "train_sequences_total": int(len(train_dataset)),
        "validation_sequences_total": int(len(validation_dataset)),
        "test_sequences_total": int(len(test_dataset)),
        "train_samples_per_epoch": int(TRAIN_SAMPLES_PER_EPOCH),
        "validation_samples_per_epoch": int(VALIDATION_SAMPLES_PER_EPOCH),
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_loss": float(checkpoint["validation_loss"]),
        "training_duration_seconds": float(training_duration),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "persistence_metrics": persistence_metrics,
        "convlstm_metrics": metrics,
        "MAE_improvement_vs_persistence_percent": float(
            100.0
            * (
                persistence_metrics["MAE_C"]
                - metrics["MAE_C"]
            )
            / persistence_metrics["MAE_C"]
        ),
        "RMSE_improvement_vs_persistence_percent": float(
            100.0
            * (
                persistence_metrics["RMSE_C"]
                - metrics["RMSE_C"]
            )
            / persistence_metrics["RMSE_C"]
        ),
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4,
        )

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        history_dataframe["epoch"],
        history_dataframe["train_loss"],
        label="Train",
    )

    axis.plot(
        history_dataframe["epoch"],
        history_dataframe["validation_loss"],
        label="Validation",
    )

    axis.set_title(
        f"ConvLSTM global — {args.centers} centres SAFRAN"
    )

    axis.set_xlabel("Époque")
    axis.set_ylabel("MSE normalisée")
    axis.legend()
    axis.grid(alpha=0.3)

    figure.tight_layout()

    figure_path = (
        FIGURE_DIR
        / f"convlstm_many_centers_{args.centers}_loss.png"
    )

    figure.savefig(
        figure_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("\n" + "=" * 80)
    print("RÉSULTATS CONVLSTM GLOBAL MULTI-POINTS")
    print("=" * 80)

    print(f"Meilleure époque : {checkpoint['epoch']}")
    print(f"Durée             : {training_duration:.1f}s")

    print("\nPersistance :")
    print(f"MAE  : {persistence_metrics['MAE_C']:.4f} °C")
    print(f"RMSE : {persistence_metrics['RMSE_C']:.4f} °C")

    print("\nConvLSTM global :")
    print(f"MAE  : {metrics['MAE_C']:.4f} °C")
    print(f"RMSE : {metrics['RMSE_C']:.4f} °C")
    print(f"Biais: {metrics['BIAS_C']:.4f} °C")
    print(f"R²   : {metrics['R2']:.4f}")

    print("\nAmélioration :")
    print(
        "MAE  : "
        f"{results['MAE_improvement_vs_persistence_percent']:.2f} %"
    )
    print(
        "RMSE : "
        f"{results['RMSE_improvement_vs_persistence_percent']:.2f} %"
    )

    print("\nFichiers créés :")
    print(f"- Modèle      : {checkpoint_path}")
    print(f"- Métriques   : {metrics_path}")
    print(f"- Historique  : {history_path}")
    print(f"- Prédictions : {predictions_path}")
    print(f"- Figure      : {figure_path}")


if __name__ == "__main__":
    main()
