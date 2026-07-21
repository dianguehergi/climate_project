from __future__ import annotations

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


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from config import (  # noqa: E402
    FIGURE_DIR,
    LOG_DIR,
    METRICS_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
    SPATIAL_PATCH_METADATA_PATH,
    SPATIAL_PATCH_PATH,
    create_output_directories,
)

from models.baseline import (  # noqa: E402
    calculate_regression_metrics,
)

from models.convlstm import (  # noqa: E402
    ConvLSTMRegressor,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1

BATCH_SIZE = 128
MAX_EPOCHS = 120
EARLY_STOPPING_PATIENCE = 20

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0

HIDDEN_CHANNELS = (
    32,
    64,
)

KERNEL_SIZE = 3
DROPOUT = 0.15

PLOT_LENGTH = 365
REQUIRE_CUDA = True


# ============================================================
# DATASET
# ============================================================

class SpatialSequenceDataset(Dataset):
    """
    Dataset paresseux construit directement à partir du
    tenseur spatial.

    Features :
        jours × canaux × hauteur × largeur

    Target :
        température normalisée du point central
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        input_length: int,
        forecast_horizon: int,
    ) -> None:
        if features.ndim != 4:
            raise ValueError(
                "features doit avoir la forme "
                "(jours, canaux, hauteur, largeur)."
            )

        if targets.ndim != 1:
            raise ValueError(
                "targets doit avoir une dimension."
            )

        if len(features) != len(targets):
            raise ValueError(
                "features et targets doivent avoir "
                "le même nombre de jours."
            )

        self.features = torch.from_numpy(
            features.astype(
                np.float32,
                copy=False,
            )
        )

        self.targets = torch.from_numpy(
            targets.astype(
                np.float32,
                copy=False,
            )
        )

        self.input_length = input_length
        self.forecast_horizon = (
            forecast_horizon
        )

        self.number_of_sequences = (
            len(features)
            - input_length
            - forecast_horizon
            + 1
        )

        if self.number_of_sequences <= 0:
            raise ValueError(
                "Pas assez de jours pour former "
                "les séquences."
            )

    def __len__(self) -> int:
        return self.number_of_sequences

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        input_end = (
            index
            + self.input_length
        )

        target_index = (
            input_end
            + self.forecast_horizon
            - 1
        )

        inputs = self.features[
            index:input_end
        ]

        target = self.targets[
            target_index
        ]

        return inputs, target


# ============================================================
# REPRODUCTIBILITÉ
# ============================================================

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# CHARGEMENT ET PRÉPARATION
# ============================================================

def load_spatial_data() -> tuple[
    np.ndarray,
    pd.DatetimeIndex,
    list[str],
    dict[str, object],
]:
    if not SPATIAL_PATCH_PATH.is_file():
        raise FileNotFoundError(
            f"Patch spatial introuvable : "
            f"{SPATIAL_PATCH_PATH}"
        )

    if not SPATIAL_PATCH_METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Métadonnées introuvables : "
            f"{SPATIAL_PATCH_METADATA_PATH}"
        )

    archive = np.load(
        SPATIAL_PATCH_PATH,
        allow_pickle=False,
    )

    data = archive["data"].astype(
        np.float32,
        copy=False,
    )

    dates_integer = archive["dates"]

    channels = [
        str(channel)
        for channel in archive[
            "channels"
        ].tolist()
    ]

    dates = pd.to_datetime(
        dates_integer.astype(str),
        format="%Y%m%d",
        errors="raise",
    )

    with SPATIAL_PATCH_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    if data.ndim != 4:
        raise ValueError(
            f"Forme spatiale incorrecte : {data.shape}"
        )

    if len(data) != len(dates):
        raise ValueError(
            "Le nombre de dates ne correspond pas "
            "au nombre de tenseurs spatiaux."
        )

    if not np.isfinite(data).all():
        raise ValueError(
            "Le patch contient des valeurs non finies."
        )

    return (
        data,
        dates,
        channels,
        metadata,
    )


def chronological_array_split(
    data: np.ndarray,
    dates: pd.DatetimeIndex,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DatetimeIndex,
    pd.DatetimeIndex,
    pd.DatetimeIndex,
]:
    number_of_days = len(data)

    train_end = int(
        number_of_days * TRAIN_RATIO
    )

    validation_end = int(
        number_of_days
        * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    train_data = data[:train_end]
    validation_data = data[
        train_end:validation_end
    ]
    test_data = data[validation_end:]

    train_dates = dates[:train_end]
    validation_dates = dates[
        train_end:validation_end
    ]
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
    Calcule une moyenne et un écart-type par canal
    à partir du train uniquement et sur tous les points.
    """

    means = train_data.mean(
        axis=(
            0,
            2,
            3,
        ),
        dtype=np.float64,
    ).astype(np.float32)

    standard_deviations = train_data.std(
        axis=(
            0,
            2,
            3,
        ),
        dtype=np.float64,
    ).astype(np.float32)

    standard_deviations = np.where(
        standard_deviations < 1e-8,
        1.0,
        standard_deviations,
    ).astype(np.float32)

    return means, standard_deviations


def standardize_features(
    data: np.ndarray,
    means: np.ndarray,
    standard_deviations: np.ndarray,
) -> np.ndarray:
    return (
        (
            data
            - means[
                None,
                :,
                None,
                None,
            ]
        )
        / standard_deviations[
            None,
            :,
            None,
            None,
        ]
    ).astype(np.float32)


def create_target_array(
    raw_data: np.ndarray,
    temperature_channel_index: int,
    center_row: int,
    center_column: int,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    center_temperature = raw_data[
        :,
        temperature_channel_index,
        center_row,
        center_column,
    ]

    return (
        (
            center_temperature
            - target_mean
        )
        / target_std
    ).astype(np.float32)


def create_loader(
    dataset: Dataset,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


# ============================================================
# APPRENTISSAGE
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

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            predictions = model(inputs)

            loss = criterion(
                predictions,
                targets,
            )

        scaler.scale(loss).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP_NORM,
        )

        scaler.step(optimizer)
        scaler.update()

        current_batch_size = inputs.shape[0]

        total_loss += (
            float(loss.detach().item())
            * current_batch_size
        )

        total_examples += current_batch_size

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

            loss = criterion(
                predictions,
                targets,
            )

        current_batch_size = inputs.shape[0]

        total_loss += (
            float(loss.item())
            * current_batch_size
        )

        total_examples += current_batch_size

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
            batch_predictions = model(
                inputs
            )

        predictions.append(
            batch_predictions
            .detach()
            .cpu()
            .numpy()
        )

        targets.append(
            batch_targets.numpy()
        )

    return (
        np.concatenate(predictions),
        np.concatenate(targets),
    )


# ============================================================
# GRAPHIQUES
# ============================================================

def save_loss_plot(
    history_dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

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
        "ConvLSTM spatial — perte"
    )

    axis.set_xlabel("Époque")
    axis.set_ylabel("MSE normalisée")
    axis.legend()
    axis.grid(alpha=0.3)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_prediction_plot(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_dataframe = dataframe.head(
        PLOT_LENGTH
    )

    figure, axis = plt.subplots(
        figsize=(14, 6)
    )

    axis.plot(
        plot_dataframe["DATE"],
        plot_dataframe["TEMPERATURE_REELLE"],
        label="Température réelle",
        linewidth=1.5,
    )

    axis.plot(
        plot_dataframe["DATE"],
        plot_dataframe["TEMPERATURE_CONVLSTM"],
        label="ConvLSTM spatial",
        linewidth=1.2,
    )

    if "TEMPERATURE_TCN" in plot_dataframe:
        axis.plot(
            plot_dataframe["DATE"],
            plot_dataframe["TEMPERATURE_TCN"],
            label="TCN temporel",
            linewidth=1.0,
            alpha=0.85,
        )

    axis.plot(
        plot_dataframe["DATE"],
        plot_dataframe[
            "TEMPERATURE_PERSISTANCE"
        ],
        label="Persistance",
        linewidth=0.9,
        alpha=0.7,
    )

    axis.set_title(
        "Centre SAFRAN 6440–22010 : "
        "comparaison à J+1"
    )

    axis.set_xlabel("Date")
    axis.set_ylabel("Température en °C")
    axis.legend()
    axis.grid(alpha=0.3)

    figure.autofmt_xdate()
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main() -> None:
    create_output_directories()
    set_random_seed(RANDOM_SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if REQUIRE_CUDA and device.type != "cuda":
        raise RuntimeError(
            "CUDA est obligatoire mais aucun GPU "
            "n'est détecté.\n"
            "Exécute dans ce terminal :\n"
            "source ~/climate_env/bin/activate\n"
            "source ndambo/scripts/"
            "activate_gpu_580126.sh"
        )

    print("=" * 80)
    print("ENTRAÎNEMENT DU CONVLSTM SPATIAL")
    print("=" * 80)

    print(f"Appareil : {device}")
    print(
        f"GPU      : "
        f"{torch.cuda.get_device_name(0)}"
    )

    (
        raw_data,
        dates,
        channels,
        metadata,
    ) = load_spatial_data()

    temperature_channel_index = (
        channels.index("T")
    )

    center_row = int(
        metadata["center_row"]
    )

    center_column = int(
        metadata["center_column"]
    )

    print(f"Tenseur brut : {raw_data.shape}")
    print(f"Canaux       : {channels}")

    print(
        f"Centre       : ligne={center_row}, "
        f"colonne={center_column}"
    )

    (
        train_raw,
        validation_raw,
        test_raw,
        train_dates,
        validation_dates,
        test_dates,
    ) = chronological_array_split(
        raw_data,
        dates,
    )

    feature_means, feature_stds = (
        calculate_feature_statistics(
            train_raw
        )
    )

    train_features = standardize_features(
        train_raw,
        feature_means,
        feature_stds,
    )

    validation_features = standardize_features(
        validation_raw,
        feature_means,
        feature_stds,
    )

    test_features = standardize_features(
        test_raw,
        feature_means,
        feature_stds,
    )

    train_center_temperature = train_raw[
        :,
        temperature_channel_index,
        center_row,
        center_column,
    ]

    target_mean = float(
        train_center_temperature.mean(
            dtype=np.float64
        )
    )

    target_std = float(
        train_center_temperature.std(
            dtype=np.float64
        )
    )

    if target_std < 1e-8:
        raise ValueError(
            "Écart-type nul pour la cible."
        )

    train_targets = create_target_array(
        train_raw,
        temperature_channel_index,
        center_row,
        center_column,
        target_mean,
        target_std,
    )

    validation_targets = create_target_array(
        validation_raw,
        temperature_channel_index,
        center_row,
        center_column,
        target_mean,
        target_std,
    )

    test_targets = create_target_array(
        test_raw,
        temperature_channel_index,
        center_row,
        center_column,
        target_mean,
        target_std,
    )

    train_dataset = SpatialSequenceDataset(
        train_features,
        train_targets,
        SEQUENCE_LENGTH,
        FORECAST_HORIZON,
    )

    validation_dataset = (
        SpatialSequenceDataset(
            validation_features,
            validation_targets,
            SEQUENCE_LENGTH,
            FORECAST_HORIZON,
        )
    )

    test_dataset = SpatialSequenceDataset(
        test_features,
        test_targets,
        SEQUENCE_LENGTH,
        FORECAST_HORIZON,
    )

    print(
        f"Séquences train      : "
        f"{len(train_dataset):,}"
    )

    print(
        f"Séquences validation : "
        f"{len(validation_dataset):,}"
    )

    print(
        f"Séquences test       : "
        f"{len(test_dataset):,}"
    )

    train_loader = create_loader(
        train_dataset,
        shuffle=True,
    )

    validation_loader = create_loader(
        validation_dataset,
        shuffle=False,
    )

    test_loader = create_loader(
        test_dataset,
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

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda"
    )

    checkpoint_path = (
        MODEL_DIR
        / "convlstm_spatial_center_best.pt"
    )

    metrics_path = (
        METRICS_DIR
        / "convlstm_spatial_center_metrics.json"
    )

    history_path = (
        LOG_DIR
        / "convlstm_spatial_center_history.csv"
    )

    predictions_path = (
        PREDICTION_DIR
        / "convlstm_spatial_center_predictions.csv"
    )

    loss_figure_path = (
        FIGURE_DIR
        / "convlstm_spatial_center_loss.png"
    )

    prediction_figure_path = (
        FIGURE_DIR
        / "convlstm_spatial_center_predictions.png"
    )

    history = []

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    start_time = time.perf_counter()

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
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
            validation_loader,
            criterion,
            device,
        )

        scheduler.step(
            validation_loss
        )

        learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": learning_rate,
            }
        )

        print(
            f"Époque {epoch:03d}/{MAX_EPOCHS} | "
            f"train={train_loss:.6f} | "
            f"val={validation_loss:.6f} | "
            f"lr={learning_rate:.2e}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "epoch": epoch,
                    "validation_loss":
                        validation_loss,
                    "channels": channels,
                    "feature_means":
                        feature_means.tolist(),
                    "feature_stds":
                        feature_stds.tolist(),
                    "target_mean":
                        target_mean,
                    "target_std":
                        target_std,
                    "sequence_length":
                        SEQUENCE_LENGTH,
                    "forecast_horizon":
                        FORECAST_HORIZON,
                    "hidden_channels":
                        HIDDEN_CHANNELS,
                    "kernel_size":
                        KERNEL_SIZE,
                    "dropout":
                        DROPOUT,
                    "center_row":
                        center_row,
                    "center_column":
                        center_column,
                    "center_lambx":
                        metadata["center_lambx"],
                    "center_lamby":
                        metadata["center_lamby"],
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
                "\nArrêt anticipé : aucune "
                f"amélioration depuis "
                f"{EARLY_STOPPING_PATIENCE} époques."
            )
            break

    training_duration = (
        time.perf_counter()
        - start_time
    )

    history_dataframe = pd.DataFrame(
        history
    )

    history_dataframe.to_csv(
        history_path,
        index=False,
    )

    save_loss_plot(
        history_dataframe,
        loss_figure_path,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    (
        normalized_predictions,
        normalized_targets,
    ) = predict(
        model,
        test_loader,
        device,
    )

    true_celsius = (
        normalized_targets
        * target_std
        + target_mean
    )

    predicted_celsius = (
        normalized_predictions
        * target_std
        + target_mean
    )

    number_of_test_sequences = len(
        test_dataset
    )

    persistence_celsius = test_raw[
        (
            SEQUENCE_LENGTH - 1
        ):
        (
            SEQUENCE_LENGTH - 1
            + number_of_test_sequences
        ),
        temperature_channel_index,
        center_row,
        center_column,
    ]

    first_target_position = (
        SEQUENCE_LENGTH
        + FORECAST_HORIZON
        - 1
    )

    target_dates = test_dates[
        first_target_position:
        (
            first_target_position
            + number_of_test_sequences
        )
    ]

    predictions_dataframe = pd.DataFrame(
        {
            "DATE": target_dates,
            "TEMPERATURE_REELLE":
                true_celsius,
            "TEMPERATURE_CONVLSTM":
                predicted_celsius,
            "TEMPERATURE_PERSISTANCE":
                persistence_celsius,
        }
    )

    tcn_predictions_path = (
        PREDICTION_DIR
        / "tcn_spatial_center_predictions.csv"
    )

    if tcn_predictions_path.is_file():
        tcn_dataframe = pd.read_csv(
            tcn_predictions_path
        )

        tcn_dataframe["DATE"] = pd.to_datetime(
            tcn_dataframe["DATE"],
            errors="coerce",
        )

        tcn_subset = tcn_dataframe[
            [
                "DATE",
                "TEMPERATURE_TCN",
            ]
        ]

        predictions_dataframe = (
            predictions_dataframe.merge(
                tcn_subset,
                on="DATE",
                how="left",
                validate="one_to_one",
            )
        )

    predictions_dataframe[
        "ERREUR_CONVLSTM"
    ] = (
        predictions_dataframe[
            "TEMPERATURE_CONVLSTM"
        ]
        - predictions_dataframe[
            "TEMPERATURE_REELLE"
        ]
    )

    predictions_dataframe.to_csv(
        predictions_path,
        index=False,
    )

    save_prediction_plot(
        predictions_dataframe,
        prediction_figure_path,
    )

    convlstm_metrics = (
        calculate_regression_metrics(
            y_true=true_celsius,
            y_pred=predicted_celsius,
        )
    )

    persistence_metrics = (
        calculate_regression_metrics(
            y_true=true_celsius,
            y_pred=persistence_celsius,
        )
    )

    comparison = {
        "MAE_improvement_vs_persistence_percent":
            (
                100.0
                * (
                    persistence_metrics["MAE_C"]
                    - convlstm_metrics["MAE_C"]
                )
                / persistence_metrics["MAE_C"]
            ),
        "RMSE_improvement_vs_persistence_percent":
            (
                100.0
                * (
                    persistence_metrics["RMSE_C"]
                    - convlstm_metrics["RMSE_C"]
                )
                / persistence_metrics["RMSE_C"]
            ),
    }

    tcn_metrics_path = (
        METRICS_DIR
        / "tcn_spatial_center_metrics.json"
    )

    tcn_reference_metrics = None

    if tcn_metrics_path.is_file():
        with tcn_metrics_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            tcn_payload = json.load(file)

        tcn_reference_metrics = (
            tcn_payload["tcn_metrics"]
        )

        comparison[
            "MAE_improvement_vs_TCN_percent"
        ] = (
            100.0
            * (
                tcn_reference_metrics["MAE_C"]
                - convlstm_metrics["MAE_C"]
            )
            / tcn_reference_metrics["MAE_C"]
        )

        comparison[
            "RMSE_improvement_vs_TCN_percent"
        ] = (
            100.0
            * (
                tcn_reference_metrics["RMSE_C"]
                - convlstm_metrics["RMSE_C"]
            )
            / tcn_reference_metrics["RMSE_C"]
        )

    results = {
        "model": "ConvLSTM_spatial_center",
        "center_lambx":
            metadata["center_lambx"],
        "center_lamby":
            metadata["center_lamby"],
        "patch_size":
            metadata["patch_size"],
        "channels": channels,
        "input_length_days":
            SEQUENCE_LENGTH,
        "forecast_horizon_days":
            FORECAST_HORIZON,
        "number_of_test_sequences":
            number_of_test_sequences,
        "best_epoch":
            int(checkpoint["epoch"]),
        "best_validation_loss":
            float(
                checkpoint[
                    "validation_loss"
                ]
            ),
        "training_duration_seconds":
            float(training_duration),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "number_of_parameters":
            number_of_parameters,
        "persistence_metrics":
            persistence_metrics,
        "tcn_reference_metrics":
            tcn_reference_metrics,
        "convlstm_metrics":
            convlstm_metrics,
        "comparison":
            comparison,
    }

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\n" + "=" * 80)
    print("RÉSULTATS DU CONVLSTM SPATIAL")
    print("=" * 80)

    print(
        f"Meilleure époque : "
        f"{checkpoint['epoch']}"
    )

    print("\nPersistance :")
    print(
        f"MAE  : "
        f"{persistence_metrics['MAE_C']:.4f} °C"
    )
    print(
        f"RMSE : "
        f"{persistence_metrics['RMSE_C']:.4f} °C"
    )

    if tcn_reference_metrics is not None:
        print("\nTCN temporel :")
        print(
            f"MAE  : "
            f"{tcn_reference_metrics['MAE_C']:.4f} °C"
        )
        print(
            f"RMSE : "
            f"{tcn_reference_metrics['RMSE_C']:.4f} °C"
        )

    print("\nConvLSTM spatial :")
    print(
        f"MAE  : "
        f"{convlstm_metrics['MAE_C']:.4f} °C"
    )
    print(
        f"RMSE : "
        f"{convlstm_metrics['RMSE_C']:.4f} °C"
    )
    print(
        f"Biais: "
        f"{convlstm_metrics['BIAS_C']:.4f} °C"
    )
    print(
        f"R²   : "
        f"{convlstm_metrics['R2']:.4f}"
    )

    print("\nComparaisons :")

    for key, value in comparison.items():
        print(
            f"{key} : {value:.2f} %"
        )

    print(
        f"\nDurée : "
        f"{training_duration:.2f} secondes"
    )

    print("\nFichiers créés :")
    print(f"- Modèle      : {checkpoint_path}")
    print(f"- Métriques   : {metrics_path}")
    print(f"- Prédictions : {predictions_path}")
    print(f"- Historique  : {history_path}")
    print(f"- Perte       : {loss_figure_path}")
    print(
        f"- Prévisions  : "
        f"{prediction_figure_path}"
    )


if __name__ == "__main__":
    main()
