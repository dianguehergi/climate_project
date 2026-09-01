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


# ============================================================
# IMPORT DU DOSSIER SRC
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from config import (  # noqa: E402
    FIGURE_DIR,
    LOG_DIR,
    METRICS_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
    create_output_directories,
)

from data_loader import (  # noqa: E402
    SequenceConfig,
    chronological_split,
    create_sequences,
    load_first_point_temperature,
    standardize_splits,
)

from datasets import NumpySequenceDataset  # noqa: E402

from models.baseline import (  # noqa: E402
    calculate_regression_metrics,
)

from models.tcn import (  # noqa: E402
    TemporalConvolutionalNetwork,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TARGET_COLUMN = "T"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

BATCH_SIZE = 128
MAX_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 20

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0

TCN_CHANNELS = (32, 64, 64)
KERNEL_SIZE = 3
DROPOUT = 0.15

PLOT_LENGTH = 365
REQUIRE_CUDA = True

# ============================================================
# REPRODUCTIBILITÉ
# ============================================================

def set_random_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# DATALOADERS
# ============================================================

def create_data_loader(
    x_array: np.ndarray,
    y_array: np.ndarray,
    batch_size: int,
    shuffle: bool,
    use_cuda: bool,
) -> DataLoader:
    dataset = NumpySequenceDataset(
        x_array=x_array,
        y_array=y_array,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=use_cuda,
        drop_last=False,
    )


# ============================================================
# ENTRAÎNEMENT D'UNE ÉPOQUE
# ============================================================

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.train()

    total_loss = 0.0
    number_of_examples = 0

    for inputs, targets in data_loader:
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
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            predictions = model(
                inputs
            )

            loss = criterion(
                predictions,
                targets,
            )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP_NORM,
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        batch_size = inputs.shape[0]

        total_loss += (
            loss.detach().item()
            * batch_size
        )

        number_of_examples += batch_size

    return total_loss / number_of_examples


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.eval()

    total_loss = 0.0
    number_of_examples = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            predictions = model(
                inputs
            )

            loss = criterion(
                predictions,
                targets,
            )

        batch_size = inputs.shape[0]

        total_loss += (
            loss.item()
            * batch_size
        )

        number_of_examples += batch_size

    return total_loss / number_of_examples


# ============================================================
# PRÉDICTIONS
# ============================================================

@torch.no_grad()
def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> np.ndarray:
    model.eval()

    predictions = []

    for inputs, _ in data_loader:
        inputs = inputs.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
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

    return np.concatenate(
        predictions
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
        "Évolution de la perte du TCN"
    )

    axis.set_xlabel(
        "Époque"
    )

    axis.set_ylabel(
        "MSE normalisée"
    )

    axis.legend()
    axis.grid(alpha=0.3)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_prediction_plot(
    predictions_dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_dataframe = predictions_dataframe.head(
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
        plot_dataframe["TEMPERATURE_PREDITE"],
        label="Prédiction TCN",
        linewidth=1.2,
    )

    axis.set_title(
        "TCN : températures réelles et prédites"
    )

    axis.set_xlabel(
        "Date"
    )

    axis.set_ylabel(
        "Température en °C"
    )

    axis.legend()
    axis.grid(alpha=0.3)

    figure.autofmt_xdate()
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


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
            "CUDA est obligatoire, mais aucun GPU n'est détecté.\n"
            "Active d'abord les bibliothèques NVIDIA avec :\n"
            "source ~/climate_env/bin/activate\n"
            "source ~/climate_project/archive_old/tcn_convlstm/scripts/"
            "activate_gpu_580126.sh"
        )

    use_cuda = device.type == "cuda"
    amp_enabled = use_cuda

    print("=" * 80)
    print("ENTRAÎNEMENT DU TCN")
    print("=" * 80)

    print(f"PyTorch           : {torch.__version__}")
    print(f"Appareil utilisé  : {device}")

    if use_cuda:
        print(
            "GPU               : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Chargement et découpage
    # --------------------------------------------------------

    dataframe = load_first_point_temperature()

    (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = chronological_split(
        dataframe=dataframe,
        train_ratio=TRAIN_RATIO,
        validation_ratio=VALIDATION_RATIO,
    )

    (
        train_scaled,
        validation_scaled,
        test_scaled,
        train_mean,
        train_std,
    ) = standardize_splits(
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        test_dataframe=test_dataframe,
        target_column=TARGET_COLUMN,
    )

    sequence_config = SequenceConfig(
        input_length=SEQUENCE_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
        target_column=TARGET_COLUMN,
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

    print(f"X train       : {x_train.shape}")
    print(f"X validation  : {x_validation.shape}")
    print(f"X test        : {x_test.shape}")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = create_data_loader(
        x_array=x_train,
        y_array=y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        use_cuda=use_cuda,
    )

    validation_loader = create_data_loader(
        x_array=x_validation,
        y_array=y_validation,
        batch_size=BATCH_SIZE,
        shuffle=False,
        use_cuda=use_cuda,
    )

    test_loader = create_data_loader(
        x_array=x_test,
        y_array=y_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        use_cuda=use_cuda,
    )

    # --------------------------------------------------------
    # Modèle
    # --------------------------------------------------------

    model = TemporalConvolutionalNetwork(
        input_channels=1,
        channels=TCN_CHANNELS,
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

    print(
        f"Champ réceptif         : "
        f"{model.receptive_field} jours"
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=6,
        min_lr=1e-6,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    # --------------------------------------------------------
    # Fichiers de sortie
    # --------------------------------------------------------

    checkpoint_path = (
        MODEL_DIR
        / "tcn_temperature_best.pt"
    )

    history_path = (
        LOG_DIR
        / "tcn_training_history.csv"
    )

    loss_figure_path = (
        FIGURE_DIR
        / "tcn_training_loss.png"
    )

    prediction_figure_path = (
        FIGURE_DIR
        / "tcn_temperature_predictions.png"
    )

    prediction_path = (
        PREDICTION_DIR
        / "tcn_temperature_predictions.csv"
    )

    metrics_path = (
        METRICS_DIR
        / "tcn_temperature_metrics.json"
    )

    # --------------------------------------------------------
    # Boucle d'entraînement
    # --------------------------------------------------------

    history = []

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    start_time = time.perf_counter()

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
        )

        validation_loss = evaluate_loss(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
        )

        scheduler.step(
            validation_loss
        )

        current_learning_rate = optimizer.param_groups[
            0
        ]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": current_learning_rate,
            }
        )

        print(
            f"Époque {epoch:03d}/{MAX_EPOCHS} | "
            f"train={train_loss:.6f} | "
            f"val={validation_loss:.6f} | "
            f"lr={current_learning_rate:.2e}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "train_mean": train_mean,
                    "train_std": train_std,
                    "sequence_length": SEQUENCE_LENGTH,
                    "forecast_horizon": FORECAST_HORIZON,
                    "channels": TCN_CHANNELS,
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
        history_dataframe=history_dataframe,
        output_path=loss_figure_path,
    )

    # --------------------------------------------------------
    # Rechargement du meilleur modèle
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Prédiction du jeu de test
    # --------------------------------------------------------

    normalized_predictions = predict(
        model=model,
        data_loader=test_loader,
        device=device,
        amp_enabled=amp_enabled,
    )

    y_test_celsius = (
        y_test * train_std
        + train_mean
    )

    predictions_celsius = (
        normalized_predictions * train_std
        + train_mean
    )

    metrics = calculate_regression_metrics(
        y_true=y_test_celsius,
        y_pred=predictions_celsius,
    )

    first_target_position = (
        SEQUENCE_LENGTH
        + FORECAST_HORIZON
        - 1
    )

    target_dates = (
        test_dataframe["DATE"]
        .iloc[
            first_target_position:
            first_target_position
            + len(y_test_celsius)
        ]
        .reset_index(drop=True)
    )

    predictions_dataframe = pd.DataFrame(
        {
            "DATE": target_dates,
            "TEMPERATURE_REELLE": y_test_celsius,
            "TEMPERATURE_PREDITE": predictions_celsius,
        }
    )

    predictions_dataframe["ERREUR"] = (
        predictions_dataframe["TEMPERATURE_PREDITE"]
        - predictions_dataframe["TEMPERATURE_REELLE"]
    )

    predictions_dataframe[
        "ERREUR_ABSOLUE"
    ] = predictions_dataframe["ERREUR"].abs()

    predictions_dataframe.to_csv(
        prediction_path,
        index=False,
    )

    save_prediction_plot(
        predictions_dataframe=predictions_dataframe,
        output_path=prediction_figure_path,
    )

    # --------------------------------------------------------
    # Comparaison avec la baseline
    # --------------------------------------------------------

    baseline_metrics_path = (
        METRICS_DIR
        / "baseline_persistence_metrics.json"
    )

    baseline_comparison = None

    if baseline_metrics_path.is_file():
        with baseline_metrics_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            baseline_metrics = json.load(
                file
            )

        baseline_mae = float(
            baseline_metrics["MAE_C"]
        )

        baseline_rmse = float(
            baseline_metrics["RMSE_C"]
        )

        baseline_comparison = {
            "baseline_MAE_C": baseline_mae,
            "baseline_RMSE_C": baseline_rmse,
            "MAE_improvement_percent": (
                100
                * (
                    baseline_mae
                    - metrics["MAE_C"]
                )
                / baseline_mae
            ),
            "RMSE_improvement_percent": (
                100
                * (
                    baseline_rmse
                    - metrics["RMSE_C"]
                )
                / baseline_rmse
            ),
        }

    metrics.update(
        {
            "model": "TCN",
            "target": TARGET_COLUMN,
            "input_length_days": SEQUENCE_LENGTH,
            "forecast_horizon_days": FORECAST_HORIZON,
            "number_of_test_sequences": int(
                len(y_test_celsius)
            ),
            "best_epoch": int(
                checkpoint["epoch"]
            ),
            "best_validation_loss": float(
                checkpoint["validation_loss"]
            ),
            "training_duration_seconds": float(
                training_duration
            ),
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(0)
                if use_cuda
                else None
            ),
            "train_mean": train_mean,
            "train_std": train_std,
            "baseline_comparison": baseline_comparison,
        }
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=4,
        )

    # --------------------------------------------------------
    # Résultats
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("RÉSULTATS DU TCN")
    print("=" * 80)

    print(
        f"Meilleure époque : "
        f"{checkpoint['epoch']}"
    )

    print(
        f"MAE              : "
        f"{metrics['MAE_C']:.4f} °C"
    )

    print(
        f"RMSE             : "
        f"{metrics['RMSE_C']:.4f} °C"
    )

    print(
        f"Biais            : "
        f"{metrics['BIAS_C']:.4f} °C"
    )

    print(
        f"R²               : "
        f"{metrics['R2']:.4f}"
    )

    print(
        f"Durée            : "
        f"{training_duration:.2f} secondes"
    )

    if baseline_comparison is not None:
        print(
            "Amélioration MAE : "
            f"{baseline_comparison['MAE_improvement_percent']:.2f} %"
        )

        print(
            "Amélioration RMSE : "
            f"{baseline_comparison['RMSE_improvement_percent']:.2f} %"
        )

    print("\nFichiers créés :")
    print(f"- Modèle       : {checkpoint_path}")
    print(f"- Métriques    : {metrics_path}")
    print(f"- Prédictions  : {prediction_path}")
    print(f"- Courbe perte : {loss_figure_path}")
    print(f"- Graphique    : {prediction_figure_path}")
    print(f"- Historique   : {history_path}")


if __name__ == "__main__":
    main()
