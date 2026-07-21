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
    create_output_directories,
)

from data_loader import chronological_split  # noqa: E402
from datasets import NumpySequenceDataset  # noqa: E402

from models.baseline import (  # noqa: E402
    calculate_regression_metrics,
)

from models.tcn import (  # noqa: E402
    TemporalConvolutionalNetwork,
)

from multivariate_data import (  # noqa: E402
    DEFAULT_FEATURE_COLUMNS,
    TARGET_COLUMN,
    create_multivariate_sequences,
    fit_feature_standardizer,
    load_multivariate_point,
    transform_features,
)


RANDOM_SEED = 42

SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1

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

REQUIRE_CUDA = True
PLOT_LENGTH = 365


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_loader(
    x_array: np.ndarray,
    y_array: np.ndarray,
    shuffle: bool,
    use_cuda: bool,
) -> DataLoader:
    dataset = NumpySequenceDataset(
        x_array=x_array,
        y_array=y_array,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=use_cuda,
        drop_last=False,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
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
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            predictions = model(inputs)

            loss = criterion(
                predictions,
                targets,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP_NORM,
        )

        scaler.step(optimizer)
        scaler.update()

        batch_size = inputs.shape[0]

        total_loss += (
            float(loss.detach().item())
            * batch_size
        )

        total_examples += batch_size

    return total_loss / total_examples


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
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
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            predictions = model(inputs)

            loss = criterion(
                predictions,
                targets,
            )

        batch_size = inputs.shape[0]

        total_loss += (
            float(loss.item())
            * batch_size
        )

        total_examples += batch_size

    return total_loss / total_examples


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> np.ndarray:
    model.eval()

    predictions = []

    for inputs, _ in loader:
        inputs = inputs.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            batch_predictions = model(inputs)

        predictions.append(
            batch_predictions
            .detach()
            .cpu()
            .numpy()
        )

    return np.concatenate(predictions)


def save_loss_plot(
    history: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        history["epoch"],
        history["train_loss"],
        label="Train",
    )

    axis.plot(
        history["epoch"],
        history["validation_loss"],
        label="Validation",
    )

    axis.set_title(
        "TCN multivarié — perte d'entraînement"
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
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_dataframe = predictions.head(
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
        label="TCN multivarié",
        linewidth=1.2,
    )

    axis.set_title(
        "TCN multivarié — prévision à J+1"
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


def load_reference_metrics() -> dict[str, object]:
    references: dict[str, object] = {}

    baseline_path = (
        METRICS_DIR
        / "baseline_persistence_metrics.json"
    )

    univariate_paths = [
        METRICS_DIR
        / "tcn_temperature_metrics_gpu.json",
        METRICS_DIR
        / "tcn_temperature_metrics.json",
    ]

    if baseline_path.is_file():
        with baseline_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            references["baseline"] = json.load(file)

    for univariate_path in univariate_paths:
        if univariate_path.is_file():
            with univariate_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                references["univariate_tcn"] = json.load(
                    file
                )
            break

    return references


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
            "source ~/climate_project/ndambo/scripts/"
            "activate_gpu_580126.sh"
        )

    use_cuda = device.type == "cuda"
    amp_enabled = use_cuda

    feature_columns = list(
        DEFAULT_FEATURE_COLUMNS
    )

    print("=" * 80)
    print("ENTRAÎNEMENT DU TCN MULTIVARIÉ")
    print("=" * 80)

    print(f"Appareil          : {device}")
    print(
        "GPU               : "
        f"{torch.cuda.get_device_name(0)}"
    )
    print(
        f"Nombre de variables : "
        f"{len(feature_columns)}"
    )
    print(
        "Variables         : "
        + ", ".join(feature_columns)
    )

    dataframe = load_multivariate_point()

    (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = chronological_split(
        dataframe,
        train_ratio=TRAIN_RATIO,
        validation_ratio=VALIDATION_RATIO,
    )

    means, standard_deviations = (
        fit_feature_standardizer(
            train_dataframe,
            feature_columns,
        )
    )

    train_scaled = transform_features(
        train_dataframe,
        feature_columns,
        means,
        standard_deviations,
    )

    validation_scaled = transform_features(
        validation_dataframe,
        feature_columns,
        means,
        standard_deviations,
    )

    test_scaled = transform_features(
        test_dataframe,
        feature_columns,
        means,
        standard_deviations,
    )

    x_train, y_train = create_multivariate_sequences(
        train_scaled,
        feature_columns,
        TARGET_COLUMN,
        SEQUENCE_LENGTH,
        FORECAST_HORIZON,
    )

    x_validation, y_validation = (
        create_multivariate_sequences(
            validation_scaled,
            feature_columns,
            TARGET_COLUMN,
            SEQUENCE_LENGTH,
            FORECAST_HORIZON,
        )
    )

    x_test, y_test = create_multivariate_sequences(
        test_scaled,
        feature_columns,
        TARGET_COLUMN,
        SEQUENCE_LENGTH,
        FORECAST_HORIZON,
    )

    print(f"X train      : {x_train.shape}")
    print(
        f"X validation : {x_validation.shape}"
    )
    print(f"X test       : {x_test.shape}")

    train_loader = create_loader(
        x_train,
        y_train,
        shuffle=True,
        use_cuda=use_cuda,
    )

    validation_loader = create_loader(
        x_validation,
        y_validation,
        shuffle=False,
        use_cuda=use_cuda,
    )

    test_loader = create_loader(
        x_test,
        y_test,
        shuffle=False,
        use_cuda=use_cuda,
    )

    model = TemporalConvolutionalNetwork(
        input_channels=len(feature_columns),
        channels=TCN_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
    ).to(device)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Paramètres entraînables : "
        f"{trainable_parameters:,}"
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
        "cuda",
        enabled=amp_enabled,
    )

    checkpoint_path = (
        MODEL_DIR
        / "tcn_multivariate_temperature_best.pt"
    )

    history_path = (
        LOG_DIR
        / "tcn_multivariate_training_history.csv"
    )

    metrics_path = (
        METRICS_DIR
        / "tcn_multivariate_temperature_metrics.json"
    )

    predictions_path = (
        PREDICTION_DIR
        / "tcn_multivariate_temperature_predictions.csv"
    )

    loss_figure_path = (
        FIGURE_DIR
        / "tcn_multivariate_training_loss.png"
    )

    prediction_figure_path = (
        FIGURE_DIR
        / "tcn_multivariate_temperature_predictions.png"
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
            amp_enabled,
        )

        validation_loss = evaluate_loss(
            model,
            validation_loader,
            criterion,
            device,
            amp_enabled,
        )

        scheduler.step(validation_loss)

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
                    "feature_columns":
                        feature_columns,
                    "feature_means": means,
                    "feature_standard_deviations":
                        standard_deviations,
                    "target_mean":
                        means[TARGET_COLUMN],
                    "target_standard_deviation":
                        standard_deviations[
                            TARGET_COLUMN
                        ],
                    "channels": TCN_CHANNELS,
                    "kernel_size": KERNEL_SIZE,
                    "dropout": DROPOUT,
                    "sequence_length":
                        SEQUENCE_LENGTH,
                    "forecast_horizon":
                        FORECAST_HORIZON,
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

    normalized_predictions = predict(
        model,
        test_loader,
        device,
        amp_enabled,
    )

    target_mean = means[TARGET_COLUMN]
    target_std = standard_deviations[
        TARGET_COLUMN
    ]

    true_celsius = (
        y_test * target_std
        + target_mean
    )

    predicted_celsius = (
        normalized_predictions
        * target_std
        + target_mean
    )

    metrics = calculate_regression_metrics(
        true_celsius,
        predicted_celsius,
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
            + len(true_celsius)
        ]
        .reset_index(drop=True)
    )

    predictions_dataframe = pd.DataFrame(
        {
            "DATE": target_dates,
            "TEMPERATURE_REELLE":
                true_celsius,
            "TEMPERATURE_PREDITE":
                predicted_celsius,
        }
    )

    predictions_dataframe["ERREUR"] = (
        predictions_dataframe[
            "TEMPERATURE_PREDITE"
        ]
        - predictions_dataframe[
            "TEMPERATURE_REELLE"
        ]
    )

    predictions_dataframe[
        "ERREUR_ABSOLUE"
    ] = predictions_dataframe["ERREUR"].abs()

    predictions_dataframe.to_csv(
        predictions_path,
        index=False,
    )

    save_prediction_plot(
        predictions_dataframe,
        prediction_figure_path,
    )

    references = load_reference_metrics()
    comparison = {}

    baseline = references.get("baseline")

    if isinstance(baseline, dict):
        comparison[
            "MAE_improvement_vs_baseline_percent"
        ] = (
            100.0
            * (
                float(baseline["MAE_C"])
                - metrics["MAE_C"]
            )
            / float(baseline["MAE_C"])
        )

        comparison[
            "RMSE_improvement_vs_baseline_percent"
        ] = (
            100.0
            * (
                float(baseline["RMSE_C"])
                - metrics["RMSE_C"]
            )
            / float(baseline["RMSE_C"])
        )

    univariate = references.get(
        "univariate_tcn"
    )

    if isinstance(univariate, dict):
        comparison[
            "univariate_TCN_MAE_C"
        ] = float(univariate["MAE_C"])

        comparison[
            "univariate_TCN_RMSE_C"
        ] = float(univariate["RMSE_C"])

        comparison[
            "MAE_improvement_vs_univariate_percent"
        ] = (
            100.0
            * (
                float(univariate["MAE_C"])
                - metrics["MAE_C"]
            )
            / float(univariate["MAE_C"])
        )

        comparison[
            "RMSE_improvement_vs_univariate_percent"
        ] = (
            100.0
            * (
                float(univariate["RMSE_C"])
                - metrics["RMSE_C"]
            )
            / float(univariate["RMSE_C"])
        )

    metrics.update(
        {
            "model": "TCN_multivariate",
            "target": TARGET_COLUMN,
            "features": feature_columns,
            "number_of_features":
                len(feature_columns),
            "input_length_days":
                SEQUENCE_LENGTH,
            "forecast_horizon_days":
                FORECAST_HORIZON,
            "number_of_test_sequences":
                int(len(true_celsius)),
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
            "comparison": comparison,
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

    print("\n" + "=" * 80)
    print("RÉSULTATS DU TCN MULTIVARIÉ")
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

    for key, value in comparison.items():
        if "percent" in key:
            print(
                f"{key} : {value:.2f} %"
            )

    print("\nFichiers créés :")
    print(f"- Modèle      : {checkpoint_path}")
    print(f"- Métriques   : {metrics_path}")
    print(f"- Prédictions : {predictions_path}")
    print(f"- Perte       : {loss_figure_path}")
    print(
        f"- Prévisions  : "
        f"{prediction_figure_path}"
    )


if __name__ == "__main__":
    main()
