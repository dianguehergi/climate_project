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
# IMPORTS DU PROJET
# ============================================================

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
    SPATIAL_CENTER_SERIES_PATH,
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
    create_multivariate_sequences,
    fit_feature_standardizer,
    transform_features,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "T",
    "TINF_H",
    "TSUP_H",
    "DOY_SIN",
    "DOY_COS",
]

TARGET_COLUMN = "T"

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
# CHARGEMENT DU POINT CENTRAL
# ============================================================

def load_center_dataframe() -> pd.DataFrame:
    if not SPATIAL_CENTER_SERIES_PATH.is_file():
        raise FileNotFoundError(
            "Série du centre introuvable : "
            f"{SPATIAL_CENTER_SERIES_PATH}"
        )

    dataframe = pd.read_csv(
        SPATIAL_CENTER_SERIES_PATH,
        low_memory=False,
    )

    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.strip()
        .str.upper()
    )

    required_columns = {
        "DATE",
        *FEATURE_COLUMNS,
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Colonnes manquantes : "
            f"{sorted(missing_columns)}"
        )

    dataframe["DATE"] = pd.to_datetime(
        dataframe["DATE"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    for column in FEATURE_COLUMNS:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe = (
        dataframe
        .dropna(
            subset=[
                "DATE",
                *FEATURE_COLUMNS,
            ]
        )
        .sort_values("DATE")
        .drop_duplicates(
            subset=["DATE"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if dataframe.empty:
        raise ValueError(
            "La série centrale est vide."
        )

    date_differences = (
        dataframe["DATE"]
        .diff()
        .dropna()
    )

    invalid_gaps = date_differences[
        date_differences
        != pd.Timedelta(days=1)
    ]

    if not invalid_gaps.empty:
        raise ValueError(
            f"{len(invalid_gaps)} rupture(s) "
            "temporelle(s) détectée(s)."
        )

    return dataframe


# ============================================================
# DATALOADER
# ============================================================

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


# ============================================================
# ENTRAÎNEMENT
# ============================================================

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
        "TCN du centre spatial — perte"
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
        plot_dataframe["TEMPERATURE_TCN"],
        label="TCN centre",
        linewidth=1.2,
    )

    axis.plot(
        plot_dataframe["DATE"],
        plot_dataframe["TEMPERATURE_PERSISTANCE"],
        label="Persistance",
        linewidth=1.0,
        alpha=0.8,
    )

    axis.set_title(
        "Centre SAFRAN 6440–22010 : "
        "prévision à J+1"
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
            "source archive_old/tcn_convlstm/scripts/"
            "activate_gpu_580126.sh"
        )

    use_cuda = device.type == "cuda"
    amp_enabled = use_cuda

    print("=" * 80)
    print("TCN DE RÉFÉRENCE DU CENTRE SPATIAL")
    print("=" * 80)

    print(f"Appareil : {device}")
    print(
        f"GPU      : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        "Variables : "
        + ", ".join(FEATURE_COLUMNS)
    )

    dataframe = load_center_dataframe()

    print(
        f"Dimensions : {dataframe.shape}"
    )

    print(
        f"Période    : "
        f"{dataframe['DATE'].min().date()} → "
        f"{dataframe['DATE'].max().date()}"
    )

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
        means,
        standard_deviations,
    ) = fit_feature_standardizer(
        train_dataframe=train_dataframe,
        feature_columns=FEATURE_COLUMNS,
    )

    train_scaled = transform_features(
        dataframe=train_dataframe,
        feature_columns=FEATURE_COLUMNS,
        means=means,
        standard_deviations=standard_deviations,
    )

    validation_scaled = transform_features(
        dataframe=validation_dataframe,
        feature_columns=FEATURE_COLUMNS,
        means=means,
        standard_deviations=standard_deviations,
    )

    test_scaled = transform_features(
        dataframe=test_dataframe,
        feature_columns=FEATURE_COLUMNS,
        means=means,
        standard_deviations=standard_deviations,
    )

    x_train, y_train = create_multivariate_sequences(
        dataframe=train_scaled,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        input_length=SEQUENCE_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
    )

    x_validation, y_validation = (
        create_multivariate_sequences(
            dataframe=validation_scaled,
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            input_length=SEQUENCE_LENGTH,
            forecast_horizon=FORECAST_HORIZON,
        )
    )

    x_test, y_test = create_multivariate_sequences(
        dataframe=test_scaled,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        input_length=SEQUENCE_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
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
        input_channels=len(FEATURE_COLUMNS),
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
        / "tcn_spatial_center_best.pt"
    )

    metrics_path = (
        METRICS_DIR
        / "tcn_spatial_center_metrics.json"
    )

    history_path = (
        LOG_DIR
        / "tcn_spatial_center_history.csv"
    )

    predictions_path = (
        PREDICTION_DIR
        / "tcn_spatial_center_predictions.csv"
    )

    loss_figure_path = (
        FIGURE_DIR
        / "tcn_spatial_center_loss.png"
    )

    prediction_figure_path = (
        FIGURE_DIR
        / "tcn_spatial_center_predictions.png"
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
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
        )

        validation_loss = evaluate_loss(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
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
                    "features":
                        FEATURE_COLUMNS,
                    "feature_means":
                        means,
                    "feature_standard_deviations":
                        standard_deviations,
                    "target_mean":
                        means[TARGET_COLUMN],
                    "target_standard_deviation":
                        standard_deviations[
                            TARGET_COLUMN
                        ],
                    "sequence_length":
                        SEQUENCE_LENGTH,
                    "forecast_horizon":
                        FORECAST_HORIZON,
                    "center_lambx": 6440,
                    "center_lamby": 22010,
                    "channels":
                        TCN_CHANNELS,
                    "kernel_size":
                        KERNEL_SIZE,
                    "dropout":
                        DROPOUT,
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
        model=model,
        loader=test_loader,
        device=device,
        amp_enabled=amp_enabled,
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

    temperature_channel_index = (
        FEATURE_COLUMNS.index("T")
    )

    persistence_celsius = (
        x_test[
            :,
            temperature_channel_index,
            -1,
        ]
        * target_std
        + target_mean
    )

    tcn_metrics = calculate_regression_metrics(
        y_true=true_celsius,
        y_pred=predicted_celsius,
    )

    persistence_metrics = (
        calculate_regression_metrics(
            y_true=true_celsius,
            y_pred=persistence_celsius,
        )
    )

    mae_improvement = (
        100.0
        * (
            persistence_metrics["MAE_C"]
            - tcn_metrics["MAE_C"]
        )
        / persistence_metrics["MAE_C"]
    )

    rmse_improvement = (
        100.0
        * (
            persistence_metrics["RMSE_C"]
            - tcn_metrics["RMSE_C"]
        )
        / persistence_metrics["RMSE_C"]
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
            "TEMPERATURE_TCN":
                predicted_celsius,
            "TEMPERATURE_PERSISTANCE":
                persistence_celsius,
        }
    )

    predictions_dataframe[
        "ERREUR_TCN"
    ] = (
        predictions_dataframe[
            "TEMPERATURE_TCN"
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

    results = {
        "model": "TCN_spatial_center_reference",
        "center_lambx": 6440,
        "center_lamby": 22010,
        "features": FEATURE_COLUMNS,
        "number_of_features":
            len(FEATURE_COLUMNS),
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
        "persistence_metrics":
            persistence_metrics,
        "tcn_metrics":
            tcn_metrics,
        "MAE_improvement_vs_persistence_percent":
            mae_improvement,
        "RMSE_improvement_vs_persistence_percent":
            rmse_improvement,
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
    print("RÉSULTATS DU TCN DU CENTRE")
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

    print("\nTCN centre :")

    print(
        f"MAE  : "
        f"{tcn_metrics['MAE_C']:.4f} °C"
    )

    print(
        f"RMSE : "
        f"{tcn_metrics['RMSE_C']:.4f} °C"
    )

    print(
        f"Biais: "
        f"{tcn_metrics['BIAS_C']:.4f} °C"
    )

    print(
        f"R²   : "
        f"{tcn_metrics['R2']:.4f}"
    )

    print(
        f"\nAmélioration MAE : "
        f"{mae_improvement:.2f} %"
    )

    print(
        f"Amélioration RMSE : "
        f"{rmse_improvement:.2f} %"
    )

    print(
        f"Durée : "
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
