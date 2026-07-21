from __future__ import annotations

import copy
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
    METRICS_DIR,
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
    TARGET_COLUMN,
    create_multivariate_sequences,
    fit_feature_standardizer,
    load_multivariate_point,
    transform_features,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEEDS = [41, 42, 43]

SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

BATCH_SIZE = 128
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0

TCN_CHANNELS = (32, 64, 64)
KERNEL_SIZE = 3
DROPOUT = 0.15


FEATURE_SETS = {
    "t_only": [
        "T",
    ],
    "t_calendar": [
        "T",
        "DOY_SIN",
        "DOY_COS",
    ],
    "temperature_family": [
        "T",
        "TINF_H",
        "TSUP_H",
        "DOY_SIN",
        "DOY_COS",
    ],
    "meteo_core": [
        "T",
        "PRETOT",
        "FF",
        "Q",
        "SSI",
        "HU",
        "ETP",
        "DOY_SIN",
        "DOY_COS",
    ],
    "full_without_extremes": [
        "T",
        "PRETOT",
        "FF",
        "Q",
        "SSI",
        "HU",
        "ETP",
        "SWI",
        "SSWI_10J",
        "DRAINC",
        "DOY_SIN",
        "DOY_COS",
    ],
    "full": [
        "T",
        "PRETOT",
        "FF",
        "Q",
        "SSI",
        "HU",
        "ETP",
        "SWI",
        "SSWI_10J",
        "DRAINC",
        "TINF_H",
        "TSUP_H",
        "DOY_SIN",
        "DOY_COS",
    ],
}


# ============================================================
# OUTILS
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_loader(
    x_array: np.ndarray,
    y_array: np.ndarray,
    shuffle: bool,
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
        pin_memory=True,
        drop_last=False,
    )


def train_epoch(
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
            set_to_none=True,
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
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP_NORM,
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
def validation_loss(
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
) -> np.ndarray:
    model.eval()

    predictions = []

    for inputs, _ in loader:
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
            batch_predictions
            .detach()
            .cpu()
            .numpy()
        )

    return np.concatenate(predictions)


# ============================================================
# UNE EXPÉRIENCE
# ============================================================

def run_experiment(
    dataframe: pd.DataFrame,
    feature_set_name: str,
    feature_columns: list[str],
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    set_seed(seed)

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

    train_loader = create_loader(
        x_train,
        y_train,
        shuffle=True,
    )

    validation_loader = create_loader(
        x_validation,
        y_validation,
        shuffle=False,
    )

    test_loader = create_loader(
        x_test,
        y_test,
        shuffle=False,
    )

    model = TemporalConvolutionalNetwork(
        input_channels=len(feature_columns),
        channels=TCN_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
    ).to(device)

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

    scaler = torch.amp.GradScaler("cuda")

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    start_time = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
        )

        current_validation_loss = validation_loss(
            model,
            validation_loader,
            criterion,
            device,
        )

        scheduler.step(current_validation_loss)

        if current_validation_loss < best_loss:
            best_loss = current_validation_loss
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            break

    duration = time.perf_counter() - start_time

    if best_state is None:
        raise RuntimeError(
            "Aucun état valide du modèle n'a été trouvé."
        )

    model.load_state_dict(best_state)

    normalized_predictions = predict(
        model,
        test_loader,
        device,
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

    return {
        "feature_set": feature_set_name,
        "seed": seed,
        "number_of_features": len(feature_columns),
        "features": ", ".join(feature_columns),
        "MAE_C": metrics["MAE_C"],
        "RMSE_C": metrics["RMSE_C"],
        "BIAS_C": metrics["BIAS_C"],
        "R2": metrics["R2"],
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "duration_seconds": duration,
    }


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main() -> None:
    create_output_directories()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA n'est pas disponible.\n"
            "Exécute d'abord :\n"
            "source ~/climate_env/bin/activate\n"
            "source ndambo/scripts/activate_gpu_580126.sh"
        )

    device = torch.device("cuda")

    print("=" * 80)
    print("ÉTUDE D'ABLATION DU TCN")
    print("=" * 80)

    print(
        f"GPU : {torch.cuda.get_device_name(0)}"
    )

    dataframe = load_multivariate_point()

    results = []

    total_experiments = (
        len(FEATURE_SETS)
        * len(SEEDS)
    )

    experiment_number = 0

    for feature_set_name, feature_columns in (
        FEATURE_SETS.items()
    ):
        for seed in SEEDS:
            experiment_number += 1

            print(
                f"\n[{experiment_number}/"
                f"{total_experiments}] "
                f"{feature_set_name} | seed={seed}"
            )

            result = run_experiment(
                dataframe=dataframe,
                feature_set_name=feature_set_name,
                feature_columns=feature_columns,
                seed=seed,
                device=device,
            )

            results.append(result)

            print(
                f"MAE={result['MAE_C']:.4f} °C | "
                f"RMSE={result['RMSE_C']:.4f} °C | "
                f"R²={result['R2']:.4f} | "
                f"epoch={result['best_epoch']} | "
                f"durée={result['duration_seconds']:.1f}s"
            )

    runs_dataframe = pd.DataFrame(results)

    runs_path = (
        METRICS_DIR
        / "tcn_ablation_runs.csv"
    )

    runs_dataframe.to_csv(
        runs_path,
        index=False,
    )

    summary = (
        runs_dataframe
        .groupby(
            [
                "feature_set",
                "number_of_features",
            ],
            as_index=False,
        )
        .agg(
            MAE_mean=("MAE_C", "mean"),
            MAE_std=("MAE_C", "std"),
            RMSE_mean=("RMSE_C", "mean"),
            RMSE_std=("RMSE_C", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            BIAS_mean=("BIAS_C", "mean"),
            duration_mean=(
                "duration_seconds",
                "mean",
            ),
        )
        .sort_values("MAE_mean")
        .reset_index(drop=True)
    )

    summary_path = (
        METRICS_DIR
        / "tcn_ablation_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    json_path = (
        METRICS_DIR
        / "tcn_ablation_summary.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary.to_dict(
                orient="records"
            ),
            file,
            ensure_ascii=False,
            indent=4,
        )

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    axis.bar(
        summary["feature_set"],
        summary["MAE_mean"],
        yerr=summary["MAE_std"],
        capsize=4,
    )

    axis.set_title(
        "Étude d'ablation — MAE moyenne sur trois graines"
    )

    axis.set_xlabel(
        "Ensemble de variables"
    )

    axis.set_ylabel(
        "MAE en °C"
    )

    axis.tick_params(
        axis="x",
        rotation=25,
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    figure_path = (
        FIGURE_DIR
        / "tcn_ablation_mae.png"
    )

    figure.savefig(
        figure_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("\n" + "=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nFichiers créés :")
    print(f"- Expériences : {runs_path}")
    print(f"- Résumé CSV  : {summary_path}")
    print(f"- Résumé JSON : {json_path}")
    print(f"- Graphique   : {figure_path}")


if __name__ == "__main__":
    main()
