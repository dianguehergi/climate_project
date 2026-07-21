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


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
SCRIPTS_DIR = PROJECT_DIR / "scripts"

for directory in (SRC_DIR, SCRIPTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


from config import (  # noqa: E402
    FIGURE_DIR,
    METRICS_DIR,
    create_output_directories,
)

from models.baseline import (  # noqa: E402
    calculate_regression_metrics,
)

from models.convlstm import (  # noqa: E402
    ConvLSTMRegressor,
)

from train_convlstm_spatial import (  # noqa: E402
    SpatialSequenceDataset,
    calculate_feature_statistics,
    chronological_array_split,
    create_loader,
    create_target_array,
    evaluate_loss,
    load_spatial_data,
    predict,
    standardize_features,
    train_one_epoch,
)


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_SIZES = [1, 3, 5]
SEEDS = [41, 42, 43]

SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1

MAX_EPOCHS = 120
EARLY_STOPPING_PATIENCE = 20

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN_CHANNELS = (32, 64)
KERNEL_SIZE = 3
DROPOUT = 0.15


# ============================================================
# REPRODUCTIBILITÉ
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# DÉCOUPAGE DU PATCH
# ============================================================

def crop_center_patch(
    data: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    """
    Extrait un patch carré centré de taille 1, 3 ou 5.

    Entrée :
        jours × canaux × hauteur × largeur
    """

    if patch_size % 2 == 0:
        raise ValueError(
            "La taille du patch doit être impaire."
        )

    full_height = data.shape[2]
    full_width = data.shape[3]

    if patch_size > full_height or patch_size > full_width:
        raise ValueError(
            f"Patch {patch_size}×{patch_size} trop grand."
        )

    center_row = full_height // 2
    center_column = full_width // 2
    radius = patch_size // 2

    row_start = center_row - radius
    row_end = center_row + radius + 1

    column_start = center_column - radius
    column_end = center_column + radius + 1

    cropped = data[
        :,
        :,
        row_start:row_end,
        column_start:column_end,
    ]

    expected_shape = (
        len(data),
        data.shape[1],
        patch_size,
        patch_size,
    )

    if cropped.shape != expected_shape:
        raise ValueError(
            f"Forme obtenue {cropped.shape}, "
            f"forme attendue {expected_shape}."
        )

    return cropped


# ============================================================
# UNE EXPÉRIENCE
# ============================================================

def run_experiment(
    patch_size: int,
    seed: int,
    train_raw_full: np.ndarray,
    validation_raw_full: np.ndarray,
    test_raw_full: np.ndarray,
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    temperature_channel_index: int,
    target_mean: float,
    target_std: float,
    number_of_channels: int,
    device: torch.device,
) -> dict[str, object]:
    set_seed(seed)

    train_raw = crop_center_patch(
        train_raw_full,
        patch_size,
    )

    validation_raw = crop_center_patch(
        validation_raw_full,
        patch_size,
    )

    test_raw = crop_center_patch(
        test_raw_full,
        patch_size,
    )

    # Les mêmes statistiques 5×5 sont utilisées pour tous
    # les patchs afin d'éviter une différence de normalisation.
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

    center_index = patch_size // 2

    train_targets = create_target_array(
        train_raw,
        temperature_channel_index,
        center_index,
        center_index,
        target_mean,
        target_std,
    )

    validation_targets = create_target_array(
        validation_raw,
        temperature_channel_index,
        center_index,
        center_index,
        target_mean,
        target_std,
    )

    test_targets = create_target_array(
        test_raw,
        temperature_channel_index,
        center_index,
        center_index,
        target_mean,
        target_std,
    )

    train_dataset = SpatialSequenceDataset(
        train_features,
        train_targets,
        SEQUENCE_LENGTH,
        FORECAST_HORIZON,
    )

    validation_dataset = SpatialSequenceDataset(
        validation_features,
        validation_targets,
        SEQUENCE_LENGTH,
        FORECAST_HORIZON,
    )

    test_dataset = SpatialSequenceDataset(
        test_features,
        test_targets,
        SEQUENCE_LENGTH,
        FORECAST_HORIZON,
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
        input_channels=number_of_channels,
        hidden_channels=HIDDEN_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
    ).to(device)

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
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

    scaler = torch.amp.GradScaler("cuda")

    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    start_time = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
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

        scheduler.step(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
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
            "Aucun meilleur état du modèle n'a été conservé."
        )

    model.load_state_dict(best_state)

    (
        normalized_predictions,
        normalized_targets,
    ) = predict(
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

    return {
        "patch_size": patch_size,
        "patch_label": f"{patch_size}x{patch_size}",
        "seed": seed,
        "number_of_parameters": number_of_parameters,
        "number_of_test_sequences": len(test_dataset),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "MAE_C": metrics["MAE_C"],
        "RMSE_C": metrics["RMSE_C"],
        "BIAS_C": metrics["BIAS_C"],
        "R2": metrics["R2"],
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
            "Exécute :\n"
            "source ~/climate_env/bin/activate\n"
            "source ndambo/scripts/activate_gpu_580126.sh"
        )

    device = torch.device("cuda")

    print("=" * 80)
    print("ÉTUDE D'ABLATION SPATIALE DU CONVLSTM")
    print("=" * 80)

    print(
        f"GPU : {torch.cuda.get_device_name(0)}"
    )

    (
        raw_data,
        dates,
        channels,
        metadata,
    ) = load_spatial_data()

    temperature_channel_index = channels.index(
        "T"
    )

    (
        train_raw_full,
        validation_raw_full,
        test_raw_full,
        _,
        _,
        _,
    ) = chronological_array_split(
        raw_data,
        dates,
    )

    # Statistiques communes calculées sur le patch 5×5
    # du jeu d'entraînement.
    feature_means, feature_stds = (
        calculate_feature_statistics(
            train_raw_full
        )
    )

    full_center_row = int(
        metadata["center_row"]
    )

    full_center_column = int(
        metadata["center_column"]
    )

    train_center_temperature = train_raw_full[
        :,
        temperature_channel_index,
        full_center_row,
        full_center_column,
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

    results = []

    total_experiments = (
        len(PATCH_SIZES)
        * len(SEEDS)
    )

    experiment_number = 0

    for patch_size in PATCH_SIZES:
        for seed in SEEDS:
            experiment_number += 1

            print(
                f"\n[{experiment_number}/"
                f"{total_experiments}] "
                f"patch={patch_size}×{patch_size} | "
                f"seed={seed}"
            )

            result = run_experiment(
                patch_size=patch_size,
                seed=seed,
                train_raw_full=train_raw_full,
                validation_raw_full=validation_raw_full,
                test_raw_full=test_raw_full,
                feature_means=feature_means,
                feature_stds=feature_stds,
                temperature_channel_index=(
                    temperature_channel_index
                ),
                target_mean=target_mean,
                target_std=target_std,
                number_of_channels=len(channels),
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
        / "convlstm_patch_ablation_runs.csv"
    )

    runs_dataframe.to_csv(
        runs_path,
        index=False,
    )

    summary = (
        runs_dataframe
        .groupby(
            [
                "patch_size",
                "patch_label",
                "number_of_parameters",
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
        .sort_values("patch_size")
        .reset_index(drop=True)
    )

    reference_mae = float(
        summary.loc[
            summary["patch_size"] == 1,
            "MAE_mean",
        ].iloc[0]
    )

    reference_rmse = float(
        summary.loc[
            summary["patch_size"] == 1,
            "RMSE_mean",
        ].iloc[0]
    )

    summary[
        "MAE_improvement_vs_1x1_percent"
    ] = (
        100.0
        * (
            reference_mae
            - summary["MAE_mean"]
        )
        / reference_mae
    )

    summary[
        "RMSE_improvement_vs_1x1_percent"
    ] = (
        100.0
        * (
            reference_rmse
            - summary["RMSE_mean"]
        )
        / reference_rmse
    )

    summary_path = (
        METRICS_DIR
        / "convlstm_patch_ablation_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    json_path = (
        METRICS_DIR
        / "convlstm_patch_ablation_summary.json"
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
        figsize=(9, 6)
    )

    axis.bar(
        summary["patch_label"],
        summary["MAE_mean"],
        yerr=summary["MAE_std"],
        capsize=5,
    )

    axis.set_title(
        "ConvLSTM — influence de la taille du patch"
    )

    axis.set_xlabel(
        "Taille du contexte spatial"
    )

    axis.set_ylabel(
        "MAE moyenne en °C"
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    figure_path = (
        FIGURE_DIR
        / "convlstm_patch_ablation_mae.png"
    )

    figure.savefig(
        figure_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("\n" + "=" * 80)
    print("RÉSUMÉ DE L'ABLATION SPATIALE")
    print("=" * 80)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print("\nFichiers créés :")
    print(f"- Exécutions : {runs_path}")
    print(f"- Résumé CSV : {summary_path}")
    print(f"- Résumé JSON: {json_path}")
    print(f"- Graphique  : {figure_path}")


if __name__ == "__main__":
    main()
