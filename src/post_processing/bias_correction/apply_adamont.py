"""Quantile mapping empirique calibré sur une période historique indépendante."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import PROCESSED_DATA_DIR, RESULTS_DIR

KEYS = ["LAMBX", "LAMBY"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROCESSED_DATA_DIR / "adamont_calibration_1960_1999.csv.gz",
        help="Historique indépendant avec T_OBS et T_PRED.",
    )
    parser.add_argument(
        "--projection",
        type=Path,
        default=RESULTS_DIR / "statistical" / "sarimax_predictions_2000_2025" / "temperature_predictions_2000_2025.csv.gz",
        help="Projections contenant T_PRED.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "statistical" / "sarimax_predictions_2000_2025" / "temperature_predictions_adamont.csv.gz",
    )
    parser.add_argument("--quantiles", type=int, default=101)
    return parser.parse_args()


def add_month(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    text = result["DATE"].astype(str).str.replace(r"\.0$", "", regex=True)
    result["MONTH"] = pd.to_datetime(text.str[:6], format="%Y%m", errors="raise").dt.month
    return result


def validate(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label}: colonnes absentes: {sorted(missing)}")


def correct_group(target: pd.DataFrame, reference: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    predicted_quantiles = np.quantile(reference["T_PRED"].dropna(), probabilities)
    observed_quantiles = np.quantile(reference["T_OBS"].dropna(), probabilities)
    corrected = target.copy()
    corrected["T_PRED_ADAMONT"] = np.interp(
        corrected["T_PRED"], predicted_quantiles, observed_quantiles
    )
    return corrected


def main() -> None:
    args = parse_args()
    calibration = add_month(pd.read_csv(args.calibration))
    projection = add_month(pd.read_csv(args.projection))
    validate(calibration, {*KEYS, "DATE", "T_OBS", "T_PRED"}, "calibration")
    validate(projection, {*KEYS, "DATE", "T_PRED"}, "projection")

    probabilities = np.linspace(0.0, 1.0, args.quantiles)
    references = {
        key: group for key, group in calibration.groupby([*KEYS, "MONTH"], sort=False)
    }
    chunks = []
    for key, group in projection.groupby([*KEYS, "MONTH"], sort=False):
        reference = references.get(key)
        if reference is None or len(reference.dropna(subset=["T_OBS", "T_PRED"])) < 10:
            raise ValueError(f"Calibration insuffisante pour le groupe {key}")
        chunks.append(correct_group(group, reference, probabilities))

    output = pd.concat(chunks, ignore_index=True).drop(columns="MONTH")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, compression="infer")
    print(f"Correction écrite dans {args.output} ({len(output):,} lignes)")


if __name__ == "__main__":
    main()
