from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PRED_PATH = Path(
    "ndambo/outputs/predictions/"
    "forecast_2000_2025_autoregressive_masked_5x5.npy"
)

DATES_PATH = Path(
    "ndambo/outputs/predictions/"
    "forecast_dates_2000_2025.npy"
)

GRID_PATH = Path(
    "ndambo/data/processed/"
    "safran_many_centers_1x1_9892centers_temperature_family.npz"
)

OUTPUT_PATH = Path(
    "ndambo/outputs/predictions/"
    "forecast_2000_2025_all_points.csv.gz"
)


def main() -> None:
    predictions = np.load(PRED_PATH, mmap_mode="r")
    dates = np.load(DATES_PATH, allow_pickle=True)

    archive = np.load(GRID_PATH)
    centers = archive["centers"]

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    print("Prédictions :", predictions.shape)
    print("Dates       :", len(dates))
    print("Points      :", centers.shape)
    print("Sortie      :", OUTPUT_PATH)

    first_write = True

    for day_index, date in enumerate(dates):
        df = pd.DataFrame(
            {
                "DATE": str(date),
                "center_index": np.arange(centers.shape[0]),
                "LAMBX": centers[:, 0].astype(int),
                "LAMBY": centers[:, 1].astype(int),
                "PRED_T_C": predictions[day_index, :],
            }
        )

        df.to_csv(
            OUTPUT_PATH,
            mode="wt" if first_write else "at",
            index=False,
            header=first_write,
            compression="gzip",
        )

        first_write = False

        if (day_index + 1) % 250 == 0:
            print(f"Jours exportés : {day_index + 1:,}/{len(dates):,}")

    print("CSV complet créé :", OUTPUT_PATH)


if __name__ == "__main__":
    main()
