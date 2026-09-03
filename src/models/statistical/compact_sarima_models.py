"""Extrait les paramètres des modèles SARIMA complets dans un CSV compact."""

import argparse
import csv
import os
import re
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from statsmodels.tsa.statespace.sarimax import SARIMAXResults


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = PROJECT_ROOT / "results/statistical/all_9892_best_model"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_FILE = BASE_DIR / "model_parameters.csv"
PATTERN = re.compile(r"sarima_lambx_(-?\d+)_lamby_(-?\d+)\.pkl$")
FIELDS = [
    "LAMBX", "LAMBY", "order", "seasonal_order", "ar.L1", "ma.L1",
    "ma.L2", "ma.S.L12", "sigma2", "source_model",
]


def extract(path_string):
    path = Path(path_string)
    match = PATTERN.match(path.name)
    if not match:
        return "error", {"source_model": path.name, "error": "nom de fichier non reconnu"}
    try:
        result = SARIMAXResults.load(path)
        params = result.params.to_dict()
        return "ok", {
            "LAMBX": int(match.group(1)),
            "LAMBY": int(match.group(2)),
            "order": "(1, 0, 2)",
            "seasonal_order": "(0, 1, 1, 12)",
            "ar.L1": params["ar.L1"],
            "ma.L1": params["ma.L1"],
            "ma.L2": params["ma.L2"],
            "ma.S.L12": params["ma.S.L12"],
            "sigma2": params["sigma2"],
            "source_model": path.name,
        }
    except Exception as exc:
        return "error", {"source_model": path.name, "error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()

    paths = sorted(MODELS_DIR.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"Aucun modèle dans {MODELS_DIR}")

    temporary = OUTPUT_FILE.with_suffix(".csv.tmp")
    errors_file = BASE_DIR / "parameter_extraction_errors.csv"
    successes = 0
    failures = 0
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        with get_context("fork").Pool(args.workers) as pool:
            for index, (status, row) in enumerate(
                pool.imap_unordered(extract, map(str, paths), chunksize=1), start=1
            ):
                if status == "ok":
                    writer.writerow(row)
                    successes += 1
                else:
                    new_file = not errors_file.exists()
                    with errors_file.open("a", encoding="utf-8", newline="") as errors:
                        error_writer = csv.DictWriter(errors, fieldnames=["source_model", "error"])
                        if new_file:
                            error_writer.writeheader()
                        error_writer.writerow(row)
                    failures += 1
                if index == 1 or index % 100 == 0 or index == len(paths):
                    stream.flush()
                    print(f"[{index}/{len(paths)}] succès={successes} échecs={failures}", flush=True)

    if failures:
        raise RuntimeError(f"Extraction incomplète : {failures} échec(s). Les modèles ne doivent pas être supprimés.")
    os.replace(temporary, OUTPUT_FILE)
    print(f"Paramètres sauvegardés : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
