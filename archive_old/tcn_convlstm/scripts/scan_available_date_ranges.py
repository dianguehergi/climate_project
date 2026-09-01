from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("data/processed")


def main() -> None:
    files = sorted(ROOT.rglob("*.csv"))

    print("=" * 80)
    print("SCAN DES PÉRIODES DISPONIBLES")
    print("=" * 80)

    for path in files:
        try:
            header = pd.read_csv(path, nrows=0)
            columns = list(header.columns)

            date_column = None

            for candidate in ["DATE", "date", "Date", "time", "TIME"]:
                if candidate in columns:
                    date_column = candidate
                    break

            if date_column is None:
                print(f"{path} : pas de colonne date détectée")
                continue

            mins = []
            maxs = []
            rows = 0

            for chunk in pd.read_csv(
                path,
                usecols=[date_column],
                chunksize=1_000_000,
            ):
                dates = pd.to_datetime(
                    chunk[date_column].astype(str),
                    errors="coerce",
                )

                mins.append(dates.min())
                maxs.append(dates.max())
                rows += len(chunk)

            print("-" * 80)
            print(f"Fichier : {path}")
            print(f"Lignes  : {rows:,}")
            print(f"Date min: {min(mins)}")
            print(f"Date max: {max(maxs)}")

        except Exception as error:
            print("-" * 80)
            print(f"Fichier : {path}")
            print(f"Erreur  : {error}")


if __name__ == "__main__":
    main()
