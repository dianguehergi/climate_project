"""
Fusion des données mensuelles SAFRAN 1960-1999
et SIM 2000-2025.

Sortie :
data/merged/monthly_1960_2025.csv
"""

from pathlib import Path
import pandas as pd


def parse_monthly_date(series: pd.Series) -> pd.Series:
    """
    Convertit une colonne DATE mensuelle en datetime.

    Formats acceptés :
    - 196001
    - "196001"
    - "1960-01-01"
    """
    date_as_string = series.astype(str).str.strip()

    parsed = pd.to_datetime(
        date_as_string,
        format="%Y%m",
        errors="coerce",
    )

    missing_mask = parsed.isna()

    if missing_mask.any():
        parsed.loc[missing_mask] = pd.to_datetime(
            date_as_string.loc[missing_mask],
            errors="coerce",
        )

    return parsed


def main() -> None:
    # Le script est situé dans climate_project/scripts/
    project_dir = Path(__file__).resolve().parent.parent

    safran_file = (
        project_dir
        / "archive_old"
        / "Yann"
        / "data"
        / "processed"
        / "safran_mens_clean.csv"
    )

    sim_file = (
        project_dir
        / "data"
        / "processed"
        / "datasets"
        / "SIM_mensuelle_2000_2025_cleaned.csv"
    )

    output_dir = project_dir / "data" / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "monthly_1960_2025.csv"

    print("=" * 70)
    print("FUSION DES DONNÉES MENSUELLES 1960-2025")
    print("=" * 70)

    # ---------------------------------------------------------
    # Vérification des fichiers
    # ---------------------------------------------------------

    if not safran_file.exists():
        raise FileNotFoundError(
            f"Fichier SAFRAN introuvable :\n{safran_file}"
        )

    if not sim_file.exists():
        raise FileNotFoundError(
            f"Fichier SIM introuvable :\n{sim_file}"
        )

    print(f"\nFichier SAFRAN :\n{safran_file}")
    print(f"\nFichier SIM :\n{sim_file}")

    # ---------------------------------------------------------
    # Chargement
    # ---------------------------------------------------------

    print("\nChargement de SAFRAN...")
    safran = pd.read_csv(safran_file)

    print("Chargement de SIM...")
    sim = pd.read_csv(sim_file)

    print(f"\nDimensions SAFRAN : {safran.shape}")
    print(f"Dimensions SIM    : {sim.shape}")

    # ---------------------------------------------------------
    # Vérification des colonnes
    # ---------------------------------------------------------

    safran_columns = list(safran.columns)
    sim_columns = list(sim.columns)

    if safran_columns != sim_columns:
        missing_in_sim = sorted(set(safran_columns) - set(sim_columns))
        missing_in_safran = sorted(set(sim_columns) - set(safran_columns))

        raise ValueError(
            "Les colonnes des deux fichiers ne correspondent pas.\n"
            f"Absentes dans SIM : {missing_in_sim}\n"
            f"Absentes dans SAFRAN : {missing_in_safran}"
        )

    print("\nColonnes compatibles : oui")

    # ---------------------------------------------------------
    # Conversion des dates
    # ---------------------------------------------------------

    safran["DATE"] = parse_monthly_date(safran["DATE"])
    sim["DATE"] = parse_monthly_date(sim["DATE"])

    invalid_safran_dates = int(safran["DATE"].isna().sum())
    invalid_sim_dates = int(sim["DATE"].isna().sum())

    if invalid_safran_dates > 0 or invalid_sim_dates > 0:
        raise ValueError(
            "Certaines dates n'ont pas pu être converties.\n"
            f"SAFRAN : {invalid_safran_dates}\n"
            f"SIM : {invalid_sim_dates}"
        )

    print("\nPériode SAFRAN :")
    print(f"{safran['DATE'].min():%Y-%m} à {safran['DATE'].max():%Y-%m}")

    print("\nPériode SIM :")
    print(f"{sim['DATE'].min():%Y-%m} à {sim['DATE'].max():%Y-%m}")

    # ---------------------------------------------------------
    # Identification de la source
    # ---------------------------------------------------------

    safran["SOURCE"] = "SAFRAN"
    sim["SOURCE"] = "SIM"

    # ---------------------------------------------------------
    # Fusion
    # ---------------------------------------------------------

    print("\nFusion des deux datasets...")

    merged = pd.concat(
        [safran, sim],
        ignore_index=True,
        sort=False,
    )

    merged = merged.sort_values(
        ["LAMBX", "LAMBY", "DATE"]
    ).reset_index(drop=True)

    print(f"Dimensions après fusion : {merged.shape}")

    # ---------------------------------------------------------
    # Vérification des doublons
    # ---------------------------------------------------------

    duplicate_mask = merged.duplicated(
        subset=["LAMBX", "LAMBY", "DATE"],
        keep=False,
    )

    duplicate_count = int(duplicate_mask.sum())

    print(
        "\nNombre de lignes appartenant à des doublons "
        f"point-date : {duplicate_count:,}"
    )

    if duplicate_count > 0:
        duplicate_file = (
            output_dir
            / "monthly_1960_2025_duplicates.csv"
        )

        merged.loc[duplicate_mask].to_csv(
            duplicate_file,
            index=False,
        )

        print(
            "Les doublons ont été enregistrés dans :"
            f"\n{duplicate_file}"
        )

        print(
            "\nSuppression des doublons en conservant "
            "la dernière observation..."
        )

        # Comme SIM constitue la série récente, la dernière ligne
        # est conservée en cas de chevauchement, notamment en 2000.
        merged = merged.drop_duplicates(
            subset=["LAMBX", "LAMBY", "DATE"],
            keep="last",
        ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Vérifications temporelles
    # ---------------------------------------------------------

    min_date = merged["DATE"].min()
    max_date = merged["DATE"].max()

    print("\nPériode finale :")
    print(f"{min_date:%Y-%m} à {max_date:%Y-%m}")

    number_of_points = (
        merged[["LAMBX", "LAMBY"]]
        .drop_duplicates()
        .shape[0]
    )

    print(f"\nNombre de points de grille : {number_of_points:,}")

    observations_per_point = (
        merged.groupby(["LAMBX", "LAMBY"])
        .size()
    )

    print("\nObservations mensuelles par point :")
    print(
        f"minimum = {observations_per_point.min():,}, "
        f"médiane = {int(observations_per_point.median()):,}, "
        f"maximum = {observations_per_point.max():,}"
    )

    # ---------------------------------------------------------
    # Vérification des dates manquantes
    # ---------------------------------------------------------

    merged["period"] = merged["DATE"].dt.to_period("M")

    previous_period = (
        merged.groupby(["LAMBX", "LAMBY"])["period"]
        .shift(1)
    )

    month_gap = (
        merged["period"].astype("int64")
        - previous_period.astype("int64")
    )

    gap_mask = previous_period.notna() & (month_gap != 1)
    gap_count = int(gap_mask.sum())

    print(
        "\nNombre de ruptures temporelles détectées : "
        f"{gap_count:,}"
    )

    if gap_count > 0:
        gaps_file = output_dir / "monthly_1960_2025_gaps.csv"

        gaps = merged.loc[
            gap_mask,
            ["LAMBX", "LAMBY", "DATE", "SOURCE"]
        ].copy()

        gaps["previous_period"] = previous_period.loc[gap_mask].astype(str)
        gaps["month_gap"] = month_gap.loc[gap_mask].astype(int)

        gaps.to_csv(gaps_file, index=False)

        print(
            "Les ruptures ont été enregistrées dans :"
            f"\n{gaps_file}"
        )

    merged = merged.drop(columns=["period"])

    # ---------------------------------------------------------
    # Valeurs manquantes
    # ---------------------------------------------------------

    missing_values = (
        merged.isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing_values = missing_values[missing_values > 0]

    print("\nValeurs manquantes :")

    if missing_values.empty:
        print("Aucune valeur manquante.")
    else:
        print(missing_values)

    # ---------------------------------------------------------
    # Sauvegarde
    # ---------------------------------------------------------

    print("\nSauvegarde du dataset fusionné...")

    merged.to_csv(
        output_file,
        index=False,
        date_format="%Y-%m-%d",
    )

    print("\n" + "=" * 70)
    print("FUSION TERMINÉE AVEC SUCCÈS")
    print("=" * 70)

    print(f"\nFichier créé :\n{output_file}")
    print(f"\nDimensions finales : {merged.shape}")
    print(f"Période finale      : {min_date:%Y-%m} à {max_date:%Y-%m}")
    print(f"Points de grille    : {number_of_points:,}")


if __name__ == "__main__":
    main()