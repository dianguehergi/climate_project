"""
=========================================================
Construction du dataset d'entraînement pour XGBoost
Projet : Prévision climatique SAFRAN
Auteur : Yann
=========================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd


def main():

    # =====================================================
    # Définition des chemins
    # =====================================================

    BASE_DIR = Path(__file__).resolve().parent.parent

    INPUT_FILE = BASE_DIR / "data" / "processed" / "safran_mens_clean.csv"

    OUTPUT_DIR = BASE_DIR / "data" / "training"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    TRAIN_FILE = OUTPUT_DIR / "train_dataset.csv"
    TEST_FILE = OUTPUT_DIR / "test_dataset.csv"

    print("=" * 60)
    print("CONSTRUCTION DU DATASET D'ENTRAÎNEMENT")
    print("=" * 60)

    print(f"\nLecture du fichier :\n{INPUT_FILE}")

    # =====================================================
    # Chargement
    # =====================================================

    df = pd.read_csv(INPUT_FILE)

    print("\nDimensions initiales :", df.shape)

    # =====================================================
    # Date
    # =====================================================

    df["DATE"] = pd.to_datetime(df["DATE"].astype(str), format="%Y%m")

    df["year"] = df["DATE"].dt.year
    df["month"] = df["DATE"].dt.month

    # =====================================================
    # Tri
    # =====================================================

    df = df.sort_values(
        ["LAMBX", "LAMBY", "DATE"]
    ).reset_index(drop=True)

    # =====================================================
    # Feature Engineering
    # =====================================================

    grouped = df.groupby(["LAMBX", "LAMBY"])

    print("\nCréation des variables temporelles...")

    df["T_lag_1"] = grouped["T"].shift(1)

    df["T_lag_12"] = grouped["T"].shift(12)

    df["T_roll_3"] = grouped["T"].transform(
        lambda x: x.rolling(3).mean()
    )

    df["T_roll_12"] = grouped["T"].transform(
        lambda x: x.rolling(12).mean()
    )

    df["T_diff"] = df["T_lag_1"] - df["T_lag_12"]

    # =====================================================
    # Saisonnalité
    # =====================================================

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    # =====================================================
    # Suppression des NaN
    # =====================================================

    before = len(df)

    df = df.dropna().reset_index(drop=True)

    after = len(df)

    print(f"\nLignes supprimées : {before - after:,}")
    
    
    print("\nPériode couverte :")
    print(df["DATE"].min())
    print(df["DATE"].max())

    # =====================================================
    # Découpage Train/Test
    # =====================================================

    train = df[df["year"] <= 2000].copy()

    test = df[df["year"] >= 2001].copy()

    # =====================================================
    # Sauvegarde
    # =====================================================

    train.to_csv(TRAIN_FILE, index=False)

    test.to_csv(TEST_FILE, index=False)

    # =====================================================
    # Résumé
    # =====================================================

    print("\n" + "=" * 60)

    print("DATASET CONSTRUIT AVEC SUCCÈS")

    print("=" * 60)

    print("\nTrain :", train.shape)

    print("Test  :", test.shape)

    print("\nColonnes :")

    for col in train.columns:
        print("-", col)

    print("\nPremières lignes :")

    print(train.head())

    print("\nFichiers créés :")

    print(TRAIN_FILE)

    print(TEST_FILE)

    print("\nTerminé.")

    print("=" * 60)


if __name__ == "__main__":
    main()