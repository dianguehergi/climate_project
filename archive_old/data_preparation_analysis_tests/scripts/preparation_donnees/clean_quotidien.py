import pandas as pd
import os

print("--- ÉTAPE 2 (Quotidien) : Nettoyage et dédoublonnage ---")

input_brut = "data/interim/safran_quot_brut.csv"

if not os.path.exists(input_brut):
    print(f"Erreur : Le fichier intermédiaire {input_brut} n'existe pas. Lance d'abord merge_quotidien.py.")
    exit()

# 1. Chargement de la base brute quotidienne
print("Lecture de la base brute fusionnée (cela peut prendre un moment)...")
df = pd.read_csv(input_brut)
total_rows_brut = len(df)

# 2. Suppression des doublons
print("Calcul et suppression des doublons stricts...")
df_clean = df.drop_duplicates()
total_rows_clean = len(df_clean)

num_duplicates = total_rows_brut - total_rows_clean
pct_duplicates = (num_duplicates / total_rows_brut) * 100

# 3. Bilan
print("\n================ BILAN DU NETTOYAGE QUOTIDIEN ================")
print(f"Lignes initiales (Brut)  : {total_rows_brut}")
print(f"Lignes finales (Clean)    : {total_rows_clean}")
print(f"Doublons quotidiens suppr : {num_duplicates} ({pct_duplicates:.2f}%)")
print("==============================================================\n")

# 4. Sauvegarde finale
os.makedirs("archive_old/xgboost_ridge/data/processed", exist_ok=True)
output_clean = "archive_old/xgboost_ridge/data/processed/safran_quot_clean.csv"

print(f"Sauvegarde de la base propre finale dans : {output_clean}...")
df_clean.to_csv(output_clean, index=False)

print("--- ÉTAPE 2 (Quotidien) terminée ! Base prête. ---")
