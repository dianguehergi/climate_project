import pandas as pd
import os

print("--- ÉTAPE 2 : Nettoyage et dédoublonnage de la base mensuelle ---")

input_brut = "data/interim/safran_mens_brut.csv"

if not os.path.exists(input_brut):
    print(f"Erreur : Le fichier intermédiaire {input_brut} n'existe pas. Lance d'abord merge_mensuel.py.")
    exit()

# 1. Chargement de la base brute fusionnée
print("Lecture de la base brute fusionnée...")
df = pd.read_csv(input_brut)
total_rows_brut = len(df)

# 2. Détection et suppression des doublons
print("Calcul et suppression des doublons en cours...")
df_clean = df.drop_duplicates()
total_rows_clean = len(df_clean)

num_duplicates = total_rows_brut - total_rows_clean
pct_duplicates = (num_duplicates / total_rows_brut) * 100

# 3. Affichage du bilan textuel pour ton équipe
print("\n================ BILAN DU NETTOYAGE ================")
print(f"Lignes initiales (Brut) : {total_rows_brut}")
print(f"Lignes finales (Clean)   : {total_rows_clean}")
print(f"Doublons supprimés       : {num_duplicates} ({pct_duplicates:.2f}%)")
print("====================================================\n")

# 4. Sauvegarde de la base finale propre
os.makedirs("data/processed", exist_ok=True)
output_clean = "data/processed/safran_mens_clean.csv"

print(f"Sauvegarde de la base propre finale dans : {output_clean}...")
df_clean.to_csv(output_clean, index=False)

print("--- ÉTAPE 2 terminée ! Votre base mensuelle est prête pour les modèles. ---")
