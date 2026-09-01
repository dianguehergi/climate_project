import pandas as pd
import glob
import os

print("--- ÉTAPE 1 (Quotidien) : Fusion des données brutes ---")

# 1. Lister les fichiers dans le nouveau sous-dossier quotidien
files = glob.glob("data/raw/quotidien/sim_quot_lot*.giga.gz")
dfs = []

# 2. Charger et concaténer
for f in files:
    print(f"Lecture et chargement en mémoire de : {f}...")
    # Changement du séparateur si besoin (généralement ';' ou ',' chez Météo-France)
    temp_df = pd.read_csv(f, compression='gzip', sep=';')
    dfs.append(temp_df)

print("Fusion des 4 lots quotidiens en cours...")
df_brut = pd.concat(dfs, ignore_index=True)
print(f"Volume de la base brute fusionnée : {df_brut.shape[0]} lignes.")

# 3. Sauvegarde dans le dossier intermédiaire
os.makedirs("data/interim", exist_ok=True)
output_brut = "data/interim/safran_quot_brut.csv"

print(f"Sauvegarde du fichier brut combiné dans : {output_brut}...")
df_brut.to_csv(output_brut, index=False)

print("--- ÉTAPE 1 (Quotidien) terminée avec succès ! ---")
