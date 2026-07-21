import pandas as pd
import glob
import os

print("--- ÉTAPE 1 : Fusion des données mensuelles brutes ---")

# 1. Lister les fichiers
files = glob.glob("data/raw/mensuel/sim_mens_lot*.giga.gz")
dfs = []

# 2. Charger et concaténer
for f in files:
    print(f"Lecture et chargement en mémoire de : {f}...")
    temp_df = pd.read_csv(f, compression='gzip', sep=';')
    dfs.append(temp_df)

print("Fusion des lots en cours...")
df_brut = pd.concat(dfs, ignore_index=True)
print(f"Volume de la base brute fusionnée : {df_brut.shape[0]} lignes.")

# 3. Sauvegarde dans un dossier intermédiaire (interim)
os.makedirs("data/interim", exist_ok=True)
output_brut = "data/interim/safran_mens_brut.csv"

print(f"Sauvegarde du fichier brut combiné dans : {output_brut}...")
df_brut.to_csv(output_brut, index=False)

print("--- ÉTAPE 1 terminée avec succès ! Fichier brut disponible. ---")
