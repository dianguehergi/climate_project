import pandas as pd
import os

print("--- ANALYSE DES TEMPÉRATURES EXTRÊMES - MENSUEL ---")
data_path = "data/processed/safran_mens_clean.csv"

if not os.path.exists(data_path):
    print(f"Erreur : Le fichier {data_path} n'existe pas.")
    exit()

print("Lecture des données...")
df = pd.read_csv(data_path, nrows=5)
colonnes_disponibles = df.columns.tolist()

nom_colonne_tmin = 'T'  # Change par le bon nom si nécessaire

if nom_colonne_tmin not in colonnes_disponibles:
    print(f"\n❌ Erreur. Les colonnes sont : {colonnes_disponibles}")
    exit()

df = pd.read_csv(data_path)
df_extremes = df[df[nom_colonne_tmin] < -10]

print(f"\nRésultat : {len(df_extremes)} lignes trouvées.")

if len(df_extremes) > 0:
    print(df_extremes.sort_values(by=nom_colonne_tmin).head(15))
    
    # --- SAUVEGARDE EN CSV ICI ---
    output_csv = "data/processed/anomalies_mensuelles.csv"
    df_extremes.to_csv(output_csv, index=False)
    print(f"\n✅ Succès : Fichier CSV sauvegardé dans : {output_csv}")
