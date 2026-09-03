import pandas as pd
import os
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parents[3] / "data")
RAW_DIR = os.path.join(BASE_DIR, "raw")
INTERIM_DIR = os.path.join(BASE_DIR, "interim")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("--- 1. MERGE DES FICHIERS QUOTIDIENS BRUTS ---")
raw_files = [
    os.path.join(RAW_DIR, "SIM_quotidienne_2000_2009.csv"),
    os.path.join(RAW_DIR, "SIM_quotidienne_2010_2019.csv"),
    os.path.join(RAW_DIR, "SIM_quotidienne_2020_2026.csv")
]

dfs = []
for file in raw_files:
    if os.path.exists(file):
        print(f"Chargement de {os.path.basename(file)}...")
        try:
            df_temp = pd.read_csv(file, compression='gzip', sep=None, engine='python', encoding='utf-8')
        except Exception:
            try:
                df_temp = pd.read_csv(file, compression='gzip', sep=None, engine='python', encoding='latin1')
            except Exception:
                try:
                    df_temp = pd.read_csv(file, sep=None, engine='python', encoding='utf-8')
                except Exception:
                    df_temp = pd.read_csv(file, sep=None, engine='python', encoding='latin1')
        dfs.append(df_temp)
    else:
        print(f"⚠️ Fichier non trouvé : {file}")

df_merged = pd.concat(dfs, ignore_index=True)
print(f"Total lignes fusionnées : {len(df_merged):,}")

# Sauvegarde dans interim
interim_path = os.path.join(INTERIM_DIR, "SIM_quotidienne_2000_2026_merged.csv")
df_merged.to_csv(interim_path, index=False)
print(f"✅ Fichier fusionné sauvegardé dans : {interim_path}")

print("\n--- 2. NETTOYAGE ET FILTRAGE QUOTIDIEN (Arrêt au 31/12/2025) ---")

date_cols = [c for c in df_merged.columns if 'date' in c.lower() or 'DATE' in c or 'time' in c.lower()]
if date_cols:
    date_col = date_cols[0]
    print(f"Colonne de date identifiée : '{date_col}'")

    # Format YYYYMMDD ou standard ISO pour les données quotidiennes
    df_merged['DATE_DT'] = pd.to_datetime(df_merged[date_col].astype(str), format='%Y%m%d', errors='coerce')
    if df_merged['DATE_DT'].isna().all():
        df_merged['DATE_DT'] = pd.to_datetime(df_merged[date_col], errors='coerce')

    df_clean = df_merged.drop_duplicates().copy()
    
    # Filtrage strict <= 31 Décembre 2025
    df_clean = df_clean[df_clean['DATE_DT'] <= '2025-12-31']
    df_clean = df_clean.drop(columns=['DATE_DT'])

    print(f"Lignes retenues (2000-2025) : {len(df_clean):,}")
else:
    print("⚠️ Attention : Colonne de date non détectée.")
    df_clean = df_merged

# Sauvegarde dans processed
processed_path = os.path.join(PROCESSED_DIR, "SIM_quotidienne_2000_2025_cleaned.csv")
df_clean.to_csv(processed_path, index=False)
print(f"\n🎉 SUCCÈS : Dataset quotidien propre généré dans : {processed_path}")
