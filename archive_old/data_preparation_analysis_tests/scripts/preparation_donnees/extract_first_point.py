import os
import glob
import pandas as pd

RAW_PATTERN = "data/raw/quotidien/sim_quot_lot*.giga.gz"
OUTPUT = "data/processed/first_point_T_daily.csv"

os.makedirs("data/processed", exist_ok=True)

files = sorted(glob.glob(RAW_PATTERN))

if not files:
    raise FileNotFoundError(f"Aucun fichier trouvé avec le pattern : {RAW_PATTERN}")

print("--- Extraction d’un point de grille SAFRAN ---")
print("Fichiers trouvés :")
for f in files:
    print("-", f)

# --------------------------------------------------
# 1. Identifier le premier point de grille
# --------------------------------------------------
first_file = files[0]

sample = pd.read_csv(
    first_file,
    compression="gzip",
    sep=";",
    usecols=["LAMBX", "LAMBY"],
    nrows=1
)

px = sample["LAMBX"].iloc[0]
py = sample["LAMBY"].iloc[0]

print("\nPoint de grille choisi :")
print("LAMBX =", px)
print("LAMBY =", py)

# --------------------------------------------------
# 2. Extraire ce point dans tous les lots
# --------------------------------------------------
all_parts = []

for f in files:
    print(f"\nLecture du fichier : {f}")

    chunk_id = 0
    rows_kept = 0

    for chunk in pd.read_csv(
        f,
        compression="gzip",
        sep=";",
        usecols=["LAMBX", "LAMBY", "DATE", "T"],
        chunksize=500_000,
        low_memory=False
    ):
        chunk_id += 1

        # garder uniquement le point choisi
        part = chunk[
            (chunk["LAMBX"] == px) &
            (chunk["LAMBY"] == py)
        ].copy()

        if not part.empty:
            rows_kept += len(part)
            all_parts.append(part[["DATE", "T"]])

        if chunk_id % 20 == 0:
            print(f"  chunks lus : {chunk_id} | lignes gardées : {rows_kept}")

    print(f"Total lignes gardées pour ce fichier : {rows_kept}")

# --------------------------------------------------
# 3. Fusionner uniquement la série du point choisi
# --------------------------------------------------
if not all_parts:
    raise ValueError("Aucune donnée trouvée pour le point choisi.")

df_point = pd.concat(all_parts, ignore_index=True)

# --------------------------------------------------
# 4. Conversion propre des dates SAFRAN
# --------------------------------------------------
df_point["DATE"] = pd.to_datetime(
    df_point["DATE"].astype(str).str.replace(".0", "", regex=False),
    format="%Y%m%d",
    errors="coerce"
)

df_point["T"] = pd.to_numeric(df_point["T"], errors="coerce")

df_point = df_point.dropna(subset=["DATE", "T"])
df_point = df_point.sort_values("DATE")

# --------------------------------------------------
# 5. Sécuriser : une seule valeur par date
# --------------------------------------------------
serie = (
    df_point
    .groupby("DATE", as_index=False)["T"]
    .mean()
    .sort_values("DATE")
)

# --------------------------------------------------
# 6. Vérifications
# --------------------------------------------------
print("\n--- Vérifications finales ---")
print("Nombre de lignes :", len(serie))
print("Période :", serie["DATE"].min(), "→", serie["DATE"].max())
print("Années présentes :")
print(serie["DATE"].dt.year.value_counts().sort_index())

expected_start = pd.Timestamp("1960-01-01")
expected_end = pd.Timestamp("1999-12-31")

if serie["DATE"].min() != expected_start:
    print("Attention : la série ne commence pas exactement au 1960-01-01.")

if serie["DATE"].max() != expected_end:
    print("Attention : la série ne finit pas exactement au 1999-12-31.")

# --------------------------------------------------
# 7. Sauvegarde finale
# --------------------------------------------------
serie.to_csv(
    OUTPUT,
    index=False,
    date_format="%Y-%m-%d"
)

print("\nFichier créé avec succès :")
print(OUTPUT)