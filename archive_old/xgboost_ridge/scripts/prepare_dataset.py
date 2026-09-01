import pandas as pd

# 1. Charger les données
df = pd.read_csv("../data/processed/safran_mens_clean.csv")

# 2. Filtrer une seule cellule (réduction complexité)
df = df[(df["LAMBX"] == 600) & (df["LAMBY"] == 24010)]

# 3. Gestion de la date
df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m")
df["year"] = df["DATE"].dt.year
df["month"] = df["DATE"].dt.month

# 4. Trier chronologiquement
df = df.sort_values("DATE")

# 5. Feature engineering (lags)
df["T_lag_1"] = df["T"].shift(1)
df["T_lag_12"] = df["T"].shift(12)

# 6. Nettoyage (suppression NaN causés par les lags)
df = df.dropna()

# 7. Features utilisées pour le modèle
features = [
    "T_lag_1",
    "T_lag_12",
    "PRETOTM",
    "EVAP",
    "ETP",
    "SWI",
    "month"
]

# 8. Définition X / y
X = df[features]
y = df["T"]

# 9. Split temporel CORRIGÉ (car 2000–2025 n'existe pas dans ta sélection)
split_idx = int(len(df) * 0.8)

train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

X_train = train[features]
y_train = train["T"]

X_test = test[features]
y_test = test["T"]

# 10. Vérification
print("Shape X_train:", X_train.shape)
print("Shape X_test:", X_test.shape)