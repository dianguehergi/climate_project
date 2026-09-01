import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge

# =========================
# 1. CHARGEMENT DATA
# =========================
df = pd.read_csv("../data/processed/safran_mens_clean.csv")

df = df[(df["LAMBX"] == 600) & (df["LAMBY"] == 24010)]

df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m")
df["year"] = df["DATE"].dt.year
df["month"] = df["DATE"].dt.month

df = df.sort_values("DATE")

# =========================
# 2. FEATURES
# =========================
df["T_lag_1"] = df["T"].shift(1)
df["T_lag_12"] = df["T"].shift(12)

df = df.dropna()

features = ["T_lag_1", "T_lag_12", "PRETOTM", "EVAP", "ETP", "SWI", "month"]

X = df[features]
y = df["T"]

# =========================
# 3. SPLIT
# =========================
split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# =========================
# 4. MODEL
# =========================
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

pred = model.predict(X_test)

# =========================
# 5. METRICS
# =========================
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\n===== METRICS =====")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

# =========================
# 6. GRAPH 1 - PRÉDICTION VS RÉEL
# =========================
plt.figure()
plt.plot(y_test.values[:100], label="Réel")
plt.plot(pred[:100], label="Prédit")
plt.title("Température - Réel vs Prédit (100 points)")
plt.legend()
plt.show()

# =========================
# 7. GRAPH 2 - ERREURS
# =========================
errors = y_test.values - pred

plt.figure()
plt.hist(errors, bins=30)
plt.title("Distribution des erreurs")
plt.xlabel("Erreur (°C)")
plt.ylabel("Fréquence")
plt.show()

# =========================
# 8. ANALYSE AUTOMATIQUE
# =========================
print("\n===== ANALYSE =====")

if r2 > 0.85:
    print("✔ Modèle très performant (forte structure saisonnière détectée)")
elif r2 > 0.7:
    print("✔ Bon modèle mais améliorable")
else:
    print("⚠ Modèle faible, revoir features")

if mae < 1:
    print("✔ Erreur moyenne faible (< 1°C) -> très bon résultat climatique")
else:
    print("⚠ Erreur élevée -> améliorer features ou modèle")

print("\nConclusion :")
print("- La température est fortement auto-corrélée")
print("- Les lags (T-1, T-12) sont les variables les plus importantes")
print("- Ridge capture déjà une grande partie du signal climatique")