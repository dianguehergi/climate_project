import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# 1. LOAD DATA
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
# 3. TRAIN / TEST SPLIT
# =========================
split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# =========================
# 4. MODELS
# =========================

models = {
    "Ridge": Ridge(alpha=1.0),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

results = {}

# =========================
# 5. TRAIN + EVALUATION
# =========================
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results[name] = {
        "pred": pred,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

    print(f"\n===== {name} =====")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

# =========================
# 6. COMPARISON TABLE
# =========================
print("\n===== COMPARISON =====")
for name, res in results.items():
    print(f"{name} -> MAE: {res['mae']:.3f} | RMSE: {res['rmse']:.3f} | R²: {res['r2']:.3f}")

# =========================
# 7. PLOT 1 - REAL VS PREDICTIONS
# =========================
plt.figure(figsize=(12,5))

plt.plot(y_test.values[:100], label="Réel", linewidth=2)

plt.plot(results["Ridge"]["pred"][:100], label="Ridge")
plt.plot(results["XGBoost"]["pred"][:100], label="XGBoost")

plt.title("Comparaison des prédictions (100 points)")
plt.legend()
plt.show()

# =========================
# 8. PLOT 2 - METRICS BAR CHART
# =========================
models_names = list(results.keys())
mae_values = [results[m]["mae"] for m in models_names]
rmse_values = [results[m]["rmse"] for m in models_names]
r2_values = [results[m]["r2"] for m in models_names]

plt.figure()

plt.bar(models_names, mae_values)
plt.title("MAE Comparison (lower is better)")
plt.show()

plt.figure()

plt.bar(models_names, r2_values)
plt.title("R² Comparison (higher is better)")
plt.show()

# =========================
# 9. AUTOMATIC ANALYSIS
# =========================
print("\n===== ANALYSE AUTOMATIQUE =====")

best_model = max(results.items(), key=lambda x: x[1]["r2"])[0]

print(f"Meilleur modèle : {best_model}")

if results["Ridge"]["r2"] > results["XGBoost"]["r2"]:
    print("➡ Le modèle linéaire est plus adapté (structure fortement saisonnière)")
else:
    print("➡ XGBoost capture mieux les non-linéarités")

print("\nInterprétation :")
print("- La température dépend fortement des lags (T-1, T-12)")
print("- La saisonnalité domine la dynamique climatique")
print("- Les modèles complexes ne sont pas toujours supérieurs sur données agrégées")


#Nous observons que le modèle Ridge est aussi performant voire légèrement supérieur à XGBoost, 
#ce qui indique que la dynamique de température est principalement linéaire et fortement dominée par la saisonnalité et l’autocorrélation.