import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("../data/processed/safran_mens_clean.csv")

df = df[(df["LAMBX"] == 600) & (df["LAMBY"] == 24010)]

df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m")
df = df.sort_values("DATE")

df["year"] = df["DATE"].dt.year
df["month"] = df["DATE"].dt.month

# =========================
# 2. FEATURE ENGINEERING (AMÉLIORATION)
# =========================

# Lags
df["T_lag_1"] = df["T"].shift(1)
df["T_lag_12"] = df["T"].shift(12)

# Rolling means (très important climat)
df["T_roll_3"] = df["T"].rolling(3).mean()
df["T_roll_12"] = df["T"].rolling(12).mean()

# Variations
df["T_diff"] = df["T_lag_1"] - df["T_lag_12"]

# Saisonnalité cyclique (TRÈS IMPORTANT)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Nettoyage
df = df.dropna()

# =========================
# 3. FEATURES FINAL
# =========================
features = [
    "T_lag_1",
    "T_lag_12",
    "T_roll_3",
    "T_roll_12",
    "T_diff",
    "month_sin",
    "month_cos",
    "PRETOTM",
    "EVAP",
    "ETP",
    "SWI"
]

X = df[features]
y = df["T"]

# =========================
# 4. SPLIT
# =========================
split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# =========================
# 5. MODEL XGBOOST
# =========================
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 6. PREDICTION
# =========================
pred = model.predict(X_test)

# =========================
# 7. METRICS
# =========================
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("\n===== XGBOOST IMPROVED =====")
print("MAE:", mae)
print("R2:", r2)

# =========================
# 8. FEATURE IMPORTANCE
# =========================
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n===== FEATURE IMPORTANCE =====")
print(importance)