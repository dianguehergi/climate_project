import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("../data/processed/safran_mens_clean.csv")

df = df[(df["LAMBX"] == 600) & (df["LAMBY"] == 24010)]

df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m")
df = df.sort_values("DATE")

df["year"] = df["DATE"].dt.year
df["month"] = df["DATE"].dt.month

# ======================
# FEATURES
# ======================
df["T_lag_1"] = df["T"].shift(1)
df["T_lag_12"] = df["T"].shift(12)
df = df.dropna()

features = ["T_lag_1", "T_lag_12", "PRETOTM", "EVAP", "ETP", "SWI", "month"]

X = df[features]
y = df["T"]

split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ======================
# MODELS
# ======================
ridge = Ridge()
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

ridge.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# ======================
# PREDICTIONS
# ======================
ridge_pred = ridge.predict(X_test)
xgb_pred = xgb.predict(X_test)

# ======================
# METRICS
# ======================
print("\n===== RIDGE =====")
print("MAE:", mean_absolute_error(y_test, ridge_pred))
print("R2:", r2_score(y_test, ridge_pred))

print("\n===== XGBOOST =====")
print("MAE:", mean_absolute_error(y_test, xgb_pred))
print("R2:", r2_score(y_test, xgb_pred))

# ======================
# VISUAL TEST (IMPORTANT)
# ======================
plt.figure(figsize=(12,5))

plt.plot(y_test.values[:100], label="Réel", linewidth=2)
plt.plot(ridge_pred[:100], label="Ridge")
plt.plot(xgb_pred[:100], label="XGBoost")

plt.title("TEST RÉEL - Température (100 points)")
plt.legend()
plt.show()