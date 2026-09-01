import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Charger données
df = pd.read_csv("../data/processed/safran_mens_clean.csv")

# 2. Filtrer une cellule spatiale
df = df[(df["LAMBX"] == 600) & (df["LAMBY"] == 24010)]

# 3. Date
df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m")
df["year"] = df["DATE"].dt.year
df["month"] = df["DATE"].dt.month

# 4. Trier
df = df.sort_values("DATE")

# 5. Features temporelles
df["T_lag_1"] = df["T"].shift(1)
df["T_lag_12"] = df["T"].shift(12)

df = df.dropna()

# 6. Features
features = [
    "T_lag_1",
    "T_lag_12",
    "PRETOTM",
    "EVAP",
    "ETP",
    "SWI",
    "month"
]

X = df[features]
y = df["T"]

# 7. Split temporel (80/20)
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# 8. Modèle XGBoost
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# 9. Prédictions
pred = model.predict(X_test)

# 10. Évaluation
print("MAE:", mean_absolute_error(y_test, pred))
print("R2:", r2_score(y_test, pred))