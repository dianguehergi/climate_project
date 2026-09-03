import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# Charger les données
df = pd.read_csv("../data/processed/safran_mens_clean.csv")

# Filtrer une cellule
df = df[(df["LAMBX"] == 600) & (df["LAMBY"] == 24010)]

# Date
df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m")
df["year"] = df["DATE"].dt.year
df["month"] = df["DATE"].dt.month

# Trier
df = df.sort_values("DATE")

# Features temporelles
df["T_lag_1"] = df["T"].shift(1)
df["T_lag_12"] = df["T"].shift(12)

df = df.dropna()

# Features
features = ["T_lag_1", "T_lag_12", "PRETOTM", "EVAP", "ETP", "SWI", "month"]

X = df[features]
y = df["T"]

# Split temporel
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Modèle Ridge
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Prédictions
pred = model.predict(X_test)

# Évaluation
print("MAE:", mean_absolute_error(y_test, pred))
print("R2:", r2_score(y_test, pred))

#L'erreur moyenne absolue en pourcentage est de 1°c Plus la RMSE et la MAPE sont faibles, meilleur est le modèle.

# C'est une très bonne chose 