import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ---- 1) Générer une série "climat" synthétique (mensuelle) ----
# 1960-1999 = 40 ans => 480 mois
np.random.seed(42)
n_months = 40 * 12
t = np.arange(n_months)

# tendance + saisonnalité + bruit
trend = 0.01 * t
season = 10 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 1.5, size=n_months)

y = 15 + trend + season + noise  # température synthétique

# ---- 2) Features tabulaires : lags + mois (calendrier) ----
def make_features(series, n_lags=12):
    X, Y = [], []
    for i in range(n_lags, len(series)):
        lags = series[i - n_lags:i]               # 12 derniers mois
        month = (i % 12)                          # mois 0..11
        X.append(np.concatenate([lags, [month]])) # lags + mois
        Y.append(series[i])
    return np.array(X), np.array(Y)

X, Y = make_features(y, n_lags=12)

# ---- 3) Split temporel (pas de shuffle !) ----
# on garde ~80% train, 20% test
split = int(0.8 * len(Y))
X_train, Y_train = X[:split], Y[:split]
X_test, Y_test = X[split:], Y[split:]

# ---- 4) Modèle Random Forest baseline ----
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, Y_train)
pred = rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_test, pred))

print("Random Forest baseline")
print(f"Train size: {len(Y_train)} | Test size: {len(Y_test)}")
print(f"RMSE: {rmse:.3f} °C")
