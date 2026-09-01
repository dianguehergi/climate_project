# Analyse du modèle XGBoost baseline

## 1. Performance globale

- MAE : **1.423 °C**
- RMSE : **1.800 °C**
- R² : **0.920**
- Biais : **0.171 °C**

## 2. Résidus

- Nombre de prédictions : **3,086,304**
- Résidu moyen : **0.171 °C**
- Écart-type des résidus : **1.792 °C**
- Médiane des résidus : **0.224 °C**

## 3. Performance temporelle

- Meilleure année selon la MAE : **2024** avec **0.840 °C**
- Année la plus difficile selon la MAE : **2003** avec **1.962 °C**

## 4. Analyse spatiale

- Nombre de points de grille évalués : **9,892**
- MAE spatiale moyenne : **1.423 °C**
- MAE spatiale médiane : **1.431 °C**
- MAE minimale : **0.875 °C**
- MAE maximale : **1.992 °C**
- R² spatial moyen : **0.904**
- Biais spatial moyen : **0.171 °C**

## 5. Cartes générées

- Température moyenne observée.
- Température moyenne prédite.
- Différence observé - prédit.
- MAE spatiale.
- RMSE spatiale.
- Biais spatial.
- R² spatial.

## 6. Note méthodologique

Les performances sont évaluées sur la période 2000–2025. Les variables temporelles du jeu de test utilisent les observations historiques disponibles pour calculer les retards. Il s'agit donc d'une évaluation mensuelle one-step-ahead et non d'une prévision récursive complète sur 26 ans.
