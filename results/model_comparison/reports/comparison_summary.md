# Comparaison XGBoost baseline vs XGBoost avancé

## 1. Résumé

Les deux modèles sont évalués sur le même jeu de test, composé de **3 086 304 observations mensuelles**, réparties sur **9 892 points de grille**, pour la période **2000–2025**.

Le modèle avancé utilise davantage de variables historiques, notamment des retards thermiques supplémentaires et des variables physiques retardées.

## 2. Performance globale

| Métrique | Baseline | Avancé | Gain |
|---|---:|---:|---:|
| MAE | 1.4227 °C | 1.3766 °C | 0.0460 °C (3.23 %) |
| RMSE | 1.7998 °C | 1.7195 °C | 4.46 % |
| R² | 0.9201 | 0.9270 | +0.0070 |
| Bias | 0.1714 °C | 0.1368 °C | réduction évaluée en valeur absolue |

## 3. Comparaison temporelle

- Nombre d'années comparées : **26**
- Années où le modèle avancé présente une meilleure MAE : **17/26 (65.38 %)**
- Plus forte amélioration annuelle : **2025**, gain de MAE = **0.2592 °C**
- Année la moins favorable au modèle avancé : **2013**, gain de MAE = **-0.3212 °C**

## 4. Comparaison prédiction par prédiction

- Nombre total de prédictions : **3,086,304**
- Modèle avancé meilleur : **1,580,823 (51.22 %)**
- Baseline meilleur : **1,505,480 (48.78 %)**
- Gain moyen d'erreur absolue : **0.0460 °C**

## 5. Comparaison spatiale

- Points de grille comparés : **9,892**
- Points où le modèle avancé a une meilleure MAE : **7,366/9,892 (74.46 %)**
- Gain spatial moyen de MAE : **0.0460 °C**
- Gain spatial médian de MAE : **0.0496 °C**
- Plus forte amélioration spatiale : point **(10040, 21770)**, gain de MAE = **0.2107 °C**
- Point le moins favorable au modèle avancé : **(10280, 21210)**, gain de MAE = **-0.2778 °C**

## 6. Variables les plus importantes du modèle avancé

- T_lag_12 : 0.6758
- T_lag_24 : 0.2104
- T_roll_12 : 0.0124
- T_roll_36 : 0.0117
- EVAP_lag_12 : 0.0084
- ETP_lag_1 : 0.0082
- EVAP_lag_1 : 0.0060
- month_cos : 0.0056
- T_roll_3 : 0.0046
- T_roll_24 : 0.0044

## 7. Interprétation

Un gain de MAE positif signifie que le modèle avancé réduit l'erreur par rapport au modèle baseline. De même, un gain positif de RMSE indique une amélioration, alors qu'un gain positif de R² correspond à une augmentation du pouvoir explicatif du modèle avancé.

Les comparaisons spatiales permettent d'identifier les régions où l'ajout des nouvelles variables améliore réellement les prédictions et celles où la baseline reste compétitive.

## 8. Note méthodologique

Les deux modèles sont comparés dans le même cadre one-step-ahead. Les variables retardées du test utilisent les observations historiques précédentes disponibles. Cette comparaison mesure donc l'amélioration de la prédiction mensuelle conditionnelle à l'historique observé, et non une simulation récursive autonome sur 2000–2025.
