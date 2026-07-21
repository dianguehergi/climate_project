# Rapport SARIMA — prévisions de température 2000–2025

## Protocole

- Modèle : SARIMA(1,0,2)(0,1,1,12), un modèle par point.
- Apprentissage final : 1960–1999 sur 9 892 points.
- Prévision : 2000–2025, soit 312 mois par point.
- Mesure réelle de performance : backtest 1996–1999 après apprentissage sur 1960–1995.
- Référence : prévision saisonnière naïve, qui reprend la température du même mois un an plus tôt.

## Résultats du backtest

- RMSE SARIMA moyen : **1.445 °C** ; médian : **1.442 °C**.
- MAE SARIMA moyen : **1.166 °C** ; médian : **1.166 °C**.
- Biais moyen : **-0.198 °C**.
- 90 % centraux des points : RMSE entre **1.187 et 1.681 °C**.
- **0.4 %** des points ont un RMSE inférieur à 1 °C ; **65.5 %** sont sous 1,5 °C.
- Référence naïve : RMSE moyen **2.059 °C**, MAE moyen **1.595 °C**.
- SARIMA bat le naïf sur le RMSE pour **100.0 %** des points.
- Gain RMSE médian par rapport au naïf : **30.4 %**.

## Projections 2000–2025

- Température moyenne prévue, tous points et mois : **10.18 °C**.
- Moyenne 2000–2004 : **10.18 °C**.
- Moyenne 2021–2025 : **10.22 °C**.
- Écart fin moins début de projection : **+0.035 °C**.
- Largeur moyenne des intervalles prédictifs ponctuels : **6.32 °C** en 2000, contre **6.78 °C** en 2025.

## Interprétation

Le RMSE et le MAE du backtest quantifient une vraie prévision hors échantillon. Le biais indique si le modèle surestime (positif) ou sous-estime (négatif) systématiquement la température. La comparaison au naïf saisonnier est essentielle : un modèle saisonnier n'est utile que s'il apporte un gain face à cette référence simple.

Les valeurs 2000–2025 sont des **projections**, pas une validation : aucune température observée postérieure à 1999 n'est présente dans le projet. L'élargissement des intervalles à 95 % montre que l'incertitude augmente avec l'horizon. De plus, ce SARIMA prolonge la dynamique historique ; sans variables ou scénarios climatiques externes, il ne doit pas être présenté comme un modèle de projection du changement climatique à long terme.

Le point représentatif de la figure 5 est (3480, 23450), avec RMSE=1.442 °C et MAE=1.178 °C sur le backtest.
