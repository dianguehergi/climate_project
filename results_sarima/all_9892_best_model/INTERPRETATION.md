# Évaluation du SARIMA sur 9 892 points

## Résultats principaux

- RMSE moyenne : **1.445 °C** ; médiane : **1.441 °C**.
- MAE moyenne : **1.166 °C**.
- Biais moyen : **-0.198 °C** : le modèle sous-estime légèrement la température.
- RMSE minimale / maximale : **0.877 / 2.148 °C**.
- Points avec RMSE ≤ 1,25 °C : **9.0 %**.
- Points avec RMSE ≤ 1,50 °C : **65.4 %**.
- Points avec |biais| ≤ 0,50 °C : **89.4 %**.

## Comparaison équitable avec SARIMAX

Sur les mêmes 500 points et la même séparation 1960–1995 / 1996–1999 :

- SARIMA : RMSE **1.437 °C**, MAE **1.160 °C**.
- SARIMAX : RMSE **1.312 °C**, MAE **1.031 °C**.
- Gain moyen de RMSE de SARIMAX face à SARIMA : **8.7 %**.
- SARIMAX gagne point par point dans **87.0 %** des cas.

## Peut-on dire que le modèle est bon ?

Oui, on peut dire que le modèle est **solide comme référence mensuelle spatiale** : il a été évalué hors échantillon sur les 9 892 points, sans échec, avec une RMSE moyenne de 1.445 °C. En revanche, on ne peut pas dire qu'il est le meilleur modèle du projet : sur les 500 points comparables, SARIMAX réduit la RMSE moyenne de 8.7 % et gagne sur 87.0 % des points.

On ne peut pas encore affirmer qu'il est meilleur que les modèles modernes « du marché » (TCN, ConvLSTM, PatchTST, Prophet, XGBoost, modèles fondation) : les résultats locaux TCN/ConvLSTM prédisent le **quotidien à J+1**, souvent sur un seul point, tandis que ce SARIMA prédit **48 mois** sur toute la grille. Une comparaison scientifique exige exactement la même cible, le même horizon, les mêmes points et les mêmes dates de test.

## Limites à annoncer

- Le biais moyen négatif indique une sous-estimation d'environ 0.20 °C.
- Les paramètres SARIMA sont identiques en structure pour tous les points ; seule leur estimation varie.
- Les extrêmes climatiques et les intervalles de confiance ne sont pas encore évalués globalement.
- Les données s'arrêtent en 1999 : une validation sur une période plus récente est nécessaire pour parler de déploiement actuel.
