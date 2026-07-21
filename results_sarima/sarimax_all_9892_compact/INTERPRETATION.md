# Évaluation SARIMAX ETP+SWI sur 9 892 points

## Résultats

- RMSE moyenne / médiane : **1.278 / 1.278 °C**.
- MAE moyenne : **1.016 °C**.
- Biais moyen : **-0.133 °C**.
- RMSE minimale / maximale : **0.879 / 1.936 °C**.
- RMSE ≤ 1,25 °C : **39.1 %** des points.
- RMSE ≤ 1,50 °C : **97.9 %** des points.
- |biais| ≤ 0,50 °C : **89.7 %** des points.

## Variables exogènes

- Coefficient ETP moyen standardisé : **3.703**, positif sur **100.0 %** des points.
- Coefficient SWI moyen standardisé : **0.318**, positif sur **88.5 %** des points.
- ETP est le déterminant linéaire dominant, ce qui confirme le rapport méthodologique. La structure et le signe de SWI varient davantage spatialement.

## Comparaison avec SARIMA

- SARIMA : RMSE **1.445 °C**, MAE **1.166 °C**.
- SARIMAX ETP+SWI : RMSE **1.278 °C**, MAE **1.016 °C**.
- Réduction de RMSE : **11.5 %**.
- SARIMAX gagne sur **95.8 %** des points.

La comparaison est strictement équitable : mêmes 9 892 points, même fichier mensuel, même entraînement 1970–1995 et même test 1996–1999. La différence mesurée isole donc l'apport de la structure SARIMAX et des variables ETP/SWI, sous réserve des ordres propres à chaque modèle.

## Conclusion défendable

Ce SARIMAX est un **très bon modèle statistique mensuel à grande échelle** : couverture complète, aucun échec, paramètres interprétables et amélioration nette face au SARIMA. Il est actuellement le meilleur modèle mensuel directement évalué à cette échelle dans le projet.

On ne peut pas encore dire qu'il bat tous les modèles « du marché ». TCN, ConvLSTM, XGBoost, PatchTST ou les modèles fondation doivent être testés sur les mêmes 9 892 points, la même cible mensuelle et la même période. Les chiffres TCN/ConvLSTM déjà présents concernent surtout du quotidien J+1 et ne constituent donc pas un classement comparable.

## Limites

- Données terminant en 1999 : validation récente nécessaire avant déploiement.
- Les valeurs futures d'ETP et SWI doivent être disponibles pour prévoir avec SARIMAX.
- Les extrêmes et intervalles de confiance doivent encore être évalués globalement.
