# Méthodologie et protocoles

## Découpage temporel

| Piste | Apprentissage/estimation | Évaluation | Remarque |
|---|---|---|---|
| Benchmark SARIMA/SARIMAX | 1960–1995 | 1996–1999 | comparaison sur les mêmes 9 892 points |
| Projections SARIMA finales | 1960–1999 | 2000–2025 | réestimation avant projection |
| Auxiliaires ETP/SWI | 1970–1999 | horizon 2000–2025 | variables futures extrapolées |
| XGBoost | 1960–1999 | 2000–2025 | one-step avec retards observés |
| ConvLSTM masqué | avant 1993 selon split interne | 1993–1999 | prévision J+1 sur entrées réelles |
| ConvLSTM rollout | 1960–1999 | 2000–2025 | réinjection auto-régressive |

Cette table tranche l’apparente contradiction 1960/1970 : **1960–1995 est la fenêtre du benchmark statistique commun** ; **1970–1999 ne décrit que les modèles auxiliaires ETP/SWI**. Les chiffres issus de protocoles différents ne doivent pas être classés directement.

## Correction inspirée d’ADAMONT

L’implémentation réalise un quantile mapping empirique par point de grille. Elle recale la distribution des sorties sur une distribution de référence. Comme le dépôt ne contient pas les champs de circulation nécessaires au conditionnement par régime météorologique, l’appellation « inspirée d’ADAMONT » est scientifiquement plus exacte que « ADAMONT complet ».

## Prévention des fuites

Les observations de validation ne doivent intervenir ni dans l’apprentissage, ni dans la sélection d’hyperparamètres, ni dans l’ajustement de la transformation de biais. Toute expérience qui utilise des retards observés 2000–2025 doit être étiquetée *one-step*, et non *projection en aveugle*.

## Métriques

Les résultats publient MAE, RMSE, biais et R² lorsque ce dernier est pertinent. Ils doivent toujours être accompagnés de la période, du nombre de points et du protocole d’inférence.
