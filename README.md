# Climate Project — SAFRAN

Projet de prévision de température sur les 9 892 points de la grille SAFRAN.
Le dépôt regroupe plusieurs familles de modèles : SARIMA, SARIMAX, TCN,
ConvLSTM, Ridge et XGBoost.

## Travail SARIMA / SARIMAX

Les scripts principaux sont organisés dans `hergi/scripts/` :

- `arima_sarima/` : entraînement et évaluation ARIMA/SARIMA ;
- `sarimax/` : modèles SARIMAX et variables exogènes ETP/SWI ;
- traitement parallèle et reprenable sur les 9 892 points ;
- sauvegarde compacte des paramètres estimés ;
- génération des cartes, distributions et rapports d'interprétation.

La comparaison équitable utilise :

- entraînement : 1970–1995 ;
- test hors échantillon : 1996–1999 ;
- mêmes 9 892 points et même fichier mensuel pour les deux modèles.

Résultats moyens :

| Modèle | RMSE | MAE |
|---|---:|---:|
| SARIMA | 1,445 °C | 1,166 °C |
| SARIMAX ETP+SWI | **1,278 °C** | **1,016 °C** |

SARIMAX réduit la RMSE de 11,5 % et gagne sur 95,8 % des points.

## Organisation

```text
climate_project/
├── data/                 # structure locale des données, fichiers lourds ignorés
├── hergi/                # ARIMA, SARIMA et SARIMAX
├── ndambo/               # TCN et ConvLSTM
├── Yann/                 # Ridge et XGBoost
├── teste/                # préparation, analyses et tests
└── results_sarima/       # métriques, paramètres compacts, rapports et figures
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tests rapides

```bash
python3 -m unittest teste.tests.test_hergi_setup -v
```

## Données

Les données SAFRAN/SIM brutes et nettoyées atteignent plusieurs dizaines de
gigaoctets et ne sont donc pas versionnées dans GitHub. Elles doivent être
placées localement dans `data/` selon les chemins attendus par les scripts.
Les données mensuelles et quotidiennes SIM sont diffusées par Météo-France sur
la grille SAFRAN.

Les paramètres compacts, métriques, figures et rapports nécessaires à l'analyse
des expériences sont conservés dans le dépôt.

