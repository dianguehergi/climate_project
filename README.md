# Prévision climatique locale sur données SAFRAN

Descente d’échelle statistique et correction de biais inspirée d’ADAMONT sur les **9 892 points** de la grille SAFRAN (France métropolitaine, maille d’environ 8 km).

Projet de recherche mené dans le cadre de l’**AI Clinic — aivancity, La Grande École de l’IA & de la Data**.

## Le projet en une page

Les modèles climatiques globaux et régionaux présentent une résolution trop grossière et des biais systématiques pour une utilisation locale directe. Ce projet utilise la réanalyse SAFRAN de Météo-France comme référence haute résolution afin d’étudier la reconstruction et la projection de la température mensuelle locale avec trois familles de modèles : statistique, machine learning et deep learning. Les sorties SARIMA/SARIMAX et ConvLSTM sont ensuite corrigées par quantile mapping empirique inspiré d’ADAMONT.

**Question scientifique.** Dans quelle mesure ces modèles peuvent-ils reconstruire et projeter la température locale sur toute la grille SAFRAN, et dans quelle mesure la correction de biais améliore-t-elle leurs projections face aux observations 2000–2025 réservées à la validation finale ?

## Résultats

Les protocoles et périodes diffèrent : les chiffres ne constituent pas un classement direct entre toutes les familles.

| Modèle | MAE (°C) | RMSE (°C) | Période évaluée | Protocole |
|---|---:|---:|---|---|
| SARIMA | 1,166 | 1,445 | 1996–1999 | Backtest hors échantillon |
| SARIMAX (ETP + SWI) | 1,016 | 1,278 | 1996–1999 | Backtest hors échantillon |
| SARIMA + ADAMONT | 1,375 | 1,731 | 2000–2025 | Projection longue en aveugle |
| SARIMAX + ADAMONT | 1,372 | 1,716 | 2000–2025 | Projection longue en aveugle |
| XGBoost | 1,418 | 1,795 | 2000–2025 | One-step à retards observés |
| ConvLSTM 5×5 masqué | 1,4221 | 1,8326 | 1993–1999 | J+1 sur entrées réelles |
| ConvLSTM brut | 10,596 | 12,617 | 2000–2025 | Rollout auto-régressif sur 26 ans |
| ConvLSTM + ADAMONT | 4,625 | 5,883 | 2000–2025 | Rollout auto-régressif corrigé |

SARIMAX apporte un gain de RMSE de 11,5 % sur SARIMA et l’améliore sur 95,8 % des points. La correction ramène le biais moyen de −0,831 °C à −0,011 °C pour SARIMA et de −0,789 °C à −0,003 °C pour SARIMAX.

Le biais froid brut combine vraisemblablement un biais de modèle et le réchauffement de 2000–2025 par rapport à la climatologie d’apprentissage. La correction neutralise les deux : elle constitue un recalage d’amplitude, pas la preuve que les modèles projettent correctement la tendance climatique. Le ConvLSTM corrigé est évalué en rollout fermé sur 26 ans ; sur sa tâche J+1, le modèle masqué atteint R² = 0,930.

## Données

| Jeu | Période | Volume | Usage |
|---|---|---:|---|
| SAFRAN quotidien | 1960–1999 | 144 522 120 lignes, 29 variables | Entraînement DL |
| SAFRAN mensuel agrégé | 1960–1999 | 4 748 160 lignes, 22 variables | Statistique et ML |
| SIM mensuel | 2000–2025 | 3 086 304 lignes | Validation finale uniquement |

Les données, trop volumineuses pour Git, sont décrites dans [docs/DONNEES.md](docs/DONNEES.md). Sources : réanalyse SAFRAN et chaîne SAFRAN-ISBA-MODCOU diffusées par Météo-France.

## Installation

```bash
git clone https://github.com/dianguehergi/climate_project.git
cd climate_project
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python 3.11 est la version de référence. Les entraînements Deep Learning nécessitent un GPU CUDA ; les expériences ont été conduites sur NVIDIA RTX 4090 24 Go. Les pistes statistique et ML fonctionnent sur CPU.

## Structure du dépôt

```text
climate_project/
├── src/
│   ├── data_processing/       # ingestion, nettoyage, agrégation, profilage
│   ├── models/
│   │   ├── statistical/       # SARIMA et SARIMAX
│   │   ├── machine_learning/  # XGBoost
│   │   └── deep_learning/     # TCN et ConvLSTM
│   ├── post_processing/       # ADAMONT, évaluation, cartes et rapports
│   └── utils/                 # configuration partagée
├── results/                   # métriques, tableaux et figures publiés
├── docs/                      # données, méthode et inventaire exécutable
├── deprecated/                # variantes exploratoires conservées pour trace
├── tests/
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

## Reproduire les résultats

```bash
# Préparation
python -m src.data_processing.preparation.prepare_monthly
python -m src.data_processing.preparation.prepare_daily
python -m src.data_processing.preparation.build_ml_dataset

# Modélisation
python -m src.models.statistical.run_sarimax_full_grid
python -m src.models.machine_learning.train_xgboost
python -m src.models.deep_learning.train_convlstm_masked_5x5

# Correction et contrôles
python -m src.post_processing.bias_correction.apply_adamont
python -m unittest discover -s tests -v
```

Les entrées, sorties, durées indicatives, besoins GPU et statuts sont détaillés dans [docs/ARBORESCENCE.md](docs/ARBORESCENCE.md).

## Méthode

Le benchmark statistique utilise 1960–1995 pour l’apprentissage et 1996–1999 pour le backtest. Les modèles finaux sont ensuite réestimés jusqu’en 1999 pour projeter 2000–2025 ; les auxiliaires ETP/SWI de SARIMAX utilisent la fenêtre 1970–1999. La piste XGBoost est entraînée jusqu’en 1999 et évaluée sur 2000–2025. Ces différences sont documentées dans [docs/METHODOLOGIE.md](docs/METHODOLOGIE.md).

La correction dite « ADAMONT » dans ce dépôt est un quantile mapping empirique par point de grille, inspiré d’ADAMONT. Faute de prédicteurs de circulation à grande échelle, ce n’est pas l’implémentation complète conditionnée par régime de temps.

## Limites et perspectives

- Les modèles ne peuvent pas apprendre une tendance postérieure à 1999.
- L’incertitude des futures valeurs ETP/SWI n’est pas propagée dans SARIMAX.
- Les protocoles doivent être harmonisés avant toute comparaison inter-familles.
- Les prochaines étapes sont l’application d’ADAMONT à XGBoost, l’inférence ConvLSTM sur observations réelles 2000–2025 et l’ajout de prédicteurs ERA5/EURO-CORDEX/CMIP.

## Équipe

| Membre | Contribution | GitHub |
|---|---|---|
| Ousmane Mombo Mombo | Coordination, ADAMONT, sélection des variables | [@OusmaneMomboMombo](https://github.com/OusmaneMomboMombo) |
| Hergi Diangue | ARIMA, SARIMA, SARIMAX | [@dianguehergi](https://github.com/dianguehergi) |
| Yann de Nzoghe | XGBoost | identifiant à compléter |
| Owen Mouketou | TCN, ConvLSTM | [@OwenDiel](https://github.com/OwenDiel) |

Encadrement : Émile Egreteau-Druet (superviseur) et Anuradha Kar (responsable de l’AI Clinic).

## Référence principale

Verfaillie, D., Déqué, M., Morin, S. et Lafaysse, M. (2017). *The method ADAMONT v1.0 for statistical adjustment of climate projections applicable to energy balance land surface models*. [https://doi.org/10.5194/gmd-10-4257-2017](https://doi.org/10.5194/gmd-10-4257-2017)

## Licence et citation

Le code est distribué sous licence MIT. Les données restent soumises à leurs licences d’origine. Pour citer le projet, voir [CITATION.cff](CITATION.cff).
