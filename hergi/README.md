# Travail de Hergi — ARIMA, SARIMA et SARIMAX

## Organisation

- `scripts/arima_sarima/` : modèles ARIMA/SARIMA sur une série ou une grille.
- `scripts/sarimax/` : modèles SARIMAX utilisant des variables exogènes.
- `config.py` : chemins principaux du projet, des données et des résultats.
- `../teste/tests/test_hergi_setup.py` : contrôles rapides, sans entraînement.

Les scripts lisent les données dans `data/processed/` et écrivent leurs sorties
dans `results_sarima/`. Leurs chemins sont calculés depuis leur emplacement : ils
peuvent donc être lancés depuis la racine ou depuis un autre répertoire.

## Vérifier l'environnement

Depuis la racine du projet :

```bash
python3 -m unittest teste.tests.test_hergi_setup -v
```

## Exemples d'exécution

```bash
python3 hergi/scripts/arima_sarima/sarima_full_test.py
python3 hergi/scripts/sarimax/sarimax_temperature_model.py
```

Les traitements sur 500 points et les recherches automatiques peuvent être
longs. Il est préférable de commencer par les scripts travaillant sur un seul
point.
