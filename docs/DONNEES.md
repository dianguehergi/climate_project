# Données attendues

Les données SAFRAN/SIM ne sont pas versionnées. Créez les dossiers suivants à la racine :

```text
data/
├── raw/                  # téléchargements d’origine, inchangés
├── interim/              # extractions et assemblages temporaires
├── processed/
│   ├── safran_quot_clean.csv
│   ├── safran_mens_clean.csv
│   └── datasets/
│       └── SIM_mensuelle_2000_2025_cleaned.csv
│   └── adamont_calibration_1960_1999.csv.gz
├── merged/               # série continue générée
└── model/                # jeux train/test générés pour XGBoost
```

Clés communes : `LAMBX`, `LAMBY`, `DATE`. La température cible est `T`; SARIMAX utilise aussi `ETP` et `SWI`. Les scripts acceptent les dates mensuelles `AAAAMM`; les fichiers quotidiens utilisent le format indiqué par leurs scripts de préparation.

SAFRAN 1960–1999 sert à l’apprentissage et au backtest. SIM 2000–2025 doit rester hors des ajustements et n’être introduit qu’aux étapes explicitement documentées de validation. Vérifiez la licence de diffusion de Météo-France avant redistribution.
