# Arborescence des scripts

> Les entraînements `src/models/deep_learning/` nécessitent le serveur GPU CUDA du campus pour reproduire les expériences complètes. Les autres scripts fonctionnent sur CPU, avec une durée potentiellement longue sur 9 892 points.

## Vue d’ensemble

| Ordre | Piste | Point d’entrée canonique | Statut |
|---:|---|---|---|
| 1 | Préparation | `src/data_processing/preparation/prepare_monthly.py` | Actif |
| 1 | Préparation | `src/data_processing/preparation/prepare_daily.py` | Actif |
| 2 | Préparation/ML | `src/data_processing/preparation/build_ml_dataset.py` | Actif |
| 3 | Statistique | `src/models/statistical/run_sarima_full_grid.py` | Actif |
| 3 | Statistique | `src/models/statistical/run_sarimax_full_grid.py` | Actif |
| 3 | ML | `src/models/machine_learning/train_xgboost.py` | Actif |
| 3 | DL | `src/models/deep_learning/train_convlstm_masked_5x5.py` | Actif · GPU |
| 4 | Post-traitement | `src/post_processing/bias_correction/apply_adamont.py` | Actif |
| — | Toutes | `deprecated/` | Exploratoire/obsolète |

## Préparation

| Chemin | Rôle | Entrées | Sorties | Ordre | Statut | Piste |
|---|---|---|---|---:|---|---|
| `src/data_processing/preparation/prepare_monthly.py` | Assemble SAFRAN et SIM mensuels | `data/processed/safran_mens_clean.csv`, `data/processed/datasets/SIM_mensuelle_2000_2025_cleaned.csv` | `data/merged/monthly_1960_2025.csv` | 1 | Actif | Préparation |
| `src/data_processing/preparation/prepare_daily.py` | Prépare les lots quotidiens SIM | `data/raw/` | `data/interim/`, `data/processed/` | 1 | Actif | Préparation |
| `src/data_processing/preparation/build_ml_dataset.py` | Construit les caractéristiques temporelles ML | série mensuelle fusionnée | `data/model/train_1960_1999.csv`, `test_2000_2025.csv` | 2 | Actif | Préparation/ML |
| `src/data_processing/cleaning/prepare_ml_dataset.py` | Variante de nettoyage ML | données traitées | dataset ML | autonome | Exploratoire | Préparation |
| `src/data_processing/extraction/*.py` | Extrait les patches spatiaux | SAFRAN quotidien | tableaux/NPZ de patches | 2 | Actif | Préparation/DL |
| `src/data_processing/profiling/*.py` | Inspecte la grille | données traitées | diagnostic console | autonome | Actif | Préparation |

## Modèles statistiques

| Chemin | Rôle | Entrées | Sorties | Ordre | Statut | Piste |
|---|---|---|---|---:|---|---|
| `src/models/statistical/run_sarima_full_grid.py` | Benchmark SARIMA 1960–1995 / 1996–1999 | mensuel SAFRAN | métriques/paramètres dans `results/statistical/` | 3 | Actif | Statistique |
| `src/models/statistical/run_sarimax_full_grid.py` | Benchmark SARIMAX ETP+SWI | mensuel SAFRAN | métriques/paramètres dans `results/statistical/` | 3 | Actif | Statistique |
| `src/models/statistical/predict_sarima_2000_2025.py` | Réestime puis projette SARIMA | mensuel SAFRAN | projections 2000–2025 | 4 | Actif | Statistique |
| `src/models/statistical/predict_sarimax_2000_2025.py` | Projette SARIMAX avec ETP/SWI extrapolés | mensuel SAFRAN | projections 2000–2025 | 4 | Actif | Statistique |
| `src/models/statistical/compact_sarima_models.py` | Compacte les paramètres | résultats SARIMA | table compacte | 4 | Actif | Statistique |

Les sept anciennes variantes SARIMA et les essais SARIMAX sont conservés sous `deprecated/sarima_sarimax/`. Ils sont **exploratoires ou obsolètes** et ne constituent pas les points d’entrée publiés.

## Machine Learning

| Chemin | Rôle | Entrées | Sorties | Ordre | Statut | Piste |
|---|---|---|---|---:|---|---|
| `src/models/machine_learning/train_xgboost.py` | Entraîne la baseline publiée | `data/model/*.csv` | `models/xgboost/`, `results/xgboost_baseline/` | 3 | Actif | ML |
| `src/models/machine_learning/train_xgboost_advanced.py` | Entraîne la variante enrichie | jeux ML | `results/xgboost_advanced/` | 3 | Exploratoire | ML |
| `src/post_processing/evaluation/analyse_xgboost_baseline.py` | Analyse erreurs et cartes | prédictions baseline | figures/tables baseline | 4 | Actif | ML |
| `src/post_processing/evaluation/compare_xgboost_models.py` | Compare les variantes | résultats baseline/advanced | `results/model_comparison/` | 4 | Exploratoire | ML |

Les anciens doublons Ridge/XGBoost se trouvent sous `deprecated/xgboost_ridge/`.

## Deep Learning

| Chemin | Rôle | Entrées | Sorties | Ordre | Statut | Piste |
|---|---|---|---|---:|---|---|
| `src/models/deep_learning/train_convlstm_masked_5x5.py` | Entraîne le ConvLSTM 5×5 masqué | NPZ des 9 892 centres | checkpoint, métriques | 3 | Actif · GPU | DL |
| `src/models/deep_learning/forecast_convlstm_2000_2025.py` | Rollout auto-régressif | historique, checkpoint | projections quotidiennes | 4 | Actif · GPU | DL |
| `src/post_processing/evaluation/evaluate_convlstm_masked_5x5.py` | Évalue point par point | données test, checkpoint | métriques par point | 4 | Actif · GPU | DL |
| `src/models/deep_learning/core/` | Architectures, datasets et boucle d’entraînement | modules Python | aucun artefact direct | support | Actif | DL |

Les ablations, TCN exploratoires et essais multi-centres sont documentés par leur nom sous `deprecated/tcn_convlstm/`.

## Post-traitement et rapports

| Chemin | Rôle | Entrées | Sorties | Ordre | Statut | Piste |
|---|---|---|---|---:|---|---|
| `src/post_processing/bias_correction/apply_adamont.py` | Corrige les projections SARIMAX | prédictions brutes | prédictions corrigées et figures | 5 | Actif | Post-traitement |
| `src/post_processing/bias_correction/postprocess_sarima.py` | Évalue/corrige SARIMA | projections + SIM | métriques corrigées | 5 | Actif | Post-traitement |
| `src/post_processing/bias_correction/postprocess_convlstm.py` | Évalue/corrige ConvLSTM | projections + SIM | métriques corrigées | 5 | Actif | Post-traitement |
| `src/post_processing/reporting/` | Génère les rapports par piste | résultats versionnés | rapports/figures | 6 | Actif | Toutes |
| `src/post_processing/spatial_analysis/` | Produit les cartes spatiales | résultats par point | figures PNG | 6 | Actif | Toutes |

## Inventaire exhaustif complémentaire

Les points d’entrée ci-dessus ont une fiche détaillée. Le tableau suivant garantit qu’aucun autre script conservé n’est ambigu : les fichiers sous `deprecated/` ne font pas partie du pipeline reproductible.

| Chemin | Rôle | Entrées | Sorties | Ordre | Statut | Piste |
|---|---|---|---|---|---|---|
| `deprecated/data_preparation_analysis_tests/scripts/analyse/script_analyse.py` | script analyse | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/analyse/script_analyse_quotidienne.py` | script analyse quotidienne | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/analyse/script_analyse_variables_mensuelle.py` | script analyse variables mensuelle | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/analyse/script_analyse_variables_mensuelles.py` | script analyse variables mensuelles | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/analyse/script_analyse_variables_quotidienne.py` | script analyse variables quotidienne | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/modeles/rf_baseline_test.py` | rf baseline test | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/post_traitement/analyse_et_rapport_sarimax.py` | analyse et rapport sarimax | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/post_traitement/apply_adamont_legacy.py` | apply adamont legacy | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/post_traitement/generate_comparative_analysis.py` | generate comparative analysis | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/post_traitement/generate_sarima_sarimax_maps.py` | generate sarima sarimax maps | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/post_traitement/generate_spatial_maps.py` | generate spatial maps | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/preparation_donnees/clean_mensuel.py` | clean mensuel | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/preparation_donnees/clean_quotidien.py` | clean quotidien | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/preparation_donnees/extract_first_point.py` | extract first point | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/preparation_donnees/merge_mensuel.py` | merge mensuel | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/preparation_donnees/merge_quotidien.py` | merge quotidien | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/preparation_donnees/pipeline_2000_2025.py` | pipeline 2000 2025 | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/profilage/run_profiling_mensuel.py` | run profiling mensuel | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/profilage/run_profiling_mensuel_brut.py` | run profiling mensuel brut | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/profilage/run_profiling_quotidien.py` | run profiling quotidien | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/profilage/run_profiling_quotidien_brut.py` | run profiling quotidien brut | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/validation/check_extremes_mens.py` | check extremes mens | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/validation/check_extremes_quot.py` | check extremes quot | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/data_preparation_analysis_tests/scripts/validation/teste_colonne_data.py` | teste colonne data | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/reorganization_previous_src/models/baselines/evaluate_baseline.py` | evaluate baseline | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/reorganization_previous_src/models/ml_models/train_ridge.py` | train ridge | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/reorganization_previous_src/models/ml_models/train_xgboost.py` | train xgboost | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/reorganization_previous_src/models/time_series/run_sarimax_pipeline.py` | run sarimax pipeline | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `deprecated/sarima_sarimax/config.py` | config | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Statistique |
| `deprecated/sarima_sarimax/scripts/arima_sarima/generate_prediction_2000_2025_report.py` | generate prediction 2000 2025 report | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Statistique |
| `deprecated/sarima_sarimax/scripts/arima_sarima/sarima_advanced_pipeline.py` | sarima advanced pipeline | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Statistique |
| `deprecated/sarima_sarimax/scripts/arima_sarima/sarima_all_9892.py` | sarima all 9892 | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Statistique |
| `deprecated/sarima_sarimax/scripts/arima_sarima/sarima_all_grid_points.py` | sarima all grid points | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Statistique |
| `deprecated/sarima_sarimax/scripts/arima_sarima/sarima_all_grid_points_improved.py` | sarima all grid points improved | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Statistique |
| `deprecated/sarima_sarimax/scripts/arima_sarima/sarima_full_test.py` | sarima full test | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Statistique |
| `deprecated/sarima_sarimax/scripts/arima_sarima/sarima_temperature_final.py` | sarima temperature final | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Statistique |
| `deprecated/sarima_sarimax/scripts/sarimax/generate_fair_comparison_figure.py` | generate fair comparison figure | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Statistique |
| `deprecated/sarima_sarimax/scripts/sarimax/sarimax_500_fast_fixed_params.py` | sarimax 500 fast fixed params | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Statistique |
| `deprecated/sarima_sarimax/scripts/sarimax/sarimax_all_grid_points_auto.py` | sarimax all grid points auto | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Statistique |
| `deprecated/sarima_sarimax/scripts/sarimax/sarimax_temperature_model.py` | sarimax temperature model | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Statistique |
| `deprecated/tcn_convlstm/scripts/analyze_grid_neighbors.py` | analyze grid neighbors | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/analyze_spatial_model_results.py` | analyze spatial model results | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/build_final_results_table.py` | build final results table | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/check_first_point.py` | check first point | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/compare_1x1_vs_masked_5x5.py` | compare 1x1 vs masked 5x5 | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/evaluate_baseline.py` | evaluate baseline | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/evaluate_many_centers_by_point.py` | evaluate many centers by point | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/evaluate_model.py` | evaluate model | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/export_forecast_2000_2025_full_csv.py` | export forecast 2000 2025 full csv | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/extract_first_point_multivariate.py` | extract first point multivariate | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/extract_many_spatial_patches.py` | extract many spatial patches | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/extract_spatial_patch.py` | extract spatial patch | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/inspect_data.py` | inspect data | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/inspect_spatial_grid.py` | inspect spatial grid | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/plot_comparison_1x1_vs_5x5_maps.py` | plot comparison 1x1 vs 5x5 maps | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/plot_many_centers_maps.py` | plot many centers maps | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/plot_not_improved_points.py` | plot not improved points | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/run_convlstm_patch_ablation.py` | run convlstm patch ablation | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/run_tcn_ablation.py` | run tcn ablation | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/scan_available_date_ranges.py` | scan available date ranges | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/train_convlstm_many_centers.py` | train convlstm many centers | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/train_convlstm_multi_center.py` | train convlstm multi center | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/train_convlstm_spatial.py` | train convlstm spatial | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/train_model.py` | train model | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/train_tcn.py` | train tcn | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/train_tcn_multivariate.py` | train tcn multivariate | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/scripts/train_tcn_spatial_center.py` | train tcn spatial center | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/tcn_convlstm/src/config.py` | config | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | DL |
| `deprecated/xgboost_ridge/check_data.py` | check data | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | ML |
| `deprecated/xgboost_ridge/scripts/01_merge_monthly_data.py` | 01 merge monthly data | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | ML |
| `deprecated/xgboost_ridge/scripts/build_training_dataset.py` | build training dataset | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | ML |
| `deprecated/xgboost_ridge/scripts/check_data.py` | check data | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | ML |
| `deprecated/xgboost_ridge/scripts/generate_report.py` | generate report | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | ML |
| `deprecated/xgboost_ridge/scripts/prepare_dataset.py` | prepare dataset | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | ML |
| `deprecated/xgboost_ridge/scripts/report_comparison.py` | report comparison | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | ML |
| `deprecated/xgboost_ridge/scripts/test_model.py` | test model | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | ML |
| `deprecated/xgboost_ridge/scripts/train_ridge.py` | train ridge | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | ML |
| `deprecated/xgboost_ridge/scripts/train_xgboost.py` | train xgboost | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | ML |
| `deprecated/xgboost_ridge/scripts/train_xgboost_improved.py` | train xgboost improved | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Exploratoire | ML |
| `src/__init__.py` |   init   | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `src/data_processing/__init__.py` |   init   | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Préparation |
| `src/data_processing/cleaning/__init__.py` |   init   | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Préparation |
| `src/data_processing/cleaning/prepare_ml_dataset.py` | prepare ml dataset | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Préparation |
| `src/data_processing/extraction/__init__.py` |   init   | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Préparation |
| `src/data_processing/extraction/extract_multi_spatial_patches.py` | extract multi spatial patches | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Préparation |
| `src/data_processing/extraction/extract_spatial_patch.py` | extract spatial patch | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Préparation |
| `src/data_processing/preparation/__init__.py` |   init   | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Préparation |
| `src/data_processing/preparation/build_ml_dataset.py` | build ml dataset | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Préparation |
| `src/data_processing/preparation/prepare_daily.py` | prepare daily | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Préparation |
| `src/data_processing/preparation/prepare_monthly.py` | prepare monthly | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Préparation |
| `src/data_processing/profiling/__init__.py` |   init   | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Préparation |
| `src/data_processing/profiling/inspect_spatial_grid.py` | inspect spatial grid | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Préparation |
| `src/models/__init__.py` |   init   | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Obsolète | Préparation |
| `src/models/deep_learning/__init__.py` |   init   | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | DL |
| `src/models/deep_learning/core/__init__.py` |   init   | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | DL |
| `src/models/deep_learning/core/architectures.py` | architectures | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | DL |
| `src/models/deep_learning/core/baseline.py` | baseline | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | DL |
| `src/models/deep_learning/core/convlstm.py` | convlstm | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | DL |
| `src/models/deep_learning/core/data_loader.py` | data loader | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | DL |
| `src/models/deep_learning/core/datasets.py` | datasets | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | DL |
| `src/models/deep_learning/core/evaluate.py` | evaluate | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | DL |
| `src/models/deep_learning/core/multivariate_data.py` | multivariate data | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | DL |
| `src/models/deep_learning/core/preprocessing.py` | preprocessing | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | DL |
| `src/models/deep_learning/core/tcn.py` | tcn | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | DL |
| `src/models/deep_learning/core/train.py` | train | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | DL |
| `src/models/deep_learning/forecast_convlstm_2000_2025.py` | forecast convlstm 2000 2025 | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | DL |
| `src/models/deep_learning/train_convlstm_masked_5x5.py` | train convlstm masked 5x5 | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | DL |
| `src/models/machine_learning/__init__.py` |   init   | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | ML |
| `src/models/machine_learning/train_xgboost.py` | train xgboost | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | ML |
| `src/models/machine_learning/train_xgboost_advanced.py` | train xgboost advanced | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | ML |
| `src/models/statistical/__init__.py` |   init   | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Statistique |
| `src/models/statistical/compact_sarima_models.py` | compact sarima models | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Statistique |
| `src/models/statistical/predict_sarima_2000_2025.py` | predict sarima 2000 2025 | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Statistique |
| `src/models/statistical/predict_sarimax_2000_2025.py` | predict sarimax 2000 2025 | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Statistique |
| `src/models/statistical/run_sarima_full_grid.py` | run sarima full grid | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Statistique |
| `src/models/statistical/run_sarimax_full_grid.py` | run sarimax full grid | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Statistique |
| `src/post_processing/__init__.py` |   init   | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Post-traitement |
| `src/post_processing/bias_correction/__init__.py` |   init   | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Post-traitement |
| `src/post_processing/bias_correction/apply_adamont.py` | apply adamont | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Post-traitement |
| `src/post_processing/bias_correction/postprocess_convlstm.py` | postprocess convlstm | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Post-traitement |
| `src/post_processing/bias_correction/postprocess_sarima.py` | postprocess sarima | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Post-traitement |
| `src/post_processing/evaluation/__init__.py` |   init   | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Post-traitement |
| `src/post_processing/evaluation/analyse_xgboost_baseline.py` | analyse xgboost baseline | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Post-traitement |
| `src/post_processing/evaluation/compare_xgboost_models.py` | compare xgboost models | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Post-traitement |
| `src/post_processing/evaluation/evaluate_convlstm_masked_5x5.py` | evaluate convlstm masked 5x5 | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Post-traitement |
| `src/post_processing/evaluation/evaluate_masked_points.py` | evaluate masked points | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Post-traitement |
| `src/post_processing/evaluation/export_full_forecasts.py` | export full forecasts | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Post-traitement |
| `src/post_processing/reporting/__init__.py` |   init   | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Post-traitement |
| `src/post_processing/reporting/generate_sarima_report.py` | generate sarima report | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Post-traitement |
| `src/post_processing/reporting/generate_sarimax_report.py` | generate sarimax report | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Post-traitement |
| `src/post_processing/reporting/generate_statistical_report.py` | generate statistical report | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Post-traitement |
| `src/post_processing/reporting/generate_xgboost_report.py` | generate xgboost report | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Post-traitement |
| `src/post_processing/spatial_analysis/__init__.py` |   init   | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Post-traitement |
| `src/post_processing/spatial_analysis/analyze_convlstm_results.py` | analyze convlstm results | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Post-traitement |
| `src/post_processing/spatial_analysis/generate_sarima_sarimax_maps.py` | generate sarima sarimax maps | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Post-traitement |
| `src/post_processing/spatial_analysis/plot_patch_comparison_maps.py` | plot patch comparison maps | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Post-traitement |
| `src/utils/__init__.py` |   init   | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Toutes |
| `src/utils/config.py` | config | Voir constantes/CLI du script | Voir constantes/CLI du script | autonome | Actif | Toutes |
