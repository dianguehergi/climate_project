# 📊 Rapport d'Analyse des Variables Significatives — SAFRAN (Quotidien)
**Auteur :** Ousmane Mombo Mombo  
**Source :** `processed/safran_quot_clean.csv`  
**Variable Cible :** `T` (Température moyenne quotidienne)

## 1. Corrélations Linéaires & Significativité (Pour Hergi - SARIMAX / Modèles Linéaires)
| Variable Quotidienne | Corrélation (Pearson) | p-value | Statut |
| :--- | :---: | :---: | :---: |
| `TSUP_H` | 0.9731 | 0.0000e+00 | ✅ Significatif |
| `TINF_H` | 0.9580 | 0.0000e+00 | ✅ Significatif |
| `Q` | 0.9257 | 0.0000e+00 | ✅ Significatif |
| `ETP` | 0.7513 | 0.0000e+00 | ✅ Significatif |
| `DLI` | 0.6986 | 0.0000e+00 | ✅ Significatif |
| `SWI` | -0.6021 | 0.0000e+00 | ✅ Significatif |
| `EVAP` | 0.5934 | 4.9094e-02 | ✅ Significatif |
| `SSI` | 0.5471 | 0.0000e+00 | ✅ Significatif |

### ⚠️ Risques de Multicolinéarité détectés (Variables fortement corrélées entre elles)
- **Alerte :** `RESR_NEIGE` et `RESR_NEIGE6` ont un coefficient de **1.00**. Risque d'instabilité si combinées.
- **Alerte :** `RESR_NEIGE` et `HTEURNEIGE` ont un coefficient de **0.96**. Risque d'instabilité si combinées.
- **Alerte :** `RESR_NEIGE6` et `HTEURNEIGE` ont un coefficient de **0.96**. Risque d'instabilité si combinées.
- **Alerte :** `RESR_NEIGE` et `HTEURNEIGE6` ont un coefficient de **0.96**. Risque d'instabilité si combinées.
- **Alerte :** `RESR_NEIGE6` et `HTEURNEIGE6` ont un coefficient de **0.96**. Risque d'instabilité si combinées.

## 2. Importance des Features (Pour Yann - XGBoost / Random Forest)
*Classement basé sur la réduction d'impureté des nœuds de décision (données quotidiennes).*

| Rang | Variable Explicative | Score d'Importance Relative |
| :---: | :--- | :---: |
| 1 | `TSUP_H` | 0.9019 |
| 2 | `TINF_H` | 0.0902 |
| 3 | `Q` | 0.0029 |
| 4 | `HU` | 0.0013 |
| 5 | `ETP` | 0.0008 |
| 6 | `SSI` | 0.0005 |
| 7 | `FF` | 0.0004 |
| 8 | `DLI` | 0.0004 |
| 9 | `SSWI_10J` | 0.0003 |
| 10 | `WG_RACINE` | 0.0003 |
| 11 | `SWI` | 0.0003 |
| 12 | `PE` | 0.0002 |
| 13 | `EVAP` | 0.0002 |
| 14 | `PRELIQ` | 0.0002 |
| 15 | `DRAINC` | 0.0001 |
| 16 | `RUNC` | 0.0001 |
| 17 | `WGI_RACINE` | 0.0001 |
| 18 | `PRENEI` | 0.0000 |
| 19 | `ECOULEMENT` | 0.0000 |
| 20 | `HTEURNEIGEX` | 0.0000 |
| 21 | `RESR_NEIGE` | 0.0000 |
| 22 | `HTEURNEIGE6` | 0.0000 |
| 23 | `RESR_NEIGE6` | 0.0000 |
| 24 | `HTEURNEIGE` | 0.0000 |
| 25 | `SNOW_FRAC` | 0.0000 |

## 3. Analyse en Composantes Principales (Pour Owen - ConvLSTM / Réseaux de Neurones)
| Axe Spatial/Temporel | Variance Expliquée | Cumulée |
| :---: | :---: | :---: |
| PC1 | 30.39% | 30.39% |
| PC2 | 18.55% | 48.94% |
| PC3 | 11.88% | 60.82% |
| PC4 | 8.68% | 69.50% |
| PC5 | 6.41% | 75.91% |
