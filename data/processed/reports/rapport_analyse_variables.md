# 📊 Rapport d'Analyse des Variables Significatives — SAFRAN
**Auteur :** Ousmane Mombo Mombo  
**Source :** `processed/safran_mens_clean.csv`  
**Variable Cible :** `T`

## 1. Corrélations Linéaires & Significativité (Pour Hergi - SARIMAX)
| Variable | Corrélation | p-value | Statut |
| :--- | :---: | :---: | :---: |
| `ETP` | 0.8600 | 0.0000e+00 | ✅ Significatif |
| `EVAP` | 0.7141 | 5.7564e-81 | ✅ Significatif |
| `SWI` | -0.6875 | 0.0000e+00 | ✅ Significatif |
| `PE` | -0.4977 | 6.2951e-79 | ✅ Significatif |
| `PRENEI` | -0.4302 | 1.8442e-118 | ✅ Significatif |
| `DRAINC` | -0.4229 | 3.3076e-20 | ✅ Significatif |
| `RUNC` | -0.3118 | 0.0000e+00 | ✅ Significatif |
| `ECOULEMENT` | -0.2967 | 3.1523e-179 | ✅ Significatif |

### ⚠️ Risques de Multicolinéarité détectés
- Aucune colinéarité critique (>0.95) détectée.

## 2. Importance des Features (Pour Yann - XGBoost)
| Rang | Variable | Score d'Importance |
| :---: | :--- | :---: |
| 1 | `ETP` | 0.6700 |
| 2 | `PRENEI` | 0.1166 |
| 3 | `EVAP` | 0.0658 |
| 4 | `SWI` | 0.0456 |
| 5 | `ECOULEMENT` | 0.0309 |
| 6 | `SSWI1` | 0.0139 |
| 7 | `DRAINC` | 0.0084 |
| 8 | `PRELIQ` | 0.0075 |
| 9 | `SPI3` | 0.0052 |
| 10 | `PE` | 0.0049 |
| 11 | `SSWI3` | 0.0046 |
| 12 | `SPI1` | 0.0045 |
| 13 | `SSWI6` | 0.0042 |
| 14 | `SPI12` | 0.0042 |
| 15 | `SSWI12` | 0.0041 |
| 16 | `SPI6` | 0.0037 |
| 17 | `PRETOTM` | 0.0030 |
| 18 | `RUNC` | 0.0029 |

## 3. Analyse en Composantes Principales (Pour Owen - ConvLSTM)
| Axe | Variance Expliquée | Cumulée |
| :---: | :---: | :---: |
| PC1 | 36.33% | 36.33% |
| PC2 | 21.16% | 57.48% |
| PC3 | 11.53% | 69.01% |
| PC4 | 8.45% | 77.46% |
| PC5 | 5.54% | 83.00% |
