import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Utilisation stricte du dossier processed
DATA_DIR = os.path.expanduser("~/climate_project/data/processed/")
FILE_NAME = "safran_mens_clean.csv"  # Change par "safran_quot_clean.csv" si tu veux le quotidien
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)
OUTPUT_PATH = os.path.join(DATA_DIR, "rapport_analyse_variables_mensuelle.md")

print(f"--- Début de l'analyse sur {FILE_NAME} ---")

if not os.path.exists(FILE_PATH):
    print(f"Erreur : Le fichier {FILE_PATH} est introuvable. Vérifie son nom dans data/processed/")
    exit()

# Chargement
df = pd.read_csv(FILE_PATH)
df.columns = [c.upper() for c in df.columns]

target = 'T'
exclude_cols = ['LAMBX', 'LAMBY', 'DATE']
predictors = [col for col in df.columns if col not in exclude_cols and col != target]

df_clean = df[[target] + predictors].dropna()
X = df_clean[predictors]
y = df_clean[target]

# 1. Analyse Linéaire & Multi-colinéarité
corr_matrix = df_clean[[target] + predictors].corr(method='pearson')
corr_with_target = corr_matrix[target].drop(target).sort_values(key=abs, ascending=False)

high_corr_pairs = []
corr_pred = X.corr().abs()
upper_tri = corr_pred.where(np.triu(np.ones(corr_pred.shape), k=1).astype(bool))
for col in upper_tri.columns:
    for row in upper_tri.index:
        if upper_tri.loc[row, col] > 0.95:
            high_corr_pairs.append((row, col, upper_tri.loc[row, col]))

X_stats = sm.add_constant(X)
model_ols = sm.OLS(y, X_stats).fit()
p_values = model_ols.pvalues.drop('const')

# 2. Importance Non Linéaire (Random Forest)
X_sample = X.sample(n=min(50000, len(X)), random_state=42)
y_sample = y.loc[X_sample.index]

rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X_sample, y_sample)
importances = pd.Series(rf.feature_importances_, index=predictors).sort_values(ascending=False)

# 3. Réduction de dimension (ACP)
X_scaled = StandardScaler().fit_transform(X_sample)
pca = PCA(n_components=min(5, len(predictors)))
pca.fit(X_scaled)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Rédaction du Markdown
md_content = f"""# 📊 Rapport d'Analyse des Variables Significatives — SAFRAN Mensuel
**Auteur :** Ousmane Mombo Mombo  
**Source :** `processed/{FILE_NAME}`  
**Variable Cible :** `{target}`

## 1. Corrélations Linéaires & Significativité (Pour Hergi - SARIMAX)
| Variable | Corrélation | p-value | Statut |
| :--- | :---: | :---: | :---: |
"""
for var in corr_with_target.index[:8]:
    p_val = p_values.get(var, 0)
    status = "✅ Significatif" if p_val < 0.05 else "❌ Non significatif"
    md_content += f"| `{var}` | {corr_with_target[var]:.4f} | {p_val:.4e} | {status} |\n"

md_content += "\n### ⚠️ Risques de Multicolinéarité détectés\n"
if high_corr_pairs:
    for var1, var2, val in high_corr_pairs[:5]:
        md_content += f"- **Alerte :** `{var1}` et `{var2}` ont une corrélation de **{val:.2f}**. Éviter de les coupler.\n"
else:
    md_content += "- Aucune colinéarité critique (>0.95) détectée.\n"

md_content += """
## 2. Importance des Features (Pour Yann - XGBoost)
| Rang | Variable | Score d'Importance |
| :---: | :--- | :---: |
"""
for idx, (var, score) in enumerate(importances.items(), 1):
    md_content += f"| {idx} | `{var}` | {score:.4f} |\n"

md_content += """
## 3. Analyse en Composantes Principales (Pour Owen - ConvLSTM)
| Axe | Variance Expliquée | Cumulée |
| :---: | :---: | :---: |
"""
for i, (exp, cum) in enumerate(zip(explained_variance, cumulative_variance), 1):
    md_content += f"| PC{i} | {exp*100:.2f}% | {cum*100:.2f}% |\n"

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"[✓] Analyse terminée avec succès !")
print(f"[✓] Rapport enregistré dans : {OUTPUT_PATH}")
