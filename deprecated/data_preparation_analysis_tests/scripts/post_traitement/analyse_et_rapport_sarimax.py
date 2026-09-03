#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ydata_profiling import ProfileReport

def main():
    print("==================================================")
    print("   CONSOLIDATION, ANALYSE & RAPPORT - SARIMAX     ")
    print("==================================================")

    base_dir = os.path.expanduser('~/climate_project/archive_old/sarima_sarimax_results/sarimax_predictions_2000_2025')
    processed_dir = os.path.expanduser('~/climate_project/data/processed')
    
    input_file = os.path.join(base_dir, 'temperature_predictions_2000_2025.csv.gz')
    sim_file = os.path.join(processed_dir, 'SIM_mensuelle_2000_2025_cleaned.csv')

    # 1. Chargement des Prédictions SARIMAX 2000-2025
    print(f"\n📂 Chargement des prédictions SARIMAX : {input_file}")
    df_pred = pd.read_csv(input_file, compression='gzip')
    df_pred['DATE'] = df_pred['DATE'].astype(str)
    df_pred['T_PRED'] = pd.to_numeric(df_pred['T_PRED'], errors='coerce').round(2)

    # 2. Chargement du VRAI fichier SIM 2000-2025
    print(f"📂 Chargement des observations SIM réelles : {sim_file}")
    df_sim = pd.read_csv(sim_file)
    df_sim['DATE'] = df_sim['DATE'].astype(str)
    df_sim['T_REAL'] = pd.to_numeric(df_sim['T'], errors='coerce').round(2)

    # 3. Fusion exacte sur (LAMBX, LAMBY, DATE)
    print("\n🔄 Fusion des datasets sur (LAMBX, LAMBY, DATE)...")
    merge_cols = ['LAMBX', 'LAMBY', 'DATE']
    df_merged = pd.merge(df_pred, df_sim[merge_cols + ['T_REAL']], on=merge_cols, how='inner')
    print(f"   └─ {len(df_merged):,} lignes fusionnées avec succès !")

    # 4. Correction ADAMONT (Quantile Mapping mensuel)
    print("\n⚙️ Application de la correction ADAMONT...")
    df_merged['MONTH'] = pd.to_datetime(df_merged['DATE'], format='%Y%m', errors='coerce').dt.month

    def quantile_mapping(group):
        if len(group) > 10:
            q_p = np.quantile(group['T_PRED'], np.linspace(0, 1, 100))
            q_r = np.quantile(group['T_REAL'], np.linspace(0, 1, 100))
            group['T_PRED_ADAMONT'] = np.interp(group['T_PRED'], q_p, q_r).round(2)
        else:
            group['T_PRED_ADAMONT'] = group['T_PRED']
        return group

    df_merged = df_merged.groupby(['MONTH'], group_keys=False).apply(quantile_mapping)

    # Nettoyage et organisation
    cols_order = ['LAMBX', 'LAMBY', 'DATE', 'T_REAL', 'T_PRED', 'T_PRED_ADAMONT', 'CI95_LOW', 'CI95_HIGH']
    cols_final = [c for c in cols_order if c in df_merged.columns]
    df_final = df_merged[cols_final].dropna(subset=['T_REAL', 'T_PRED', 'T_PRED_ADAMONT']).copy()

    # Sauvegarde CSV Consolidé
    final_csv = os.path.join(base_dir, 'temperature_SARIMAX_2000_2025_CONSOLIDATED.csv.gz')
    df_final.to_csv(final_csv, index=False, compression='gzip')
    print(f"\n✅ Dataset consolidé sauvegardé : {final_csv}")

    # 5. Métriques
    def calc_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        bias = np.mean(y_pred - y_true)
        return mae, rmse, r2, bias

    m_brut = calc_metrics(df_final['T_REAL'], df_final['T_PRED'])
    m_adam = calc_metrics(df_final['T_REAL'], df_final['T_PRED_ADAMONT'])
    m_diff = calc_metrics(df_final['T_PRED'], df_final['T_PRED_ADAMONT'])

    print("\n--- MÉTRIQUES D'ÉVALUATION (2000-2025) ---")
    print(f"Brut vs Réel   -> MAE: {m_brut[0]:.3f}°C | RMSE: {m_brut[1]:.3f}°C | Bias: {m_brut[3]:.3f}°C | R²: {m_brut[2]:.3f}")
    print(f"ADAMONT vs Réel-> MAE: {m_adam[0]:.3f}°C | RMSE: {m_adam[1]:.3f}°C | Bias: {m_adam[3]:.3f}°C | R²: {m_adam[2]:.3f}")

    # 6. Rapport Markdown
    report_md = f"""# 📊 Rapport d'Analyse Comparative SARIMAX vs ADAMONT (2000-2025)

## 1. Synthèse des Performances

| Comparaison | MAE (°C) | RMSE (°C) | R² Score | Biais Moyen (°C) |
| :--- | :---: | :---: | :---: | :---: |
| **T REAL vs T PRED (Brut)** | {m_brut[0]:.3f} | {m_brut[1]:.3f} | {m_brut[2]:.3f} | {m_brut[3]:.3f} |
| **T REAL vs T PRED ADAMONT** | **{m_adam[0]:.3f}** | **{m_adam[1]:.3f}** | **{m_adam[2]:.3f}** | **{m_adam[3]:.3f}** |
| **T PRED Brut vs ADAMONT** | {m_diff[0]:.3f} | {m_diff[1]:.3f} | {m_diff[2]:.3f} | {m_diff[3]:.3f} |
"""
    md_path = os.path.join(base_dir, 'RAPPORT_ANALYSE_SARIMAX.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    print(f"📝 Rapport Markdown généré : {md_path}")

    # 7. Scatter Plots
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    sample_df = df_final.sample(min(5000, len(df_final)))
    
    sns.scatterplot(data=sample_df, x='T_REAL', y='T_PRED', ax=axes[0], color='#e74c3c', alpha=0.3)
    axes[0].plot([df_final['T_REAL'].min(), df_final['T_REAL'].max()], [df_final['T_REAL'].min(), df_final['T_REAL'].max()], 'k--', lw=2)
    axes[0].set_title(f"T REAL vs T PRED Brut (R² = {m_brut[2]:.2f})", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("T REAL (°C)")
    axes[0].set_ylabel("T PRED Brut (°C)")

    sns.scatterplot(data=sample_df, x='T_REAL', y='T_PRED_ADAMONT', ax=axes[1], color='#2ecc71', alpha=0.3)
    axes[1].plot([df_final['T_REAL'].min(), df_final['T_REAL'].max()], [df_final['T_REAL'].min(), df_final['T_REAL'].max()], 'k--', lw=2)
    axes[1].set_title(f"T REAL vs T PRED ADAMONT (R² = {m_adam[2]:.2f})", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("T REAL (°C)")
    axes[1].set_ylabel("T PRED ADAMONT (°C)")

    plt.tight_layout()
    scatter_path = os.path.join(base_dir, 'scatter_comparaison_adamont.png')
    plt.savefig(scatter_path)
    plt.close()
    print(f"📊 Graphique Scatter enregistré : {scatter_path}")

    # 8. Profiling HTML
    print("\n📊 Génération du Profiling HTML...")
    profile = ProfileReport(df_final, title="Profiling - Dataset Consolidé SARIMAX (2000-2025)", minimal=True)
    profile_path = os.path.join(base_dir, 'profile_SARIMAX_consolidated.html')
    profile.to_file(profile_path)
    print(f"✅ Profiling HTML sauvegardé : {profile_path}")

if __name__ == '__main__':
    main()
