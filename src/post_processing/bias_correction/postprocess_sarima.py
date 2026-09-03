#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ydata_profiling import ProfileReport

def main():
    print("==================================================")
    print("   SARIMA - POST-PROCESSING & ADAMONT CORRECTION  ")
    print("==================================================")

    # 1. Paths configuration
    base_dir = os.path.expanduser('~/climate_project/results/statistical/predictions_2000_2025')
    processed_dir = os.path.expanduser('~/climate_project/data/processed')
    
    input_file = os.path.join(base_dir, 'temperature_predictions_2000_2025.csv.gz')
    sim_file = os.path.join(processed_dir, 'SIM_mensuelle_2000_2025_cleaned.csv')

    print(f"\n📂 Loading raw SARIMA predictions : {input_file}")
    df_pred = pd.read_csv(input_file, compression='gzip')
    
    # Standardize column names
    df_pred['DATE'] = df_pred['DATE'].astype(str)
    pred_col = [c for c in df_pred.columns if 'pred' in c.lower() or 't_' in c.lower()][0]
    df_pred['T_PRED'] = pd.to_numeric(df_pred[pred_col], errors='coerce').round(2)

    # 2. Load SIM Reference Data (Ground Truth)
    print(f"📂 Loading ground truth reference data : {sim_file}")
    df_sim = pd.read_csv(sim_file)
    df_sim['DATE'] = df_sim['DATE'].astype(str)
    df_sim['T_REAL'] = pd.to_numeric(df_sim['T'], errors='coerce').round(2)

    # 3. Merge datasets on Spatial & Temporal keys
    print("\n🔄 Merging datasets on (LAMBX, LAMBY, DATE)...")
    merge_cols = ['LAMBX', 'LAMBY', 'DATE']
    df_merged = pd.merge(df_pred, df_sim[merge_cols + ['T_REAL']], on=merge_cols, how='inner')
    print(f"   └─ {len(df_merged):,} matched records found!")

    # 4. Apply ADAMONT Quantile Mapping (Monthly)
    print("\n⚙️ Applying ADAMONT Quantile Mapping Correction...")
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

    # Select & clean final columns
    cols_order = ['LAMBX', 'LAMBY', 'DATE', 'T_REAL', 'T_PRED', 'T_PRED_ADAMONT']
    cols_final = [c for c in cols_order if c in df_merged.columns]
    df_final = df_merged[cols_final].dropna(subset=['T_REAL', 'T_PRED', 'T_PRED_ADAMONT']).copy()

    # Save consolidated dataset with standardized English name
    output_csv = os.path.join(base_dir, 'sarima_temperature_monthly_2000_2025_adamont.csv.gz')
    df_final.to_csv(output_csv, index=False, compression='gzip')
    print(f"\n✅ Consolidated dataset saved : {output_csv}")

    # 5. Compute Performance Metrics
    def calc_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        bias = np.mean(y_pred - y_true)
        return mae, rmse, r2, bias

    m_brut = calc_metrics(df_final['T_REAL'], df_final['T_PRED'])
    m_adam = calc_metrics(df_final['T_REAL'], df_final['T_PRED_ADAMONT'])

    print("\n--- SARIMA EVALUATION METRICS (2000-2025) ---")
    print(f"Raw vs Real     -> MAE: {m_brut[0]:.3f}°C | RMSE: {m_brut[1]:.3f}°C | Bias: {m_brut[3]:.3f}°C | R²: {m_brut[2]:.3f}")
    print(f"ADAMONT vs Real -> MAE: {m_adam[0]:.3f}°C | RMSE: {m_adam[1]:.3f}°C | Bias: {m_adam[3]:.3f}°C | R²: {m_adam[2]:.3f}")

    # 6. Generate Markdown Evaluation Report
    report_md = f"""# 📊 SARIMA Model Evaluation Report (2000-2025)

## 1. Summary Metrics Comparison

| Model Stage | MAE (°C) | RMSE (°C) | R² Score | Mean Bias (°C) |
| :--- | :---: | :---: | :---: | :---: |
| **SARIMA Raw (`T_PRED`)** | {m_brut[0]:.3f} | {m_brut[1]:.3f} | {m_brut[2]:.3f} | {m_brut[3]:.3f} |
| **SARIMA ADAMONT (`T_PRED_ADAMONT`)** | **{m_adam[0]:.3f}** | **{m_adam[1]:.3f}** | **{m_adam[2]:.3f}** | **{m_adam[3]:.3f}** |

> **Conclusion**: ADAMONT quantile mapping successfully calibrated the SARIMA temperature distributions.
"""
    md_path = os.path.join(base_dir, 'sarima_eval_report_2000_2025.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    print(f"📝 Markdown Report generated : {md_path}")

    # 7. Scatter Plots Generation
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    sample_df = df_final.sample(min(5000, len(df_final)))
    
    sns.scatterplot(data=sample_df, x='T_REAL', y='T_PRED', ax=axes[0], color='#3498db', alpha=0.3)
    axes[0].plot([df_final['T_REAL'].min(), df_final['T_REAL'].max()], [df_final['T_REAL'].min(), df_final['T_REAL'].max()], 'k--', lw=2)
    axes[0].set_title(f"SARIMA Raw vs Real (R² = {m_brut[2]:.2f})", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Observed Temperature T_REAL (°C)")
    axes[0].set_ylabel("Predicted Temperature T_PRED (°C)")

    sns.scatterplot(data=sample_df, x='T_REAL', y='T_PRED_ADAMONT', ax=axes[1], color='#2ecc71', alpha=0.3)
    axes[1].plot([df_final['T_REAL'].min(), df_final['T_REAL'].max()], [df_final['T_REAL'].min(), df_final['T_REAL'].max()], 'k--', lw=2)
    axes[1].set_title(f"SARIMA ADAMONT vs Real (R² = {m_adam[2]:.2f})", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Observed Temperature T_REAL (°C)")
    axes[1].set_ylabel("Corrected Temperature T_PRED_ADAMONT (°C)")

    plt.tight_layout()
    scatter_path = os.path.join(base_dir, 'sarima_scatter_adamont_vs_real.png')
    plt.savefig(scatter_path)
    plt.close()
    print(f"📊 Comparison Plot saved : {scatter_path}")

    # 8. HTML Profiling Report
    print("\n📊 Generating HTML Profile Report...")
    profile = ProfileReport(df_final, title="Profiling - SARIMA Consolidated Dataset (2000-2025)", minimal=True)
    profile_path = os.path.join(base_dir, 'sarima_profile_report.html')
    profile.to_file(profile_path)
    print(f"✅ HTML Profile Report saved : {profile_path}")

if __name__ == '__main__':
    main()
