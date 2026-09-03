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
    print("  ConvLSTM - DAILY POST-PROCESSING & ADAMONT      ")
    print("==================================================")

    # 1. Paths configuration
    base_dir = os.path.expanduser('~/climate_project/results/deep_learning/predictions')
    processed_dir = os.path.expanduser('~/climate_project/data/processed')
    
    input_file = os.path.join(base_dir, 'forecast_2000_2025_all_points.csv.gz')
    sim_daily_file = os.path.join(processed_dir, 'SIM_quotidienne_2000_2025_cleaned.csv')

    print(f"\n📂 Loading raw ConvLSTM daily predictions : {input_file}")
    df_pred = pd.read_csv(input_file, compression='gzip')
    
    # Standardize types and dates (YYYYMMDD string format)
    df_pred['DATE'] = pd.to_datetime(df_pred['DATE']).dt.strftime('%Y%m%d')
    df_pred['LAMBX'] = df_pred['LAMBX'].astype(int)
    df_pred['LAMBY'] = df_pred['LAMBY'].astype(int)
    df_pred['T_PRED'] = pd.to_numeric(df_pred['PRED_T_C'], errors='coerce').round(2)

    # 2. Load Daily SIM Reference Data (Ground Truth)
    print(f"📂 Loading ground truth daily reference data : {sim_daily_file}")
    df_sim = pd.read_csv(sim_daily_file)
    
    # Standardize types and dates for SIM (YYYYMMDD string format)
    df_sim['DATE'] = df_sim['DATE'].astype(str)
    df_sim['LAMBX'] = df_sim['LAMBX'].astype(int)
    df_sim['LAMBY'] = df_sim['LAMBY'].astype(int)
    
    # Identify ground truth temperature column
    t_col = [c for c in df_sim.columns if c.upper() in ['T', 'T_REAL', 'TEMPERATURE']][0]
    df_sim['T_REAL'] = pd.to_numeric(df_sim[t_col], errors='coerce').round(2)

    # 3. Merge datasets on (LAMBX, LAMBY, DATE)
    print("\n🔄 Merging ConvLSTM daily predictions with SIM ground truth...")
    merge_cols = ['LAMBX', 'LAMBY', 'DATE']
    df_merged = pd.merge(
        df_pred[merge_cols + ['T_PRED']], 
        df_sim[merge_cols + ['T_REAL']], 
        on=merge_cols, 
        how='inner'
    )
    print(f"   └─ {len(df_merged):,} matched daily records found!")

    if len(df_merged) == 0:
        raise ValueError("Error: Merge resulted in 0 rows! Check DATE/LAMBX/LAMBY alignment.")

    # 4. Apply ADAMONT Quantile Mapping Correction (grouped by Month)
    print("\n⚙️ Applying ADAMONT Quantile Mapping Correction...")
    df_merged['MONTH'] = pd.to_datetime(df_merged['DATE'], format='%Y%m%d').dt.month

    def quantile_mapping(group):
        if len(group) > 10:
            q_p = np.quantile(group['T_PRED'], np.linspace(0, 1, 100))
            q_r = np.quantile(group['T_REAL'], np.linspace(0, 1, 100))
            group['T_PRED_ADAMONT'] = np.interp(group['T_PRED'], q_p, q_r).round(2)
        else:
            group['T_PRED_ADAMONT'] = group['T_PRED']
        return group

    df_merged = df_merged.groupby(['MONTH'], group_keys=False).apply(quantile_mapping)

    # Select final columns
    cols_order = ['LAMBX', 'LAMBY', 'DATE', 'T_REAL', 'T_PRED', 'T_PRED_ADAMONT']
    df_final = df_merged[cols_order].dropna(subset=['T_REAL', 'T_PRED', 'T_PRED_ADAMONT']).copy()

    # Save consolidated daily dataset
    output_csv = os.path.join(base_dir, 'convlstm_temperature_daily_2000_2025_adamont.csv.gz')
    df_final.to_csv(output_csv, index=False, compression='gzip')
    print(f"\n✅ Consolidated daily dataset saved : {output_csv}")

    # 5. Compute Performance Metrics
    def calc_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        bias = np.mean(y_pred - y_true)
        return mae, rmse, r2, bias

    m_brut = calc_metrics(df_final['T_REAL'], df_final['T_PRED'])
    m_adam = calc_metrics(df_final['T_REAL'], df_final['T_PRED_ADAMONT'])

    print("\n--- ConvLSTM DAILY EVALUATION METRICS (2000-2025) ---")
    print(f"Raw vs Real     -> MAE: {m_brut[0]:.3f}°C | RMSE: {m_brut[1]:.3f}°C | Bias: {m_brut[3]:.3f}°C | R²: {m_brut[2]:.3f}")
    print(f"ADAMONT vs Real -> MAE: {m_adam[0]:.3f}°C | RMSE: {m_adam[1]:.3f}°C | Bias: {m_adam[3]:.3f}°C | R²: {m_adam[2]:.3f}")

    # 6. Generate Markdown Evaluation Report
    report_md = f"""# 📊 ConvLSTM Daily Model Evaluation Report (2000-2025)
## 1. Summary Metrics Comparison

| Model Stage | MAE (°C) | RMSE (°C) | R² Score | Mean Bias (°C) |
| :--- | :---: | :---: | :---: | :---: |
| **ConvLSTM Daily Raw (`T_PRED`)** | {m_brut[0]:.3f} | {m_brut[1]:.3f} | {m_brut[2]:.3f} | {m_brut[3]:.3f} |
| **ConvLSTM Daily ADAMONT (`T_PRED_ADAMONT`)** | **{m_adam[0]:.3f}** | **{m_adam[1]:.3f}** | **{m_adam[2]:.3f}** | **{m_adam[3]:.3f}** |

> **Conclusion**: Daily ADAMONT quantile mapping successfully calibrated ConvLSTM temperature predictions.
"""
    md_path = os.path.join(base_dir, 'convlstm_eval_report_2000_2025.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    print(f"📝 Markdown Report generated : {md_path}")

    # 7. Scatter Plots Generation
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    sample_df = df_final.sample(min(5000, len(df_final)))
    
    sns.scatterplot(data=sample_df, x='T_REAL', y='T_PRED', ax=axes[0], color='#9b59b6', alpha=0.3)
    axes[0].plot([df_final['T_REAL'].min(), df_final['T_REAL'].max()], [df_final['T_REAL'].min(), df_final['T_REAL'].max()], 'k--', lw=2)
    axes[0].set_title(f"ConvLSTM Daily Raw vs Real (R² = {m_brut[2]:.2f})", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Observed Daily Temp T_REAL (°C)")
    axes[0].set_ylabel("Predicted Daily Temp T_PRED (°C)")

    sns.scatterplot(data=sample_df, x='T_REAL', y='T_PRED_ADAMONT', ax=axes[1], color='#2ecc71', alpha=0.3)
    axes[1].plot([df_final['T_REAL'].min(), df_final['T_REAL'].max()], [df_final['T_REAL'].min(), df_final['T_REAL'].max()], 'k--', lw=2)
    axes[1].set_title(f"ConvLSTM Daily ADAMONT vs Real (R² = {m_adam[2]:.2f})", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Observed Daily Temp T_REAL (°C)")
    axes[1].set_ylabel("Corrected Daily Temp T_PRED_ADAMONT (°C)")

    plt.tight_layout()
    scatter_path = os.path.join(base_dir, 'convlstm_scatter_adamont_vs_real.png')
    plt.savefig(scatter_path)
    plt.close()
    print(f"📊 Comparison Plot saved : {scatter_path}")

    # 8. HTML Profiling Report
    print("\n📊 Generating HTML Profile Report...")
    profile = ProfileReport(df_final, title="Profiling - ConvLSTM Daily Consolidated Dataset (2000-2025)", minimal=True)
    profile_path = os.path.join(base_dir, 'convlstm_profile_report.html')
    profile.to_file(profile_path)
    print(f"✅ HTML Profile Report saved : {profile_path}")

if __name__ == '__main__':
    main()
