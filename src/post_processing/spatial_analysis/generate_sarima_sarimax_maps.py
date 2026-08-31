#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_model_spatial_map(df, model_name, out_dir):
    # Uniformisation des colonnes
    df.columns = [c.upper() for c in df.columns]

    pred_col = 'T_PRED'
    adam_col = 'T_PRED_ADAMONT'
    real_col = 'T_REAL'

    # Calcul du biais spatial
    df['ERR_RAW'] = df[pred_col] - df[real_col]
    df['ERR_ADAM'] = df[adam_col] - df[real_col]

    print(f"🗺️ Calcul du biais spatial moyen ({model_name})...")
    spatial_bias = df.groupby(['LAMBX', 'LAMBY'])[['ERR_RAW', 'ERR_ADAM']].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # 1. Carte Brut
    sc1 = axes[0].scatter(spatial_bias['LAMBX'], spatial_bias['LAMBY'], c=spatial_bias['ERR_RAW'], 
                           cmap='coolwarm', vmin=-3, vmax=3, s=25, marker='s')
    fig.colorbar(sc1, ax=axes[0], label='Biais Moyen (°C)')
    axes[0].set_title(f"Biais Spatial Moyen - {model_name} Brut", fontsize=12, fontweight='bold')
    axes[0].set_aspect('equal')

    # 2. Carte Post-ADAMONT
    sc2 = axes[1].scatter(spatial_bias['LAMBX'], spatial_bias['LAMBY'], c=spatial_bias['ERR_ADAM'], 
                           cmap='coolwarm', vmin=-1, vmax=1, s=25, marker='s')
    fig.colorbar(sc2, ax=axes[1], label='Biais Moyen (°C)')
    axes[1].set_title(f"Biais Spatial Moyen - {model_name} Post-ADAMONT", fontsize=12, fontweight='bold')
    axes[1].set_aspect('equal')

    plt.tight_layout()
    out_file = os.path.join(out_dir, f'{model_name.lower()}_spatial_bias_map.png')
    plt.savefig(out_file)
    plt.close()
    print(f"✅ Carte sauvegardée : {out_file}")

def main():
    print("==================================================")
    print("  CARTO SPATIALE - SARIMA ET SARIMAX             ")
    print("==================================================")

    out_dir = os.path.expanduser('~/climate_project/results_sarima')

    p_sarima  = os.path.expanduser('~/climate_project/results_sarima/predictions_2000_2025/sarima_monthly_predictions_2000_2025_adamont.csv.gz')
    p_sarimax = os.path.expanduser('~/climate_project/results_sarima/sarimax_predictions_2000_2025/sarimax_monthly_predictions_2000_2025_adamont.csv.gz')

    # 1. Traitement SARIMA
    if os.path.exists(p_sarima):
        print("\n📂 Chargement des données pour SARIMA...")
        df_sarima = pd.read_csv(p_sarima, compression='gzip')
        df_sarima.columns = [c.upper() for c in df_sarima.columns]
        generate_model_spatial_map(df_sarima, "SARIMA", out_dir)

    # 2. Traitement SARIMAX (avec fusion de T_REAL si absent)
    if os.path.exists(p_sarimax):
        print("\n📂 Chargement des données pour SARIMAX...")
        df_sarimax = pd.read_csv(p_sarimax, compression='gzip')
        df_sarimax.columns = [c.upper() for c in df_sarimax.columns]

        if 'T_REAL' not in df_sarimax.columns:
            print("   ℹ️  Recouvrement de 'T_REAL' depuis les données SARIMA...")
            df_real_ref = df_sarima[['LAMBX', 'LAMBY', 'DATE', 'T_REAL']].drop_duplicates()
            df_sarimax = pd.merge(df_sarimax, df_real_ref, on=['LAMBX', 'LAMBY', 'DATE'], how='inner')

        generate_model_spatial_map(df_sarimax, "SARIMAX", out_dir)

if __name__ == '__main__':
    main()
