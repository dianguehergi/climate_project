#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_event_map(df, target_date, title_event, output_filename, out_dir):
    """ Génère une comparaison 3-cartes (Réel vs ConvLSTM Brut vs ADAMONT) pour une date précise. """
    # Convertir target_date en format string 'YYYYMMDD'
    df_day = df[df['DATE'].astype(str) == str(target_date)]

    if len(df_day) == 0:
        print(f"⚠️ Aucune donnée trouvée pour la date {target_date}")
        return

    print(f"🗺️ Génération de la carte pour le : {target_date} ({title_event})...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    # Échelle de température commune pour le Réel et ADAMONT
    vmin_t = min(df_day['T_REAL'].min(), df_day['T_PRED_ADAMONT'].min())
    vmax_t = max(df_day['T_REAL'].max(), df_day['T_PRED_ADAMONT'].max())

    cmap_choice = 'YlOrRd' if 'Canicule' in title_event else 'Blues_r'

    # 1. Vérité Terrain (T_REAL)
    sc0 = axes[0].scatter(df_day['LAMBX'], df_day['LAMBY'], c=df_day['T_REAL'], 
                           cmap=cmap_choice, vmin=vmin_t, vmax=vmax_t, s=25, marker='s')
    fig.colorbar(sc0, ax=axes[0], label='Température (°C)')
    axes[0].set_title(f"Réel (T_REAL)\n{title_event} ({target_date})", fontsize=11, fontweight='bold')
    axes[0].set_aspect('equal')

    # 2. ConvLSTM Brut (T_PRED)
    sc1 = axes[1].scatter(df_day['LAMBX'], df_day['LAMBY'], c=df_day['T_PRED'], 
                           cmap=cmap_choice, vmin=vmin_t - 10, vmax=vmax_t - 10, s=25, marker='s')
    fig.colorbar(sc1, ax=axes[1], label='Température (°C)')
    axes[1].set_title(f"ConvLSTM Brut (T_PRED)\n[Biais froid visible]", fontsize=11, fontweight='bold')
    axes[1].set_aspect('equal')

    # 3. ConvLSTM Débiaisé (T_PRED_ADAMONT)
    sc2 = axes[2].scatter(df_day['LAMBX'], df_day['LAMBY'], c=df_day['T_PRED_ADAMONT'], 
                           cmap=cmap_choice, vmin=vmin_t, vmax=vmax_t, s=25, marker='s')
    fig.colorbar(sc2, ax=axes[2], label='Température (°C)')
    axes[2].set_title(f"ConvLSTM ADAMONT (T_PRED_ADAMONT)\n[Calibré]", fontsize=11, fontweight='bold')
    axes[2].set_aspect('equal')

    plt.tight_layout()
    out_path = os.path.join(out_dir, output_filename)
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Carte sauvegardée : {out_path}")

def main():
    print("==================================================")
    print("  CARTO SPATIALE - ÉVÉNEMENTS CLIMATIQUES CIBLÉS  ")
    print("==================================================")

    out_dir = os.path.expanduser('~/climate_project/archive_old/tcn_convlstm/outputs/predictions')
    p_conv = os.path.join(out_dir, 'convlstm_temperature_daily_2000_2025_adamont.csv.gz')

    if not os.path.exists(p_conv):
        print("❌ Fichier ConvLSTM introuvable !")
        return

    print("📂 Chargement des données ConvLSTM...")
    df = pd.read_csv(p_conv, compression='gzip')

    # 1. Carte du Biais Spatial Moyen
    print("\n🗺️ Calcul du Biais Spatial Moyen (2000-2025)...")
    df['ERR_RAW'] = df['T_PRED'] - df['T_REAL']
    df['ERR_ADAM'] = df['T_PRED_ADAMONT'] - df['T_REAL']

    spatial_bias = df.groupby(['LAMBX', 'LAMBY'])[['ERR_RAW', 'ERR_ADAM']].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    sc1 = axes[0].scatter(spatial_bias['LAMBX'], spatial_bias['LAMBY'], c=spatial_bias['ERR_RAW'], 
                           cmap='coolwarm', vmin=-15, vmax=5, s=25, marker='s')
    fig.colorbar(sc1, ax=axes[0], label='Biais (°C)')
    axes[0].set_title("Biais Moyen - ConvLSTM Brut", fontsize=12, fontweight='bold')
    axes[0].set_aspect('equal')

    sc2 = axes[1].scatter(spatial_bias['LAMBX'], spatial_bias['LAMBY'], c=spatial_bias['ERR_ADAM'], 
                           cmap='coolwarm', vmin=-1, vmax=1, s=25, marker='s')
    fig.colorbar(sc2, ax=axes[1], label='Biais (°C)')
    axes[1].set_title("Biais Moyen - Post-ADAMONT", fontsize=12, fontweight='bold')
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'convlstm_spatial_bias_map.png'))
    plt.close()

    # 2. Cartes pour des dates ciblées historiques
    plot_event_map(df, '20030812', 'Canicule 2003', 'convlstm_canicule_2003_map.png', out_dir)
    plot_event_map(df, '20120212', 'Vague de Froid 2012', 'convlstm_vague_froid_2012_map.png', out_dir)

if __name__ == '__main__':
    main()
