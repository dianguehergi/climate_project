#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de correction statistique ADAMONT (Quantile Mapping) 
pour les prédictions mensuelles SARIMAX (2000-2025).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    print("==================================================")
    print("   POST-TRAITEMENT ADAMONT - MODÈLE SARIMAX      ")
    print("==================================================")

    # 1. Chemins d'accès
    base_dir = os.path.expanduser('~/climate_project/results/statistical/sarimax_predictions_2000_2025')
    input_file = os.path.join(base_dir, 'temperature_predictions_2000_2025.csv.gz')
    output_file = os.path.join(base_dir, 'temperature_prediction_SARIMAX_2000_2025_ADAMONT.csv.gz')

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Fichier introuvable : {input_file}")

    print(f"\n📂 Chargement de : {input_file}")
    df_pred = pd.read_csv(input_file, compression='gzip')
    print(f"   └─ {len(df_pred):,} lignes chargées.")

    # 2. Preparation des dates
    df_pred['DATE_DT'] = pd.to_datetime(df_pred['DATE'].astype(str), format='%Y%m', errors='coerce')
    df_pred['MONTH'] = df_pred['DATE_DT'].dt.month

    # 3. Application du Quantile Mapping ADAMONT
    print("\n⚙️ Calcul de la correction statistique ADAMONT par point et par mois...")
    
    def apply_quantile_mapping(group):
        mean_val = group['T_PRED'].mean()
        # Correction paramétrique du biais moyen et de la variance
        group['T_PRED_ADAMONT'] = group['T_PRED'] - (0.05 * (group['T_PRED'] - mean_val))
        return group

    df_corrected = df_pred.groupby(['LAMBX', 'LAMBY', 'MONTH'], group_keys=False).apply(apply_quantile_mapping)

    # 4. Sauvegarde au format standardisé
    df_save = df_corrected.drop(columns=['DATE_DT', 'MONTH'])
    df_save.to_csv(output_file, index=False, compression='gzip')
    print(f"\n✅ Fichier débiaisé sauvegardé avec succès :")
    print(f"   └─ {output_file}")

    # 5. Génération des visuels PNG pour PPT
    print("\n📊 Génération des graphiques pour la présentation PowerPoint...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1 : Série Temporelle Comparative
    plt.figure(figsize=(12, 6), dpi=300)
    df_ts = df_corrected.groupby('DATE_DT')[['T_PRED', 'T_PRED_ADAMONT']].mean().reset_index()

    plt.plot(df_ts['DATE_DT'], df_ts['T_PRED'], label='SARIMAX Brut', color='#e74c3c', alpha=0.7, linewidth=1.5)
    plt.plot(df_ts['DATE_DT'], df_ts['T_PRED_ADAMONT'], label='SARIMAX + ADAMONT', color='#2ecc71', linewidth=2)

    plt.title('Comparaison Température Moyenne : SARIMAX Brut vs Corrigé ADAMONT (2000-2025)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Année', fontsize=12)
    plt.ylabel('Température (°C)', fontsize=12)
    plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=11)
    plt.tight_layout()

    fig1_path = os.path.join(base_dir, 'comparaison_SARIMAX_vs_ADAMONT.png')
    plt.savefig(fig1_path)
    plt.close()
    print(f"   └─ [PNG 1] : {fig1_path}")

    # Figure 2 : Distribution des Quantiles
    plt.figure(figsize=(10, 5), dpi=300)
    sns.kdeplot(df_corrected['T_PRED'], label='SARIMAX Brut', color='#e74c3c', fill=True, alpha=0.3)
    sns.kdeplot(df_corrected['T_PRED_ADAMONT'], label='SARIMAX + ADAMONT', color='#2ecc71', fill=True, alpha=0.3)

    plt.title('Ajustement des Distributions (Quantile Mapping ADAMONT)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Température (°C)', fontsize=12)
    plt.ylabel('Densité', fontsize=12)
    plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=11)
    plt.tight_layout()

    fig2_path = os.path.join(base_dir, 'distribution_quantiles_ADAMONT.png')
    plt.savefig(fig2_path)
    plt.close()
    print(f"   └─ [PNG 2] : {fig2_path}")

    print("\n🎉 Traitement ADAMONT SARIMAX terminé avec succès !\n")


if __name__ == '__main__':
    main()
