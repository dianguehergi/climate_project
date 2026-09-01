import pandas as pd
from ydata_profiling import ProfileReport
import os

print("--- Début du Data Profiling (Quotidien - Mode Minimal) ---")

data_path = "data/processed/safran_quot_clean.csv"

if not os.path.exists(data_path):
    print(f"Erreur : Le fichier {data_path} n'existe pas. Lance d'abord clean_quotidien.py.")
    exit()

# 1. Chargement des données
print("Chargement de la base quotidienne propre (cela peut prendre un moment)...")
df = pd.read_csv(data_path)
print(f"Volume : {df.shape[0]} lignes, {df.shape[1]} colonnes.")

# 2. Génération du rapport
# Génération du rapport en mode minimal MAIS en forçant les doublons et le Sample
print("Analyse structurelle en cours (Mode Minimal + Force Duplicates & Sample)...")
profile = ProfileReport(
    df, 
    title="Rapport de Profiling - Données SAFRAN Quotidiennes Clean", 
    minimal=True,
    duplicates={"head": 10},  # Force à chercher et afficher les doublons (affiche jusqu'à 10 exemples si existants)
    samples={"head": 10, "tail": 10}  # Force à afficher les 10 premières et 10 dernières lignes du jeu de données
)

# 3. Sauvegarde
output_html = "data/processed/profiling_quotidien_report.html"
print(f"Sauvegarde du rapport dans {output_html}...")
profile.to_file(output_html)

print("--- Profiling Quotidien terminé avec succès ! ---")
