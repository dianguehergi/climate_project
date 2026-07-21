import pandas as pd
from ydata_profiling import ProfileReport
import os

print("--- Data Profiling : Quotidien BRUT (Mode Minimal) ---")

data_path = "data/interim/safran_quot_brut.csv"

if not os.path.exists(data_path):
    print(f"Erreur : Le fichier brut {data_path} n'existe pas. Lance d'abord merge_quotidien.py.")
    exit()

print("Chargement de la base quotidienne brute (cela peut prendre un moment)...")
df = pd.read_csv(data_path)
print(f"Volume brut : {df.shape[0]} lignes, {df.shape[1]} colonnes.")

# Génération du rapport en mode minimal MAIS en forçant les doublons et le Sample
print("Analyse structurelle en cours (Mode Minimal + Force Duplicates & Sample)...")
profile = ProfileReport(
    df, 
    title="Rapport de Profiling - Données SAFRAN Quotidiennes Clean", 
    minimal=True,
    duplicates={"head": 10},  # Force à chercher et afficher les doublons (affiche jusqu'à 10 exemples si existants)
    samples={"head": 10, "tail": 10}  # Force à afficher les 10 premières et 10 dernières lignes du jeu de données
)

output_html = "data/interim/profiling_quotidien_brut_report.html"
print(f"Sauvegarde du rapport brut dans {output_html}...")
profile.to_file(output_html)

print("--- Profiling Quotidien Brut terminé ! ---")
