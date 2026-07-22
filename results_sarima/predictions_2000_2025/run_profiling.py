import pandas as pd
from ydata_profiling import ProfileReport

print("--> Chargement du fichier temperature_predictions_2000_2025.csv.gz...")
df = pd.read_csv("temperature_predictions_2000_2025.csv.gz")

print("\n" + "="*50)
print("Aperçu des premières lignes (df.head()) :")
print("="*50)
print(df.head())

print("\n" + "="*50)
print("Informations sur le DataFrame (df.info()) :")
print("="*50)
print(df.info())

print("\n--> Génération du rapport ydata-profiling (sans corrélations)...")
profile = ProfileReport(
    df,
    title="Profiling Prédictions SARIMA (2000-2025)",
    minimal=True,
    correlations=None,
    explorative=False
)

output_html = "profiling_predictions_2000_2025.html"
profile.to_file(output_html)
print(f"\n✅ Rapport HTML généré avec succès : {output_html}")
