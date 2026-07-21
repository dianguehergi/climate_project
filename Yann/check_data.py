import pandas as pd
import numpy as np


path = "Yann/data/processed/safran_mens_clean.csv"

df = pd.read_csv(path)

print("Dimensions :")
print(df.shape)

print("\nColonnes :")
print(df.columns.tolist())

print("\nTypes :")
print(df.dtypes)

print("\nPremières lignes :")
print(df.head())

print("\nValeurs manquantes :")
print(df.isnull().sum())