import pandas as pd

df = pd.read_csv(
    "data/processed/safran_quot_clean.csv",
    nrows=5
)

print(df.columns.tolist())