import pandas as pd


# Script to covert GBM_Govt Date's into "YYYY-MM-DD format"

df = pd.read_csv("data/GBM_Govt/2025-11-17_GBM_Govt_to_send.csv")

df['Date'] = pd.to_datetime(
    df['Date'], origin='1899-12-30', unit='D').dt.strftime('%Y-%m-%d')

df.to_csv("data/2025-11-17_GBM_Govt.csv", index=False)
