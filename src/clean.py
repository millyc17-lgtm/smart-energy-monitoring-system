import os

import pandas as pd

from energy_lib import clean_readings

df = pd.read_csv("data/raw/block_0.csv")
df.columns = ["household_id", "timestamp", "energy_kwh"]   # simple names for the three columns

print("Rows before cleaning:", len(df))
raw_energy = pd.to_numeric(df["energy_kwh"], errors="coerce")
print("Missing or non-numeric energy values:", int(raw_energy.isna().sum()))
print("Duplicate readings:", int(df.duplicated(["household_id", "timestamp"]).sum()))

df = clean_readings(df)

print(df["energy_kwh"].describe())
print("Negative readings:", int((df["energy_kwh"] < 0).sum()))
print("Rows after cleaning:", len(df))
print("Date range:", df["timestamp"].min(), "to", df["timestamp"].max())
print("Households:", df["household_id"].nunique())

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/energy_clean.csv", index=False)
