import os
import pandas as pd

df = pd.read_csv("data/raw/block_0.csv")

# Give the columns simple names
df.columns = ["household_id", "timestamp", "energy_kwh"]

# Fix data types
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["energy_kwh"] = pd.to_numeric(df["energy_kwh"], errors="coerce")

# Check for problems
print("Rows before cleaning:", len(df))
print("Missing values:")
print(df.isna().sum())
print("Duplicate readings:", df.duplicated(["household_id", "timestamp"]).sum())

# Fix them
df = df.drop_duplicates(["household_id", "timestamp"])
df = df.dropna(subset=["energy_kwh"])
df = df.sort_values(["household_id", "timestamp"])

# Look for impossible values
print(df["energy_kwh"].describe())
print("Negative readings:", (df["energy_kwh"] < 0).sum())

print("Rows after cleaning:", len(df))
print("Date range:", df["timestamp"].min(), "to", df["timestamp"].max())
print("Households:", df["household_id"].nunique())

# Save the cleaned copy (the raw file stays untouched)
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/energy_clean.csv", index=False)