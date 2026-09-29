import pandas as pd

df = pd.read_csv("data/processed/energy_features.csv", parse_dates=["timestamp"])
df["month"] = df["timestamp"].dt.to_period("M")
print(df.groupby("month")["household_id"].nunique())