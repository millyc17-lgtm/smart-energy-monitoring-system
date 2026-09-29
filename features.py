import pandas as pd

PRICE_PER_KWH = 0.25  # placeholder tariff in £, made configurable in the dashboard later

df = pd.read_csv("data/processed/energy_clean.csv", parse_dates=["timestamp"])

# Time columns
df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_name()
df["month"] = df["timestamp"].dt.month
df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5

# Energy columns (the data is already in kWh per half hour, so no conversion needed)
df["power_kw"] = df["energy_kwh"] * 2      # kWh per 0.5 h -> average kW
df["cost"] = df["energy_kwh"] * PRICE_PER_KWH

print(df.head())
print(df.dtypes)

df.to_csv("data/processed/energy_features.csv", index=False)