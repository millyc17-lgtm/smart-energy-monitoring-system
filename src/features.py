import pandas as pd

from energy_lib import add_features

PRICE_PER_KWH = 0.25  # placeholder tariff in GBP; the dashboard lets the user change it

df = pd.read_csv("data/processed/energy_clean.csv", parse_dates=["timestamp"])
df = add_features(df, PRICE_PER_KWH)

print(df.head())
print(df.dtypes)

df.to_csv("data/processed/energy_features.csv", index=False)
