import pandas as pd



pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

df = pd.read_csv("data/processed/energy_features.csv", parse_dates=["timestamp"])
df = df.sort_values(["household_id", "timestamp"])

# Half-hour slot of the day: 0 to 47
df["slot"] = df["timestamp"].dt.hour * 2 + df["timestamp"].dt.minute // 30

# "Normal" = mean and spread of the previous 14 readings at the same slot
# (shift(1) makes sure a reading is never compared against itself)
g = df.groupby(["household_id", "slot"])["energy_kwh"]
df["normal_mean"] = g.transform(lambda s: s.shift(1).rolling(14, min_periods=7).mean())
df["normal_std"] = g.transform(lambda s: s.shift(1).rolling(14, min_periods=7).std())

# Flag readings more than 3 standard deviations above normal.
# The 0.05 floor stops tiny changes being flagged for households with very steady usage.
K = 3
df["upper_limit"] = df["normal_mean"] + K * df["normal_std"].clip(lower=0.05)
df["is_anomaly"] = df["energy_kwh"] > df["upper_limit"]

checked = df["upper_limit"].notna()
print("Readings checked:", checked.sum())
print("Anomalies found:", df["is_anomaly"].sum())
print("Anomaly rate:", round(100 * df["is_anomaly"].sum() / checked.sum(), 2), "%")

print("\nAnomalies by hour of day:")
print(df[df["is_anomaly"]].groupby("hour").size())

print("\nBiggest anomalies:")
cols = ["household_id", "timestamp", "energy_kwh", "normal_mean", "upper_limit"]
print(df[df["is_anomaly"]].nlargest(10, "energy_kwh")[cols].round(2))

df.to_csv("data/processed/energy_anomalies.csv", index=False)

df[df["is_anomaly"]].to_csv("data/processed/anomalies_only.csv", index=False)