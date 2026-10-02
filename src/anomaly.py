import pandas as pd

from detectors import rule_baseline, rule_flags, rule_upper_limit

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

K = 3  # sensitivity: flag readings more than K standard deviations above normal

df = pd.read_csv("data/processed/energy_features.csv", parse_dates=["timestamp"])
df = df.sort_values(["household_id", "timestamp"]).reset_index(drop=True)

df = rule_baseline(df)
df["upper_limit"] = rule_upper_limit(df, K)
df["is_anomaly"] = rule_flags(df, K)

checked = df["upper_limit"].notna()
print("Readings checked:", checked.sum())
print("Anomalies found:", df["is_anomaly"].sum())
print("Anomaly rate:", round(100 * df["is_anomaly"].sum() / checked.sum(), 2), "%")

print("\nAnomalies by hour of day:")
print(df[df["is_anomaly"]].groupby("hour").size())

print("\nBiggest anomalies:")
cols = ["household_id", "timestamp", "energy_kwh", "normal_mean", "upper_limit"]
top = df[df["is_anomaly"]].nlargest(10, "energy_kwh")[cols]
print(top.round({"energy_kwh": 2, "normal_mean": 2, "upper_limit": 2}))

df.to_csv("data/processed/energy_anomalies.csv", index=False)
df[df["is_anomaly"]].to_csv("data/processed/anomalies_only.csv", index=False)
