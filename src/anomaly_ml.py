import pandas as pd

from detectors import isolation_forest_scores

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

df = pd.read_csv("data/processed/energy_anomalies.csv", parse_dates=["timestamp"])
df = df.sort_values(["household_id", "timestamp"]).reset_index(drop=True)

df["ml_score"] = isolation_forest_scores(df)       # lower = more unusual, NaN without enough history
data = df.dropna(subset=["ml_score"]).copy()
data["ml_anomaly"] = data["ml_score"] < 0          # the model's own normal/anomaly boundary

print("ML anomalies:", data["ml_anomaly"].sum(), "of", len(data))

print("\nRule-based (rows) vs ML (columns):")
print(pd.crosstab(data["is_anomaly"], data["ml_anomaly"]))

print("\nMost unusual according to ML:")
cols = ["household_id", "timestamp", "energy_kwh", "normal_mean", "ml_score"]
print(data[data["ml_anomaly"]].nsmallest(10, "ml_score")[cols].round({"energy_kwh": 2, "normal_mean": 2, "ml_score": 3}))

data[data["ml_anomaly"]].to_csv("data/processed/ml_anomalies_only.csv", index=False)
