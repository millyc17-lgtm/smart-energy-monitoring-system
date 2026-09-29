import pandas as pd
from sklearn.ensemble import IsolationForest

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

df = pd.read_csv("data/processed/energy_anomalies.csv", parse_dates=["timestamp"])
df = df.sort_values(["household_id", "timestamp"])

# Build features
g = df.groupby("household_id")["energy_kwh"]
df["prev_energy"] = g.shift(1)
df["rolling_avg"] = g.transform(lambda s: s.shift(1).rolling(6).mean())  # last 3 hours
df["hh_mean"] = g.transform("mean").clip(lower=0.01)
df["weekday"] = df["timestamp"].dt.dayofweek

df["rel_energy"] = df["energy_kwh"] / df["hh_mean"]
df["rel_prev"] = df["prev_energy"] / df["hh_mean"]
df["rel_rolling"] = df["rolling_avg"] / df["hh_mean"]

features = ["rel_energy", "rel_prev", "rel_rolling", "hour", "weekday"]
data = df.dropna(subset=features).copy()

# Train the model. contamination = share of readings we expect to be unusual.
model = IsolationForest(n_estimators=100, contamination=0.03, random_state=42, n_jobs=-1)
data["ml_anomaly"] = model.fit_predict(data[features]) == -1
data["ml_score"] = model.decision_function(data[features])  # lower = more unusual

print("ML anomalies:", data["ml_anomaly"].sum(), "of", len(data))

# Compare with the rule-based detector
print("\nRule-based (rows) vs ML (columns):")
print(pd.crosstab(data["is_anomaly"], data["ml_anomaly"]))

print("\nMost unusual according to ML:")
cols = ["household_id", "timestamp", "energy_kwh", "normal_mean", "ml_score"]
print(data[data["ml_anomaly"]].nsmallest(10, "ml_score")[cols].round({"energy_kwh": 2, "normal_mean": 2, "ml_score": 3}))

data[data["ml_anomaly"]].to_csv("data/processed/ml_anomalies_only.csv", index=False)