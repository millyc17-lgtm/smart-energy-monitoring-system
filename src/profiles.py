"""Group households by the shape of their typical day (load profile).

Reads data/energy.db and writes models/household_profiles.csv.
Run from the project folder:  python src/profiles.py
"""
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

start = time.time()
DB = Path("data/energy.db")
OUT = Path("models/household_profiles.csv")
MIN_READINGS = 5000      # skip households with very little data (about 100 days)
K = 3                    # number of groups (see the silhouette scores printed below)

# 1. Average power (kW) for each household at each hour of the day
con = sqlite3.connect(DB)
hourly = pd.read_sql_query(
    """SELECT household_id, hour, AVG(power_kw) AS power_kw, COUNT(*) AS readings
       FROM energy_readings GROUP BY household_id, hour""",
    con,
)
con.close()

counts = hourly.groupby("household_id")["readings"].sum()
keep = counts[counts >= MIN_READINGS].index
hourly = hourly[hourly["household_id"].isin(keep)]
print(f"Households used: {len(keep)} (skipped {len(counts) - len(keep)} with too little data)")

profile = hourly.pivot(index="household_id", columns="hour", values="power_kw")
profile = profile.dropna()                      # need all 24 hours
avg_kw = profile.mean(axis=1)

# 2. Compare SHAPE, not size: divide each household by its own average,
#    so a big and a small household with the same daily pattern end up together.
shape = profile.div(avg_kw, axis=0)

# 3. How many groups? A higher silhouette score means more clearly separated groups.
print("\nSilhouette score for different numbers of groups (higher is better):")
for k in range(2, 7):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(shape)
    print(f"  {k} groups: {silhouette_score(shape, labels):.3f}")

# 4. Final grouping
labels = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(shape)
result = pd.DataFrame({"household_id": shape.index, "raw": labels, "avg_kw": avg_kw.values})

# Number the groups by the hour their average shape peaks, so Group 1 is always the earliest peak
group_peak = {c: int(shape[labels == c].mean().idxmax()) for c in range(K)}
order = sorted(group_peak, key=group_peak.get)
rename = {c: i + 1 for i, c in enumerate(order)}
result["group"] = result["raw"].map(rename)
result = result.drop(columns="raw")

# 5. Save in long format: one row per household per hour
shape_long = shape.stack().rename("relative").reset_index()
power_long = profile.stack().rename("power_kw").reset_index()
out = (
    result.merge(power_long, on="household_id")
    .merge(shape_long, on=["household_id", "hour"])
    .sort_values(["group", "household_id", "hour"])
)
OUT.parent.mkdir(exist_ok=True)
out.to_csv(OUT, index=False)

# 6. Summary
print("\nGroups:")
for g, rows in result.groupby("group"):
    mean_shape = shape.loc[rows["household_id"]].mean()
    print(f"  Group {g}: {len(rows)} households, average {rows['avg_kw'].mean():.2f} kW, "
          f"peak at {int(mean_shape.idxmax()):02d}:00, quietest at {int(mean_shape.idxmin()):02d}:00")
print(f"\nSaved {OUT}")
print(f"--- profiles.py finished in {time.time() - start:.0f}s")