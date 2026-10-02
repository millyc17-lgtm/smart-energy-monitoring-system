import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from detectors import isolation_forest_scores, rule_baseline, rule_flags

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

SEED = 42
SIZES = [0.5, 1.0, 2.0, 4.0]   # extra kWh added to each affected half-hour reading
DURATIONS = [1, 6]              # readings per event: 1 = short spike, 6 = sustained for 3 hours
EVENTS_PER_HOUSEHOLD = 12
MIN_HISTORY = 20 * 48           # keep events away from the start of each household's data

# ---------- Load (from Oct 2012, when the panel is stable) ----------
conn = sqlite3.connect("data/energy.db")
df = pd.read_sql_query(
    "SELECT household_id, timestamp, energy_kwh FROM energy_readings WHERE date >= '2012-10-01'",
    conn,
    parse_dates=["timestamp"],
)
conn.close()
df = df.sort_values(["household_id", "timestamp"]).reset_index(drop=True)
df["event_id"] = -1
df["event_size"] = 0.0
df["event_len"] = 0

# ---------- Inject artificial events (in memory only) ----------
rng = np.random.default_rng(SEED)
combos = [(size, dur) for size in SIZES for dur in DURATIONS]
event_id = 0
for hh, idx in df.groupby("household_id").indices.items():
    usable = len(idx) - MIN_HISTORY - max(DURATIONS)
    n_events = min(EVENTS_PER_HOUSEHOLD, usable // 400)
    if n_events < 1:
        continue
    seg = usable // n_events
    for e in range(n_events):
        start = MIN_HISTORY + e * seg + int(rng.integers(0, seg - max(DURATIONS)))
        size, dur = combos[event_id % len(combos)]
        rows = idx[start:start + dur]
        df.loc[rows, "energy_kwh"] += size
        df.loc[rows, "event_id"] = event_id
        df.loc[rows, "event_size"] = size
        df.loc[rows, "event_len"] = dur
        event_id += 1

injected = df["event_id"] >= 0
print(f"Injected {event_id} events ({injected.sum()} readings) across {df['household_id'].nunique()} households")

# ---------- Run both detectors on the modified data ----------
print("Running the rolling rule...")
base = rule_baseline(df)
checked_rule = base["normal_mean"].notna()
print("Running the Isolation Forest (this takes a few minutes)...")
scores = isolation_forest_scores(df)
checked_if = scores.notna()
hh_level = df.loc[~injected].groupby("household_id")["energy_kwh"].mean()


def evaluate(method, setting, flagged, checked):
    flagged = flagged & checked
    ev = (
        df.loc[injected]
        .assign(flag=flagged[injected])
        .groupby("event_id")
        .agg(caught=("flag", "any"), size=("event_size", "first"),
             dur=("event_len", "first"), hh=("household_id", "first"))
    )
    untouched = (~injected) & checked
    flag_rate = flagged[untouched].mean()
    summary = {
        "method": method,
        "setting": setting,
        "flag_rate_untouched_%": 100 * flag_rate,
        "flags_per_household_per_week": flag_rate * 48 * 7,
        "event_recall_%": 100 * ev["caught"].mean(),
        "precision_lower_bound_%": 100 * flagged[injected].sum() / max(flagged.sum(), 1),
    }
    return summary, ev


rows, details = [], {}
for k in [2, 3, 4, 5]:
    s, ev = evaluate("Rolling rule", f"K = {k}", rule_flags(base, k), checked_rule)
    rows.append(s)
    details[("Rolling rule", f"K = {k}")] = ev

valid_scores = scores[checked_if]
for c in [0.01, 0.03, 0.05]:
    flagged = scores < valid_scores.quantile(c)
    s, ev = evaluate("Isolation Forest", f"flag lowest {c:.0%}", flagged, checked_if)
    rows.append(s)
    details[("Isolation Forest", f"flag lowest {c:.0%}")] = ev

summary = pd.DataFrame(rows).round(2)
print("\nSummary (event_recall = % of planted events detected):")
print(summary.to_string(index=False))
summary.to_csv("models/anomaly_validation.csv", index=False)

# ---------- Detail for the default settings ----------
defaults = {
    "Rolling rule (K = 3)": details[("Rolling rule", "K = 3")],
    "Isolation Forest (lowest 3%)": details[("Isolation Forest", "flag lowest 3%")],
}
for name, ev in defaults.items():
    print(f"\n{name}: % of events detected, by extra kWh (rows) and event length in readings (columns)")
    print((ev.pivot_table(index="size", columns="dur", values="caught", aggfunc="mean") * 100).round(1))
    ev["ratio"] = ev["size"] / ev["hh"].map(hh_level)
    bucket = pd.cut(ev["ratio"], [0, 1, 3, np.inf], labels=["under 1x", "1x to 3x", "over 3x"])
    r = ev.groupby(bucket, observed=True)["caught"].agg(recall="mean", events="count")
    r["recall"] = (r["recall"] * 100).round(1)
    print("Detection by spike size relative to the household's typical half-hour usage:")
    print(r)

# ---------- Chart ----------
fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(SIZES))
w = 0.2
series = []
for name, ev in (("Rule", defaults["Rolling rule (K = 3)"]), ("Isolation Forest", defaults["Isolation Forest (lowest 3%)"])):
    pt = ev.pivot_table(index="size", columns="dur", values="caught", aggfunc="mean") * 100
    for dur in DURATIONS:
        series.append((f"{name}, {'1 reading' if dur == 1 else '3 hours'}", pt[dur].reindex(SIZES).values))
for i, (label, vals) in enumerate(series):
    ax.bar(x + (i - 1.5) * w, vals, w, label=label)
ax.set_xticks(x)
ax.set_xticklabels([f"+{s} kWh" for s in SIZES])
ax.set_xlabel("Extra energy added to each affected half hour")
ax.set_ylabel("Events detected (%)")
ax.set_title("Detection rate for injected anomalies (default settings)")
ax.set_ylim(0, 105)
ax.legend()
plt.tight_layout()
plt.savefig("outputs/anomaly_validation.png")
plt.close()
print("\nSaved models/anomaly_validation.csv and outputs/anomaly_validation.png")