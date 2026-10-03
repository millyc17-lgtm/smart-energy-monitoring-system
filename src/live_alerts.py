"""A simple rule for spotting unusual readings in a live stream.

Each reading is compared with the previous `window` readings from the same device
(48 half-hours = the previous day). A reading is "unusual" if it is more than k
standard deviations above that recent average. This is a lighter rule than the
Anomalies page, because a live stream has little history to learn from.
"""
import pandas as pd

MIN_STD = 0.05      # stops tiny wobbles in a very steady device from counting as unusual


def flag_unusual(df, window=48, k=3.0, min_history=12):
    """Add `baseline`, `upper_limit` and `is_unusual` columns to a table of readings.

    `df` needs `timestamp` and `power_kw` columns for ONE device. The first `min_history`
    readings are never flagged, because there is nothing to compare them with yet.
    """
    out = df.sort_values("timestamp").reset_index(drop=True).copy()
    previous = out["power_kw"].shift(1)
    roll = previous.rolling(window, min_periods=min_history)
    out["baseline"] = roll.mean()
    out["upper_limit"] = out["baseline"] + k * roll.std().fillna(0).clip(lower=MIN_STD)
    out["is_unusual"] = (out["power_kw"] > out["upper_limit"]).fillna(False)
    return out