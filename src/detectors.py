import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def rule_baseline(df):
    """Add the rolling 'normal' mean and std for each household and half-hour slot.

    df needs household_id, timestamp and energy_kwh, sorted by household_id then timestamp.
    The previous 14 readings at the same slot are used (shifted by one, so a reading is
    never part of its own baseline). At least 7 are needed, otherwise the baseline is NaN.
    """
    df = df.copy()
    df["slot"] = df["timestamp"].dt.hour * 2 + df["timestamp"].dt.minute // 30
    g = df.groupby(["household_id", "slot"])["energy_kwh"]
    df["normal_mean"] = g.transform(lambda s: s.shift(1).rolling(14, min_periods=7).mean())
    df["normal_std"] = g.transform(lambda s: s.shift(1).rolling(14, min_periods=7).std())
    return df


def rule_upper_limit(df, k=3):
    """Upper limit of normal: mean + k standard deviations (std floored at 0.05 kWh)."""
    return df["normal_mean"] + k * df["normal_std"].clip(lower=0.05)


def rule_flags(df, k=3):
    """True where a reading is more than k standard deviations above its normal."""
    upper = rule_upper_limit(df, k)
    return (df["energy_kwh"] > upper) & upper.notna()


def isolation_forest_scores(df, random_state=42):
    """Anomaly score per row (lower = more unusual). Rows without enough history get NaN.

    df must be sorted by household_id then timestamp.
    """
    df = df.copy()
    g = df.groupby("household_id")["energy_kwh"]
    df["prev_energy"] = g.shift(1)
    df["rolling_avg"] = g.transform(lambda s: s.shift(1).rolling(6).mean())
    hh_mean = g.transform("mean").clip(lower=0.01)
    df["rel_energy"] = df["energy_kwh"] / hh_mean
    df["rel_prev"] = df["prev_energy"] / hh_mean
    df["rel_rolling"] = df["rolling_avg"] / hh_mean
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek

    features = ["rel_energy", "rel_prev", "rel_rolling", "hour", "weekday"]
    ok = df[features].notna().all(axis=1)
    model = IsolationForest(n_estimators=100, contamination=0.03, random_state=random_state, n_jobs=-1)
    model.fit(df.loc[ok, features])
    scores = pd.Series(np.nan, index=df.index)
    scores[ok] = model.decision_function(df.loc[ok, features])
    return scores
