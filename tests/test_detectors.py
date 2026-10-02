import numpy as np
import pandas as pd

from detectors import isolation_forest_scores, rule_baseline, rule_flags, rule_upper_limit


def make_readings(days=30, households=("A",), seed=0):
    """Half-hourly readings with a daily pattern and a little noise."""
    rng = np.random.default_rng(seed)
    frames = []
    for h in households:
        ts = pd.date_range("2013-01-01", periods=days * 48, freq="30min")
        base = 0.3 + 0.2 * np.sin(2 * np.pi * (ts.hour * 2 + ts.minute // 30) / 48)
        frames.append(pd.DataFrame({
            "household_id": h,
            "timestamp": ts,
            "energy_kwh": base + rng.normal(0, 0.02, len(ts)),
        }))
    return pd.concat(frames, ignore_index=True).sort_values(["household_id", "timestamp"]).reset_index(drop=True)


def test_spike_is_flagged_and_ordinary_readings_mostly_are_not():
    df = make_readings()
    spike_at = df.index[df["timestamp"] == pd.Timestamp("2013-01-25 12:00:00")][0]
    df.loc[spike_at, "energy_kwh"] += 3.0
    flags = rule_flags(rule_baseline(df), k=3)
    assert flags[spike_at]
    assert flags.mean() < 0.02


def test_spike_cannot_hide_in_its_own_baseline():
    df = make_readings()
    spike_at = df.index[df["timestamp"] == pd.Timestamp("2013-01-25 12:00:00")][0]
    clean_base = rule_baseline(df).loc[spike_at, "normal_mean"]
    df.loc[spike_at, "energy_kwh"] += 50.0
    with_spike = rule_baseline(df).loc[spike_at, "normal_mean"]
    assert with_spike == clean_base          # the reading is not part of its own baseline


def test_nothing_is_flagged_without_enough_history():
    df = make_readings(days=30)
    first_days = df["timestamp"] < pd.Timestamp("2013-01-08")       # fewer than 7 earlier readings per slot
    df.loc[first_days & (df["timestamp"].dt.hour == 12), "energy_kwh"] = 99.0
    base = rule_baseline(df)
    flags = rule_flags(base, k=3)
    assert not flags[first_days].any()
    assert base.loc[first_days, "normal_mean"].isna().all()


def test_higher_k_never_flags_more():
    df = rule_baseline(make_readings())
    counts = [int(rule_flags(df, k).sum()) for k in (2, 3, 4, 5)]
    assert counts == sorted(counts, reverse=True)
    assert (rule_upper_limit(df, 5).dropna() >= rule_upper_limit(df, 2).dropna()).all()


def test_households_are_compared_only_with_themselves():
    df = make_readings(households=("A", "B"))
    df.loc[df["household_id"] == "B", "energy_kwh"] += 5.0          # B always uses far more than A
    flags = rule_flags(rule_baseline(df), k=3)
    assert flags.mean() < 0.02                                       # neither household is flagged for being heavy


def test_isolation_forest_flags_a_sustained_event():
    df = make_readings(days=30, households=("A", "B"))
    start = df.index[(df["household_id"] == "A") & (df["timestamp"] == pd.Timestamp("2013-01-25 03:00:00"))][0]
    event = list(range(start, start + 6))                            # 3 hours of extra use
    df.loc[event, "energy_kwh"] += 4.0
    scores = isolation_forest_scores(df)
    assert scores.isna().sum() > 0                                   # the first readings have no history
    # a single-reading spike can be missed by this detector (see the README validation),
    # but at least one reading of a 3-hour event should be among the 3% most unusual
    assert scores[event].min() < scores.dropna().quantile(0.03)
