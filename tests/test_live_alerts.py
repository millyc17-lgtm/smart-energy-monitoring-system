import pandas as pd

from live_alerts import flag_unusual


def make(values):
    return pd.DataFrame({
        "timestamp": pd.date_range("2013-09-13", periods=len(values), freq="30min"),
        "power_kw": values,
    })


def test_a_spike_is_flagged():
    values = [0.4, 0.5] * 30 + [7.0] + [0.4, 0.5] * 5
    out = flag_unusual(make(values))
    assert out["is_unusual"].sum() == 1
    assert out.loc[out["is_unusual"], "power_kw"].iloc[0] == 7.0


def test_steady_readings_are_not_flagged():
    out = flag_unusual(make([0.5] * 100))
    assert not out["is_unusual"].any()


def test_early_readings_are_never_flagged():
    out = flag_unusual(make([0.1, 9.0, 0.1, 9.0, 0.1]))
    assert not out["is_unusual"].any()
    assert out["baseline"].isna().all()


def test_unsorted_input_is_sorted_first():
    df = make([0.4, 0.5] * 30 + [7.0]).sample(frac=1, random_state=1)
    out = flag_unusual(df)
    assert out["timestamp"].is_monotonic_increasing and out["is_unusual"].sum() == 1