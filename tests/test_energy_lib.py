import numpy as np
import pandas as pd

from energy_lib import add_features, clean_readings, make_lag_frame


def raw(rows):
    return pd.DataFrame(rows, columns=["household_id", "timestamp", "energy_kwh"])


def test_clean_removes_duplicates_and_bad_values():
    df = raw([
        ("A", "2013-01-01 00:00:00", 0.5),
        ("A", "2013-01-01 00:00:00", 0.5),      # duplicate reading
        ("A", "2013-01-01 00:30:00", "Null"),   # text instead of a number
        ("A", "2013-01-01 01:00:00", 0.2),
    ])
    out = clean_readings(df)
    assert len(out) == 2
    assert out["energy_kwh"].tolist() == [0.5, 0.2]
    assert out["energy_kwh"].dtype == float


def test_clean_sorts_by_household_then_time_and_keeps_input_unchanged():
    df = raw([
        ("B", "2013-01-01 00:30:00", 1.0),
        ("A", "2013-01-01 01:00:00", 2.0),
        ("A", "2013-01-01 00:00:00", 3.0),
    ])
    before = df.copy()
    out = clean_readings(df)
    assert out["household_id"].tolist() == ["A", "A", "B"]
    assert out["timestamp"].is_monotonic_increasing is False  # sorted per household, not globally
    assert out.loc[out["household_id"] == "A", "timestamp"].is_monotonic_increasing
    pd.testing.assert_frame_equal(df, before)


def test_add_features_time_power_and_cost():
    df = clean_readings(raw([
        ("A", "2013-06-15 19:30:00", 0.8),   # a Saturday
        ("A", "2013-06-17 03:00:00", 0.1),   # a Monday
    ]))
    out = add_features(df, price_per_kwh=0.30)
    sat, mon = out.iloc[0], out.iloc[1]
    assert sat["hour"] == 19 and sat["day_of_week"] == "Saturday" and bool(sat["is_weekend"])
    assert mon["day_of_week"] == "Monday" and not bool(mon["is_weekend"])
    assert sat["month"] == 6
    assert abs(sat["power_kw"] - 1.6) < 1e-9          # 0.8 kWh in half an hour = 1.6 kW
    assert abs(sat["cost"] - 0.24) < 1e-9


def make_daily(households=("A",), days=40, start="2013-01-01"):
    rows = []
    for h_i, h in enumerate(households):
        for d in range(days):
            rows.append((h, pd.Timestamp(start) + pd.Timedelta(days=d), 10.0 * (h_i + 1) + d))
    return pd.DataFrame(rows, columns=["household_id", "date", "energy_kwh"])


def test_lags_match_calendar_days():
    daily = make_daily(days=40)
    full, target = make_lag_frame(daily)
    assert target == pd.Timestamp("2013-02-10")
    row = full[full["date"] == pd.Timestamp("2013-02-05")].iloc[0]   # day index 35, energy 45
    assert row["energy_kwh"] == 45
    assert row["lag_1"] == 44
    assert row["lag_7"] == 38
    assert abs(row["roll_7"] - np.mean([38, 39, 40, 41, 42, 43, 44])) < 1e-9
    assert abs(row["wk_avg"] - np.mean([38, 31, 24, 17])) < 1e-9


def test_target_day_row_exists_with_lags_but_no_energy():
    daily = make_daily(days=40)
    full, target = make_lag_frame(daily)
    row = full[full["date"] == target].iloc[0]
    assert np.isnan(row["energy_kwh"])
    assert row["lag_1"] == 49          # last observed day, 10 + 39
    assert row["lag_7"] == 43


def test_missing_day_gives_nan_not_a_shifted_value():
    daily = make_daily(days=40)
    daily = daily[daily["date"] != pd.Timestamp("2013-01-20")]       # drop one day
    full, _ = make_lag_frame(daily)
    day_after = full[full["date"] == pd.Timestamp("2013-01-21")].iloc[0]
    assert np.isnan(day_after["lag_1"])                               # yesterday is missing
    assert day_after["lag_7"] == 10 + 13                              # 2013-01-14 is still found by date


def test_households_do_not_leak_into_each_other():
    daily = make_daily(households=("A", "B"), days=40)
    full, _ = make_lag_frame(daily)
    a = full[(full["household_id"] == "A") & (full["date"] == pd.Timestamp("2013-02-01"))].iloc[0]
    b = full[(full["household_id"] == "B") & (full["date"] == pd.Timestamp("2013-02-01"))].iloc[0]
    assert a["lag_1"] == 10 + 30 and b["lag_1"] == 20 + 30
    assert a["lag_7"] == 10 + 24 and b["lag_7"] == 20 + 24
