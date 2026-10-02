"""Reusable data functions for the pipeline (unit tested in tests/)."""
import warnings

import numpy as np
import pandas as pd


def clean_readings(df):
    """Clean raw readings with columns household_id, timestamp, energy_kwh.

    Parses timestamps, turns non-numeric energy values (such as the text 'Null') into
    missing, drops duplicate household/timestamp readings and missing energy values,
    and sorts by household then time. Returns a new DataFrame.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["energy_kwh"] = pd.to_numeric(df["energy_kwh"], errors="coerce")
    df = df.drop_duplicates(["household_id", "timestamp"])
    df = df.dropna(subset=["energy_kwh"])
    return df.sort_values(["household_id", "timestamp"]).reset_index(drop=True)


def add_features(df, price_per_kwh=0.25):
    """Add date/time columns, average power (kW) and cost.

    The data is energy per half hour, so average power in kW is kWh x 2.
    """
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
    df["power_kw"] = df["energy_kwh"] * 2
    df["cost"] = df["energy_kwh"] * price_per_kwh
    return df


def make_lag_frame(daily, extra_days=1):
    """Build lag features on a complete daily calendar.

    daily needs household_id, date (datetime) and energy_kwh (one row per household per day).
    Lags are matched by calendar date, so a missing day gives NaN and is never filled with
    a different day's value, and households never mix.

    Returns (full, target_day). `full` has one row per household per calendar day, including
    `extra_days` days after the last observed day (energy_kwh is NaN there), with columns:
    energy_kwh, lag_1 (yesterday), lag_7 (same weekday last week), roll_7 (mean of the last
    7 days) and wk_avg (mean of the same weekday over the last 4 weeks).
    """
    wide = daily.pivot(index="date", columns="household_id", values="energy_kwh")
    target_day = wide.index.max() + pd.Timedelta(days=extra_days)
    wide = wide.reindex(pd.date_range(wide.index.min(), target_day, freq="D"))
    wide.index.name = "date"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wk_avg = np.nanmean(np.stack([wide.shift(k).values for k in (7, 14, 21, 28)]), axis=0)

    frames = {
        "energy_kwh": wide,
        "lag_1": wide.shift(1),
        "lag_7": wide.shift(7),
        "roll_7": wide.shift(1).rolling(7, min_periods=5).mean(),
        "wk_avg": pd.DataFrame(wk_avg, index=wide.index, columns=wide.columns),
    }
    full = pd.concat({name: f.unstack() for name, f in frames.items()}, axis=1)
    full = full.rename_axis(["household_id", "date"]).reset_index()
    return full.sort_values(["household_id", "date"]).reset_index(drop=True), target_day
