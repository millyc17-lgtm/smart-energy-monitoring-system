import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/energy_features.csv", parse_dates=["timestamp"])
os.makedirs("outputs", exist_ok=True)

# 1. Average power by hour of day
by_hour = df.groupby("hour")["power_kw"].mean()
print("Busiest hour:", by_hour.idxmax(), "->", round(by_hour.max(), 3), "kW")
by_hour.plot(kind="bar", figsize=(10, 4), title="Average power by hour (kW)")
plt.xlabel("Hour of day")
plt.tight_layout()
plt.savefig("outputs/power_by_hour.png")
plt.close()

# 2. Daily energy per household (keep only full days: 48 half-hour readings)
daily = (
    df.groupby(["household_id", "date"])
    .agg(energy_kwh=("energy_kwh", "sum"), readings=("energy_kwh", "count"))
    .reset_index()
)
daily = daily[daily["readings"] == 48].copy()
daily["date"] = pd.to_datetime(daily["date"])
daily["day_of_week"] = daily["date"].dt.day_name()
daily["is_weekend"] = daily["date"].dt.dayofweek >= 5
daily["month"] = daily["date"].dt.to_period("M")

# 3. Average daily energy by day of week
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
by_day = daily.groupby("day_of_week")["energy_kwh"].mean().reindex(order)
print(by_day.round(2))
by_day.plot(kind="bar", figsize=(8, 4), title="Average daily energy per household (kWh)")
plt.tight_layout()
plt.savefig("outputs/energy_by_weekday.png")
plt.close()

# 4. Weekday vs weekend
print(daily.groupby("is_weekend")["energy_kwh"].mean().round(2))

# 5. Monthly trend (average kWh per household per day)
monthly = daily.groupby("month")["energy_kwh"].mean()
monthly = monthly[monthly.index >= pd.Period("2012-10")]
monthly.index = monthly.index.astype(str)
monthly.plot(figsize=(10, 4), title="Average daily energy per household by month (kWh)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/monthly_trend.png")
plt.close()

# 6. Power extremes
print("Max power:", round(df["power_kw"].max(), 2), "kW")
print("Average power:", round(df["power_kw"].mean(), 3), "kW")
print("Min power:", round(df["power_kw"].min(), 2), "kW")