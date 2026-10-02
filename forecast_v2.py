import os
import sqlite3
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

# ---------- 1. Daily energy per household (full days only) from SQLite ----------
conn = sqlite3.connect("data/energy.db")
daily = pd.read_sql_query(
    """SELECT household_id, date, SUM(energy_kwh) AS energy_kwh
       FROM energy_readings GROUP BY household_id, date HAVING COUNT(*) = 48""",
    conn,
)
conn.close()
daily["date"] = pd.to_datetime(daily["date"])
daily = daily[daily["date"] >= "2012-10-01"]

# ---------- 2. Lag features on a complete daily calendar ----------
wide = daily.pivot(index="date", columns="household_id", values="energy_kwh")
last_day = wide.index.max()
target_day = last_day + pd.Timedelta(days=1)  # the day we will forecast at the end
wide = wide.reindex(pd.date_range(wide.index.min(), target_day, freq="D"))
wide.index.name = "date"

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    wk_avg = np.nanmean(np.stack([wide.shift(k).values for k in (7, 14, 21, 28)]), axis=0)

frames = {
    "energy_kwh": wide,
    "lag_1": wide.shift(1),                                       # yesterday
    "lag_7": wide.shift(7),                                       # same weekday last week
    "roll_7": wide.shift(1).rolling(7, min_periods=5).mean(),     # average of the last 7 days
    "wk_avg": pd.DataFrame(wk_avg, index=wide.index, columns=wide.columns),  # same weekday, last 4 weeks
}
full = pd.concat({name: f.stack() for name, f in frames.items()}, axis=1).reset_index()

# ---------- 3. Calendar, holiday, weather and tariff features ----------
full["day_of_week"] = full["date"].dt.dayofweek
full["is_weekend"] = (full["day_of_week"] >= 5).astype(int)
full["month"] = full["date"].dt.month

bank = pd.read_csv("data/raw/uk_bank_holidays.csv")
holiday_dates = set(pd.to_datetime(bank["Bank holidays"]).dt.normalize())
full["is_holiday"] = full["date"].isin(holiday_dates).astype(int)

weather = pd.read_csv("data/raw/weather_daily_darksky.csv", parse_dates=["time"])
weather["date"] = weather["time"].dt.normalize()
weather = weather.rename(columns={
    "temperatureMax": "temp_max", "temperatureMin": "temp_min",
    "windSpeed": "wind_speed", "cloudCover": "cloud_cover",
})
WEATHER = ["temp_max", "temp_min", "humidity", "wind_speed", "cloud_cover"]
weather = weather[["date"] + WEATHER].drop_duplicates("date")
full = full.merge(weather, on="date", how="left")
full[WEATHER] = full[WEATHER].fillna(full[WEATHER].median())

info = pd.read_csv("data/raw/informations_households.csv")
tou_ids = set(info.loc[info["stdorToU"] == "ToU", "LCLid"])
mine = set(full["household_id"].unique())
print(f"Households in the data: {len(mine)}, on the time-of-use tariff: {len(mine & tou_ids)}")
# Assumption: the dynamic tariff was in force during calendar year 2013
full["tou_active"] = (full["household_id"].isin(tou_ids) & (full["date"].dt.year == 2013)).astype(int)

# Sanity check: does colder weather go with higher usage?
check = full.groupby("date")[["energy_kwh", "temp_max"]].mean().corr().round(2)
print("\nCorrelation, daily energy vs max temperature:", check.loc["energy_kwh", "temp_max"])

# ---------- 4. Feature sets to compare ----------
BASE = ["lag_1", "lag_7", "roll_7", "wk_avg", "day_of_week", "is_weekend", "month"]
FEATURE_SETS = {
    "1 original features": ["lag_1", "lag_7", "day_of_week", "is_weekend", "month"],
    "2 + rolling averages": BASE,
    "3 + bank holidays": BASE + ["is_holiday"],
    "4 + weather": BASE + ["is_holiday"] + WEATHER,
    "5 + tariff group": BASE + ["is_holiday"] + WEATHER + ["tou_active"],
}


def make_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
    }


need = ["energy_kwh", "lag_1", "lag_7", "roll_7", "wk_avg"]
data = full.dropna(subset=need).copy()
print("Rows used for training and testing:", len(data))

# ---------- 5. Walk-forward validation: train on the past, test on the next 2 months, repeat ----------
FOLD_STARTS = list(pd.to_datetime(["2013-04-01", "2013-06-01", "2013-08-01", "2013-10-01", "2013-12-01"]))
FOLD_ENDS = FOLD_STARTS[1:] + [data["date"].max() + pd.Timedelta(days=1)]
labels = [f"{s:%b %Y}" for s in FOLD_STARTS]

rows = []
for start, end, label in zip(FOLD_STARTS, FOLD_ENDS, labels):
    train = data[data["date"] < start]
    test = data[(data["date"] >= start) & (data["date"] < end)]
    for name, col in (("Baseline: same as yesterday", "lag_1"), ("Baseline: same day last week", "lag_7")):
        rows.append({"features": "-", "model": name, "fold": label,
                     "mae": mean_absolute_error(test["energy_kwh"], test[col])})
    for fs_name, cols in FEATURE_SETS.items():
        for model_name, model in make_models().items():
            model.fit(train[cols], train["energy_kwh"])
            pred = model.predict(test[cols])
            rows.append({"features": fs_name, "model": model_name, "fold": label,
                         "mae": mean_absolute_error(test["energy_kwh"], pred)})
    print("finished fold starting", label)

res = pd.DataFrame(rows)
table = res.pivot_table(index=["features", "model"], columns="fold", values="mae", sort=False)[labels]
table["mean MAE"] = table.mean(axis=1)
print("\nMAE in kWh per household per day (lower is better):")
print(table.round(3).to_string())

os.makedirs("models", exist_ok=True)
table.round(3).reset_index().to_csv("models/forecast_walkforward.csv", index=False)

# ---------- 6. Pick the best, refit on all data, forecast the next day ----------
models_only = table[table.index.get_level_values("features") != "-"]
best_fs, best_model_name = models_only["mean MAE"].round(3).idxmin()
best_mae = models_only["mean MAE"].min()
b1 = table.loc[("-", "Baseline: same as yesterday"), "mean MAE"]
b7 = table.loc[("-", "Baseline: same day last week"), "mean MAE"]
print(f"\nBest: {best_model_name} with '{best_fs}', mean MAE {best_mae:.3f}")
print(f"Baselines: same as yesterday {b1:.3f}, same day last week {b7:.3f}")
print(f"Improvement over 'same as yesterday': {100 * (1 - best_mae / b1):.1f}%")

cols = FEATURE_SETS[best_fs]
final_model = make_models()[best_model_name]
final_model.fit(data[cols], data["energy_kwh"])
joblib.dump({"model": final_model, "features": cols}, "models/best_forecast_model_v2.joblib")

target_rows = full[full["date"] == target_day].dropna(subset=["lag_1", "lag_7", "roll_7", "wk_avg"]).copy()
target_rows["forecast_kwh"] = final_model.predict(target_rows[cols])
target_rows["target_date"] = target_day.date()
target_rows[["household_id", "target_date", "forecast_kwh"]].to_csv("models/forecast_latest.csv", index=False)
print(f"Forecast for {target_day.date()}: {target_rows['forecast_kwh'].mean():.1f} kWh per household "
      f"({len(target_rows)} households)")

# ---------- 7. Chart: actual vs predicted over the final test fold ----------
start, end = FOLD_STARTS[-1], FOLD_ENDS[-1]
m = make_models()[best_model_name]
m.fit(data[data["date"] < start][cols], data[data["date"] < start]["energy_kwh"])
test = data[(data["date"] >= start) & (data["date"] < end)].copy()
test["predicted"] = m.predict(test[cols])
plot = test.groupby("date")[["energy_kwh", "predicted"]].mean()
plot.plot(figsize=(10, 4), title=f"Actual vs predicted, {labels[-1]} onward ({best_model_name})")
plt.ylabel("kWh per household per day")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/actual_vs_predicted_v2.png")
plt.close()