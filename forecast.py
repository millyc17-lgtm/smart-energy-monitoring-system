import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option("display.width", 200)

df = pd.read_csv("data/processed/energy_features.csv", parse_dates=["timestamp"])

# Daily energy per household (full days only), from the reliable period onwards
daily = (
    df.groupby(["household_id", "date"])
    .agg(energy_kwh=("energy_kwh", "sum"), readings=("energy_kwh", "count"))
    .reset_index()
)
daily = daily[daily["readings"] == 48].drop(columns="readings")
daily["date"] = pd.to_datetime(daily["date"])
daily = daily[daily["date"] >= "2012-10-01"]


def add_lag(d, days, name):
    """Attach the same household's energy from `days` days earlier (matched by calendar date)."""
    lag = d[["household_id", "date", "energy_kwh"]].copy()
    lag["date"] = lag["date"] + pd.Timedelta(days=days)
    return d.merge(lag.rename(columns={"energy_kwh": name}), on=["household_id", "date"], how="left")


daily = add_lag(daily, 1, "lag_1")   # yesterday
daily = add_lag(daily, 7, "lag_7")   # same day last week
daily["day_of_week"] = daily["date"].dt.dayofweek
daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
daily["month"] = daily["date"].dt.month
daily = daily.dropna(subset=["lag_1", "lag_7"])

features = ["lag_1", "lag_7", "day_of_week", "is_weekend", "month"]

# Time-based split: train on the past, test on the future
dates = np.sort(daily["date"].unique())
cutoff = dates[int(len(dates) * 0.8)]
train = daily[daily["date"] < cutoff]
test = daily[daily["date"] >= cutoff].copy()
print("Training days:", train["date"].min().date(), "to", train["date"].max().date())
print("Testing days: ", test["date"].min().date(), "to", test["date"].max().date())
print("Average daily energy in test set:", round(test["energy_kwh"].mean(), 2), "kWh\n")

y_train, y_test = train["energy_kwh"], test["energy_kwh"]


def score(name, pred):
    return {
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
    }


results = [
    score("Baseline: same as yesterday", test["lag_1"]),
    score("Baseline: same as last week", test["lag_7"]),
]

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}
fitted = {}
for name, model in models.items():
    model.fit(train[features], y_train)
    test[name] = model.predict(test[features])
    fitted[name] = model
    results.append(score(name, test[name]))

table = pd.DataFrame(results).round(3)
print(table.to_string(index=False))

# Pick the best of the three models by MAE
model_rows = table[~table["Model"].str.startswith("Baseline")]
best_name = model_rows.sort_values("MAE").iloc[0]["Model"]
print("\nBest model by MAE:", best_name)

os.makedirs("models", exist_ok=True)
joblib.dump(fitted[best_name], "models/best_forecast_model.joblib")
table.to_csv("models/forecast_results.csv", index=False)

# Actual vs predicted, averaged across households for each test day
plot_data = test.groupby("date")[["energy_kwh", best_name]].mean()
plot_data.plot(figsize=(10, 4), title=f"Actual vs predicted daily energy per household ({best_name})")
plt.ylabel("kWh")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/actual_vs_predicted.png")
plt.close()