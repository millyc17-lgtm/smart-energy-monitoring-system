import joblib
import pandas as pd
import streamlit as st

from common import get_filters, run_query

f = get_filters()
household, price = f["household"], f["price"]
st.title("Predictions")

FULL_DAYS = """
    SELECT household_id, date, SUM(energy_kwh) AS energy_kwh
    FROM energy_readings
    GROUP BY household_id, date
    HAVING COUNT(*) = 48
"""


@st.cache_resource
def load_model():
    return joblib.load("models/best_forecast_model.joblib")


last_full = run_query(f"SELECT MAX(date) AS d FROM ({FULL_DAYS})")["d"][0]
target = pd.Timestamp(last_full) + pd.Timedelta(days=1)
week_ago = (target - pd.Timedelta(days=7)).strftime("%Y-%m-%d")

yesterday = run_query(f"SELECT household_id, energy_kwh AS lag_1 FROM ({FULL_DAYS}) WHERE date = ?", (last_full,))
last_week = run_query(f"SELECT household_id, energy_kwh AS lag_7 FROM ({FULL_DAYS}) WHERE date = ?", (week_ago,))
X = yesterday.merge(last_week, on="household_id")
if household != "All households":
    X = X[X["household_id"] == household]

if X.empty:
    st.info("Not enough recent data to forecast for this household.")
else:
    X["day_of_week"] = target.dayofweek
    X["is_weekend"] = int(target.dayofweek >= 5)
    X["month"] = target.month
    predicted = load_model().predict(X[["lag_1", "lag_7", "day_of_week", "is_weekend", "month"]])
    forecast_kwh = predicted.mean()

    a, b = st.columns(2)
    a.metric(f"Forecast for {target.date()} (per household)", f"{forecast_kwh:.1f} kWh")
    b.metric("Estimated cost", f"£{forecast_kwh * price:.2f}")
    st.caption(
        "The forecast is for the day after the last full day in the historical data. "
        "It ignores the date filter, and the model does not use weather or holidays."
    )

st.subheader("Model comparison")
st.image("outputs/actual_vs_predicted.png", caption="Actual vs predicted daily energy (test period, averaged across households)")
st.dataframe(pd.read_csv("models/forecast_results.csv"), width="stretch", hide_index=True)