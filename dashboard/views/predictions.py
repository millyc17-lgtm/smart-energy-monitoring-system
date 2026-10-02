import pandas as pd
import streamlit as st

from common import get_filters

f = get_filters()
household, price = f["household"], f["price"]
st.title("Predictions")


@st.cache_data
def load_latest():
    return pd.read_csv("models/forecast_latest.csv")


@st.cache_data
def load_walkforward():
    return pd.read_csv("models/forecast_walkforward.csv")


latest = load_latest()
target_date = latest["target_date"].iloc[0]
if household != "All households":
    latest = latest[latest["household_id"] == household]

if latest.empty:
    st.info("No forecast available for this household (not enough recent history).")
else:
    forecast_kwh = latest["forecast_kwh"].mean()
    a, b = st.columns(2)
    a.metric(f"Forecast for {target_date} (per household)", f"{forecast_kwh:.1f} kWh")
    b.metric("Estimated cost", f"£{forecast_kwh * price:.2f}")
    st.caption(
        "The forecast is for the day after the last full day in the historical data and ignores the date filter. "
        "Weather inputs are the observed weather for that day. A real deployment would use a weather forecast, "
        "so real-world accuracy would be lower."
    )

st.subheader("How accurate is it?")
wf = load_walkforward()
base = wf.loc[wf["model"] == "Baseline: same as yesterday", "mean MAE"].iloc[0]
models_only = wf[wf["features"] != "-"]
best = models_only.loc[models_only["mean MAE"].idxmin()]
st.write(
    "Tested with walk-forward validation: trained on the past and tested on the following two months, "
    "five times in 2013. "
    f"The best model ({best['model']}, features: {best['features']}) has an average error of "
    f"{best['mean MAE']:.2f} kWh per household per day, "
    f"{100 * (1 - best['mean MAE'] / base):.0f}% better than simply repeating yesterday ({base:.2f} kWh)."
)
st.dataframe(wf, width="stretch", hide_index=True)
st.caption("Mean absolute error in kWh per household per day for each test period. Lower is better.")
st.image("outputs/actual_vs_predicted_v2.png", caption="Actual vs predicted daily energy, final test period (averaged across households)")