import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib

st.set_page_config(page_title="Smart Energy Monitor", layout="wide")


@st.cache_data
def load_data():
    conn = sqlite3.connect("data/energy.db")
    data = pd.read_sql("SELECT * FROM energy_readings", conn, parse_dates=["timestamp"])
    conn.close()
    return data

df = load_data()

# Sidebar filters
st.sidebar.title("⚡ Smart Energy Monitor")
households = ["All households"] + sorted(df["household_id"].unique())
household = st.sidebar.selectbox("Household", households)

dates = st.sidebar.date_input(
    "Date range",
    value=(pd.Timestamp("2012-10-01").date(), df["timestamp"].max().date()),
    min_value=df["timestamp"].min().date(),
    max_value=df["timestamp"].max().date(),
)
if len(dates) != 2:
    st.info("Select a start date and an end date.")
    st.stop()
start, end = dates

price = st.sidebar.number_input("Electricity price (£ per kWh)", min_value=0.0, value=0.25, step=0.01)

# Filter the data
data = df[(df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] < pd.Timestamp(end) + pd.Timedelta(days=1))]
if household != "All households":
    data = data[data["household_id"] == household]
if data.empty:
    st.warning("No data for this selection.")
    st.stop()
data = data.assign(cost=data["energy_kwh"] * price)

# Title and KPI cards
st.title("Smart Energy Monitoring System")
st.caption("Historical London smart meter data (2011-2014). The price is an adjustable assumption, not a real tariff.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total energy", f"{data['energy_kwh'].sum():,.0f} kWh")
c2.metric("Estimated cost", f"£{data['cost'].sum():,.0f}")
c3.metric("Peak power", f"{data['power_kw'].max():.2f} kW")
c4.metric("Average power", f"{data['power_kw'].mean():.2f} kW")

# Charts
by_hour = data.groupby("hour")["power_kw"].mean().reset_index()
fig1 = px.bar(by_hour, x="hour", y="power_kw", title="Average power by hour of day (kW)",
              labels={"hour": "Hour of day", "power_kw": "Average power (kW)"})

daily = (
    data.groupby(["household_id", "date"])
    .agg(energy_kwh=("energy_kwh", "sum"), readings=("energy_kwh", "count"))
    .reset_index()
)
daily = daily[daily["readings"] == 48]
daily = daily.groupby("date")["energy_kwh"].mean().reset_index()
daily["date"] = pd.to_datetime(daily["date"])
daily["cost"] = daily["energy_kwh"] * price
fig2 = px.line(daily, x="date", y="energy_kwh", title="Daily energy per household (kWh)",
               labels={"date": "Date", "energy_kwh": "Energy (kWh)"})
fig3 = px.line(daily, x="date", y="cost", title="Daily cost per household (£)",
               labels={"date": "Date", "cost": "Cost (£)"})


left, right = st.columns(2)
left.plotly_chart(fig1, use_container_width=True)
right.plotly_chart(fig2, use_container_width=True)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Unusual usage detected")


@st.cache_data
def load_anomalies():
    return pd.read_csv("data/processed/anomalies_only.csv", parse_dates=["timestamp"])


anoms = load_anomalies()
anoms = anoms[
    (anoms["timestamp"] >= pd.Timestamp(start))
    & (anoms["timestamp"] < pd.Timestamp(end) + pd.Timedelta(days=1))
]
if household != "All households":
    anoms = anoms[anoms["household_id"] == household]

st.write(f"{len(anoms):,} unusual readings in this selection")
top = anoms.nlargest(10, "energy_kwh")[
    ["household_id", "timestamp", "energy_kwh", "normal_mean", "upper_limit"]
].round({"energy_kwh": 2, "normal_mean": 2, "upper_limit": 2})
st.dataframe(top, width="stretch")

st.subheader("Machine-learning anomalies (Isolation Forest)")


@st.cache_data
def load_ml_anomalies():
    return pd.read_csv("data/processed/ml_anomalies_only.csv", parse_dates=["timestamp"])


ml = load_ml_anomalies()
ml = ml[
    (ml["timestamp"] >= pd.Timestamp(start))
    & (ml["timestamp"] < pd.Timestamp(end) + pd.Timedelta(days=1))
]
if household != "All households":
    ml = ml[ml["household_id"] == household]

st.write(
    f"{len(ml):,} readings flagged by the ML model, "
    f"{int(ml['is_anomaly'].sum()):,} of them also flagged by the rule-based detector (higher confidence)"
)
top_ml = ml.nsmallest(10, "ml_score")[
    ["household_id", "timestamp", "energy_kwh", "normal_mean", "ml_score", "is_anomaly"]
].rename(columns={"is_anomaly": "also_flagged_by_rule"})
top_ml = top_ml.round({"energy_kwh": 2, "normal_mean": 2, "ml_score": 3})
st.dataframe(top_ml, width="stretch")

st.subheader("Energy forecast")


@st.cache_resource
def load_model():
    return joblib.load("models/best_forecast_model.joblib")


@st.cache_data
def load_daily():
    d = (
        df.groupby(["household_id", "date"])
        .agg(energy_kwh=("energy_kwh", "sum"), readings=("energy_kwh", "count"))
        .reset_index()
    )
    d = d[d["readings"] == 48].drop(columns="readings")
    d["date"] = pd.to_datetime(d["date"])
    return d


all_daily = load_daily()
last_day = all_daily["date"].max()
target = last_day + pd.Timedelta(days=1)

yesterday = all_daily[all_daily["date"] == last_day][["household_id", "energy_kwh"]].rename(columns={"energy_kwh": "lag_1"})
last_week = all_daily[all_daily["date"] == target - pd.Timedelta(days=7)][["household_id", "energy_kwh"]].rename(columns={"energy_kwh": "lag_7"})
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

    f1, f2 = st.columns(2)
    f1.metric(f"Forecast for {target.date()} (per household)", f"{forecast_kwh:.1f} kWh")
    f2.metric("Estimated cost", f"£{forecast_kwh * price:.2f}")
    st.caption(
        "The forecast is for the day after the last full day in the historical data. "
        "It ignores the date filter, and the model does not use weather or holidays."
    )

st.image("outputs/actual_vs_predicted.png", caption="Actual vs predicted daily energy (test period, averaged across households)")
st.dataframe(pd.read_csv("models/forecast_results.csv"), width="stretch")