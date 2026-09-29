import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st

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
fig1 = px.bar(by_hour, x="hour", y="power_kw", title="Average power by hour of day (kW)")

daily = (
    data.groupby(["household_id", "date"])
    .agg(energy_kwh=("energy_kwh", "sum"), readings=("energy_kwh", "count"))
    .reset_index()
)
daily = daily[daily["readings"] == 48]
daily = daily.groupby("date")["energy_kwh"].mean().reset_index()
daily["date"] = pd.to_datetime(daily["date"])
daily["cost"] = daily["energy_kwh"] * price
fig2 = px.line(daily, x="date", y="energy_kwh", title="Daily energy per household (kWh)")
fig3 = px.line(daily, x="date", y="cost", title="Daily cost per household (£)")

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