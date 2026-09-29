import sqlite3
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = "data/energy.db"

st.set_page_config(page_title="Smart Energy Monitor", layout="wide")


@st.cache_data
def run_query(sql, params=()):
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


# ---------- Sidebar ----------
bounds = run_query("SELECT MIN(date) AS first_day, MAX(date) AS last_day FROM energy_readings")
first_day = pd.Timestamp(bounds["first_day"][0]).date()
last_day = pd.Timestamp(bounds["last_day"][0]).date()
household_ids = run_query(
    "SELECT DISTINCT household_id FROM energy_readings ORDER BY household_id"
)["household_id"].tolist()

st.sidebar.title("⚡ Smart Energy Monitor")
household = st.sidebar.selectbox("Household", ["All households"] + household_ids)
dates = st.sidebar.date_input(
    "Date range",
    value=(pd.Timestamp("2012-10-01").date(), last_day),
    min_value=first_day,
    max_value=last_day,
)
if len(dates) != 2:
    st.info("Select a start date and an end date.")
    st.stop()
start, end = dates
price = st.sidebar.number_input("Electricity price (£ per kWh)", min_value=0.0, value=0.25, step=0.01)

# Filters are passed to SQL as ? placeholders, never pasted into the query text
where = "date >= ? AND date <= ?"
params = [start.isoformat(), end.isoformat()]
if household != "All households":
    where += " AND household_id = ?"
    params.append(household)
params = tuple(params)

# ---------- KPI cards ----------
kpi = run_query(
    f"""SELECT COUNT(*) AS n, SUM(energy_kwh) AS total_kwh,
               MAX(power_kw) AS peak_kw, AVG(power_kw) AS avg_kw
        FROM energy_readings WHERE {where}""",
    params,
)
if kpi["n"][0] == 0:
    st.warning("No data for this selection.")
    st.stop()

st.title("Smart Energy Monitoring System")
st.caption("Historical London smart meter data (2011-2014). The price is an adjustable assumption, not a real tariff.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total energy", f"{kpi['total_kwh'][0]:,.0f} kWh")
c2.metric("Estimated cost", f"£{kpi['total_kwh'][0] * price:,.0f}")
c3.metric("Peak power", f"{kpi['peak_kw'][0]:.2f} kW")
c4.metric("Average power", f"{kpi['avg_kw'][0]:.2f} kW")

# ---------- Charts ----------
by_hour = run_query(
    f"SELECT hour, AVG(power_kw) AS power_kw FROM energy_readings WHERE {where} GROUP BY hour ORDER BY hour",
    params,
)
daily = run_query(
    f"""SELECT date, AVG(day_kwh) AS energy_kwh FROM (
            SELECT household_id, date, SUM(energy_kwh) AS day_kwh
            FROM energy_readings WHERE {where}
            GROUP BY household_id, date
            HAVING COUNT(*) = 48
        ) GROUP BY date ORDER BY date""",
    params,
)
daily["date"] = pd.to_datetime(daily["date"])
daily["cost"] = daily["energy_kwh"] * price

fig1 = px.bar(by_hour, x="hour", y="power_kw", title="Average power by hour of day (kW)",
              labels={"hour": "Hour of day", "power_kw": "Average power (kW)"})
fig2 = px.line(daily, x="date", y="energy_kwh", title="Daily energy per household (kWh)",
               labels={"date": "Date", "energy_kwh": "Energy (kWh)"})
fig3 = px.line(daily, x="date", y="cost", title="Daily cost per household (£)",
               labels={"date": "Date", "cost": "Cost (£)"})

left, right = st.columns(2)
left.plotly_chart(fig1, width="stretch")
right.plotly_chart(fig2, width="stretch")
st.plotly_chart(fig3, width="stretch")

# ---------- Rule-based anomalies ----------
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
st.dataframe(top, width="stretch", hide_index=True)

# ---------- ML anomalies ----------
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
st.dataframe(top_ml, width="stretch", hide_index=True)

# ---------- Forecast ----------
st.subheader("Energy forecast")

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

    f1, f2 = st.columns(2)
    f1.metric(f"Forecast for {target.date()} (per household)", f"{forecast_kwh:.1f} kWh")
    f2.metric("Estimated cost", f"£{forecast_kwh * price:.2f}")
    st.caption(
        "The forecast is for the day after the last full day in the historical data. "
        "It ignores the date filter, and the model does not use weather or holidays."
    )

st.image("outputs/actual_vs_predicted.png", caption="Actual vs predicted daily energy (test period, averaged across households)")
st.dataframe(pd.read_csv("models/forecast_results.csv"), width="stretch", hide_index=True)