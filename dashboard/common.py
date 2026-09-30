import sqlite3

import pandas as pd
import streamlit as st

DB_PATH = "data/energy.db"


@st.cache_data
def run_query(sql, params=()):
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def sidebar_filters():
    """Draw the sidebar controls once and store the choices for every page to use."""
    bounds = run_query("SELECT MIN(date) AS first_day, MAX(date) AS last_day FROM energy_readings")
    first_day = pd.Timestamp(bounds["first_day"][0]).date()
    last_day = pd.Timestamp(bounds["last_day"][0]).date()
    household_ids = run_query(
        "SELECT DISTINCT household_id FROM energy_readings ORDER BY household_id"
    )["household_id"].tolist()

    st.sidebar.header("⚡ Filters")
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

    st.sidebar.header("Assumptions")
    price = st.sidebar.number_input("Electricity price (£ per kWh)", min_value=0.0, value=0.25, step=0.01)
    factor = st.sidebar.number_input(
        "Emissions factor (kg CO₂e per kWh)",
        min_value=0.0, value=0.13096, step=0.001, format="%.5f",
        help="UK grid electricity, DESNZ/DEFRA 2026 conversion factors. Change it to test other assumptions.",
    )

    # Filters are passed to SQL as ? placeholders, never pasted into the query text
    where = "date >= ? AND date <= ?"
    params = [start.isoformat(), end.isoformat()]
    if household != "All households":
        where += " AND household_id = ?"
        params.append(household)

    st.session_state["filters"] = {
        "household": household, "start": start, "end": end,
        "price": price, "factor": factor, "where": where, "params": tuple(params),
    }


def get_filters():
    return st.session_state["filters"]


def household_days(where, params):
    """Total energy per household per full day (48 readings), aggregated in SQL."""
    d = run_query(
        f"""SELECT household_id, date, day_of_week, SUM(energy_kwh) AS energy_kwh
            FROM energy_readings WHERE {where}
            GROUP BY household_id, date
            HAVING COUNT(*) = 48""",
        params,
    )
    d["date"] = pd.to_datetime(d["date"])
    return d