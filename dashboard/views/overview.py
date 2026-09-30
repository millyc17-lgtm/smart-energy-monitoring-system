import plotly.express as px
import streamlit as st

from common import get_filters, household_days, run_query

f = get_filters()
where, params, price, factor = f["where"], f["params"], f["price"], f["factor"]

st.title("Smart Energy Monitoring System")
st.caption("Historical London smart meter data (2011-2014). The price is an adjustable assumption, not a real tariff.")

kpi = run_query(
    f"""SELECT COUNT(*) AS n, SUM(energy_kwh) AS total_kwh,
               MAX(power_kw) AS peak_kw, AVG(power_kw) AS avg_kw
        FROM energy_readings WHERE {where}""",
    params,
)
if kpi["n"][0] == 0:
    st.warning("No data for this selection.")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total energy", f"{kpi['total_kwh'][0]:,.0f} kWh")
c2.metric("Estimated cost", f"£{kpi['total_kwh'][0] * price:,.0f}")
c3.metric("Estimated emissions", f"{kpi['total_kwh'][0] * factor:,.0f} kg CO₂e")
c4.metric("Peak power", f"{kpi['peak_kw'][0]:.2f} kW")
c5.metric("Average power", f"{kpi['avg_kw'][0]:.2f} kW")
st.caption(
    "Emissions are an estimate (energy x the factor in the sidebar), not a measurement. "
    "The data is from 2011-2014, when the grid was more carbon-intensive than the 2026 factor suggests."
)

by_hour = run_query(
    f"SELECT hour, AVG(power_kw) AS power_kw FROM energy_readings WHERE {where} GROUP BY hour ORDER BY hour",
    params,
)
daily = household_days(where, params).groupby("date", as_index=False)["energy_kwh"].mean()

fig1 = px.bar(by_hour, x="hour", y="power_kw", title="Average power by hour of day (kW)",
              labels={"hour": "Hour of day", "power_kw": "Average power (kW)"})
fig2 = px.line(daily, x="date", y="energy_kwh", title="Daily energy per household (kWh)",
               labels={"date": "Date", "energy_kwh": "Energy (kWh)"})

left, right = st.columns(2)
left.plotly_chart(fig1, width="stretch")
right.plotly_chart(fig2, width="stretch")