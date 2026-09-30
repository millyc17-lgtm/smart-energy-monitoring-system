import plotly.express as px
import streamlit as st

from common import get_filters, household_days

f = get_filters()
st.title("Consumption")

days = household_days(f["where"], f["params"])
if days.empty:
    st.warning("No full days of data for this selection.")
    st.stop()

# Weekday vs weekend (average kWh per household per day)
is_weekend = days["date"].dt.dayofweek >= 5
a, b = st.columns(2)
a.metric("Weekday average", f"{days.loc[~is_weekend, 'energy_kwh'].mean():.1f} kWh per household per day")
b.metric("Weekend average", f"{days.loc[is_weekend, 'energy_kwh'].mean():.1f} kWh per household per day")

daily = days.groupby("date", as_index=False)["energy_kwh"].mean()
daily["cost"] = daily["energy_kwh"] * f["price"]
fig_cost = px.line(daily, x="date", y="cost", title="Daily cost per household (£)",
                   labels={"date": "Date", "cost": "Cost (£)"})
st.plotly_chart(fig_cost, width="stretch")

order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
by_weekday = days.groupby("day_of_week")["energy_kwh"].mean().reindex(order).reset_index()
fig_week = px.bar(by_weekday, x="day_of_week", y="energy_kwh",
                  title="Average daily energy by weekday (kWh per household)",
                  labels={"day_of_week": "Day", "energy_kwh": "Energy (kWh)"})

days["month"] = days["date"].dt.to_period("M").astype(str)
monthly = days.groupby("month", as_index=False)["energy_kwh"].mean()
fig_month = px.line(monthly, x="month", y="energy_kwh", markers=True,
                    title="Average daily energy by month (kWh per household)",
                    labels={"month": "Month", "energy_kwh": "Energy (kWh)"})

left, right = st.columns(2)
left.plotly_chart(fig_week, width="stretch")
right.plotly_chart(fig_month, width="stretch")