import pandas as pd
import plotly.express as px
import streamlit as st

from common import get_filters, run_query

f = get_filters()
where, params = f["where"], f["params"]

st.title("Tariff what-if")
st.caption(
    "Compare a flat price with a time-of-use tariff (cheap at some hours, expensive at others) using the "
    "household and dates selected in the sidebar. All prices here are illustrative placeholders, not real tariffs."
)

by_hour = run_query(
    f"""SELECT hour, SUM(energy_kwh) AS kwh, COUNT(*) AS readings
        FROM energy_readings WHERE {where} GROUP BY hour ORDER BY hour""",
    params,
)
if by_hour.empty or by_hour["readings"].sum() == 0:
    st.warning("No data for this selection.")
    st.stop()

# ---------- Controls ----------
st.subheader("Flat tariff")
flat = st.number_input("Flat price (£ per kWh)", min_value=0.0, value=float(f.get("price", 0.25)), step=0.01)

st.subheader("Time-of-use tariff")
c1, c2, c3 = st.columns(3)
cheap_hours = c1.slider("Cheap hours (from, to)", 0, 24, (0, 7))
cheap_price = c1.number_input("Cheap price (£ per kWh)", min_value=0.0, value=0.12, step=0.01)
peak_hours = c2.slider("Peak hours (from, to)", 0, 24, (16, 20))
peak_price = c2.number_input("Peak price (£ per kWh)", min_value=0.0, value=0.35, step=0.01)
std_price = c3.number_input("Price at all other hours (£ per kWh)", min_value=0.0, value=0.25, step=0.01)
c3.caption("Hours are 'from' up to but not including 'to'. If the two windows overlap, peak wins.")

# ---------- Calculation ----------
def band(h):
    if peak_hours[0] <= h < peak_hours[1]:
        return "Peak"
    if cheap_hours[0] <= h < cheap_hours[1]:
        return "Cheap"
    return "Standard"


by_hour["band"] = by_hour["hour"].map(band)
by_hour["price"] = by_hour["band"].map({"Peak": peak_price, "Cheap": cheap_price, "Standard": std_price})

household_days = by_hour["readings"].sum() / 48          # one household-day = 48 half-hour readings
total_kwh = by_hour["kwh"].sum()
peak_kwh = by_hour.loc[by_hour["band"] == "Peak", "kwh"].sum()
has_cheap = by_hour["band"].eq("Cheap").any()

shift = st.slider(
    "What if households moved this share of their peak-hour usage into the cheap hours?",
    0, 100, 0, format="%d%%", disabled=not has_cheap,
)
moved = peak_kwh * shift / 100 if has_cheap else 0.0

flat_cost = total_kwh * flat
tou_cost = (by_hour["kwh"] * by_hour["price"]).sum() - moved * (peak_price - cheap_price)

flat_day = flat_cost / household_days
tou_day = tou_cost / household_days
diff_pct = (tou_cost / flat_cost - 1) * 100 if flat_cost > 0 else 0.0

# ---------- Results ----------
m1, m2, m3 = st.columns(3)
m1.metric("Flat tariff", f"£{flat_day:.2f} per household per day")
m2.metric("Time-of-use tariff", f"£{tou_day:.2f} per household per day")
m3.metric("Difference", f"£{tou_day - flat_day:+.2f} per day", delta=f"{diff_pct:+.1f}%", delta_color="inverse")

st.write(
    f"{peak_kwh / total_kwh:.0%} of the selected usage falls in your peak hours. "
    + ("The time-of-use tariff would cost less." if tou_cost < flat_cost else "The flat tariff would cost less or the same.")
)
if peak_hours[0] < cheap_hours[1] and cheap_hours[0] < peak_hours[1]:
    st.warning("The cheap and peak windows overlap. Hours in both are charged at the peak price.")

plot = by_hour.assign(kwh_per_day=by_hour["kwh"] / household_days)
fig = px.bar(
    plot, x="hour", y="kwh_per_day", color="band",
    color_discrete_map={"Cheap": "#2a9d8f", "Standard": "#8d99ae", "Peak": "#e76f51"},
    category_orders={"band": ["Cheap", "Standard", "Peak"]},
    title="Average energy by hour of day, coloured by price band (kWh per household per day)",
    labels={"hour": "Hour of day", "kwh_per_day": "Energy (kWh)", "band": "Price band"},
)
st.plotly_chart(fig, width="stretch")

st.caption(
    "This is a simple what-if on historical 2011-2014 usage. It assumes usage stays the same unless you move some "
    "peak usage with the slider, and real households may behave differently. Tariff prices are placeholders."
)