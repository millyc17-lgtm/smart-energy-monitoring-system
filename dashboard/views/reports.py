import pandas as pd
import streamlit as st

from common import get_filters, run_query

f = get_filters()
household, price, factor = f["household"], f["price"], f["factor"]
st.title("Monthly report")

months = run_query(
    "SELECT DISTINCT substr(date, 1, 7) AS month FROM energy_readings WHERE date >= '2012-10-01' ORDER BY month"
)["month"].tolist()
month = st.selectbox("Month", months, index=max(len(months) - 2, 0))


def month_stats(m):
    sql = """
        SELECT COUNT(*) AS household_days, SUM(day_kwh) AS total_kwh,
               AVG(day_kwh) AS avg_day_kwh, MAX(peak_kw) AS peak_kw
        FROM (
            SELECT household_id, date, SUM(energy_kwh) AS day_kwh, MAX(power_kw) AS peak_kw
            FROM energy_readings
            WHERE substr(date, 1, 7) = ?"""
    p = [m]
    if household != "All households":
        sql += " AND household_id = ?"
        p.append(household)
    sql += """
            GROUP BY household_id, date
            HAVING COUNT(*) = 48
        )"""
    return run_query(sql, tuple(p)).iloc[0]


cur = month_stats(month)
prev_month = (pd.Period(month) - 1).strftime("%Y-%m")
prev = month_stats(prev_month) if prev_month in months else None

if cur["household_days"] == 0:
    st.info("No full days of data for this month and household.")
else:
    total_kwh = cur["total_kwh"]
    change = None
    if prev is not None and prev["household_days"] > 0:
        change = (cur["avg_day_kwh"] / prev["avg_day_kwh"] - 1) * 100

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total energy", f"{total_kwh:,.0f} kWh")
    r2.metric("Estimated cost", f"£{total_kwh * price:,.0f}")
    r3.metric("Estimated emissions", f"{total_kwh * factor:,.0f} kg CO₂e")
    r4.metric(
        "Average per household per day",
        f"{cur['avg_day_kwh']:.1f} kWh",
        delta=None if change is None else f"{change:+.1f}% vs previous month",
        delta_color="inverse",
    )

    report = pd.DataFrame({
        "Metric": [
            "Month", "Household", "Total energy (kWh)", "Estimated cost (GBP)",
            "Estimated emissions (kg CO2e)", "Average per household per day (kWh)",
            "Peak power (kW)", "Change vs previous month (%)",
        ],
        "Value": [
            month, household, f"{total_kwh:.1f}", f"{total_kwh * price:.2f}",
            f"{total_kwh * factor:.1f}", f"{cur['avg_day_kwh']:.2f}", f"{cur['peak_kw']:.2f}",
            "n/a" if change is None else f"{change:.1f}",
        ],
    })
    st.dataframe(report, width="stretch", hide_index=True)
    st.download_button(
        "Download report (CSV)",
        report.to_csv(index=False),
        file_name=f"energy_report_{month}.csv",
        mime="text/csv",
    )
    st.caption(
        "Full days only (48 readings). Uses the price and emissions factor from the sidebar and ignores the date filter. "
        "The comparison uses per-household averages because the number of reporting households varies."
    )