import streamlit as st

st.title("About this project")

st.markdown(
    """
A portfolio project that turns half-hourly household electricity readings into a cleaned database,
anomaly detection, a forecast and an interactive dashboard, with a prototype of a live monitoring pipeline.

**Data.** Low Carbon London smart meter data (UK Power Networks, released under a CC-BY licence),
accessed through the Kaggle copy "Smart meters in London". This app uses one file: 50 households,
half-hourly kWh, December 2011 to February 2014. It is historical data, not live usage. Trends and
forecasts use October 2012 onwards, when 43 to 50 households were reporting.

**How it works**
- Data is cleaned in Python and stored in SQLite, and the dashboard queries it with SQL.
- Anomalies are flagged two ways: a rolling statistical rule and an Isolation Forest. They overlap on
  about a quarter of flagged readings. Tested on artificial injected spikes, the rule detected about 87% of events
  and the Isolation Forest about 63% at similar flag rates, and small spikes are the hardest to catch.
- The forecast predicts a household's next-day energy use. Models are tested with walk-forward validation
  across five periods in 2013. The best (a Random Forest using recent usage, calendar and weather features)
  has an average error of about 3.5 kWh per household per day, about 11% better than simply repeating yesterday.
  Weather inputs use observed weather, so real-world accuracy would be lower.
- **Tariff what-if** compares a flat price with cheap, standard and peak hours on the selected usage.
- **Load profiles** group households by the shape of their typical day. The groups are rough, not sharply
  separate types, and differ mainly in when the evening peak happens.
- **Live monitoring** is a working prototype: a local API receives readings, stores them in SQLite, and the page shows
  them with a REPLAY or LIVE DEVICE label. On this public site it replays historical readings inside the app.
  The live version runs on a local computer, and readings from a real home are never published here.

**Assumptions**
- The electricity price (default £0.25 per kWh) is an adjustable placeholder, not a real tariff. The same applies
  to the prices on the tariff page.
- Emissions use an adjustable factor (default 0.13096 kg CO₂e per kWh, UK grid electricity,
  DESNZ/DEFRA 2026). It is applied to 2011-2014 data, so emissions for that period are likely understated.

**Limitations.** A small sample of 50 households, not representative of the UK average. Only 2 of the 50 households
were on a dynamic time-of-use tariff in 2013. Forecasts use observed, not forecast, weather. The live pipeline has been
tested with replayed data and not yet with a real device.
"""
)

st.markdown("Code and full write-up: [GitHub repository](https://github.com/millyc17-lgtm/smart-energy-monitoring-system)")