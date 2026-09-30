import pandas as pd
import streamlit as st

from common import get_filters

f = get_filters()
household, start, end = f["household"], f["start"], f["end"]
st.title("Anomalies")
st.caption(
    "Flags mean 'unusually high for this household at this time of day'. "
    "There are no labelled anomalies, so accuracy cannot be measured, and a flag does not explain the cause."
)


def in_selection(d):
    d = d[(d["timestamp"] >= pd.Timestamp(start)) & (d["timestamp"] < pd.Timestamp(end) + pd.Timedelta(days=1))]
    if household != "All households":
        d = d[d["household_id"] == household]
    return d


@st.cache_data
def load_anomalies():
    return pd.read_csv("data/processed/anomalies_only.csv", parse_dates=["timestamp"])


@st.cache_data
def load_ml_anomalies():
    return pd.read_csv("data/processed/ml_anomalies_only.csv", parse_dates=["timestamp"])


st.subheader("Rule-based detector")
anoms = in_selection(load_anomalies())
st.write(f"{len(anoms):,} unusual readings in this selection")
top = anoms.nlargest(10, "energy_kwh")[
    ["household_id", "timestamp", "energy_kwh", "normal_mean", "upper_limit"]
].round({"energy_kwh": 2, "normal_mean": 2, "upper_limit": 2})
st.dataframe(top, width="stretch", hide_index=True)

st.subheader("Isolation Forest (machine learning)")
ml = in_selection(load_ml_anomalies())
st.write(
    f"{len(ml):,} readings flagged by the ML model, "
    f"{int(ml['is_anomaly'].sum()):,} of them also flagged by the rule-based detector (higher confidence)"
)
top_ml = ml.nsmallest(10, "ml_score")[
    ["household_id", "timestamp", "energy_kwh", "normal_mean", "ml_score", "is_anomaly"]
].rename(columns={"is_anomaly": "also_flagged_by_rule"})
top_ml = top_ml.round({"energy_kwh": 2, "normal_mean": 2, "ml_score": 3})
st.dataframe(top_ml, width="stretch", hide_index=True)