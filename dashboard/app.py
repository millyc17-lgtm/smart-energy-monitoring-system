import streamlit as st

from common import sidebar_filters

st.set_page_config(page_title="Smart Energy Monitor", page_icon="⚡", layout="wide")

pages = [
    st.Page("views/overview.py", title="Overview", icon="📊", default=True),
    st.Page("views/consumption.py", title="Consumption", icon="📈"),
    st.Page("views/predictions.py", title="Predictions", icon="🔮"),
    st.Page("views/profiles.py", title="Load profiles", icon="👥"),
    st.Page("views/anomalies.py", title="Anomalies", icon="⚠️"),
    st.Page("views/reports.py", title="Reports", icon="📄"),
    st.Page("views/live.py", title="Live monitoring", icon="📡"),
    st.Page("views/about.py", title="About", icon="ℹ️"),
]
page = st.navigation(pages)
sidebar_filters()
page.run()