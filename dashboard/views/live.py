import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import run_query

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from live_alerts import flag_unusual  # noqa: E402

LIVE_DB = ROOT / "data" / "live.db"
REFRESH_SECONDS = 5

# Demo scenarios: the same three days for two households (the second has an unusual evening spike)
SCENARIOS = {
    "Typical household (MAC000002, 13-15 Sep 2013)": ("MAC000002", "2013-09-13", 3),
    "Household with an unusual evening (MAC003428, 13-15 Sep 2013)": ("MAC003428", "2013-09-13", 3),
}
DEMO_START = 24          # readings already showing when a demo starts (12 hours)


def read_live(sql, params=()):
    """Read from data/live.db (read-only). Returns an empty table if the file or table does not exist yet."""
    if not LIVE_DB.exists():
        return pd.DataFrame()
    try:
        con = sqlite3.connect(f"{LIVE_DB.as_uri()}?mode=ro", uri=True)
        try:
            return pd.read_sql_query(sql, con, params=params)
        finally:
            con.close()
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        return pd.DataFrame()


def show_readings(df, device, n, demo=False):
    """Banner, numbers and chart for one device. `df` holds readings oldest to newest."""
    df = flag_unusual(df)
    shown = df.tail(n)
    latest = df.iloc[-1]

    if latest["source"] == "replay":
        where = "inside this app" if demo else "through the live pipeline"
        st.warning(
            f"**REPLAY.** These are historical household readings from 2011-2014, replayed {where}. "
            "They are not live measurements."
        )
    else:
        st.success("**LIVE DEVICE.** These readings come from a real device.")

    previous = df["power_kw"].iloc[-2] if len(df) > 1 else None
    a, b, c, d = st.columns(4)
    a.metric("Latest reading", f"{latest['power_kw']:.3f} kW",
             delta=None if previous is None else f"{latest['power_kw'] - previous:+.3f} kW")
    b.metric("Reading time", latest["timestamp"].strftime("%d %b %Y %H:%M"))
    c.metric("Average shown", f"{shown['power_kw'].mean():.3f} kW")
    d.metric("Unusual readings shown", int(shown["is_unusual"].sum()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=shown["timestamp"], y=shown["power_kw"], mode="lines", name="Power"))
    fig.add_trace(go.Scatter(x=shown["timestamp"], y=shown["upper_limit"], mode="lines", name="Usual upper limit",
                             line=dict(dash="dot", width=1)))
    odd = shown[shown["is_unusual"]]
    fig.add_trace(go.Scatter(x=odd["timestamp"], y=odd["power_kw"], mode="markers", name="Unusual",
                             marker=dict(size=10, color="#e76f51")))
    fig.update_layout(title=f"{device}: power over the last {len(shown)} readings",
                      xaxis_title="Time", yaxis_title="Power (kW)", legend_title_text="")
    st.plotly_chart(fig, width="stretch")

    if demo:
        st.caption("Demo: readings are replayed inside this app from the historical dataset.")
    else:
        received = datetime.strptime(latest["received_at"], "%Y-%m-%d %H:%M:%S")
        st.caption(f"Last data received by the API at {received:%H:%M:%S}. "
                   f"This panel refreshes every {REFRESH_SECONDS} seconds.")
    st.caption(
        "A reading is marked unusual if it is more than 3 standard deviations above the previous day's average "
        "for this device. This is a simple rule for a live stream, lighter than the Anomalies page."
    )


# ---------------- Local live feed (needs the API running on your own computer) ----------------
@st.fragment(run_every=REFRESH_SECONDS)
def live_panel():
    devices = read_live("SELECT DISTINCT device_id FROM live_readings ORDER BY device_id")
    if devices.empty:
        st.info("No live readings yet. Start the API with `python src/live_api.py`, then send data "
                "with `python src/replay.py`.")
        return
    c1, c2 = st.columns([2, 1])
    device = c1.selectbox("Device", devices["device_id"].tolist())
    n = c2.slider("Readings to show", 12, 288, 96, step=12)
    df = read_live(
        """SELECT source, device_id, timestamp, power_kw, received_at FROM live_readings
           WHERE device_id = ? ORDER BY timestamp DESC LIMIT ?""",
        (device, n + 48),                        # 48 extra so the unusual-reading rule has history
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    show_readings(df.sort_values("timestamp"), device, n)
    with st.expander("Latest readings"):
        st.dataframe(df.head(10)[["timestamp", "device_id", "power_kw", "source", "received_at"]],
                     hide_index=True, width="stretch")


# ---------------- Demo replay (works anywhere, including the public website) ----------------
def demo_panel():
    name = st.selectbox("Scenario", list(SCENARIOS))
    household, start, days = SCENARIOS[name]
    end = (datetime.strptime(start, "%Y-%m-%d") + pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    data = run_query(
        """SELECT timestamp, power_kw FROM energy_readings
           WHERE household_id = ? AND timestamp >= ? AND timestamp < ? ORDER BY timestamp""",
        [household, start, end],
    )
    if data.empty:
        st.warning("No data found for this scenario.")
        return
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["source"] = "replay"
    data["device_id"] = household

    # Remember where the replay is up to, and start again if the scenario changes
    if st.session_state.get("demo_scenario") != name:
        st.session_state["demo_scenario"] = name
        st.session_state["demo_pos"] = min(DEMO_START, len(data))
        st.session_state["demo_playing"] = False
    playing = st.session_state["demo_playing"]

    b1, b2, b3 = st.columns([1, 1, 2])
    if b1.button("Pause" if playing else "Play", width="stretch"):
        if st.session_state["demo_pos"] >= len(data):
            st.session_state["demo_pos"] = min(DEMO_START, len(data))
        st.session_state["demo_playing"] = not playing
        st.rerun()
    if b2.button("Restart", width="stretch"):
        st.session_state["demo_pos"] = min(DEMO_START, len(data))
        st.session_state["demo_playing"] = False
        st.rerun()
    speed = b3.slider("Speed (readings per second)", 1, 8, 2)

    @st.fragment(run_every=1.0 if playing else None)
    def ticking():
        if st.session_state["demo_playing"]:
            st.session_state["demo_pos"] = min(st.session_state["demo_pos"] + speed, len(data))
        pos = st.session_state["demo_pos"]
        show_readings(data.iloc[:pos].copy(), household, 96, demo=True)
        st.progress(pos / len(data), text=f"{pos} of {len(data)} readings replayed")
        if st.session_state["demo_playing"] and pos >= len(data):
            st.session_state["demo_playing"] = False
            st.rerun(scope="app")

    ticking()


# ---------------- Page ----------------
st.title("Live monitoring")

has_live = bool(len(read_live("SELECT 1 FROM live_readings LIMIT 1")))
if has_live:
    mode = st.radio("Source", ["Local live feed", "Demo replay"], horizontal=True)
else:
    mode = "Demo replay"
    st.info("There is no local live feed on this site, so this page runs a **replay** of historical readings "
            "inside the app.")

if mode == "Local live feed":
    live_panel()
else:
    demo_panel()

with st.expander("How the real-time version works"):
    st.markdown(
        """
Readings arrive at a small **API** that checks them and saves them to a **SQLite database**. This page reads that
database and refreshes itself. A **replay script** feeds historical household readings through the API as if they were
arriving live. A real device, such as a smart plug measuring a desk's electricity use, can send readings to the same API
and is labelled **LIVE DEVICE** instead of **REPLAY**.

The API runs on a local computer, so the live version is not part of this public website. The demo above plays the same
kind of data inside the app. Real household readings from a home are never published here.
"""
    )