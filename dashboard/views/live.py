import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from live_alerts import flag_unusual  # noqa: E402

LIVE_DB = ROOT / "data" / "live.db"
REFRESH_SECONDS = 5

st.title("Live monitoring")


def read(sql, params=()):
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


@st.fragment(run_every=REFRESH_SECONDS)
def live_panel():
    devices = read("SELECT DISTINCT device_id FROM live_readings ORDER BY device_id")
    if devices.empty:
        st.info(
            "No live readings yet. This page shows readings received by the local live API. "
            "Start the API with `python src/live_api.py`, then send data with `python src/replay.py`."
        )
        return

    c1, c2 = st.columns([2, 1])
    device = c1.selectbox("Device", devices["device_id"].tolist())
    n = c2.slider("Readings to show", 12, 288, 96, step=12)

    df = read(
        """SELECT source, device_id, timestamp, power_kw, received_at FROM live_readings
           WHERE device_id = ? ORDER BY timestamp DESC LIMIT ?""",
        (device, n + 48),                        # 48 extra so the unusual-reading rule has history
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = flag_unusual(df)
    shown = df.tail(n)
    latest = df.iloc[-1]

    # ---------- Clear label for where the data comes from ----------
    if latest["source"] == "replay":
        st.warning(
            "**REPLAY.** These are historical household readings from 2011-2014, replayed through the live "
            "pipeline. They are not live measurements."
        )
    else:
        st.success("**LIVE DEVICE.** These readings come from a real device.")

    # ---------- Numbers ----------
    previous = df["power_kw"].iloc[-2] if len(df) > 1 else None
    a, b, c, d = st.columns(4)
    a.metric("Latest reading", f"{latest['power_kw']:.3f} kW",
             delta=None if previous is None else f"{latest['power_kw'] - previous:+.3f} kW")
    b.metric("Reading time", latest["timestamp"].strftime("%d %b %Y %H:%M"))
    c.metric("Average shown", f"{shown['power_kw'].mean():.3f} kW")
    d.metric("Unusual readings shown", int(shown["is_unusual"].sum()))

    # ---------- Chart ----------
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

    received = datetime.strptime(latest["received_at"], "%Y-%m-%d %H:%M:%S")
    st.caption(
        f"Last data received by the API at {received:%H:%M:%S}. This panel refreshes every {REFRESH_SECONDS} seconds. "
        "A reading is marked unusual if it is more than 3 standard deviations above the previous day's average "
        "for this device. This is a simple rule for a live stream, lighter than the Anomalies page."
    )
    with st.expander("Latest readings"):
        st.dataframe(df.tail(10).iloc[::-1][["timestamp", "device_id", "power_kw", "source", "received_at"]],
                     hide_index=True, width="stretch")


live_panel()