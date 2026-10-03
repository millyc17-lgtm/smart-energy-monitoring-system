from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROFILES = Path(__file__).resolve().parents[2] / "models" / "household_profiles.csv"

st.title("Household load profiles")
st.caption(
    "A load profile is the shape of a household's typical day. Households are grouped by how similar those shapes "
    "are, using all available data. The sidebar filters do not apply on this page."
)

if not PROFILES.exists():
    st.error("models/household_profiles.csv was not found. Run: python src/profiles.py")
    st.stop()

df = pd.read_csv(PROFILES)
groups = sorted(df["group"].unique())
colours = dict(zip(groups, px.colors.qualitative.Set2))
labels = {g: f"Group {g}" for g in groups}
df["label"] = df["group"].map(labels)

# ---------- Summary table ----------
rows = []
for g in groups:
    d = df[df["group"] == g]
    shape = d.groupby("hour")["relative"].mean()
    rows.append({
        "Group": labels[g],
        "Households": d["household_id"].nunique(),
        "Average use (kW)": round(d.groupby("household_id")["avg_kw"].first().mean(), 2),
        "Busiest hour": f"{int(shape.idxmax()):02d}:00",
        "Quietest hour": f"{int(shape.idxmin()):02d}:00",
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ---------- Group chart ----------
view = st.radio(
    "Show",
    ["Shape of the day (relative to each household's own average)", "Actual power (kW)"],
    horizontal=True,
)
col = "relative" if view.startswith("Shape") else "power_kw"
ylabel = "Relative use (1.0 = own daily average)" if col == "relative" else "Average power (kW)"

avg = df.groupby(["label", "group", "hour"], as_index=False)[col].mean()
fig = px.line(
    avg, x="hour", y=col, color="label", markers=True,
    color_discrete_map={labels[g]: colours[g] for g in groups},
    labels={"hour": "Hour of day", col: ylabel, "label": ""},
    title="Average day for each group",
)
fig.update_xaxes(dtick=2)
st.plotly_chart(fig, width="stretch")

if col == "relative":
    st.caption("Dividing by each household's own average lets a big house and a small house with the same routine count as similar.")

# ---------- Single household ----------
st.subheader("Look up a household")
hid = st.selectbox("Household", sorted(df["household_id"].unique()))
mine = df[df["household_id"] == hid]
g = int(mine["group"].iloc[0])
st.write(f"**{hid}** is in **{labels[g]}**, with an average use of {mine['avg_kw'].iloc[0]:.2f} kW.")

compare = pd.concat([
    mine.assign(line=hid)[["hour", col, "line"]],
    avg[avg["group"] == g].assign(line=f"{labels[g]} average")[["hour", col, "line"]],
])
fig2 = px.line(
    compare, x="hour", y=col, color="line", markers=True,
    labels={"hour": "Hour of day", col: ylabel, "line": ""},
    title=f"{hid} compared with its group",
)
fig2.update_xaxes(dtick=2)
st.plotly_chart(fig2, width="stretch")

st.info(
    "**How to read this:** the groups are not sharply separate types. Households vary smoothly, and the groups are "
    "a way of slicing that range, mainly by when in the evening usage peaks. The grouping score is low (about 0.2 "
    "out of 1), so treat the groups as a rough guide, not a hard classification."
)
st.caption("Only 50 households, from one trial, so the groups may not match other populations.")