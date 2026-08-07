"""
Streamlit dashboard to review pipeline output after running main.py.

Usage:
    streamlit run dashboard.py
"""
import sqlite3
import os

import pandas as pd
import streamlit as st

from config import config

st.set_page_config(page_title="Laptop Ownership & Handover Dashboard", layout="wide")
st.title("Laptop Ownership & Handover Tracking — Dashboard")

if not os.path.exists(config.EVENT_LOG_SQLITE):
    st.warning(
        f"No event log found at `{config.EVENT_LOG_SQLITE}`. "
        "Run `python main.py --input your_video.mp4` first."
    )
    st.stop()

conn = sqlite3.connect(config.EVENT_LOG_SQLITE)
df = pd.read_sql_query("SELECT * FROM events ORDER BY frame_idx", conn)
conn.close()

col1, col2, col3 = st.columns(3)
col1.metric("Total events", len(df))
col2.metric("Distinct laptops tracked", df["laptop_id"].nunique() if not df.empty else 0)
col3.metric("Handovers", int((df["event_type"] == "handover").sum()) if not df.empty else 0)

st.subheader("Event log")
laptop_filter = st.multiselect("Filter by laptop ID", sorted(df["laptop_id"].unique()) if not df.empty else [])
shown = df[df["laptop_id"].isin(laptop_filter)] if laptop_filter else df
st.dataframe(shown, use_container_width=True)

st.subheader("Ownership timeline per laptop")
if not df.empty:
    for laptop_id, group in df.groupby("laptop_id"):
        st.markdown(f"**Laptop #{laptop_id}**")
        chain = " → ".join(
            f"{row.new_owner if row.new_owner is not None else 'unattended'}"
            for row in group.itertuples()
        )
        st.text(chain)

st.subheader("Annotated video")
if os.path.exists(config.ANNOTATED_VIDEO_PATH):
    st.video(config.ANNOTATED_VIDEO_PATH)
else:
    st.info("Annotated video not found yet — run main.py first.")
