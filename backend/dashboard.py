import altair as alt
import pandas as pd
import streamlit as st

import assistant
from detect import LOCAL_TZ, find_gaps, load_data

st.set_page_config(page_title="Smart Room Monitor", layout="wide")


@st.cache_data(ttl=30)
def get_readings():
    return load_data()


@st.cache_data(ttl=30)
def get_anomalies():
    return assistant.load_anomalies()


try:
    df = get_readings()
except Exception as e:
    st.error(f"Could not read readings.db: {e}. Run the dashboard from the backend folder.")
    st.stop()
anoms = get_anomalies()

# ------------------------------------------------------------------ sidebar
st.sidebar.header("View")
period = st.sidebar.radio("Period (counted back from the newest reading)",
                          ["Last 6 hours", "Last 24 hours", "Last 3 days", "All data"], index=1)
step = st.sidebar.selectbox("Averaging", ["1min", "5min", "15min"], index=1)
if st.sidebar.button("Refresh now"):
    st.cache_data.clear()
    st.rerun()

end = df.index.max()
hours = {"Last 6 hours": 6, "Last 24 hours": 24, "Last 3 days": 72}.get(period)
start = end - pd.Timedelta(hours=hours) if hours else df.index.min()
view = df[df.index >= start]
view_anoms = anoms[anoms.index >= start]

st.title("Smart Room Monitor")
tab_overview, tab_anoms, tab_chat = st.tabs(["Overview", "Anomalies", "Ask the assistant"])

# ----------------------------------------------------------------- overview
with tab_overview:
    latest = view.iloc[-1]
    age = pd.Timestamp.now(tz="UTC") - view.index[-1]
    gaps = find_gaps(view, 120)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Temperature now", f"{latest['temp']:.0f} °C")
    c2.metric("Humidity now", f"{latest['hum']:.0f} %")
    c3.metric("Mean temp (period)", f"{view['temp'].mean():.1f} °C")
    c4.metric("Max humidity (period)", f"{view['hum'].max():.0f} %")
    c5.metric("Anomalies (period)", len(view_anoms))

    if age > pd.Timedelta(minutes=2):
        st.warning(f"Last reading was {int(age.total_seconds() // 60)} min ago "
                   f"({view.index[-1].tz_convert(LOCAL_TZ):%Y-%m-%d %H:%M}). "
                   "The collector or the ESP32 may be offline.")
    if len(gaps):
        st.caption(f"{len(gaps)} data gap(s) over 2 min in this period, "
                   f"longest {gaps['duration'].max()}.")

    plot = view.resample(step).mean().dropna()
    plot.index = plot.index.tz_convert(LOCAL_TZ).tz_localize(None)
    plot = plot.rename_axis("time").reset_index()

    pts = view_anoms.copy()
    pts.index = pts.index.tz_convert(LOCAL_TZ).tz_localize(None)
    pts = pts.rename_axis("time").reset_index()

    def chart(col, title):
        line = alt.Chart(plot).mark_line().encode(
            x=alt.X("time:T", title=f"Local time ({LOCAL_TZ})"),
            y=alt.Y(f"{col}:Q", title=title, scale=alt.Scale(zero=False)))
        red = alt.Chart(pts).mark_circle(color="red", size=80).encode(
            x="time:T", y=f"{col}:Q",
            tooltip=["time:T", "temp:Q", "hum:Q", "score:Q"])
        return (line + red).properties(height=260)

    st.altair_chart(chart("temp", "Temperature (°C)"), use_container_width=True)
    st.altair_chart(chart("hum", "Humidity (%)"), use_container_width=True)
    st.caption("Red dots = moments flagged by the Isolation Forest (run detect.py to update).")

# ---------------------------------------------------------------- anomalies
with tab_anoms:
    if view_anoms.empty:
        st.info("No anomalies in this period. Run `python detect.py` to (re)compute them.")
    else:
        table = view_anoms.copy()
        table.index = table.index.tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
        table.index.name = "local time"
        st.dataframe(table.sort_values("score"), use_container_width=True)
        st.caption("Lower score = more unusual. Flags are statistical, not proof of an event.")

# ------------------------------------------------------------------- chat
with tab_chat:
    st.write("Ask about the room in plain language. The assistant only sees a summary "
             "computed from your database, never the raw file.")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    examples = ["Summarize the last 24 hours",
                "What was the strangest moment and what could explain it?",
                "Was humidity high this morning?"]
    cols = st.columns(len(examples))
    clicked = None
    for col, ex in zip(cols, examples):
        if col.button(ex):
            clicked = ex

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m.get("context"):
                with st.expander("Data the assistant saw"):
                    st.code(m["context"])

    question = st.chat_input("Ask about the room data") or clicked
    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply, context = assistant.answer(question)
                except Exception as e:
                    reply, context = f"Could not get an answer: {e}", None
            st.write(reply)
            if context:
                with st.expander("Data the assistant saw"):
                    st.code(context)
        st.session_state.chat.append({"role": "assistant", "content": reply, "context": context})