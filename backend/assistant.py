import json
import math
import re
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from detect import DB_PATH, LOCAL_TZ, find_gaps, load_data, load_events
from llm import ask_llm

MAX_ANOMALIES = 15   # most anomalous moments to show the LLM
TARGET_ROWS = 24     # rows in the time-series table sent to the LLM

RANGE_SYSTEM = (
    "You turn a question about sensor data into a time range. "
    'Reply with ONLY a JSON object: {"start": "YYYY-MM-DD HH:MM" or null, '
    '"end": "YYYY-MM-DD HH:MM" or null}. Times are local time. '
    "Use null when the question does not limit the period."
)

ANSWER_SYSTEM = (
    "You are a monitoring assistant for a room with a temperature and humidity sensor. "
    "Rules: use ONLY the data summary you are given and never invent numbers. "
    "Separate what the data shows from possible explanations, and label explanations "
    "as hypotheses (a sensor cannot know the cause). If an event logged by the user "
    "matches an anomaly, say so. If data is missing for the period, say so. "
    "Give a recommendation only when the data supports one. "
    "Answer in the language of the question, in under 150 words."
)


def fmt(ts):
    """UTC timestamp -> local time string."""
    return ts.tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")


def load_anomalies():
    """Read the 'anomalies' table written by detect.py (empty if it doesn't exist yet)."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM anomalies", conn)
    except Exception:
        df = pd.DataFrame(columns=["ts", "temp", "hum", "score"])
    finally:
        conn.close()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").set_index("ts")


def _to_utc(text):
    if not text:
        return None
    return pd.Timestamp(text).tz_localize(LOCAL_TZ).tz_convert("UTC")


def parse_range(question, data_start, data_end):
    """Step 1: ask the LLM for the time range the question is about."""
    now = datetime.now(ZoneInfo(LOCAL_TZ)).strftime("%Y-%m-%d %H:%M (%A)")
    prompt = (f"Current local time: {now}\n"
              f"Data available from {fmt(data_start)} to {fmt(data_end)}\n"
              f"Question: {question}")
    try:
        raw = ask_llm(RANGE_SYSTEM, prompt, max_tokens=100)
        rng = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        start, end = _to_utc(rng.get("start")), _to_utc(rng.get("end"))
    except Exception:
        start = end = None          # fall back to all data
    start, end = start or data_start, end or data_end
    if start > end:
        start, end = data_start, data_end
    return start, end


def summarize(df, anomalies, events, start, end):
    """Step 2: a compact text summary of the period, built with pandas."""
    lines = [f"Period: {fmt(start)} to {fmt(end)} (local time)"]
    part = df[(df.index >= start) & (df.index <= end)]
    if part.empty:
        lines.append("No readings exist in this period.")
        return "\n".join(lines)

    span = part.index.max() - part.index.min()
    expected = span.total_seconds() / 10 + 1          # sensor sends one reading per 10 s
    lines.append(f"Readings: {len(part)} between {fmt(part.index.min())} and "
                 f"{fmt(part.index.max())} (coverage {len(part) / expected:.0%} of expected)")

    for col, label, unit in [("temp", "Temperature", "°C"), ("hum", "Humidity", "%")]:
        s = part[col]
        lines.append(f"{label}: mean {s.mean():.1f}{unit}, "
                     f"min {s.min():.0f}{unit} at {fmt(s.idxmin())}, "
                     f"max {s.max():.0f}{unit} at {fmt(s.idxmax())}")

    step = max(10, math.ceil(span.total_seconds() / 60 / TARGET_ROWS))
    table = part.resample(f"{step}min").mean().dropna()
    lines.append(f"Averages every {step} min (local time | temp | humidity):")
    for ts, row in table.iterrows():
        lines.append(f"  {fmt(ts)} | {row['temp']:.1f} | {row['hum']:.1f}")

    a = anomalies[(anomalies.index >= start) & (anomalies.index <= end)]
    lines.append(f"Detected anomalies in period: {len(a)} "
                 "(Isolation Forest on 1-min data; lower score = more unusual)")
    for ts, row in a.sort_values("score").head(MAX_ANOMALIES).sort_index().iterrows():
        lines.append(f"  {fmt(ts)} | temp {row['temp']:.1f} | hum {row['hum']:.1f} "
                     f"| score {row['score']:.3f}")

    ev = [e for e in events if start <= e["ts"] <= end]
    if ev:
        lines.append("Events logged by the user:")
        lines += [f"  {fmt(e['ts'])} | {e['label']}" for e in ev]

    gaps = find_gaps(part, 120)
    if len(gaps):
        lines.append(f"Data gaps over 2 min: {len(gaps)}")
        for _, g in gaps.head(5).iterrows():
            lines.append(f"  {fmt(g['start'])} -> {fmt(g['end'])}")
    return "\n".join(lines)


def answer(question):
    """Full pipeline. Returns (reply, data_summary) so the UI can show what the model saw."""
    df = load_data()
    anomalies = load_anomalies()
    events = load_events()
    start, end = parse_range(question, df.index.min(), df.index.max())
    context = summarize(df, anomalies, events, start, end)
    reply = ask_llm(ANSWER_SYSTEM, f"DATA SUMMARY\n{context}\n\nQUESTION\n{question}")
    return reply, context


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Summarize the data and explain the strongest anomaly."
    reply, ctx = answer(q)
    print(ctx, "\n" + "-" * 60 + "\n" + reply)