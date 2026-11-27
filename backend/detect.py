import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------- settings
HERE = Path(__file__).parent
DB_PATH = HERE / "readings.db"
EVENTS_PATH = HERE.parent / "notes" / "events.txt"
PLOT_PATH = HERE / "anomalies.png"
REPORT_PATH = HERE / "report.txt"   # everything printed is also saved here

LOCAL_TZ = "Africa/Tunis"   # your event notes are in this time zone
MAX_GAP_S = 30              # a gap is a pause longer than this (readings come every 10 s)
GRID = "1min"               # resample step: DHT11 is coarse, 1 min is smoother
INTERP_LIMIT = 3            # fill gaps of up to 3 grid steps; longer ones stay empty
WINDOW = 10                 # rolling window in grid steps (10 min)
TOLERANCE_MIN = 5           # a flag within +-5 min of an event counts as "caught"
MIN_ROWS = 50               # refuse to train on less than this
HOURS_NEEDED = 20           # use hour-of-day features only if data covers this many hours of the day


# ------------------------------------------------------------ 1. load data
def _pick(columns, prefix):
    """Find the column whose name starts with prefix (temp / temp_c, hum / humidity_pct)."""
    for c in columns:
        if c.lower().startswith(prefix):
            return c
    raise KeyError(f"No column starting with '{prefix}' in {list(columns)}")


def load_data(db_path=DB_PATH):
    """Read the readings table into a DataFrame indexed by UTC timestamp."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM readings", conn)
    conn.close()

    df = df.rename(columns={_pick(df.columns, "temp"): "temp",
                            _pick(df.columns, "hum"): "hum"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").drop_duplicates("ts").set_index("ts")
    return df[["temp", "hum"]]


# ------------------------------------------------------------- 2. find gaps
def find_gaps(df, max_gap_s=MAX_GAP_S):
    """Return a table of pauses in the data (ESP32 unplugged, Wi-Fi lost, ...)."""
    ts = df.index.to_series()
    delta = ts.diff()
    mask = delta > pd.Timedelta(seconds=max_gap_s)
    gaps = pd.DataFrame({
        "start": ts.shift()[mask],
        "end": ts[mask],
        "duration": delta[mask],
    })
    return gaps.reset_index(drop=True)


# -------------------------------------------------------------- 3. resample
def resample(df):
    """Average to a fixed 1-minute grid. Short holes are interpolated,
    long holes stay NaN and are dropped later (so they are never 'anomalies')."""
    grid = df.resample(GRID).mean()
    grid = grid.interpolate(limit=INTERP_LIMIT, limit_area="inside")
    return grid


# ------------------------------------------------------------- 4. features
def make_features(grid):
    """Turn raw values into numbers that describe how 'unusual' a moment is."""
    feat = pd.DataFrame(index=grid.index)
    for col in ["temp", "hum"]:
        # full window required: the first WINDOW minutes after a long gap have
        # no proper history, so they are left out instead of scored from a few points
        roll = grid[col].rolling(WINDOW, min_periods=WINDOW)
        feat[col] = grid[col]                              # the value itself
        feat[f"{col}_dev"] = grid[col] - roll.mean()       # distance from recent average
        feat[f"{col}_std"] = roll.std()                    # how jumpy it has been
        feat[f"{col}_diff"] = grid[col].diff()             # change since last minute

    # hour of day only makes sense if the data actually covers most of the day.
    # (Judging by time span is wrong: a long gap makes the span big while whole
    # hours are missing, and then a normal morning looks "new" to the model.)
    hours_covered = grid.dropna().index.tz_convert(LOCAL_TZ).hour.nunique()
    if hours_covered >= HOURS_NEEDED:
        local = grid.index.tz_convert(LOCAL_TZ)
        hour = local.hour + local.minute / 60
        feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)   # sin/cos so 23:59 is close to 00:00
        feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return feat.dropna()


# -------------------------------------------------------------- 5. detect
def detect(feat, contamination):
    """Fit an Isolation Forest. score: lower = stranger. flag: True = anomaly."""
    model = IsolationForest(n_estimators=200,
                            contamination=contamination,
                            random_state=42)
    model.fit(feat)
    result = pd.DataFrame(index=feat.index)
    result["flag"] = model.predict(feat) == -1
    result["score"] = model.decision_function(feat)
    return result


# ---------------------------------------------------- 6. events + evaluate
def load_events(path=EVENTS_PATH):
    """Read 'YYYY-MM-DD HH:MM description' lines (local time) and convert to UTC."""
    events = []
    if not path.exists():
        return events
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            stamp = pd.Timestamp(line[:16])
        except ValueError:
            print(f"  (skipping unreadable event line: {line!r})")
            continue
        events.append({"ts": stamp.tz_localize(LOCAL_TZ).tz_convert("UTC"),
                       "label": line[16:].strip()})
    return events


def evaluate(result, events, data_start, data_end):
    """Did the model flag something near each event you caused on purpose?"""
    tol = pd.Timedelta(minutes=TOLERANCE_MIN)
    flagged = result.index[result["flag"]]

    inside = [e for e in events if data_start <= e["ts"] <= data_end]
    if not inside:
        print("No events from your notes fall inside the data range.")
        return

    # An effect cannot come before its cause, so only flags in the window
    # [event, event + tolerance] count. An event is only testable if the model
    # scored enough minutes inside that window (not at a gap or at the data's end).
    caught = testable = 0
    for e in inside:
        when = e["ts"].tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
        scored = ((result.index >= e["ts"]) & (result.index <= e["ts"] + tol)).sum()
        if scored < TOLERANCE_MIN:
            print(f"  no data  {when}  {e['label']}  (too few scored minutes after it)")
            continue
        testable += 1
        hit = ((flagged >= e["ts"]) & (flagged <= e["ts"] + tol)).any()
        caught += int(hit)
        print(f"  {'CAUGHT' if hit else 'missed'}  {when}  {e['label']}")

    print(f"Events caught: {caught}/{testable} testable "
          f"({len(inside) - testable} had no scored data)")

    # Chance level: how often would a random moment have a flag in the next
    # TOLERANCE_MIN minutes? A catch rate near this number proves nothing.
    near = np.zeros(len(result), dtype=bool)
    for f in flagged:
        near |= (result.index >= f - tol) & (result.index <= f)
    print(f"Chance level: {near.mean():.0%} of minutes have a flag within "
          f"{TOLERANCE_MIN} min after them")

    unexplained = sum(
        not any(e["ts"] <= t <= e["ts"] + tol for e in inside) for t in flagged
    )
    print(f"Flagged minutes not after any event: {unexplained} of {len(flagged)}")


# --------------------------------------------------------------- 7. plot
def plot(grid, result, gaps, events, path=PLOT_PATH, show=False):
    g = grid.tz_convert(LOCAL_TZ)
    flagged_idx = result.index[result["flag"]].tz_convert(LOCAL_TZ)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
    panels = [("temp", "Temperature (°C)"), ("hum", "Humidity (%)")]

    for ax, (col, label) in zip(axes, panels):
        ax.plot(g.index, g[col], lw=1, label=col)
        pts = g.loc[g.index.intersection(flagged_idx), col]
        ax.scatter(pts.index, pts.values, color="red", zorder=3, label="anomaly")
        for _, gap in gaps[gaps["duration"] > pd.Timedelta(minutes=2)].iterrows():
            ax.axvspan(gap["start"].tz_convert(LOCAL_TZ),
                       gap["end"].tz_convert(LOCAL_TZ),
                       color="grey", alpha=0.25)
        for e in events:
            ax.axvline(e["ts"].tz_convert(LOCAL_TZ), color="green", ls="--", alpha=0.7)
        ax.set_ylabel(label)
        ax.legend(loc="upper left")

    axes[0].set_title("Room monitor: red = anomaly, grey = data gap, green dashes = your events")
    axes[1].set_xlabel(f"Time ({LOCAL_TZ})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------- 8. save
def save(grid, result, db_path=DB_PATH):
    """Store the flagged moments in an 'anomalies' table for the API / dashboard / LLM."""
    rows = grid.loc[result.index[result["flag"]]].copy()
    rows["score"] = result.loc[rows.index, "score"]
    rows = rows.reset_index()                                   # ts becomes a column
    rows["ts"] = rows["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")  # same format as readings

    conn = sqlite3.connect(db_path, timeout=30)
    rows.to_sql("anomalies", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved {len(rows)} anomalies to table 'anomalies'")


# ------------------------------------------------------------------ main
class Tee:
    """Send everything that print() writes to the screen AND to a file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def run(args):
    print(f"Report generated {datetime.now():%Y-%m-%d %H:%M}")
    df = load_data()
    print(f"Loaded {len(df)} readings from {df.index.min()} to {df.index.max()}")

    gaps = find_gaps(df)
    if len(gaps):
        print(f"Found {len(gaps)} gaps over {MAX_GAP_S}s, longest: {gaps['duration'].max()}")
    else:
        print("No gaps found")

    grid = resample(df)
    feat = make_features(grid)
    print(f"{len(feat)} usable {GRID} points after cleaning")
    if len(feat) < MIN_ROWS:
        print(f"Not enough data yet (need at least {MIN_ROWS} points). Let it collect longer.")
        return

    result = detect(feat, args.contamination)
    print(f"Flagged {int(result['flag'].sum())} points (contamination={args.contamination})")

    top = result[result["flag"]].sort_values("score").head(10)
    print("Most anomalous moments (local time, lowest score = strangest):")
    for ts, row in top.iterrows():
        t = grid.loc[ts]
        print(f"  {ts.tz_convert(LOCAL_TZ):%Y-%m-%d %H:%M}  "
              f"temp={t['temp']:.1f}  hum={t['hum']:.1f}  score={row['score']:.3f}")

    events = load_events()
    print(f"Loaded {len(events)} events from {EVENTS_PATH}")
    evaluate(result, events, df.index.min(), df.index.max())

    plot(grid, result, gaps, events, show=args.show)
    save(grid, result)


def main():
    parser = argparse.ArgumentParser(description="Anomaly detection on room readings")
    parser.add_argument("--contamination", type=float, default=0.02,
                        help="expected fraction of anomalies (default 0.02)")
    parser.add_argument("--show", action="store_true", help="open the plot window")
    args = parser.parse_args()

    # everything printed inside run() goes to the screen and to report.txt
    with open(REPORT_PATH, "w", encoding="utf-8") as report:
        original = sys.stdout
        sys.stdout = Tee(original, report)
        try:
            run(args)
            print(f"Report saved to {REPORT_PATH}")
        finally:
            sys.stdout = original


if __name__ == "__main__":
    main()