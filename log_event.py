import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

path = Path(__file__).parent / "notes" / "events.txt"
path.parent.mkdir(exist_ok=True)
now = datetime.now(ZoneInfo("Africa/Tunis")).strftime("%Y-%m-%d %H:%M")
label = " ".join(sys.argv[1:]) or "event"
with open(path, "a") as f:
    f.write(f"{now} {label}\n")
print(f"logged: {now} {label}")