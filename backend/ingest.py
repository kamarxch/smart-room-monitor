import json
import sqlite3
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

topic = "room/lab1/telemetry"

db = sqlite3.connect("readings.db")
db.execute("CREATE TABLE IF NOT EXISTS readings (ts TEXT, temp REAL, hum REAL)")
db.commit()

def on_connect(client, db, flags, reason_code, properties):
    print("connected to broker:", reason_code)
    client.subscribe(topic)
    

def on_message(_client, db, msg):
    data = json.loads(msg.payload)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db.execute("Insert into readings values(?,?,?)", (ts, data["temp_c"], data["humidity_pct"]))
    db.commit()
    print(ts,data)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, userdata =db)
client.on_connect = on_connect
client.on_message = on_message
client.connect("localhost", 1883)
client.loop_forever()
