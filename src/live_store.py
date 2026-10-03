"""Storage and validation for live readings (no web code here, so it is easy to test).

Readings go into their own SQLite file (data/live.db), separate from the historical
data/energy.db.
"""
import math
import sqlite3
from datetime import datetime
from pathlib import Path

SOURCES = {"replay", "device"}
MAX_POWER_KW = 50.0        # anything above this is rejected as a faulty reading


def connect(path):
    """Open (and create if needed) the live database."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        """CREATE TABLE IF NOT EXISTS live_readings (
               id          INTEGER PRIMARY KEY,
               source      TEXT NOT NULL,
               device_id   TEXT NOT NULL,
               timestamp   TEXT NOT NULL,
               power_kw    REAL NOT NULL,
               received_at TEXT NOT NULL,
               UNIQUE (device_id, timestamp)
           )"""
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_live_device_time ON live_readings (device_id, timestamp)")
    return con


def validate(item):
    """Check one incoming reading. Returns (clean_reading, None) or (None, error_message)."""
    if not isinstance(item, dict):
        return None, "each reading must be a JSON object"

    device_id = item.get("device_id")
    if not isinstance(device_id, str) or not device_id.strip() or len(device_id) > 64:
        return None, "device_id must be a non-empty text value of at most 64 characters"

    try:
        ts = datetime.fromisoformat(str(item.get("timestamp")))
    except ValueError:
        return None, "timestamp must look like 2013-05-01 18:30:00"

    power = item.get("power_kw")
    if isinstance(power, bool) or not isinstance(power, (int, float)) or not math.isfinite(power):
        return None, "power_kw must be a number"
    if power < 0 or power > MAX_POWER_KW:
        return None, f"power_kw must be between 0 and {MAX_POWER_KW:g}"

    source = item.get("source", "replay")
    if source not in SOURCES:
        return None, "source must be 'replay' or 'device'"

    clean = {
        "source": source,
        "device_id": device_id.strip(),
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "power_kw": float(power),
    }
    return clean, None


def add_readings(con, readings):
    """Save validated readings. Returns how many were new (repeats of the same device and time are ignored)."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    before = con.total_changes
    con.executemany(
        """INSERT OR IGNORE INTO live_readings (source, device_id, timestamp, power_kw, received_at)
           VALUES (:source, :device_id, :timestamp, :power_kw, :now)""",
        [{**r, "now": now} for r in readings],
    )
    con.commit()
    return con.total_changes - before


def latest(con, device_id=None, limit=100):
    """Most recent readings, newest first."""
    limit = max(1, min(int(limit), 1000))
    sql = "SELECT source, device_id, timestamp, power_kw FROM live_readings"
    args = []
    if device_id:
        sql += " WHERE device_id = ?"
        args.append(device_id)
    sql += " ORDER BY timestamp DESC, id DESC LIMIT ?"
    args.append(limit)
    cols = ["source", "device_id", "timestamp", "power_kw"]
    return [dict(zip(cols, row)) for row in con.execute(sql, args)]