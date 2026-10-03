import sqlite3

import replay


def make_history(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE energy_readings (household_id TEXT, timestamp TEXT, power_kw REAL)")
    con.executemany(
        "INSERT INTO energy_readings VALUES (?, ?, ?)",
        [
            ("A", "2013-06-01 00:00:00", 0.4),
            ("B", "2013-06-01 00:00:00", 0.9),
            ("A", "2013-06-01 00:30:00", 0.5),
            ("A", "2013-06-02 23:30:00", 0.6),
            ("A", "2013-06-03 00:00:00", 9.9),     # after the window
            ("A", "2013-05-31 23:30:00", 9.9),     # before the window
            ("C", "2013-06-01 00:00:00", 9.9),     # not requested
        ],
    )
    con.commit()
    con.close()


def test_load_readings_filters_and_orders(tmp_path):
    db = tmp_path / "h.db"
    make_history(db)
    rows = replay.load_readings(str(db), ["A", "B"], "2013-06-01", 2)
    assert [(r["device_id"], r["timestamp"]) for r in rows] == [
        ("A", "2013-06-01 00:00:00"),
        ("B", "2013-06-01 00:00:00"),
        ("A", "2013-06-01 00:30:00"),
        ("A", "2013-06-02 23:30:00"),
    ]
    assert all(r["source"] == "replay" for r in rows)


def test_load_readings_unknown_household_is_empty(tmp_path):
    db = tmp_path / "h.db"
    make_history(db)
    assert replay.load_readings(str(db), ["nobody"], "2013-06-01", 2) == []