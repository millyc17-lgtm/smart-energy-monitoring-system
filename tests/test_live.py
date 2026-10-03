import pytest

import live_api
import live_store

GOOD = {"device_id": "MAC000002", "timestamp": "2013-05-01 18:30:00", "power_kw": 0.8}


def test_validate_accepts_good_reading_and_fills_defaults():
    clean, error = live_store.validate(GOOD)
    assert error is None
    assert clean["source"] == "replay"
    assert clean["timestamp"] == "2013-05-01 18:30:00"


@pytest.mark.parametrize("change", [
    {"device_id": ""},
    {"device_id": 5},
    {"timestamp": "yesterday"},
    {"timestamp": None},
    {"power_kw": -0.1},
    {"power_kw": 51},
    {"power_kw": "high"},
    {"power_kw": True},
    {"power_kw": float("nan")},
    {"source": "robot"},
])
def test_validate_rejects_bad_readings(change):
    clean, error = live_store.validate({**GOOD, **change})
    assert clean is None and error


def test_store_ignores_repeats_and_returns_newest_first(tmp_path):
    con = live_store.connect(tmp_path / "live.db")
    a = live_store.validate(GOOD)[0]
    b = live_store.validate({**GOOD, "timestamp": "2013-05-01 19:00:00", "power_kw": 1.2})[0]
    assert live_store.add_readings(con, [a, b]) == 2
    assert live_store.add_readings(con, [a]) == 0          # same device + time again
    rows = live_store.latest(con, "MAC000002")
    assert [r["power_kw"] for r in rows] == [1.2, 0.8]
    assert live_store.latest(con, "someone-else") == []


@pytest.fixture
def client(tmp_path):
    return live_api.create_app(str(tmp_path / "live.db")).test_client()


def test_api_round_trip(client):
    assert client.get("/health").status_code == 200
    r = client.post("/readings", json=[GOOD, {**GOOD, "timestamp": "2013-05-01 19:00:00"}])
    assert r.status_code == 201 and r.get_json()["saved"] == 2
    again = client.post("/readings", json=GOOD)
    assert again.get_json()["duplicates"] == 1
    rows = client.get("/readings?device_id=MAC000002&limit=1").get_json()
    assert len(rows) == 1 and rows[0]["timestamp"] == "2013-05-01 19:00:00"


def test_api_rejects_bad_input_and_saves_nothing(client):
    assert client.post("/readings", json=[GOOD, {**GOOD, "power_kw": -1}]).status_code == 400
    assert client.post("/readings", data="not json").status_code == 400
    assert client.post("/readings", json=[]).status_code == 400
    assert client.get("/readings").get_json() == []         # the good one in the bad batch was not saved


def test_api_key_is_enforced_when_set(client, monkeypatch):
    monkeypatch.setenv("LIVE_API_KEY", "secret")
    assert client.post("/readings", json=GOOD).status_code == 401
    assert client.post("/readings", json=GOOD, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/readings", json=GOOD, headers={"X-API-Key": "secret"}).status_code == 201