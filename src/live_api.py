"""A small local web service that receives readings and stores them in data/live.db.

Start it from the project folder:   python src/live_api.py
It listens on 127.0.0.1 (your own computer only), port 5000.

Optional: set the environment variable LIVE_API_KEY and every POST must then send
the same value in an "X-API-Key" header.
"""
import hmac
import os

from flask import Flask, jsonify, request

import live_store

MAX_BATCH = 1000


def create_app(db_path="data/live.db"):
    app = Flask(__name__)
    app.config["DB_PATH"] = db_path

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.post("/readings")
    def post_readings():
        key = os.environ.get("LIVE_API_KEY")
        if key and not hmac.compare_digest(request.headers.get("X-API-Key", ""), key):
            return jsonify(error="missing or wrong API key"), 401

        body = request.get_json(silent=True)
        items = body if isinstance(body, list) else [body]
        if body is None or not items or len(items) > MAX_BATCH:
            return jsonify(error=f"send one reading, or a list of 1 to {MAX_BATCH} readings, as JSON"), 400

        clean = []
        for i, item in enumerate(items):
            reading, error = live_store.validate(item)
            if error:
                return jsonify(error=f"reading {i}: {error}"), 400
            clean.append(reading)

        con = live_store.connect(app.config["DB_PATH"])
        try:
            saved = live_store.add_readings(con, clean)
        finally:
            con.close()
        return jsonify(received=len(clean), saved=saved, duplicates=len(clean) - saved), 201

    @app.get("/readings")
    def get_readings():
        con = live_store.connect(app.config["DB_PATH"])
        try:
            rows = live_store.latest(
                con,
                device_id=request.args.get("device_id"),
                limit=request.args.get("limit", 100, type=int),
            )
        finally:
            con.close()
        return jsonify(rows)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000)