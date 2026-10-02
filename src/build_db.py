import sqlite3
import pandas as pd

df = pd.read_csv("data/processed/energy_features.csv", parse_dates=["timestamp"])
df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
df["is_weekend"] = df["is_weekend"].astype(int)

columns = ["household_id", "timestamp", "date", "hour", "day_of_week",
           "month", "is_weekend", "energy_kwh", "power_kw"]

conn = sqlite3.connect("data/energy.db")
df[columns].to_sql("energy_readings", conn, if_exists="replace", index=False)
conn.execute("CREATE INDEX IF NOT EXISTS idx_household_time ON energy_readings (household_id, timestamp)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON energy_readings (date)")
conn.commit()

print(pd.read_sql("SELECT COUNT(*) AS rows, COUNT(DISTINCT household_id) AS households FROM energy_readings", conn))
conn.close()