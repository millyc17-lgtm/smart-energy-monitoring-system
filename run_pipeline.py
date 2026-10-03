"""Run the whole pipeline in order: python run_pipeline.py [--with-validation]"""
import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    "clean.py",
    "features.py",
    "build_db.py",
    "analysis.py",
    "anomaly.py",
    "anomaly_ml.py",
    "forecast.py",
    "profiles.py",
]
OPTIONAL = ["validate_anomalies.py"]          # about 5 minutes, run with --with-validation
REQUIRED_RAW = [
    "block_0.csv",
    "weather_daily_darksky.csv",
    "uk_bank_holidays.csv",
    "informations_households.csv",
]

root = Path(__file__).resolve().parent
missing = [f for f in REQUIRED_RAW if not (root / "data" / "raw" / f).exists()]
if missing:
    sys.exit(f"Missing files in data/raw/: {', '.join(missing)}\nDownload them from the Kaggle dataset first (see README).")

steps = STEPS + (OPTIONAL if "--with-validation" in sys.argv else [])
for step in steps:
    print(f"\n=== {step} ===", flush=True)
    start = time.time()
    result = subprocess.run([sys.executable, str(root / "src" / step)], cwd=root)
    if result.returncode != 0:
        sys.exit(f"\nStopped: {step} failed (exit code {result.returncode}).")
    print(f"--- {step} finished in {time.time() - start:.0f}s", flush=True)

print("\nPipeline finished. Start the dashboard with: streamlit run dashboard/app.py")
