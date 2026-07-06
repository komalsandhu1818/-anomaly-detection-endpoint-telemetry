"""
daemon.py

Lightweight monitoring daemon that simulates polling live endpoint telemetry,
scores each window with the trained Isolation Forest, and raises real-time
alerts for anomalous processes (memory leak / CPU spike / disk spike).

In a production deployment, `poll_live_telemetry()` would be replaced with
real OS-level polling (e.g. psutil on Linux/macOS, or the Win32
ProcessMonitor project in this repo's sibling folder on Windows). Here it
replays held-out samples from the dataset to demonstrate the alerting loop
end-to-end without needing OS-level hooks.
"""

import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

BASE = Path(__file__).parent.parent
MODEL_DIR = BASE / "models"
ALERT_LOG = BASE / "alerts.log"


class AnomalyDaemon:
    def __init__(self):
        self.scaler = joblib.load(MODEL_DIR / "scaler.joblib")
        self.model = joblib.load(MODEL_DIR / "isolation_forest.joblib")
        op = joblib.load(MODEL_DIR / "operating_point.joblib")
        self.threshold = op["threshold"]
        self.feature_cols = op["feature_cols"]

    def score_window(self, feature_row: dict) -> tuple[float, bool]:
        x = np.array([[feature_row[c] for c in self.feature_cols]])
        x_s = self.scaler.transform(x)
        score = self.model.decision_function(x_s)[0]
        is_anomaly = score < self.threshold
        return score, is_anomaly

    def raise_alert(self, process: str, score: float, feature_row: dict):
        alert = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "process": process,
            "anomaly_score": round(float(score), 4),
            "cpu_mean": round(feature_row["cpu_mean"], 1),
            "mem_growth_rate": round(feature_row["mem_growth_rate"], 2),
            "disk_max": round(feature_row["disk_max"], 1),
        }
        print(f"[ALERT] {alert['timestamp']} | {process} | score={alert['anomaly_score']} "
              f"| cpu_mean={alert['cpu_mean']}% | mem_growth={alert['mem_growth_rate']} MB/tick "
              f"| disk_max={alert['disk_max']} KB/s")
        with open(ALERT_LOG, "a") as f:
            f.write(json.dumps(alert) + "\n")

    def run(self, source_csv: Path, interval_sec: float, limit: int | None):
        df = pd.read_csv(source_csv)
        if limit:
            df = df.head(limit)

        print(f"AnomalyDaemon started — monitoring {len(df)} telemetry windows "
              f"(polling every {interval_sec}s). Alerts logged to {ALERT_LOG}")

        n_alerts = 0
        for _, row in df.iterrows():
            feature_row = row[self.feature_cols].to_dict()
            score, is_anomaly = self.score_window(feature_row)
            if is_anomaly:
                self.raise_alert(row["process"], score, feature_row)
                n_alerts += 1
            time.sleep(interval_sec)

        print(f"\nDaemon finished. Raised {n_alerts} alert(s) over {len(df)} windows.")


def main():
    parser = argparse.ArgumentParser(description="Endpoint telemetry anomaly daemon")
    parser.add_argument("--source", type=str, default=str(BASE / "data" / "endpoint_telemetry.csv"),
                         help="CSV of telemetry windows to replay")
    parser.add_argument("--interval", type=float, default=0.0,
                         help="Seconds to sleep between polls (0 = run as fast as possible)")
    parser.add_argument("--limit", type=int, default=200,
                         help="Number of windows to replay (omit/0 for full file)")
    args = parser.parse_args()

    daemon = AnomalyDaemon()
    daemon.run(Path(args.source), args.interval, args.limit or None)


if __name__ == "__main__":
    main()
