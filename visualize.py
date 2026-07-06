"""
visualize.py

Generates two plots for the project README / portfolio:
  1. Score distribution for normal vs anomalous samples with the chosen
     operating threshold marked
  2. Example memory-leak window vs a normal window, side by side
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent.parent
DATA_PATH = BASE / "data" / "endpoint_telemetry.csv"
MODEL_DIR = BASE / "models"
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def plot_score_distribution():
    df = pd.read_csv(DATA_PATH)
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    model = joblib.load(MODEL_DIR / "isolation_forest.joblib")
    op = joblib.load(MODEL_DIR / "operating_point.joblib")

    X = df[op["feature_cols"]].values
    X_s = scaler.transform(X)
    scores = model.decision_function(X_s)
    df["score"] = scores

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df.loc[df.label == 0, "score"], bins=50, alpha=0.6, label="Normal", color="#4C72B0")
    ax.hist(df.loc[df.label == 1, "score"], bins=50, alpha=0.6, label="Anomalous", color="#C44E52")
    ax.axvline(op["threshold"], color="black", linestyle="--", label="Operating threshold")
    ax.set_xlabel("Isolation Forest anomaly score (lower = more anomalous)")
    ax.set_ylabel("Count")
    ax.set_title("Anomaly Score Distribution: Normal vs Anomalous Telemetry Windows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "score_distribution.png", dpi=150)
    print(f"Saved {PLOTS_DIR / 'score_distribution.png'}")


def plot_example_windows():
    rng = np.random.default_rng(7)
    window = 30
    normal_mem = 200 + rng.normal(0, 8, window).cumsum() * 0.08
    leak_mem = 200 + np.arange(window) * 9 + rng.normal(0, 10, window)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(normal_mem, label="Normal process", color="#4C72B0", linewidth=2)
    ax.plot(leak_mem, label="Memory-leak process", color="#C44E52", linewidth=2)
    ax.set_xlabel("Polling tick within window")
    ax.set_ylabel("Memory usage (MB)")
    ax.set_title("Example: Normal vs Memory-Leak Telemetry Window")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "memory_leak_example.png", dpi=150)
    print(f"Saved {PLOTS_DIR / 'memory_leak_example.png'}")


if __name__ == "__main__":
    plot_score_distribution()
    plot_example_windows()
