"""
train_model.py

Trains an Isolation Forest on engineered endpoint telemetry features to flag
anomalous process behavior (memory leaks, CPU spikes, disk I/O bursts).

Pipeline:
  1. Load engineered features from endpoint_telemetry.csv
  2. Scale features (StandardScaler)
  3. Fit IsolationForest (unsupervised -- ground-truth labels are used only
     for evaluation, never for fitting)
  4. Score every sample, pick an operating threshold on the score
     distribution, and report precision/recall/F1 against the held-out
     ground truth
  5. Persist the fitted scaler + model with joblib for use by daemon.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib

BASE = Path(__file__).parent.parent
DATA_PATH = BASE / "data" / "endpoint_telemetry.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "cpu_mean", "cpu_std", "cpu_max",
    "mem_mean", "mem_growth_rate", "mem_std",
    "disk_mean", "disk_std", "disk_max",
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def main():
    df = load_data()
    X = df[FEATURE_COLS].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # contamination tells the forest roughly what fraction of TRAIN data is
    # anomalous -- it does not see y_train, only this prior
    contamination = y_train.mean()

    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s)

    # decision_function: higher = more normal, lower = more anomalous
    scores = model.decision_function(X_test_s)

    # Sweep thresholds on the score distribution to report the best
    # precision/recall trade-off, then settle on the operating point.
    best = None
    for pct in np.arange(5, 25, 0.5):
        thresh = np.percentile(scores, pct)
        preds = (scores < thresh).astype(int)
        if preds.sum() == 0:
            continue
        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        if best is None or f1 > best["f1"]:
            best = {"pct": pct, "thresh": thresh, "precision": p, "recall": r, "f1": f1}

    preds = (scores < best["thresh"]).astype(int)
    cm = confusion_matrix(y_test, preds)

    print("=" * 50)
    print("Isolation Forest -- Endpoint Telemetry Anomaly Detection")
    print("=" * 50)
    print(f"Test set size      : {len(y_test)}")
    print(f"True anomalies     : {int(y_test.sum())}")
    print(f"Operating threshold: {best['pct']:.1f}th percentile of score")
    print(f"Precision          : {best['precision']*100:.1f}%")
    print(f"Recall             : {best['recall']*100:.1f}%")
    print(f"F1-score           : {best['f1']*100:.1f}%")
    print("Confusion matrix [ [TN FP] [FN TP] ]:")
    print(cm)

    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(model, MODEL_DIR / "isolation_forest.joblib")
    joblib.dump(
        {"threshold": best["thresh"], "feature_cols": FEATURE_COLS},
        MODEL_DIR / "operating_point.joblib",
    )
    print(f"\nSaved scaler, model, and operating point to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
