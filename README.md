# -anomaly-detection-endpoint-telemetry
An unsupervised ML pipeline that flags abnormal CPU, memory, and disk
behavior on endpoint processes — built around `scikit-learn`'s Isolation
Forest — paired with a lightweight daemon that raises real-time alerts.

## Why this project

Endpoint security and IT operations teams need to catch misbehaving
processes (memory leaks, runaway CPU usage, disk thrashing) before they
degrade a device or signal something worse — without hand-labeling every
failure mode in advance. Isolation Forest is well suited here: it isolates
anomalies based on how *few* splits it takes to separate a point from the
rest of the data, so it doesn't need anomaly examples to train on.

## Architecture

```
data/generate_telemetry.py   → synthetic endpoint telemetry generator
src/train_model.py           → feature scaling + Isolation Forest training + evaluation
src/daemon.py                → real-time scoring loop that raises alerts
src/visualize.py             → evaluation plots
models/                      → persisted scaler + trained model (joblib)
plots/                       → generated PNG charts
```

**Pipeline:**
1. Telemetry is collected in fixed-size polling windows per process
   (CPU %, memory MB, disk I/O KB/s).
2. Each window is reduced to 9 engineered features: mean/std/max of CPU,
   mean/growth-rate/std of memory, mean/std/max of disk I/O.
3. Features are standardized (`StandardScaler`) and fit with an
   `IsolationForest` (200 trees, contamination set from the training
   prior).
4. An operating threshold is chosen by sweeping percentiles of the
   anomaly-score distribution and selecting the best F1 trade-off.
5. `daemon.py` reuses the fitted scaler/model/threshold to score incoming
   windows live and write structured alerts to `alerts.log`.

## Results

On a held-out test split (1,500 windows, 10% true anomaly rate):

| Metric    | Value |
|-----------|-------|
| Precision | ~91%  |
| Recall    | ~96%  |
| F1-score  | ~93%  |

See `plots/score_distribution.png` for how cleanly the anomaly scores
separate normal vs. anomalous windows at the chosen threshold, and
`plots/memory_leak_example.png` for a sample memory-leak trace.

> Note: results are generated from a synthetic but intentionally
> noisy/overlapping dataset (per-process baseline profiles, severity-graded
> anomalies, and ~6% benign fluctuations injected into "normal" samples) to
> avoid a trivially separable toy problem. Exact numbers will vary slightly
> across reruns since severity/noise are randomized — set `RNG_SEED` in
> `generate_telemetry.py` for reproducibility.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate synthetic telemetry (5,000 windows, ~10% anomalous)
python data/generate_telemetry.py

# 2. Train the Isolation Forest and print evaluation metrics
python src/train_model.py

# 3. Generate evaluation plots
python src/visualize.py

# 4. Run the alerting daemon (replays windows and raises alerts live)
python src/daemon.py --limit 200 --interval 0.1
```

## Extending to real endpoints

`daemon.py`'s `poll_live_telemetry()` hook is the integration point for
real OS-level monitoring — e.g. via `psutil` on Linux/macOS, or by piping
in CSV output from this repo's sibling project,
[`windows-process-monitor`](../windows-process-monitor), which collects
the same CPU/memory/disk signals natively on Windows via the Win32 API.

## Tech stack

Python · scikit-learn (IsolationForest, StandardScaler) · pandas · NumPy ·
Matplotlib · joblib
