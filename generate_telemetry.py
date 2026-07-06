"""
generate_telemetry.py

Simulates endpoint telemetry (CPU%, memory MB, disk I/O KB/s) for a fleet of
processes over time, and injects three classes of anomalies:
  - memory_leak : sustained upward memory drift within a window
  - cpu_spike    : sudden, short-lived CPU saturation
  - disk_spike   : abnormal burst of disk I/O

Each row in the output CSV represents one (process, time-window) sample with
engineered features ready for model training, plus a ground-truth label
(0 = normal, 1 = anomaly) used only for evaluation, never as a model input.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
N_SAMPLES = 5000
ANOMALY_FRACTION = 0.10  # ~10% of samples are anomalous
WINDOW_SIZE = 30  # telemetry points per window (e.g. 30 seconds of polling)

OUTPUT_PATH = Path(__file__).parent / "endpoint_telemetry.csv"


# Per-process baseline profiles -- some apps are naturally heavier than
# others, which adds realistic overlap between "normal but heavy" and
# "lightly anomalous" samples instead of one global clean threshold.
PROCESS_PROFILES = {
    "chrome.exe":   dict(cpu=22, mem=420, disk=70),
    "explorer.exe": dict(cpu=8,  mem=120, disk=30),
    "svchost.exe":  dict(cpu=6,  mem=90,  disk=25),
    "outlook.exe":  dict(cpu=14, mem=260, disk=55),
    "teams.exe":    dict(cpu=20, mem=310, disk=60),
    "python.exe":   dict(cpu=18, mem=180, disk=40),
    "code.exe":     dict(cpu=16, mem=240, disk=45),
    "slack.exe":    dict(cpu=13, mem=200, disk=35),
    "edge.exe":     dict(cpu=19, mem=350, disk=65),
}
PROCESS_NAMES = list(PROCESS_PROFILES.keys())


def make_normal_window(rng, process):
    p = PROCESS_PROFILES[process]
    cpu = np.clip(rng.normal(p["cpu"], p["cpu"] * 0.35, WINDOW_SIZE), 0, 100)
    mem = p["mem"] + rng.normal(0, p["mem"] * 0.04, WINDOW_SIZE).cumsum() * 0.08
    disk = np.clip(rng.normal(p["disk"], p["disk"] * 0.45, WINDOW_SIZE), 0, None)

    # ~6% of "normal" windows get a mild, benign fluctuation (e.g. a real
    # but harmless burst from a browser tab loading) -- this is what makes
    # the boundary fuzzy instead of perfectly separable.
    if rng.random() < 0.06:
        i = rng.integers(0, WINDOW_SIZE - 3)
        cpu[i:i+3] = np.clip(cpu[i:i+3] + rng.uniform(15, 35), 0, 100)
    return cpu, mem, disk


def make_memory_leak_window(rng, process):
    p = PROCESS_PROFILES[process]
    cpu = np.clip(rng.normal(p["cpu"] * 1.1, p["cpu"] * 0.4, WINDOW_SIZE), 0, 100)
    # severity varies -- some leaks are slow/subtle, some are fast
    severity = rng.choice(["mild", "moderate", "severe"], p=[0.35, 0.4, 0.25])
    leak_rate = {"mild": rng.uniform(1.5, 4), "moderate": rng.uniform(4, 10),
                 "severe": rng.uniform(10, 22)}[severity]
    mem = p["mem"] + np.arange(WINDOW_SIZE) * leak_rate + rng.normal(0, p["mem"] * 0.05, WINDOW_SIZE)
    disk = np.clip(rng.normal(p["disk"], p["disk"] * 0.45, WINDOW_SIZE), 0, None)
    return cpu, mem, disk


def make_cpu_spike_window(rng, process):
    p = PROCESS_PROFILES[process]
    cpu = np.clip(rng.normal(p["cpu"], p["cpu"] * 0.35, WINDOW_SIZE), 0, 100)
    severity = rng.choice(["mild", "moderate", "severe"], p=[0.3, 0.4, 0.3])
    spike_level = {"mild": rng.uniform(45, 65), "moderate": rng.uniform(65, 85),
                   "severe": rng.uniform(85, 100)}[severity]
    spike_len = rng.integers(2, 8) if severity != "mild" else rng.integers(2, 4)
    spike_start = rng.integers(5, WINDOW_SIZE - 8)
    cpu[spike_start:spike_start + spike_len] = np.clip(
        rng.normal(spike_level, 8, spike_len), 0, 100
    )
    mem = p["mem"] + rng.normal(0, p["mem"] * 0.04, WINDOW_SIZE).cumsum() * 0.08
    disk = np.clip(rng.normal(p["disk"], p["disk"] * 0.45, WINDOW_SIZE), 0, None)
    return cpu, mem, disk


def make_disk_spike_window(rng, process):
    p = PROCESS_PROFILES[process]
    cpu = np.clip(rng.normal(p["cpu"], p["cpu"] * 0.35, WINDOW_SIZE), 0, 100)
    mem = p["mem"] + rng.normal(0, p["mem"] * 0.04, WINDOW_SIZE).cumsum() * 0.08
    disk = np.clip(rng.normal(p["disk"], p["disk"] * 0.45, WINDOW_SIZE), 0, None)
    severity = rng.choice(["mild", "moderate", "severe"], p=[0.3, 0.4, 0.3])
    spike_level = {"mild": rng.uniform(200, 400), "moderate": rng.uniform(400, 700),
                   "severe": rng.uniform(700, 1100)}[severity]
    spike_len = rng.integers(2, 7)
    spike_start = rng.integers(5, WINDOW_SIZE - 8)
    disk[spike_start:spike_start + spike_len] = np.clip(
        rng.normal(spike_level, spike_level * 0.15, spike_len), 0, None
    )
    return cpu, mem, disk


def extract_features(cpu, mem, disk):
    return {
        "cpu_mean": cpu.mean(),
        "cpu_std": cpu.std(),
        "cpu_max": cpu.max(),
        "mem_mean": mem.mean(),
        "mem_growth_rate": (mem[-1] - mem[0]) / WINDOW_SIZE,
        "mem_std": mem.std(),
        "disk_mean": disk.mean(),
        "disk_std": disk.std(),
        "disk_max": disk.max(),
    }


def main():
    rng = np.random.default_rng(RNG_SEED)
    n_anomalies = int(N_SAMPLES * ANOMALY_FRACTION)
    n_normal = N_SAMPLES - n_anomalies

    # split anomalies roughly evenly across the three anomaly types
    n_leak = n_anomalies // 3
    n_cpu = n_anomalies // 3
    n_disk = n_anomalies - n_leak - n_cpu

    rows = []

    for i in range(n_normal):
        process = rng.choice(PROCESS_NAMES)
        cpu, mem, disk = make_normal_window(rng, process)
        feats = extract_features(cpu, mem, disk)
        feats.update(process=process, label=0, anomaly_type="none")
        rows.append(feats)

    for i in range(n_leak):
        process = rng.choice(PROCESS_NAMES)
        cpu, mem, disk = make_memory_leak_window(rng, process)
        feats = extract_features(cpu, mem, disk)
        feats.update(process=process, label=1, anomaly_type="memory_leak")
        rows.append(feats)

    for i in range(n_cpu):
        process = rng.choice(PROCESS_NAMES)
        cpu, mem, disk = make_cpu_spike_window(rng, process)
        feats = extract_features(cpu, mem, disk)
        feats.update(process=process, label=1, anomaly_type="cpu_spike")
        rows.append(feats)

    for i in range(n_disk):
        process = rng.choice(PROCESS_NAMES)
        cpu, mem, disk = make_disk_spike_window(rng, process)
        feats = extract_features(cpu, mem, disk)
        feats.update(process=process, label=1, anomaly_type="disk_spike")
        rows.append(feats)

    df = pd.DataFrame(rows).sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} samples ({df['label'].sum()} anomalous) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
