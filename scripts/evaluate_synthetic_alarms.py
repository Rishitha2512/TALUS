"""Run a reproducible walk-forward alarm exercise on simulated sensors only."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.research import alarm_metrics, simulate_sensor_series

OUTPUT_JSON = ROOT / "results" / "synthetic_alarm_evaluation.json"
OUTPUT_CSV = ROOT / "results" / "synthetic_alarm_pareto.csv"


def evaluate() -> tuple[dict, pd.DataFrame]:
    """Generate, split by rolling origin, and score threshold alarms."""
    records = simulate_sensor_series(
        sample_count=2000,
        seed=2026,
        baseline_mm=5,
        drift_mm_per_sample=0.002,
        noise_std_mm=0.05,
        acceleration_mm_per_sample2=0.00005,
        acceleration_start_index=1000,
        dropout_rate=0.02,
        outlier_rate=0.01,
    )
    raw = np.array([
        np.nan if item["dropout"] else float(item["displacement_mm"])
        for item in records
    ])
    observed = pd.Series(raw).interpolate(limit_direction="both").to_numpy()
    window = 20
    velocities = (observed[window:] - observed[:-window]) / window
    velocity_times = np.arange(window, len(observed))
    crossings = np.flatnonzero(observed >= 20.0)
    event_times = [int(crossings[0])] if len(crossings) else []

    origins = (1000, 1200, 1400, 1600)
    rows: list[dict[str, float | int]] = []
    for origin in origins:
        train = velocities[velocity_times < origin]
        test_mask = velocity_times >= origin
        test_times = velocity_times[test_mask]
        test_velocity = velocities[test_mask]
        if len(train) < 20 or len(test_velocity) == 0:
            continue
        for quantile in (0.90, 0.95, 0.99):
            threshold = float(np.quantile(train, quantile))
            alarm_locations = np.flatnonzero(test_velocity > threshold)
            alarms = [int(test_times[alarm_locations[0]] - origin)] if len(alarm_locations) else []
            future_events = [time - origin for time in event_times if time >= origin]
            metrics = alarm_metrics(
                alarms,
                future_events,
                observation_count=len(test_velocity),
                samples_per_month=30.0,
            )
            rows.append({
                "rolling_origin": origin,
                "training_velocity_quantile": quantile,
                "threshold_mm_per_sample": threshold,
                "test_observations": len(test_velocity),
                "events": metrics["events"],
                "event_recall": metrics["event_recall"],
                "false_alarms_per_30_test_samples": metrics["false_alarms_per_month"],
                "mean_lead_samples": metrics["mean_lead_samples"],
                "alarms": len(alarms),
                "data_type_synthetic": 1,
            })

    table = pd.DataFrame(rows)
    summary = {
        "dataset_type": "DEMO/SYNTHETIC",
        "n_samples": len(records),
        "event_definition": "First generated displacement sample >= 20 mm; this is a simulator threshold, not an observed failure.",
        "velocity_rule": "Mean displacement rate over a trailing 20-sample interval; threshold selected as a training-window quantile.",
        "walk_forward_origins": list(origins),
        "threshold_quantiles": [0.90, 0.95, 0.99],
        "results": rows,
        "interpretation": (
            "Software demonstration of rolling-origin splitting and alarm metrics "
            "on generated sensor signals. Not evidence of lead time, false-alarm "
            "rate, or reliability at a real mine; simulated event timing is defined "
            "by construction."
        ),
    }
    return summary, table


def main() -> None:
    summary, table = evaluate()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    table.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_JSON} and {OUTPUT_CSV}")
    print("DEMO/SYNTHETIC ONLY — no real-mine alarm performance is measured.")


if __name__ == "__main__":
    main()
