"""Audit how closely synthetic risk labels follow their generating rule.

Run with ``python scripts/audit_synthetic_labels.py``. The audit never treats
synthetic metrics as external validation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "slope_monitoring_synthetic.csv"
DEFAULT_JSON = ROOT / "results" / "synthetic_label_audit.json"
DEFAULT_ROWS = ROOT / "results" / "synthetic_label_audit_rows.csv"

REQUIRED_COLUMNS = {
    "factor_of_safety",
    "deformation_rate_mm_day",
    "pore_pressure_kpa",
    "blast_intensity",
    "recent_blasting",
    "risk_class",
}
CLASS_THRESHOLDS = (0.65, 1.05, 1.55)
CLASS_NAMES = ("Low", "Moderate", "High", "Critical")
NOISE_STD = 0.08


def deterministic_risk_score(frame: pd.DataFrame) -> np.ndarray:
    """Recompute the label score without its generator's random noise term."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    missing.discard("risk_class")
    if missing:
        raise ValueError(f"Missing synthetic label inputs: {sorted(missing)}")

    fos = pd.to_numeric(frame["factor_of_safety"], errors="raise").to_numpy(dtype=float)
    deformation = pd.to_numeric(frame["deformation_rate_mm_day"], errors="raise").to_numpy(dtype=float)
    pore_pressure = pd.to_numeric(frame["pore_pressure_kpa"], errors="raise").to_numpy(dtype=float)
    blast_intensity = pd.to_numeric(frame["blast_intensity"], errors="raise").to_numpy(dtype=float)
    recent_blasting = pd.to_numeric(frame["recent_blasting"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(np.column_stack(
        [fos, deformation, pore_pressure, blast_intensity, recent_blasting]
    )).all():
        raise ValueError("Synthetic label inputs must be finite.")

    return (
        (1.5 / np.maximum(0.4, fos)) * 0.35
        + (deformation / 15.0) * 0.30
        + (pore_pressure / 120.0) * 0.20
        + np.where(recent_blasting != 0, blast_intensity / 10.0, 0.0) * 0.15
    )


def classify_risk_score(scores: np.ndarray) -> np.ndarray:
    """Apply the project's four synthetic class thresholds to numeric scores."""
    return np.digitize(np.asarray(scores, dtype=float), CLASS_THRESHOLDS).astype(int)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        if 0 <= actual < len(CLASS_NAMES) and 0 <= predicted < len(CLASS_NAMES):
            confusion[int(actual), int(predicted)] += 1
    f1_values: list[float] = []
    for class_id in range(len(CLASS_NAMES)):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        denom = 2 * tp + fp + fn
        f1_values.append(2 * tp / denom if denom else 0.0)

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(np.mean(f1_values)),
        "confusion_matrix": confusion.tolist(),
    }


def audit_frame(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Measure agreement between actual labels and the noise-free label rule."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required audit columns: {sorted(missing)}")
    clean = frame.dropna(subset=sorted(REQUIRED_COLUMNS)).copy()
    if clean.empty:
        raise ValueError("No complete records are available for the label audit.")

    scores = deterministic_risk_score(clean)
    predicted = classify_risk_score(scores)
    actual = pd.to_numeric(clean["risk_class"], errors="raise").to_numpy(dtype=int)
    metrics = _classification_metrics(actual, predicted)
    model_feature_columns = [
        "slope_height_m",
        "slope_angle_deg",
        "cohesion_kpa",
        "friction_angle_deg",
        "unit_weight_kn_m3",
        "rainfall_mm",
        "groundwater_level",
        "pore_pressure_kpa",
        "deformation_rate_mm_day",
        "displacement_mm",
        "recent_blasting",
        "blast_intensity",
        "equipment_load",
        "factor_of_safety",
    ]
    score_inputs = [
        "factor_of_safety",
        "deformation_rate_mm_day",
        "pore_pressure_kpa",
        "blast_intensity",
        "recent_blasting",
    ]
    result: dict[str, Any] = {
        "dataset_type": "DEMO/SYNTHETIC",
        "records_in_file": int(len(frame)),
        "records_audited": int(len(clean)),
        "records_excluded_for_missing_inputs": int(len(frame) - len(clean)),
        "noise_free_rule": (
            "0.35*(1.5/max(0.4,FoS)) + 0.30*(deformation_mm_day/15) "
            "+ 0.20*(pore_pressure_kPa/120) "
            "+ 0.15*(blast_intensity/10 if recent_blasting else 0)"
        ),
        "thresholds": list(CLASS_THRESHOLDS),
        "class_names": list(CLASS_NAMES),
        "generator_noise": {
            "distribution": "Normal(0, 0.08)",
            "stored_per_record": False,
            "limitation": "The sampled per-record noise is not persisted, so it cannot be recovered from this CSV.",
        },
        "noise_free_rule_agreement": metrics,
        "score_inputs_also_used_as_model_features": [
            name for name in score_inputs if name in model_feature_columns
        ],
        "direct_input_overlap_count": len(
            [name for name in score_inputs if name in model_feature_columns]
        ),
        "interpretation": (
            "Agreement measures how often thresholding the documented score without "
            "random noise reproduces the stored synthetic class. It is a circularity "
            "diagnostic, not a model validation score and not real-mine performance."
        ),
    }
    audited_rows = clean.copy()
    audited_rows["noise_free_risk_score"] = scores
    audited_rows["noise_free_risk_class"] = predicted
    audited_rows["noise_free_risk_label"] = [CLASS_NAMES[i] for i in predicted]
    audited_rows["noise_free_rule_matches_label"] = predicted == actual
    return result, audited_rows


def write_audit(
    input_path: Path = DEFAULT_INPUT,
    json_path: Path = DEFAULT_JSON,
    rows_path: Path = DEFAULT_ROWS,
) -> dict[str, Any]:
    """Read the source CSV and persist summary and per-record audit artifacts."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Synthetic input CSV not found: {input_path}")
    frame = pd.read_csv(input_path)
    result, audited_rows = audit_frame(frame)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    audited_rows.to_csv(rows_path, index=False)
    return result


if __name__ == "__main__":
    report = write_audit()
    print(json.dumps(report, indent=2))
