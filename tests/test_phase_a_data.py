"""Tests for Phase A dataset provenance and synthetic-label audit behavior."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from scripts.audit_synthetic_labels import (
    audit_frame,
    classify_risk_score,
    deterministic_risk_score,
)
from scripts.download_nasa_glc import download_catalog, validate_catalog_csv


def test_synthetic_risk_score_reproduces_documented_formula() -> None:
    frame = pd.DataFrame(
        {
            "factor_of_safety": [1.5, 0.3],
            "deformation_rate_mm_day": [3.0, 15.0],
            "pore_pressure_kpa": [60.0, 120.0],
            "blast_intensity": [8.0, 8.0],
            "recent_blasting": [1, 0],
        }
    )
    scores = deterministic_risk_score(frame)
    assert scores == pytest.approx([0.35 + 0.06 + 0.10 + 0.12, 0.35 * (1.5 / 0.4) + 0.30 + 0.20])


def test_synthetic_class_thresholds_match_generator_boundaries() -> None:
    scores = np.array([0.6499, 0.65, 1.0499, 1.05, 1.5499, 1.55])
    assert classify_risk_score(scores).tolist() == [0, 1, 1, 2, 2, 3]


def test_audit_reports_actual_rule_agreement_and_feature_overlap() -> None:
    frame = pd.DataFrame(
        {
            "factor_of_safety": [2.0, 0.5],
            "deformation_rate_mm_day": [1.0, 20.0],
            "pore_pressure_kpa": [10.0, 180.0],
            "blast_intensity": [0.0, 8.0],
            "recent_blasting": [0, 1],
            "risk_class": [0, 3],
        }
    )
    summary, rows = audit_frame(frame)
    assert summary["dataset_type"] == "DEMO/SYNTHETIC"
    assert summary["records_audited"] == 2
    assert summary["noise_free_rule_agreement"]["accuracy"] == 1.0
    assert summary["direct_input_overlap_count"] == 5
    assert rows["noise_free_risk_class"].tolist() == [0, 3]


def test_synthetic_label_audit_rejects_missing_and_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="Missing required audit columns"):
        audit_frame(pd.DataFrame({"risk_class": [0]}))

    frame = pd.DataFrame(
        {
            "factor_of_safety": [np.nan],
            "deformation_rate_mm_day": [1.0],
            "pore_pressure_kpa": [10.0],
            "blast_intensity": [0.0],
            "recent_blasting": [0],
            "risk_class": [0],
        }
    )
    with pytest.raises(ValueError, match="No complete records"):
        audit_frame(frame)


def test_catalog_csv_validation_requires_inventory_fields_and_data() -> None:
    valid_csv = b"event_date,latitude,longitude\n2020-01-01,1.0,2.0\n"
    assert validate_catalog_csv(valid_csv)[0] == 1

    with pytest.raises(ValueError, match="expected landslide event date"):
        validate_catalog_csv(b"name,value\nx,1\n")
    with pytest.raises(ValueError, match="no data records"):
        validate_catalog_csv(b"event_date,latitude,longitude\n")


def test_catalog_download_validates_before_writing(tmp_path, monkeypatch) -> None:
    from scripts import download_nasa_glc

    payload = b"event_date,latitude,longitude\n2020-01-01,1.0,2.0\n"

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(download_nasa_glc, "urlopen", lambda *_args, **_kwargs: FakeResponse(payload))
    destination = tmp_path / "public" / "catalog.csv"
    assert download_catalog("https://example.invalid/catalog.csv", destination) == 1
    assert destination.read_bytes() == payload
