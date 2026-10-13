"""Smoke tests for the Streamlit research, audit, and sensor visualizations."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit
from collections.abc import Iterator

from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from backend.main import app as api_app

ROOT = Path(__file__).resolve().parents[1]
APP_FILE = ROOT / "frontend" / "app.py"


@contextmanager
def _render_module(module_name: str) -> Iterator[AppTest]:
    """Render one dashboard page against an in-process API, never production."""
    with TestClient(api_app) as client:
        def local_get(url: str, **kwargs):
            return client.get(urlsplit(url).path, headers=kwargs.get("headers"))

        def local_post(url: str, json=None, **kwargs):
            return client.post(
                urlsplit(url).path,
                json=json,
                headers=kwargs.get("headers"),
            )

        with (
            patch("requests.get", side_effect=local_get),
            patch("requests.post", side_effect=local_post),
            patch.dict("os.environ", {"TALUS_API_URL": "http://127.0.0.1:8000"}),
        ):
            dashboard = AppTest.from_file(str(APP_FILE), default_timeout=60).run()
            dashboard.radio[0].set_value(module_name).run()
            yield dashboard


def test_audit_module_renders_visual_evidence_without_errors():
    with _render_module("Synthetic vs Real / Leakage Audit") as dashboard:
        assert not dashboard.exception
        assert len(dashboard.get("plotly_chart")) >= 3
        assert any("REAL EXTERNAL VALIDATION: NOT AVAILABLE" in item.value for item in dashboard.warning)


def test_sensor_module_generates_and_charts_a_synthetic_batch():
    with _render_module("Live Sensor Stream") as dashboard:
        assert not dashboard.exception
        button = next(item for item in dashboard.button if "Generate & publish batch" in item.label)
        button.click().run()

        assert not dashboard.exception
        assert len(dashboard.get("plotly_chart")) == 1
        assert any("Published 60 readings" in item.value for item in dashboard.success)


def test_engineering_module_displays_charts_for_each_calculation():
    with _render_module("Engineering Research") as dashboard:
        actions = (
            (2, "Calculate slice FoS"),
            (4, "Compare Monte Carlo and FORM"),
            (5, "Fit equivalent Mohr-Coulomb"),
            (6, "Estimate cumulative infiltration"),
            (7, "Screen planar kinematics"),
        )

        assert not dashboard.exception
        for expected_chart_count, label in actions:
            button = next(item for item in dashboard.button if item.label == label)
            button.click().run()
            assert not dashboard.exception
            assert len(dashboard.get("plotly_chart")) == expected_chart_count
