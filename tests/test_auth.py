"""Password, token, and backend role-enforcement tests."""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from backend.auth import authenticate_user, create_access_token, create_user, decode_access_token
from backend.main import app

client = TestClient(app)


def test_password_hashing_and_signed_token(monkeypatch):
    from backend.auth import hash_password, verify_password

    monkeypatch.setenv("TALUS_AUTH_SECRET", "test-only-signing-secret-at-least-32-chars")
    encoded = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", encoded)
    assert not verify_password("wrong password", encoded)
    token = create_access_token("engineer", "Engineer")
    assert decode_access_token(token) == {"username": "engineer", "role": "Engineer"}


def test_api_rejects_anonymous_and_wrong_role_for_protected_prediction(monkeypatch):
    monkeypatch.setenv("TALUS_AUTH_ENABLED", "true")
    monkeypatch.setenv("TALUS_AUTH_SECRET", "test-only-signing-secret-at-least-32-chars")
    username = f"viewer-{uuid.uuid4().hex[:10]}"
    create_user(username, "correct horse battery staple", "Viewer", "test")
    viewer_token = create_access_token(username, "Viewer")

    assert client.get("/auth/me").status_code == 401
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {viewer_token}"}).json()["role"] == "Viewer"
    payload = {
        "sector_id": "S1", "slope_height_m": 40, "slope_angle_deg": 30,
        "cohesion_kpa": 65, "friction_angle_deg": 35, "unit_weight_kn_m3": 24,
        "rainfall_mm": 5, "groundwater_level": 5, "pore_pressure_kpa": 15,
        "deformation_rate_mm_day": 0.5, "displacement_mm": 8,
        "recent_blasting": 0, "blast_intensity": 0, "equipment_load": 10,
        "scenario_name": "auth test",
    }
    assert client.post("/predict", json=payload).status_code == 401
    assert client.post(
        "/predict", json=payload,
        headers={"Authorization": f"Bearer {viewer_token}"},
    ).status_code == 403


def test_login_rejects_bad_password_and_returns_user_role(monkeypatch):
    monkeypatch.setenv("TALUS_AUTH_ENABLED", "true")
    monkeypatch.setenv("TALUS_AUTH_SECRET", "test-only-signing-secret-at-least-32-chars")
    username = f"engineer-{uuid.uuid4().hex[:10]}"
    create_user(username, "correct horse battery staple", "Engineer", "test")
    assert authenticate_user(username, "not the password") is None
    response = client.post("/auth/login", json={"username": username, "password": "correct horse battery staple"})
    assert response.status_code == 200
    assert response.json()["role"] == "Engineer"
    with pytest.raises(ValueError, match="12 characters"):
        create_user(f"short-{uuid.uuid4().hex[:5]}", "short", "Viewer", "test")
