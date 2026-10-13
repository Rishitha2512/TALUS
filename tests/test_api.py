import os, sys
import pytest
from fastapi.testclient import TestClient
ROOT_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path: sys.path.insert(0,ROOT_DIR)
from backend.main import app
from backend.physics import calculate_factor_of_safety, categorize_physics_stability

client=TestClient(app)

def test_physics_engine():
    stable=calculate_factor_of_safety(35,25,70,35,24,5)
    assert stable >= 1.5
    assert categorize_physics_stability(stable)[0] == "Low Risk"
    critical=calculate_factor_of_safety(120,65,15,20,26,220,recent_blasting=1,blast_intensity=8.5,equipment_load=50)
    assert critical < 1.0
    assert categorize_physics_stability(critical)[0] == "Critical"

def test_health_and_metadata():
    assert client.get("/").status_code == 200
    h=client.get("/health"); assert h.status_code == 200 and h.json()["status"] == "healthy"
    m=client.get("/model-info"); assert m.status_code == 200 and "selected_model" in m.json()

def test_data_summary_is_dataset_driven():
    r=client.get("/data-summary"); assert r.status_code == 200
    data=r.json(); assert data["records"] >= 1000; assert len(data["sectors"]) >= 8

def test_overview_has_all_processed_sectors():
    r=client.get("/mine/overview"); assert r.status_code == 200
    data=r.json(); assert data["total_sectors"] >= 8
    ids=[x["sector_id"] for x in data["sectors"]]
    assert len(ids)==len(set(ids))

def test_prediction_persists_and_returns_xai():
    payload={
        "sector_id":"S1","slope_height_m":40,"slope_angle_deg":30,"cohesion_kpa":65,"friction_angle_deg":35,
        "unit_weight_kn_m3":24,"rainfall_mm":5,"groundwater_level":5,"pore_pressure_kpa":15,
        "deformation_rate_mm_day":0.5,"displacement_mm":8,"recent_blasting":0,"blast_intensity":0,
        "equipment_load":10,"scenario_name":"Automated test"
    }
    r=client.post("/predict",json=payload); assert r.status_code==200
    data=r.json(); assert "factor_of_safety" in data and len(data["top_contributors"])>0 and data["saved_db_id"]>0

def test_validation():
    payload={"sector_id":"S1","slope_height_m":-10,"slope_angle_deg":120,"cohesion_kpa":35,"friction_angle_deg":30,"unit_weight_kn_m3":24,"rainfall_mm":25,"groundwater_level":10,"pore_pressure_kpa":50,"deformation_rate_mm_day":3.5,"displacement_mm":25}
    assert client.post("/predict",json=payload).status_code==422

def test_assistant_alert_scenario_and_system_endpoints():
    assistant = client.post("/assistant/query", json={"query": "Which sector is most critical?"})
    assert assistant.status_code == 200 and assistant.json().get("intent") == "most_critical"
    scenario = client.get("/scenario/rainfall")
    assert scenario.status_code == 200 and len(scenario.json().get("precautions", [])) >= 3
    health = client.get("/system/health-detail")
    assert health.status_code == 200 and health.json().get("synthetic_data") is True
    created = client.post("/alerts/test", json={"sector": "S9", "level": 2, "reason": "unit test alert", "demo": True})
    assert created.status_code == 200
    created_data = created.json()
    assert created_data["notice"] == "Demo alert only; no external notification is sent."
    alert_id = created_data["alert"]["id"]
    duplicate = client.post("/alerts/test", json={"sector": "S9", "level": 2, "reason": "unit test alert", "demo": True})
    assert duplicate.status_code == 200 and duplicate.json()["created"] is False
    ack = client.post(f"/alerts/{alert_id}/ack", json={"acknowledged_by": "pytest"})
    assert ack.status_code == 200 and ack.json()["alert"]["status"] == "ACKNOWLEDGED"


def test_pof_report_and_ttf_endpoints():
    payload = {
        "sector_id": "S1", "slope_height_m": 60, "slope_angle_deg": 40,
        "cohesion_kpa": 35, "friction_angle_deg": 30, "unit_weight_kn_m3": 24,
        "rainfall_mm": 25, "groundwater_level": 10, "pore_pressure_kpa": 50,
        "deformation_rate_mm_day": 3.5, "displacement_mm": 25,
        "recent_blasting": 0, "blast_intensity": 0, "equipment_load": 10,
        "scenario_name": "test"
    }
    pof = client.post("/simulate/pof", json=payload)
    assert pof.status_code == 200 and 0 <= pof.json()["probability_of_failure"] <= 1
    report = client.get("/report/S1.pdf")
    assert report.status_code == 200 and report.headers["content-type"] == "application/pdf"
    ttf = client.get("/sector/S1/ttf")
    assert ttf.status_code == 200 and "available" in ttf.json()


def test_material_image_classifier_endpoint():
    from pathlib import Path
    image_path = next((Path(ROOT_DIR) / "data" / "raw" / "images").rglob("*.jpg"))
    with open(image_path, "rb") as image_file:
        response = client.post("/image/classify", files={"file": (image_path.name, image_file, "image/jpeg")})
    assert response.status_code == 200
    result = response.json()
    assert result["predicted_material"]
    assert 0 <= result["confidence"] <= 1


def test_workers_geofence_and_advanced_simulation():
    w = client.get("/workers/status")
    assert w.status_code == 200 and len(w.json()["workers"]) >= 5
    payload = {
        "sector_id": "S1", "slope_height_m": 60, "slope_angle_deg": 40,
        "cohesion_kpa": 35, "friction_angle_deg": 30, "unit_weight_kn_m3": 24,
        "rainfall_mm": 25, "groundwater_level": 10, "pore_pressure_kpa": 50,
        "deformation_rate_mm_day": 3.5, "displacement_mm": 25,
        "recent_blasting": 0, "blast_intensity": 0, "equipment_load": 10,
        "scenario_name": "test"
    }
    sens = client.post("/simulate/sensitivity", json=payload)
    assert sens.status_code == 200 and len(sens.json()["sensitivity_parameters"]) > 0
    recs = client.post("/simulate/recommendations", json=payload)
    assert recs.status_code == 200 and len(recs.json()["ranked_recommendations"]) > 0
    drift = client.get("/model/drift")
    assert drift.status_code == 200


def test_yolo_hazard_detection_and_tta():
    # Test TTA profiles and simulation
    prof = client.get("/tta/profiles")
    assert prof.status_code == 200 and "mine_alpha" in prof.json()["profiles"]
    tta = client.post("/simulate/tta?mine_key=mine_alpha")
    assert tta.status_code == 200 and "adapted_tta" in tta.json()
    
    # Test YOLO detection on sample image
    from pathlib import Path
    image_path = next((Path(ROOT_DIR) / "data" / "raw" / "images").rglob("*.jpg"))
    with open(image_path, "rb") as f:
        yolo_res = client.post("/yolo/detect", files={"file": (image_path.name, f, "image/jpeg")})
    assert yolo_res.status_code == 200
    data = yolo_res.json()
    assert "safety_assessment" in data and "annotated_image_base64" in data


def test_whisper_assistant_endpoint():
    dummy_wav = b"RIFF$ \x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00 \x00\x00" + b"\x00" * 200
    r = client.post("/assistant/whisper", files={"file": ("speech.wav", dummy_wav, "audio/wav")})
    assert r.status_code == 200
    res = r.json()
    assert "transcript" in res and "text" in res

