"""TALUS production-style FastAPI service."""
from __future__ import annotations
import os, sys, io, time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Path, Query, UploadFile, File, Form
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from backend.schemas import SectorPredictRequest, SectorPredictResponse
from backend.predictor import predict_sector_risk, get_model_info
from backend.database import get_sector_history, get_latest_mine_overview, init_db, get_data_summary
from backend.advanced import (
    init_alerts, list_alerts, create_alert, create_alert_with_status, acknowledge_alert, resolve_alert,
    parse_assistant, monte_carlo_pof, time_to_failure, classify_material,
    get_simulated_workers, tornado_sensitivity, geotechnical_recommendations, model_drift_analysis
)

VERSION = "3.0.0"
@asynccontextmanager
async def lifespan(app):
    init_db()
    init_alerts()
    yield

app = FastAPI(
    title="TALUS API — Physics-Informed Mine Slope Risk Intelligence",
    description="REST API for physics-informed slope risk assessment, telemetry history and model intelligence.",
    version=VERSION,
    lifespan=lifespan,
)

origins = [x.strip() for x in os.getenv("TALUS_CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", summary="API status")
def root_endpoint():
    return {"service": "TALUS API", "version": VERSION, "status": "Operational", "docs": "/docs"}

@app.get("/health", summary="Health check")
def health_check():
    try:
        init_db()
        info = get_model_info()
        return {"status": "healthy", "database": "connected", "model_loaded": bool(info.get("selected_model")), "version": VERSION}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Service not ready: {exc}")

@app.get("/model-info")
def model_info():
    try:
        return get_model_info()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/data-summary")
def data_summary():
    return get_data_summary()

@app.post("/predict", response_model=SectorPredictResponse)
def predict_endpoint(request: SectorPredictRequest):
    try:
        result = predict_sector_risk(request.model_dump())
        # Prototype TARP trigger logic: thresholds are illustrative and must be site-calibrated.
        fos = float(result.get("factor_of_safety", 99))
        probability = float(result.get("ml_risk_probability", 0))
        label = str(result.get("ml_risk_label", "Low"))
        deformation = float(request.deformation_rate_mm_day)
        if fos < 1.0 or (label == "Critical" and probability >= 0.75):
            create_alert(result["sector_id"], 3, f"Prototype trigger: FoS={fos:.2f}, class={label}, probability={probability:.2f}")
        elif fos < 1.3 or probability >= 0.80 or deformation >= 15:
            create_alert(result["sector_id"], 2, f"Prototype trigger: FoS={fos:.2f}, class={label}, deformation={deformation:.1f} mm/day")
        elif fos < 1.5 or probability >= 0.60 or deformation >= 8:
            create_alert(result["sector_id"], 1, f"Prototype advisory: FoS={fos:.2f}, class={label}, deformation={deformation:.1f} mm/day")
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction processing error: {exc}")

@app.get("/sector/{sector_id}/history")
def sector_history_endpoint(
    sector_id: str = Path(..., pattern=r"^S[A-Za-z0-9_-]+$"),
    limit: int = Query(60, ge=1, le=500),
):
    history = get_sector_history(sector_id.upper(), limit)
    return {"sector_id": sector_id.upper(), "records_count": len(history), "history": history}

@app.get("/mine/overview")
def mine_overview_endpoint():
    sectors = get_latest_mine_overview()
    return {"total_sectors": len(sectors), "sectors": sectors}


# ---------------------------------------------------------------------------
# Auxiliary demo services: alerts, voice assistant, Monte Carlo, materials, PDF
# ---------------------------------------------------------------------------
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

class AssistantQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)

class AlertTestRequest(BaseModel):
    sector: str = Field(default="S4", pattern=r"^S[A-Za-z0-9_-]+$")
    level: int = Field(default=3, ge=0, le=3)
    reason: str = Field(default="Demo alert: simulated high-risk scenario", max_length=500)
    demo: bool = True

class AlertAckRequest(BaseModel):
    acknowledged_by: str = Field(default="Engineer", min_length=1, max_length=100)

@app.post("/assistant/query", summary="Rule-based TALUS Voice intent parser")
def assistant_query(request: AssistantQuery):
    try:
        sectors = get_latest_mine_overview()
        alerts = list_alerts(active_only=False)
        return parse_assistant(request.query, sectors, alerts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Assistant query failed: {exc}")

@app.post("/assistant/whisper", summary="OpenAI Whisper speech audio transcriber & intent executor")
async def assistant_whisper(file: UploadFile = File(...)):
    try:
        raw_audio = await file.read()
        from backend.whisper_engine import transcribe_audio_bytes
        transcript = transcribe_audio_bytes(raw_audio, file.filename or "audio.wav")
        sectors = get_latest_mine_overview()
        alerts = list_alerts(active_only=False)
        result = parse_assistant(transcript, sectors, alerts)
        result["transcript"] = transcript
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Whisper transcription failed: {exc}")

@app.post("/demo/critical", summary="Run a clearly labeled severe S4 demo scenario")
def demo_critical():
    # Deliberately severe synthetic inputs for classroom demonstration only.
    payload = {"sector_id":"S4", "slope_height_m":120.0, "slope_angle_deg":65.0,
        "cohesion_kpa":15.0, "friction_angle_deg":20.0, "unit_weight_kn_m3":26.0,
        "rainfall_mm":150.0, "groundwater_level":28.0, "pore_pressure_kpa":220.0,
        "deformation_rate_mm_day":40.0, "displacement_mm":250.0, "recent_blasting":1,
        "blast_intensity":8.5, "equipment_load":50.0,
        "scenario_name":"DEMO ONLY — severe synthetic S4 scenario"}
    try:
        result = predict_sector_risk(payload)
        alert = create_alert("S4", 3, f"DEMO ONLY severe scenario: FoS={result['factor_of_safety']:.2f}; model label={result['ml_risk_label']}", True)
        return {"demo_only": True, "prediction": result, "alert": alert,
                "notice": "Synthetic demo inputs were persisted to S4 history. This is not a real field event."}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Demo scenario failed: {exc}")

@app.get("/alerts/active", summary="List active alerts")
def alerts_active():
    return {"count": len(list_alerts(active_only=True)), "alerts": list_alerts(active_only=True)}

@app.get("/alerts/history", summary="Alert history")
def alerts_history():
    alerts = list_alerts(active_only=False)
    return {"count": len(alerts), "alerts": alerts}

@app.post("/alerts/test", summary="Create a demo/test alert")
def alerts_test(request: AlertTestRequest):
    alert, created = create_alert_with_status(request.sector.upper(), request.level, request.reason, request.demo)
    return {
        "created": created,
        "alert": alert,
        "notice": "Demo alert only; no external notification is sent.",
    }

@app.post("/alerts/{alert_id}/ack", summary="Acknowledge an active alert")
def alerts_ack(alert_id: int, request: AlertAckRequest):
    result = acknowledge_alert(alert_id, request.acknowledged_by)
    if result is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"acknowledged": result.get("status") != "ACTIVE", "alert": result}

@app.post("/alerts/{alert_id}/resolve", summary="Resolve an alert")
def alerts_resolve(alert_id: int):
    result = resolve_alert(alert_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"resolved": True, "alert": result}

@app.post("/simulate/pof", summary="Illustrative Monte Carlo probability of failure")
def simulate_pof(request: SectorPredictRequest):
    try:
        return monte_carlo_pof(request.model_dump(), n=5000)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"PoF simulation failed: {exc}")

@app.post("/physics/response-surfaces", summary="Physics response grids for the analytics UI")
def physics_response_surfaces(request: SectorPredictRequest):
    import numpy as np
    from backend.physics import calculate_factor_of_safety
    p = request.model_dump()
    angles = np.linspace(20, 65, 25).tolist()
    pressures = np.linspace(0, 250, 25).tolist()
    cohesion = np.linspace(5, 180, 25).tolist()
    friction = np.linspace(10, 55, 25).tolist()
    surface_angle_pressure = [[calculate_factor_of_safety(p["slope_height_m"], a, p["cohesion_kpa"], p["friction_angle_deg"], p["unit_weight_kn_m3"], u, p["recent_blasting"], p["blast_intensity"], p["equipment_load"]) for a in angles] for u in pressures]
    surface_cohesion_friction = [[calculate_factor_of_safety(p["slope_height_m"], p["slope_angle_deg"], c, f, p["unit_weight_kn_m3"], p["pore_pressure_kpa"], p["recent_blasting"], p["blast_intensity"], p["equipment_load"]) for c in cohesion] for f in friction]
    return {"angles": angles, "pressures": pressures, "fos_angle_pressure": surface_angle_pressure,
            "cohesion": cohesion, "friction": friction, "fos_cohesion_friction": surface_cohesion_friction,
            "assumption": "Grids are generated from the simplified infinite-slope prototype equation, not a calibrated numerical model."}

@app.get("/sector/{sector_id}/ttf", summary="Indicative inverse-velocity estimate")
def sector_ttf(sector_id: str = Path(..., pattern=r"^S[A-Za-z0-9_-]+$")):
    return time_to_failure(sector_id.upper())

@app.get("/workers/status", summary="Worker geofence tracking and danger zone detection")
def workers_status():
    return get_simulated_workers()

@app.post("/simulate/sensitivity", summary="Tornado FoS parameter sensitivity analysis")
def simulate_sensitivity(request: SectorPredictRequest):
    try:
        return tornado_sensitivity(request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Sensitivity calculation failed: {exc}")

@app.post("/simulate/recommendations", summary="Ranked geotechnical mitigation actions")
def simulate_recommendations(request: SectorPredictRequest):
    try:
        return geotechnical_recommendations(request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Recommendations calculation failed: {exc}")

@app.get("/model/drift", summary="Model and telemetry data drift metrics")
def model_drift():
    return model_drift_analysis()

@app.post("/image/classify", summary="Classify uploaded rock/soil photo")
async def image_classify(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, WEBP, or BMP image.")
    raw = await file.read()
    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 8 MB or smaller.")
    try:
        result = classify_material(raw)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Material classification failed: {exc}")

@app.get("/scenario/{scenario_type}", summary="Scenario guidance and simulation parameters")
def scenario_info(scenario_type: str):
    scenarios = {
        "rainfall": ("Heavy rainfall", "Pore pressure may rise and effective stress may fall.", ["Follow the site trigger-action-response plan", "Restrict access when directed", "Check drainage and piezometer readings", "Inspect displacement trends", "Notify the geotechnical control room"]),
        "blasting": ("Blasting vibration", "Dynamic loading can increase driving demand and worsen existing cracks.", ["Pause work if site rules require it", "Confirm blast exclusion zones", "Inspect after the blast", "Review vibration and displacement records", "Require engineer clearance before re-entry"]),
        "deformation": ("Accelerating deformation", "Increasing displacement rate can indicate a changing slope condition.", ["Verify sensor quality", "Review velocity trends", "Escalate through site procedures", "Keep personnel outside restricted areas", "Seek geotechnical review"]),
        "equipment": ("Crest surcharge", "Heavy loads near the crest can increase driving stresses.", ["Respect engineered stand-off limits", "Relocate loads only under approved instructions", "Check crest cracks", "Inspect deformation sensors", "Document the change"]),
        "groundwater": ("Groundwater rise", "Water pressure can reduce effective normal stress.", ["Inspect drainage systems", "Review piezometer trends", "Check for blocked outlets", "Avoid unapproved pumping changes", "Escalate abnormal readings"]),
        "baseline": ("Stable baseline", "A reference scenario for comparing stable-looking behaviour; it is not a guarantee of safety.", ["Continue routine monitoring", "Verify instrument status", "Inspect after weather or blast events", "Maintain access controls", "Follow approved site procedures"]),
    }
    key = scenario_type.lower().replace(" ", "_")
    if key not in scenarios:
        raise HTTPException(status_code=404, detail=f"Unknown scenario. Choose: {', '.join(scenarios)}")
    title, mechanism, precautions = scenarios[key]
    return {"scenario": key, "title": title, "mechanism": mechanism, "precautions": precautions, "disclaimer": "Educational simulation only. Follow the mine's approved TARP and certified geotechnical instructions."}

@app.get("/report/{sector_id}.pdf", summary="Download a sector PDF report")
def sector_report(sector_id: str = Path(..., pattern=r"^S[A-Za-z0-9_-]+$")):
    sector_id = sector_id.upper()
    sectors = get_latest_mine_overview()
    record = next((s for s in sectors if s.get("sector_id", "").upper() == sector_id), None)
    if not record:
        raise HTTPException(status_code=404, detail="Sector not found")
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        y = height - 55
        pdf.setTitle(f"TALUS Sector {sector_id} Prototype Report")
        pdf.setFont("Helvetica-Bold", 18); pdf.drawString(48, y, f"TALUS | Sector {sector_id} Assessment")
        y -= 24; pdf.setFont("Helvetica", 9); pdf.drawString(48, y, f"Generated: {datetime.now().isoformat(timespec='seconds')}")
        y -= 32; pdf.setFont("Helvetica-Bold", 12); pdf.drawString(48, y, "Current prototype snapshot")
        fields = [("Risk label", record.get("ml_risk_label")), ("Risk probability", f"{float(record.get('ml_risk_probability', 0))*100:.1f}%"), ("Factor of Safety", record.get("factor_of_safety")), ("Slope angle (deg)", record.get("slope_angle_deg")), ("Pore pressure (kPa)", record.get("pore_pressure_kpa")), ("Deformation (mm/day)", record.get("deformation_rate_mm_day")), ("Scenario", record.get("scenario_name"))]
        y -= 22; pdf.setFont("Helvetica", 10)
        for label, value in fields:
            pdf.drawString(55, y, f"{label}: {value}"); y -= 17
        y -= 10; pdf.setFont("Helvetica-Bold", 11); pdf.drawString(48, y, "Recommendation")
        y -= 18; pdf.setFont("Helvetica", 9)
        text = str(record.get("decision_recommendation", "Review with the responsible engineer."))
        for i in range(0, len(text), 95):
            pdf.drawString(55, y, text[i:i+95]); y -= 13
        y -= 20; pdf.setFont("Helvetica-Bold", 10); pdf.drawString(48, y, "Important limitation")
        y -= 16; pdf.setFont("Helvetica", 8)
        disclaimer = "PROTOTYPE DECISION-SUPPORT USING SYNTHETIC DATA. Not field evidence or a certified safety determination. Final decisions require qualified geotechnical engineering review."
        for i in range(0, len(disclaimer), 100):
            pdf.drawString(55, y, disclaimer[i:i+100]); y -= 12
        pdf.showPage(); pdf.save(); buffer.seek(0)
        return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=TALUS_{sector_id}_report.pdf"})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not generate report: {exc}")

@app.get("/system/health-detail", summary="System health detail")
def system_health_detail():
    started = time.perf_counter()
    try:
        info = get_model_info()
        data = get_data_summary()
        db_size = os.path.getsize(os.path.join(ROOT_DIR, "database", "talus.db")) if os.path.exists(os.path.join(ROOT_DIR, "database", "talus.db")) else 0
        return {"status": "healthy", "api_version": VERSION, "model": info.get("selected_model"), "dataset_records": data.get("records", 0), "database_bytes": db_size, "response_ms": round((time.perf_counter()-started)*1000, 2), "synthetic_data": True}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.post("/yolo/detect", summary="YOLOv8 mine hazard and equipment detection")
async def yolo_detect_endpoint(file: UploadFile = File(...)):
    import base64
    from backend.yolo_detector import detect_mine_hazards
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, WEBP, or BMP image.")
    raw = await file.read()
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 12 MB or smaller.")
    try:
        run_yolo = os.getenv("TALUS_YOLO_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
        result = detect_mine_hazards(raw, run_yolo=run_yolo)
        annotated_b64 = base64.b64encode(result.pop("annotated_image_bytes")).decode("utf-8")
        result["annotated_image_base64"] = annotated_b64
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"YOLO detection failed: {exc}")

@app.post("/simulate/tta", summary="Test-Time Adaptation (TTA) for unseen mine environments")
def simulate_tta_endpoint(mine_key: str = Query("mine_alpha")):
    from backend.tta_engine import run_test_time_adaptation
    try:
        return run_test_time_adaptation(mine_key)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"TTA adaptation failed: {exc}")

@app.get("/tta/profiles", summary="Available unseen mine site profiles for adaptation")
def tta_profiles_endpoint():
    from backend.tta_engine import MINE_PROFILES
    return {"profiles": MINE_PROFILES}
