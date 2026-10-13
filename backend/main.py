"""TALUS production-style FastAPI service."""
from __future__ import annotations
import os, sys, io, time, math, uuid, logging, asyncio
from collections import deque
from datetime import datetime
from typing import Any
import numpy as np
from fastapi import FastAPI, HTTPException, Path, Query, UploadFile, File, Form, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from backend.schemas import SectorPredictRequest, SectorPredictResponse
from backend.predictor import predict_sector_risk, get_model_info
from backend.database import (
    DATABASE_URL,
    DB_PATH,
    get_db_connection,
    get_sector_history,
    get_latest_mine_overview,
    init_db,
    get_data_summary,
)
from backend.advanced import (
    init_alerts, list_alerts, create_alert, create_alert_with_status, acknowledge_alert, resolve_alert,
    parse_assistant, monte_carlo_pof, time_to_failure, classify_material,
    get_simulated_workers, tornado_sensitivity, geotechnical_recommendations, model_drift_analysis
)
from backend.database import save_sensor_readings
from backend.research import simulate_sensor_series
from backend.streaming import telemetry_hub
from backend.research import (
    MonotonicLogisticModel,
    abstention_curve,
    Slice,
    alarm_metrics,
    bishop_simplified,
    cost_sensitive_threshold,
    conformal_prediction_sets,
    fit_physics_residual,
    form_hlrf,
    filter_sensor_series,
    green_ampt_infiltration_mm,
    hoek_brown_equivalent_mc,
    inverse_velocity_forecast,
    janbu_simplified,
    kinematic_planar_feasible,
    monte_carlo_probability_of_failure,
    score_sensor_health,
    sample_correlated_random_field,
)
from backend.auth import (
    authenticate_user,
    create_access_token,
    create_user,
    ensure_admin_user,
    get_current_user,
    record_audit_event,
    require_roles,
)

VERSION = "3.0.0"
logger = logging.getLogger("talus.api")
REQUEST_LATENCIES_MS: deque[float] = deque(maxlen=5000)
REQUEST_COUNT = 0
REQUEST_ERROR_COUNT = 0
@asynccontextmanager
async def lifespan(app):
    init_db()
    init_alerts()
    ensure_admin_user()
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

@app.middleware("http")
async def request_observability(request, call_next):
    global REQUEST_COUNT, REQUEST_ERROR_COUNT
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    started = time.perf_counter()
    try:
        response = await call_next(request)
        if response.status_code >= 500:
            REQUEST_ERROR_COUNT += 1
    except Exception:
        REQUEST_ERROR_COUNT += 1
        logger.exception("request failed request_id=%s path=%s", request_id, request.url.path)
        raise
    finally:
        REQUEST_COUNT += 1
        REQUEST_LATENCIES_MS.append((time.perf_counter() - started) * 1000)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/", summary="API status")
def root_endpoint():
    return {"service": "TALUS API", "version": VERSION, "status": "Operational", "docs": "/docs"}

@app.get("/health", summary="Health check")
def health_check():
    try:
        init_db()
        info = get_model_info()
        return {
            "status": "healthy",
            "database": "connected",
            "model_loaded": bool(info.get("selected_model")),
            "auth_enabled": os.getenv("TALUS_AUTH_ENABLED", "false").strip().lower() in {"1", "true", "yes"},
            "version": VERSION,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Service not ready: {exc}")


@app.get("/metrics", response_class=PlainTextResponse, summary="Prometheus text exposition")
def prometheus_metrics():
    durations = sorted(REQUEST_LATENCIES_MS)
    def quantile(fraction: float) -> float:
        if not durations:
            return 0.0
        return durations[min(len(durations) - 1, int((len(durations) - 1) * fraction))]
    return (
        "# TYPE talus_http_requests_total counter\n"
        f"talus_http_requests_total {REQUEST_COUNT}\n"
        "# TYPE talus_http_5xx_total counter\n"
        f"talus_http_5xx_total {REQUEST_ERROR_COUNT}\n"
        "# TYPE talus_http_request_duration_milliseconds summary\n"
        f"talus_http_request_duration_milliseconds_count {len(durations)}\n"
        f'talus_http_request_duration_milliseconds{{quantile="0.5"}} {quantile(0.5):.3f}\n'
        f'talus_http_request_duration_milliseconds{{quantile="0.95"}} {quantile(0.95):.3f}\n'
    )


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    client_id, queue = await telemetry_hub.connect(websocket)
    try:
        await websocket.send_json({"type": "connected", "dataset_type": "DEMO/SYNTHETIC"})
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20)
                await websocket.send_json(event)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        pass
    finally:
        telemetry_hub.disconnect(client_id)


@app.post("/simulate/telemetry", summary="Generate, persist, and stream synthetic sensor records")
def simulate_telemetry(payload: dict[str, Any]):
    try:
        sector_id = str(payload.get("sector_id", "S1")).upper()
        if not sector_id.startswith("S") or len(sector_id) > 32:
            raise ValueError("sector_id must be a short identifier beginning with S.")
        readings = simulate_sensor_series(
            sample_count=int(payload.get("sample_count", 120)),
            seed=int(payload.get("seed", 42)),
            baseline_mm=float(payload.get("baseline_mm", 5.0)),
            drift_mm_per_sample=float(payload.get("drift_mm_per_sample", 0.03)),
            noise_std_mm=float(payload.get("noise_std_mm", 0.1)),
            dropout_rate=float(payload.get("dropout_rate", 0.03)),
            outlier_rate=float(payload.get("outlier_rate", 0.01)),
        )
        mqtt_published = False
        if os.getenv("TALUS_MQTT_URL"):
            try:
                from backend.mqtt import publish_sensor_events
                mqtt_published = publish_sensor_events([
                    {"sector_id": sector_id, **reading} for reading in readings
                ])
            except (OSError, RuntimeError, ValueError) as exc:
                logger.warning("MQTT publish failed; persisting locally instead: %s", exc)
        persisted = 0 if mqtt_published else save_sensor_readings(sector_id, readings)
        for reading in readings:
            telemetry_hub.publish({
                "type": "telemetry",
                "sector_id": sector_id,
                **reading,
            })
        return {
            "sector_id": sector_id,
            "dataset_type": "DEMO/SYNTHETIC",
            "readings_generated": len(readings),
            "readings_persisted": persisted,
            "mqtt_published": mqtt_published,
            "ingestion_notice": (
                "Sent to the MQTT broker; the ingestion worker persists records asynchronously."
                if mqtt_published else "Saved directly to the configured database."
            ),
            "readings": readings,
            "notice": "Generated demonstration sensor data; not real mine telemetry.",
        }
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=f"Sensor simulation failed: {exc}") from exc

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
def predict_endpoint(
    request: SectorPredictRequest,
    current_user: dict[str, str] = Depends(require_roles("Engineer", "Admin")),
):
    try:
        result = predict_sector_risk(request.model_dump())
        record_audit_event(current_user["username"], "predict", str(result["sector_id"]))
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
from fastapi.responses import StreamingResponse, PlainTextResponse

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
def demo_critical(current_user: dict[str, str] = Depends(require_roles("Engineer", "Admin"))):
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
def alerts_test(
    request: AlertTestRequest,
    current_user: dict[str, str] = Depends(require_roles("Engineer", "Admin")),
):
    alert, created = create_alert_with_status(request.sector.upper(), request.level, request.reason, request.demo)
    if created:
        record_audit_event(current_user["username"], "dispatch_alert", f"{request.sector.upper()}:L{request.level}")
    return {
        "created": created,
        "alert": alert,
        "notice": "Demo alert only; no external notification is sent.",
    }

@app.post("/alerts/{alert_id}/ack", summary="Acknowledge an active alert")
def alerts_ack(
    alert_id: int,
    request: AlertAckRequest,
    current_user: dict[str, str] = Depends(require_roles("Engineer", "Admin")),
):
    result = acknowledge_alert(alert_id, request.acknowledged_by)
    if result is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    record_audit_event(current_user["username"], "acknowledge_alert", str(alert_id))
    return {"acknowledged": result.get("status") != "ACTIVE", "alert": result}

@app.post("/alerts/{alert_id}/resolve", summary="Resolve an alert")
def alerts_resolve(
    alert_id: int,
    current_user: dict[str, str] = Depends(require_roles("Engineer", "Admin")),
):
    result = resolve_alert(alert_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    record_audit_event(current_user["username"], "resolve_alert", str(alert_id))
    return {"resolved": True, "alert": result}


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=256)


class CreateUserRequest(LoginRequest):
    role: str = Field(..., pattern="^(Viewer|Engineer|Admin)$")


@app.post("/auth/login", summary="Exchange username/password for a signed access token")
def auth_login(request: LoginRequest):
    if os.getenv("TALUS_AUTH_ENABLED", "false").strip().lower() not in {"1", "true", "yes"}:
        raise HTTPException(status_code=503, detail="Backend authentication is disabled.")
    user = authenticate_user(request.username, request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return {"access_token": create_access_token(user["username"], user["role"]), "token_type": "bearer", "role": user["role"]}


@app.get("/auth/me", summary="Return the authenticated TALUS role")
def auth_me(current_user: dict[str, str] = Depends(get_current_user)):
    return current_user


@app.post("/auth/users", summary="Create a role-based TALUS user")
def auth_create_user(
    request: CreateUserRequest,
    current_user: dict[str, str] = Depends(require_roles("Admin")),
):
    try:
        create_user(request.username, request.password, request.role, current_user["username"])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
            raise HTTPException(status_code=409, detail="Username already exists.") from exc
        raise
    return {"created": True, "username": request.username, "role": request.role}


@app.get("/auth/users", summary="List TALUS usernames and roles")
def auth_list_users(current_user: dict[str, str] = Depends(require_roles("Admin"))):
    init_db()
    conn = get_db_connection()
    try:
        users = conn.execute(
            "SELECT username,role,created_at FROM talus_users ORDER BY username"
        ).fetchall()
        return {"users": [dict(row) for row in users]}
    finally:
        conn.close()


@app.get("/auth/audit", summary="Read authentication and user-action audit records")
def auth_audit_log(
    limit: int = Query(100, ge=1, le=1000),
    current_user: dict[str, str] = Depends(require_roles("Admin")),
):
    init_db()
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT username,action,resource,created_at FROM audit_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return {"records": [dict(row) for row in rows]}
    finally:
        conn.close()

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

@app.post("/research/limit-equilibrium", summary="Bishop and uncorrected Janbu circular-slice screening")
def research_limit_equilibrium(payload: dict[str, Any]):
    try:
        slices = [Slice(**item) for item in payload["slices"]]
        return {
            "bishop_simplified_fos": bishop_simplified(slices),
            "janbu_simplified_uncorrected_fos": janbu_simplified(slices),
            "assumption": "User-supplied circular slices; Janbu correction factor omitted. Research screening only.",
        }
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid limit-equilibrium input: {exc}") from exc

@app.post("/research/hoek-brown", summary="Equivalent Mohr-Coulomb fit from Hoek-Brown")
def research_hoek_brown(payload: dict[str, Any]):
    try:
        cohesion, friction = hoek_brown_equivalent_mc(
            intact_ucs_mpa=float(payload["intact_ucs_mpa"]),
            mi=float(payload["mi"]),
            gsi=float(payload["gsi"]),
            disturbance_factor=float(payload["disturbance_factor"]),
            sigma3_max_mpa=float(payload["sigma3_max_mpa"]),
        )
        return {
            "equivalent_cohesion_kpa": cohesion,
            "equivalent_friction_angle_deg": friction,
            "assumption": "Least-squares fit across the supplied confining stress range; equivalent parameters are range-dependent.",
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid Hoek-Brown input: {exc}") from exc

@app.post("/research/kinematics/planar", summary="Planar sliding kinematic feasibility screen")
def research_planar_kinematics(payload: dict[str, Any]):
    try:
        feasible = kinematic_planar_feasible(
            slope_dip_deg=float(payload["slope_dip_deg"]),
            slope_dip_direction_deg=float(payload["slope_dip_direction_deg"]),
            joint_dip_deg=float(payload["joint_dip_deg"]),
            joint_dip_direction_deg=float(payload["joint_dip_direction_deg"]),
            friction_angle_deg=float(payload["friction_angle_deg"]),
        )
        return {
            "planar_feasible_screen": feasible,
            "assumption": "Single joint, acute direction window of 20 degrees, no cohesion/water/lateral restraint.",
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid planar kinematics input: {exc}") from exc

@app.post("/research/infiltration", summary="Green-Ampt rainfall infiltration screening")
def research_infiltration(payload: dict[str, Any]):
    try:
        infiltration = green_ampt_infiltration_mm(
            rainfall_mm=float(payload["rainfall_mm"]),
            duration_hours=float(payload["duration_hours"]),
            saturated_conductivity_mm_h=float(payload["saturated_conductivity_mm_h"]),
            suction_head_mm=float(payload["suction_head_mm"]),
            moisture_deficit=float(payload["moisture_deficit"]),
        )
        return {
            "infiltration_mm": infiltration,
            "assumption": "Homogeneous Green-Ampt soil column, explicit 1-minute steps, no runoff routing.",
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid infiltration input: {exc}") from exc

@app.post("/research/reliability", summary="Monte Carlo and one-variable FORM comparison")
def research_reliability(payload: dict[str, Any]):
    try:
        mean = float(payload["factor_of_safety_mean"])
        deviation = float(payload["factor_of_safety_standard_deviation"])
        threshold = float(payload.get("failure_threshold", 1.0))
        count = int(payload.get("sample_count", 10000))
        seed = int(payload.get("seed", 42))
        if deviation <= 0 or not 100 <= count <= 500000:
            raise ValueError("Standard deviation must be positive and sample_count must be 100..500000.")
        pof, samples = monte_carlo_probability_of_failure(
            means=[mean], standard_deviations=[deviation],
            factor_of_safety=lambda values: float(values[0]),
            sample_count=count, seed=seed, failure_threshold=threshold,
        )
        beta, _design_point = form_hlrf(
            means=[mean], standard_deviations=[deviation],
            limit_state=lambda values: float(values[0]) - threshold,
        )
        signed_beta = beta if mean > threshold else -beta
        convergence = []
        for subset in (min(100, count), min(500, count), min(1000, count), count):
            convergence.append({
                "samples": subset,
                "probability_of_failure": float(
                    sum(value < threshold for value in samples[:subset]) / subset
                ),
            })
        return {
            "monte_carlo_probability_of_failure": pof,
            "form_reliability_index_beta": signed_beta,
            "normal_approximation_probability_from_beta": float(0.5 * (1 - math.erf(signed_beta / math.sqrt(2)))),
            "convergence": convergence,
            "seed": seed,
            "assumption": "FoS is modelled as one normally distributed scalar; FORM assumes an independent normal variable. This is not uncertainty propagation through slope mechanics.",
        }
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=f"Reliability analysis failed: {exc}") from exc

@app.post("/research/random-field", summary="Sample a 1D correlated Gaussian property field")
def research_random_field(payload: dict[str, Any]):
    try:
        field = sample_correlated_random_field(
            positions_m=payload["positions_m"],
            mean=float(payload["mean"]),
            standard_deviation=float(payload["standard_deviation"]),
            correlation_length_m=float(payload["correlation_length_m"]),
            sample_count=int(payload.get("sample_count", 1000)),
            seed=int(payload.get("seed", 42)),
        )
        correlation = np.corrcoef(field, rowvar=False)
        return {
            "sample_count": int(field.shape[0]),
            "locations": int(field.shape[1]),
            "mean_by_location": field.mean(axis=0).tolist(),
            "standard_deviation_by_location": field.std(axis=0).tolist(),
            "empirical_correlation": correlation.tolist(),
            "assumption": "Stationary 1D Gaussian property field with exponential covariance; not coupled to the PoF mechanics endpoint.",
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Random-field simulation failed: {exc}") from exc

@app.post("/research/cost-threshold", summary="Empirical binary decision-cost threshold")
def research_cost_threshold(payload: dict[str, Any]):
    try:
        threshold, expected_cost = cost_sensitive_threshold(
            payload["probabilities"], payload["labels"],
            false_negative_cost=float(payload["false_negative_cost"]),
            false_positive_cost=float(payload["false_positive_cost"]),
        )
        return {
            "threshold": threshold,
            "mean_cost_per_observation": expected_cost,
            "assumption": "Costs are user-supplied and empirical on the supplied labels; not site-calibrated.",
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid decision-cost input: {exc}") from exc

@app.post("/research/conformal", summary="Split-conformal prediction sets")
def research_conformal(payload: dict[str, Any]):
    try:
        result = conformal_prediction_sets(
            payload["calibration_probabilities"],
            payload["calibration_labels"],
            payload["test_probabilities"],
            alpha=float(payload.get("alpha", 0.1)),
        )
        labels = payload.get("test_labels")
        if labels is not None:
            sets = result["prediction_sets"]
            if len(labels) != len(sets):
                raise ValueError("test_labels must align with test_probabilities.")
            result["empirical_test_coverage"] = float(
                sum(int(label) in prediction_set for label, prediction_set in zip(labels, sets))
                / max(1, len(labels))
            )
            result["coverage_is_empirical_only"] = True
        return result
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Conformal prediction failed: {exc}") from exc

@app.post("/research/abstention-curve", summary="Confidence-threshold coverage/accuracy curve")
def research_abstention(payload: dict[str, Any]):
    try:
        return {
            "curve": abstention_curve(
                payload["probabilities"], payload["labels"],
                confidence_thresholds=payload.get("confidence_thresholds", [0.5, 0.7, 0.9]),
            ),
            "assumption": "Evaluation uses the supplied labelled records; it is not an independent validation split unless the caller ensures one.",
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Abstention analysis failed: {exc}") from exc

@app.post("/research/monotonic-classifier", summary="Fit binary logistic classifier with projected sign constraints")
def research_monotonic_classifier(payload: dict[str, Any]):
    try:
        model = MonotonicLogisticModel(payload["monotonic_constraints"]).fit(
            payload["features"], payload["labels"]
        )
        probabilities = model.predict_proba(payload.get("prediction_features", payload["features"]))
        return {
            "positive_class_probabilities": probabilities[:, 1].tolist(),
            "coefficients_standardized": model.coef_.tolist(),
            "constraints": list(payload["monotonic_constraints"]),
            "assumption": "Binary logistic regression with coefficient sign projection; feature ordering and direction constraints are supplied by the user.",
            "limitation": "This is a research baseline, not the existing serving model; it has not been trained or validated on real mine labels.",
        }
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=f"Monotonic classifier failed: {exc}") from exc

@app.post("/research/physics-residual", summary="Fit physics FoS residual from supplied observed FoS")
def research_physics_residual(payload: dict[str, Any]):
    try:
        return fit_physics_residual(
            payload["features"], payload["physics_factor_of_safety"],
            payload["observed_factor_of_safety"],
            l2=float(payload.get("l2", 1.0)),
        )
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        raise HTTPException(status_code=422, detail=f"Physics residual fit failed: {exc}") from exc

@app.post("/research/inverse-velocity", summary="Indicative inverse-velocity forecast")
def research_inverse_velocity(payload: dict[str, Any]):
    try:
        forecast, slope = inverse_velocity_forecast(payload["times"], payload["velocities"])
        return {
            "forecast_time": forecast,
            "inverse_velocity_slope": slope,
            "assumption": "Linear extrapolation of inverse velocity. This is not a reliable failure-time prediction without site data and uncertainty analysis.",
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid inverse-velocity series: {exc}") from exc

@app.post("/research/sensor-health", summary="Transparent sensor range and completeness score")
def research_sensor_health(payload: dict[str, Any]):
    try:
        return score_sensor_health(
            payload["values"], expected_min=float(payload["expected_min"]),
            expected_max=float(payload["expected_max"]),
            missing_count=int(payload.get("missing_count", 0)),
            outlier_count=int(payload.get("outlier_count", 0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid sensor-health input: {exc}") from exc

@app.post("/research/sensor-clean", summary="MAD outlier flagging and bounded-gap interpolation")
def research_sensor_clean(payload: dict[str, Any]):
    try:
        return filter_sensor_series(
            payload["values"],
            max_interpolated_gap=int(payload.get("max_interpolated_gap", 3)),
            mad_threshold=float(payload.get("mad_threshold", 3.5)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Sensor cleaning failed: {exc}") from exc

@app.post("/research/cusum", summary="One-sided standardized upward-shift CUSUM")
def research_cusum(payload: dict[str, Any]):
    try:
        from backend.research import cusum_change_points
        return {
            "alarm_indices": cusum_change_points(
                payload["values"],
                drift=float(payload.get("drift", 0.5)),
                threshold=float(payload.get("threshold", 5.0)),
            ),
            "assumption": "Standardized one-sided upward CUSUM. Thresholds require site-specific calibration.",
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"CUSUM calculation failed: {exc}") from exc

@app.post("/research/alarm-metrics", summary="Lead-time and false-alarm screening metrics")
def research_alarm_metrics(payload: dict[str, Any]):
    try:
        return {
            **alarm_metrics(
                payload["alarm_indices"], payload["event_indices"],
                observation_count=int(payload["observation_count"]),
                samples_per_month=float(payload.get("samples_per_month", 30)),
            ),
            "assumption": "One-to-one greedy prior-alarm matching. Not walk-forward or site-validated.",
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid alarm-evaluation input: {exc}") from exc

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


@app.get("/shift-report.pdf", summary="Generate synthetic/demo shift handover report")
def shift_report(current_user: dict[str, str] = Depends(require_roles("Viewer", "Engineer", "Admin"))):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        sectors = get_latest_mine_overview()
        alerts = list_alerts(active_only=False)
        active_alerts = [alert for alert in alerts if alert.get("status") in {"ACTIVE", "ACKNOWLEDGED"}]
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        page_width, page_height = letter
        y = page_height - 48
        pdf.setTitle("TALUS Synthetic Shift Handover")
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(42, y, "TALUS | Shift Handover (DEMO/SYNTHETIC)")
        y -= 22
        pdf.setFont("Helvetica", 9)
        pdf.drawString(42, y, f"Generated: {datetime.now().isoformat(timespec='seconds')} | Report user: {current_user['username']}")
        y -= 24

        def section(title: str) -> None:
            nonlocal y
            if y < 80:
                pdf.showPage()
                y = page_height - 48
            pdf.setFont("Helvetica-Bold", 11)
            pdf.drawString(42, y, title)
            y -= 17
            pdf.setFont("Helvetica", 8)

        section("Top current sector risks")
        for item in sectors[:10]:
            line = (
                f"{item.get('sector_id')} | {item.get('ml_risk_label')} | "
                f"FoS={item.get('factor_of_safety')} | "
                f"pore pressure={item.get('pore_pressure_kpa')} kPa | "
                f"deformation={item.get('deformation_rate_mm_day')} mm/day"
            )
            pdf.drawString(48, y, line[:120])
            y -= 12

        section("Open / acknowledged alerts")
        if not active_alerts:
            pdf.drawString(48, y, "No active or acknowledged alerts in the local alert ledger.")
            y -= 12
        for alert in active_alerts[:30]:
            line = (
                f"#{alert.get('id')} {alert.get('sector')} L{alert.get('level')} "
                f"{alert.get('status')} | ack by={alert.get('acknowledged_by') or 'not acknowledged'} "
                f"| {alert.get('reason', '')}"
            )
            pdf.drawString(48, y, line[:130])
            y -= 12
            if y < 60:
                pdf.showPage()
                y = page_height - 48

        section("Data and safety limitation")
        disclaimer = (
            "All telemetry in this prototype is synthetic/demo data. This report is "
            "not a real shift log, site instruction, or certified geotechnical safety "
            "determination. Confirm sensor status and follow the mine's approved TARP "
            "and responsible engineer's instructions."
        )
        for offset in range(0, len(disclaimer), 110):
            pdf.drawString(48, y, disclaimer[offset:offset + 110])
            y -= 12
        pdf.showPage()
        pdf.save()
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=TALUS_demo_shift_handover.pdf"},
        )
    except Exception as exc:
        logger.exception("shift report generation failed")
        raise HTTPException(status_code=500, detail=f"Could not generate shift report: {exc}") from exc

@app.get("/system/health-detail", summary="System health detail")
def system_health_detail():
    started = time.perf_counter()
    try:
        info = get_model_info()
        data = get_data_summary()
        db_size = (
            os.path.getsize(DB_PATH)
            if not (DATABASE_URL and DATABASE_URL.startswith(("postgres://", "postgresql://")))
            and os.path.exists(DB_PATH)
            else None
        )
        return {
            "status": "healthy",
            "api_version": VERSION,
            "model": info.get("selected_model"),
            "dataset_records": data.get("records", 0),
            "database_bytes": db_size,
            "database_backend": (
                "postgresql"
                if DATABASE_URL and DATABASE_URL.startswith(("postgres://", "postgresql://"))
                else "sqlite"
            ),
            "response_ms": round((time.perf_counter()-started)*1000, 2),
            "synthetic_data": True,
        }
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
