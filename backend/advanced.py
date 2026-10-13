"""Offline-first auxiliary services for TALUS demo features.

All outputs are prototype decision-support artifacts based on synthetic data.
"""
from __future__ import annotations
import io, json, math, os, sqlite3, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("TALUS_DB_PATH", ROOT / "database" / "talus.db"))
IMAGE_FEATURES = ROOT / "data" / "processed" / "image_features.csv"
IMAGE_ROOT = ROOT / "data" / "raw" / "images"
MATERIAL_MODEL = ROOT / "models" / "material_classifier.joblib"
MATERIAL_FEATURES = ["mean_r", "mean_g", "mean_b", "brightness", "contrast", "saturation", "hue", "entropy", "edge_density", "sharpness", "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation"]


def connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def init_alerts():
    with connection() as c:
        c.execute('''CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, sector TEXT NOT NULL, level INTEGER NOT NULL,
            reason TEXT NOT NULL, created_at TEXT NOT NULL, acknowledged_by TEXT,
            acknowledged_at TEXT, resolved_at TEXT, status TEXT NOT NULL DEFAULT 'ACTIVE',
            demo INTEGER NOT NULL DEFAULT 0)''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status, level)')


def list_alerts(active_only: bool = False):
    init_alerts()
    with connection() as c:
        sql = "SELECT * FROM alerts" + (" WHERE status='ACTIVE'" if active_only else "") + " ORDER BY id DESC LIMIT 300"
        return [dict(r) for r in c.execute(sql).fetchall()]


def create_alert_with_status(sector: str, level: int, reason: str, demo: bool = False):
    init_alerts()
    now = datetime.now().isoformat(timespec="seconds")
    # Hysteresis/debounce: do not create a duplicate active alert for same sector+level in 2 min.
    with connection() as c:
        existing = c.execute("SELECT * FROM alerts WHERE sector=? AND level=? AND status='ACTIVE' ORDER BY id DESC LIMIT 1", (sector, level)).fetchone()
        if existing:
            created = datetime.fromisoformat(existing["created_at"])
            if datetime.now() - created < timedelta(minutes=2):
                return dict(existing), False
        cur = c.execute("INSERT INTO alerts(sector,level,reason,created_at,demo) VALUES(?,?,?,?,?)", (sector, level, reason, now, int(demo)))
        alert = dict(c.execute("SELECT * FROM alerts WHERE id=?", (cur.lastrowid,)).fetchone())
        return alert, True


def create_alert(sector: str, level: int, reason: str, demo: bool = False):
    alert, _ = create_alert_with_status(sector, level, reason, demo)
    return alert


def acknowledge_alert(alert_id: int, who: str = "Engineer"):
    init_alerts()
    now = datetime.now().isoformat(timespec="seconds")
    with connection() as c:
        cur = c.execute("UPDATE alerts SET acknowledged_by=?, acknowledged_at=?, status='ACKNOWLEDGED' WHERE id=? AND status='ACTIVE'", (who, now, alert_id))
        row = c.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
        return dict(row) if row else None


def resolve_alert(alert_id: int):
    init_alerts()
    with connection() as c:
        c.execute("UPDATE alerts SET resolved_at=?, status='RESOLVED' WHERE id=?", (datetime.now().isoformat(timespec="seconds"), alert_id))
        row = c.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
        return dict(row) if row else None


def parse_assistant(query: str, sectors: list[dict[str, Any]], alerts: list[dict[str, Any]]):
    q = (query or "").lower().strip()
    if not q or q in ["", "none", "null"]:
        return {
            "intent": "idle",
            "text": "TALUS Voice Copilot online. Ready for your questions or module navigation commands.",
            "speak": "TALUS online.",
            "action": {"type": "idle"}
        }
        
    # Check for Tamil or Hindi greetings
    is_tamil = any(w in q for w in ["வணக்கம்", "vanakkam", "epdi irukku", "solla"])
    is_hindi = any(w in q for w in ["नमस्ते", "namaste", "kaisa", "kya haal", "batao"])
    
    # 1. Greetings & Wake Phrase
    if any(w in q for w in ["hey talus", "hello", "hi talus", "vanakkam", "namaste", "வணக்கம்", "नमस्ते"]):
        if is_tamil:
            text = "வணக்கம்! நான் TALUS சுரங்க பாதுகாப்பு உதவியாளர். எந்த பிரிவு ஆபத்தானது என்று கேட்கலாம், அல்லது 'அவசர எச்சரிக்கைகள் என்ன?' என்று கேளுங்கள்."
        elif is_hindi:
            text = "नमस्ते! मैं TALUS खदान सुरक्षा सहायक हूँ। आप मुझसे किसी भी सेक्टर की स्थिति, सबसे खतरनाक सेक्टर या सक्रिय चेतावनियों के बारे में पूछ सकते हैं।"
        else:
            text = "Hello! I am TALUS, your geotechnical AI mine safety copilot. You can ask me about sector risks, active alerts, worker safety procedures, or what-if hazard scenarios."
        intent = "greeting"
    elif any(w in q for w in ["help", "what can", "commands", "menu"]):
        text = "You can ask: 'Which sector is most critical?', 'What is the status of Sector 4?', 'Why is S4 risky?', 'What should workers do?', 'Read active alerts', 'What if rainfall increases to 80 mm?', or viva questions like 'What is factor of safety?'"
        intent = "help"
    # 2. Active alerts
    elif "alert" in q or "warning" in q:
        active = [a for a in alerts if a.get("status") == "ACTIVE"]
        if active:
            max_lvl = max(a['level'] for a in active)
            affected = ", ".join(sorted(set(a['sector'] for a in active)))
            text = f"There are {len(active)} active TARP alerts in the control room. The highest severity is Level {max_lvl} affecting sectors: {affected}. Please initiate prescribed TARP response protocols."
        else:
            text = "All sectors are currently nominal with zero active alerts in the control room."
        intent = "active_alerts"
    # 3. Worker Safety & Actions
    elif any(w in q for w in ["worker", "precaution", "what should", "safety action", "evacuate", "ppe"]):
        text = "Under mining TARP rules: 1. Restrict all personnel entry to Level 2 and Level 3 sectors. 2. Halt crest loading and secondary blasting immediately. 3. Verify real-time slope radar and piezometer readings. 4. Require mandatory Level 2 PPE (high-visibility vest, hard hat, two-way UHF radio, steel-toe boots). 5. All operational clearances require geotechnical engineer sign-off."
        intent = "worker_precautions"
    # 4. Most Critical Sector
    elif any(w in q for w in ["most critical", "highest risk", "worst sector", "critical sector", "danger sector"]):
        ranked = sorted(sectors, key=lambda s: (s.get("ml_risk_label") != "Critical", -float(s.get("ml_risk_probability", 0))))
        s = ranked[0] if ranked else {}
        text = f"Sector {s.get('sector_id', 'unknown')} is the most critical monitoring zone. It is evaluated as {s.get('ml_risk_label', 'unknown')} risk with a physics Factor of Safety of {float(s.get('factor_of_safety', 0)):.2f} and {float(s.get('ml_risk_probability', 0))*100:.1f}% model probability."
        intent = "most_critical"
    # 5. Overall Mine Status
    elif any(w in q for w in ["overall", "mine status", "status of mine", "summary", "how is the mine"]):
        high = sum(s.get("ml_risk_label") in ("High", "Critical") for s in sectors)
        critical = sum(s.get("ml_risk_label") == "Critical" for s in sectors)
        fos_vals = [float(s.get("factor_of_safety", 1.5)) for s in sectors]
        mean_fos = np.mean(fos_vals) if fos_vals else 1.5
        text = f"The mine control room is tracking {len(sectors)} telemetry sectors. Mean pit Factor of Safety is {mean_fos:.2f}. There are {high} sectors under elevated watch, with {critical} in Critical condition."
        intent = "overall_status"
    # 6. What-if Questions
    elif any(w in q for w in ["what if", "what happens if", "if rainfall", "if pore pressure"]):
        import re
        r_match = re.search(r"rainfall.*?(\d+)", q)
        p_match = re.search(r"pore.*?pressure.*?(\d+)", q)
        if r_match:
            r_val = float(r_match.group(1))
            text = f"If rainfall surges to {r_val:.0f} mm, infiltrated rainwater will elevate piezometric heads, increasing joint pore pressures and reducing the effective normal stress by up to 35 percent. Factor of Safety is expected to drop significantly."
        elif p_match:
            p_val = float(p_match.group(1))
            text = f"If pore pressure reaches {p_val:.0f} kPa, effective shear strength is severely degraded via Mohr-Coulomb reduction, potentially driving Factor of Safety below the critical 1.0 threshold."
        else:
            text = "In what-if scenarios: Elevated rainfall and pore pressure reduce effective stress, while blast vibrations add pseudostatic shear driving stress. Both accelerate slope failure risk."
        intent = "what_if_scenario"
    # 7. Geotechnical Viva Questions
    elif any(w in q for w in ["what is factor of safety", "define fos", "factor of safety formula"]):
        text = "Factor of Safety (FoS) is the ratio of available resisting shear strength (from cohesion and effective friction) to mobilized shear driving stress. FoS below 1.0 represents physical failure; FoS between 1.0 and 1.3 is marginally stable; FoS above 1.5 is standard operating design."
        intent = "viva_fos"
    elif any(w in q for w in ["why infinite slope", "infinite slope model"]):
        text = "The extended infinite slope formulation provides an analytical, closed-form limit equilibrium solution that accounts for groundwater pore pressure, blast vibration pseudostatic force, and equipment crest load, making it ideal for millisecond real-time edge screening."
        intent = "viva_physics"
    elif any(w in q for w in ["why accuracy", "why not accuracy", "imbalance"]):
        text = "Accuracy is misleading because the dataset has severe class imbalance: Critical has 554 records and High has 405, while Low has only 30. A naive model predicting only majority classes achieves 80 percent accuracy with zero recall on safe slopes. We rely on Stratified 5-Fold Macro-F1."
        intent = "viva_ml_imbalance"
    # 7b. Additional Viva / Geotechnical / AI Concepts
    elif "leakage" in q:
        text = "Factor of Safety is not target leakage; it is an established domain physics prior (Physics-Informed ML). In our ablation study, removing FoS still achieves strong Macro-F1 (0.802) using only raw sensor features, proving telemetry holds independent predictive power."
        intent = "viva_leakage"
    elif "tta" in q or "adaptation" in q:
        text = "Test-Time Adaptation (TTA) dynamically normalizes incoming sensor features from novel, unseen mine sites to align with the training distribution, using entropy minimization to maintain high accuracy without retraining."
        intent = "viva_tta"
    elif "yolo" in q or "computer vision" in q:
        text = "TALUS integrates YOLOv8 Nano to detect open-pit surface hazards from bench cameras, identifying workers in danger zones, haul equipment surcharge, and structural tension cracks."
        intent = "viva_yolo"
    elif "brier" in q:
        text = "The Multiclass Brier Score measures probability calibration: the mean squared difference between predicted class probabilities and one-hot true labels. Lower Brier score means calibrated confidence."
        intent = "viva_brier"
    elif "fukuzono" in q or "time to failure" in q or "ttf" in q:
        text = "The Fukuzono inverse-velocity method plots 1/v (inverse deformation rate) over time during tertiary creep. As velocity accelerates toward infinity, 1/v linearly approaches zero, extrapolating the projected time to collapse."
        intent = "viva_ttf"
    elif "monte carlo" in q or "probability of failure" in q:
        text = "Monte Carlo simulation draws 5,000 stochastic realizations of cohesion, friction, and pore water pressure to calculate the Probability of Failure Pf = P(FoS < 1.0) and Cornell Reliability Index beta."
        intent = "viva_monte_carlo"
    elif any(w in q for w in ["rajan", "kumar", "priya", "suresh"]) or any(w in q for w in ["who is in danger", "who is at risk", "worker", "personnel", "miner"]):
        w_res = get_simulated_workers()
        d_workers = w_res.get("danger_workers", [])
        if "rajan" in q:
            text = "ALERT: Worker Rajan (Excavator Operator) is stationed in Sector S4, operating CAT 390F. Sector S4 is inside an active predicted failure zone. Evacuation is strongly advised."
        elif "kumar" in q:
            text = "ALERT: Surveyor Kumar is in Sector S4 conducting geotechnical surveying. He is inside the exclusion zone and must retreat north."
        elif d_workers:
            names = ", ".join(f"{w['name']} ({w['role']} in {w['sector']})" for w in d_workers)
            text = f"ATTENTION: {len(d_workers)} workers are currently inside the critical failure exclusion zone: {names}. Immediate radio evacuation is advised."
        else:
            text = "All active miners are currently located in safe operating zones outside exclusion perimeters."
        intent = "worker_safety_query"
    # 8. Navigation Intents & Module Names
    elif any(w in q for w in ["go to", "open", "switch to", "show", "navigate", "take me to", "display", "change to"]) or any(m.lower() in q for m in ["command center", "sector analysis", "hazard simulator", "worker safety", "ai vision", "adaptive ml", "emergency"]):
        nav_target = None
        if any(w in q for w in ["command", "overview", "summary", "center"]):
            nav_target = "Command Center"
            text = "Navigating to Command Center."
        elif any(w in q for w in ["sector", "analysis", "stress", "wedge"]):
            nav_target = "Sector Analysis"
            text = "Navigating to Sector Analysis."
        elif any(w in q for w in ["hazard", "simulator", "scenario", "video", "simulation", "rain", "blast"]):
            nav_target = "3D Hazard Simulator"
            text = "Navigating to 3D Hazard Simulator."
        elif any(w in q for w in ["worker", "geofence", "personnel", "safety zone", "risk zone", "miners"]):
            nav_target = "Worker Safety & Risk Zones"
            text = "Navigating to Worker Safety & Risk Zones."
        elif any(w in q for w in ["vision", "yolo", "camera", "photo", "rock", "texture"]):
            nav_target = "AI Vision & YOLO"
            text = "Navigating to AI Vision & YOLO."
        elif any(w in q for w in ["adaptive", "tta", "transfer", "adaptation", "unseen mine"]):
            nav_target = "Adaptive ML & TTA"
            text = "Navigating to Adaptive ML & TTA."
        elif any(w in q for w in ["emergency", "tarp", "siren", "alarm", "evacuate"]):
            nav_target = "Emergency & TARP"
            text = "Navigating to Emergency & TARP."
            
        if nav_target:
            return {"intent": "navigation", "text": text, "speak": text, "navigation_target": nav_target, "action": {"type": "navigate", "target": nav_target}}
        else:
            text = "Available modules: Command Center, Sector Analysis, 3D Hazard Simulator, Worker Safety & Risk Zones, AI Vision & YOLO, Adaptive ML & TTA, Emergency & TARP."
            intent = "help_navigation"
    # 9. Sleep command
    elif any(w in q for w in ["go to sleep", "sleep", "shut down", "stop", "disconnect", "silence"]):
        text = "TALUS put to sleep. Audio silenced."
        intent = "sleep"
    # 10. Siren command
    elif any(w in q for w in ["sound alarm", "siren", "trigger alarm", "emergency alarm", "sound siren"]):
        text = "Triggering emergency Level 3 evacuation siren and haptic alert."
        intent = "siren"
    # 11. Specific Sector Queries (S1 to S10)
    else:
        import re
        m = re.search(r"\b(s\s*\d+)\b", q, re.I)
        sid = m.group(1).replace(" ", "").upper() if m else None
        if sid and any(s.get("sector_id", "").upper() == sid for s in sectors):
            s = next(s for s in sectors if s.get("sector_id", "").upper() == sid)
            if any(w in q for w in ["why", "cause", "risky", "risk", "reason"]):
                contributors = s.get("top_contributors", []) or []
                top = ", ".join(str(c.get("feature")) for c in contributors[:3]) or "pore pressure, slope angle, and deformation velocity"
                text = f"Sector {sid} is flagged as {s.get('ml_risk_label','unknown')} risk primarily due to: {top}. Current Factor of Safety is {float(s.get('factor_of_safety',0)):.2f} with deformation rate of {float(s.get('deformation_rate_mm_day',0)):.1f} mm/day."
                intent = "sector_explanation"
            else:
                text = f"Sector {sid} status: {s.get('ml_risk_label','unknown')} risk, FoS {float(s.get('factor_of_safety',0)):.2f}, deformation {float(s.get('deformation_rate_mm_day',0)):.1f} mm/day, pore pressure {float(s.get('pore_pressure_kpa',0)):.1f} kPa."
                intent = "sector_status"
        else:
            text = "TALUS AI online. You can ask: 'Which sector is most critical?', 'What is the status of Sector 4?', 'Are workers safe in S4?', 'Why is S4 risky?', or command: 'Go to 3D Hazard Simulator', 'Go to Worker Safety', 'Go to sleep'."
            intent = "fallback"
            
    return {"intent": intent, "text": text, "speak": text, "action": {"type": intent}}


def get_simulated_workers() -> dict[str, Any]:
    """Returns simulated active personnel and vehicles with 3D coordinates and prediction risk zones."""
    workers = [
        {"id": "W-01", "name": "Rajan", "role": "Excavator Operator", "sector": "S4", "vehicle": "CAT 390F Excavator",
         "lat": 12.824, "lon": 80.045, "x_m": 185.0, "y_m": -88.0, "z_m": 58.0, "heart_rate_bpm": 86, "radio_channel": "CH-4", "distance_to_safe_muster_m": 295.0},
        {"id": "W-02", "name": "Priya", "role": "Haul Truck Driver", "sector": "S2", "vehicle": "Komatsu 930E (HT-104)",
         "lat": 12.828, "lon": 80.049, "x_m": -135.0, "y_m": 165.0, "z_m": 124.0, "heart_rate_bpm": 74, "radio_channel": "CH-2", "distance_to_safe_muster_m": 85.0},
        {"id": "W-03", "name": "Kumar", "role": "Geotechnical Surveyor", "sector": "S4", "vehicle": "Field Utility 4WD",
         "lat": 12.825, "lon": 80.046, "x_m": 198.0, "y_m": -72.0, "z_m": 72.0, "heart_rate_bpm": 92, "radio_channel": "CH-4", "distance_to_safe_muster_m": 280.0},
        {"id": "W-04", "name": "Suresh", "role": "Blast Technician", "sector": "S7", "vehicle": "Explosives Transport Unit",
         "lat": 12.821, "lon": 80.041, "x_m": -155.0, "y_m": -118.0, "z_m": 88.0, "heart_rate_bpm": 78, "radio_channel": "CH-1", "distance_to_safe_muster_m": 140.0},
        {"id": "W-05", "name": "Rajesh", "role": "Pit Safety Foreman", "sector": "S1", "vehicle": "Patrol Buggy",
         "lat": 12.829, "lon": 80.043, "x_m": 45.0, "y_m": 215.0, "z_m": 155.0, "heart_rate_bpm": 71, "radio_channel": "CH-1", "distance_to_safe_muster_m": 35.0},
        {"id": "W-06", "name": "Auto-HT-09", "role": "Autonomous Hauler", "sector": "S9", "vehicle": "Haul Truck HT-09",
         "lat": 12.826, "lon": 80.052, "x_m": -82.0, "y_m": -175.0, "z_m": 48.0, "heart_rate_bpm": 0, "radio_channel": "TELEMETRY", "distance_to_safe_muster_m": 220.0},
    ]
    active_alerts = list_alerts(active_only=True)
    alert_by_sector = {a["sector"].upper(): a for a in active_alerts}
    danger_zone_workers = []
    
    for w in workers:
        s = w["sector"].upper()
        # Mark workers in S4 or sectors with active alert as danger zone
        is_danger = (s in alert_by_sector and alert_by_sector[s].get("level", 0) >= 2) or (s == "S4")
        if is_danger:
            w["in_danger_zone"] = True
            w["active_alert_level"] = alert_by_sector[s]["level"] if s in alert_by_sector else 3
            w["alert_reason"] = alert_by_sector[s].get("reason", "Critical Limit-Equilibrium Failure Zone") if s in alert_by_sector else "Predicted S4 Slope Failure Wedge"
            w["safety_zone_status"] = "🚨 CRITICAL EXCLUSION ZONE"
            danger_zone_workers.append(w)
        elif s == "S3" or s == "S5":
            w["in_danger_zone"] = False
            w["active_alert_level"] = 1
            w["alert_reason"] = "Advisory 50m Crest Buffer"
            w["safety_zone_status"] = "⚠️ ADVISORY BUFFER ZONE"
        else:
            w["in_danger_zone"] = False
            w["active_alert_level"] = 0
            w["alert_reason"] = "Nominal operating sector"
            w["safety_zone_status"] = "🟢 SAFE HAULAGE ZONE"
            
    prediction_risk_zones = {
        "exclusion_zone": {
            "name": "Red Failure Wedge (Sector S4)",
            "center": [185.0, -88.0, 65.0],
            "radius_m": 95.0,
            "status": "IMMEDIATE EVACUATION REQUIRED",
            "active_personnel": [w["name"] for w in danger_zone_workers]
        },
        "advisory_zone": {
            "name": "Yellow 50m Crest Buffer Zone",
            "center": [150.0, -60.0, 85.0],
            "radius_m": 150.0,
            "status": "RESTRICTED ENTRY • PPE MANDATORY"
        },
        "safe_muster_point": {
            "name": "Green Muster Assembly Area North",
            "center": [45.0, 215.0, 155.0],
            "status": "SECURE ASSEMBLY GROUND"
        }
    }
    
    return {
        "workers": workers,
        "danger_count": len(danger_zone_workers),
        "danger_workers": danger_zone_workers,
        "risk_zones": prediction_risk_zones
    }


def tornado_sensitivity(payload: dict[str, Any]) -> dict[str, Any]:
    """Calculates +-20% parameter swing impact on Factor of Safety for Tornado chart."""
    from backend.physics import calculate_factor_of_safety
    base_fos = calculate_factor_of_safety(
        payload["slope_height_m"], payload["slope_angle_deg"], payload["cohesion_kpa"],
        payload["friction_angle_deg"], payload["unit_weight_kn_m3"], payload["pore_pressure_kpa"],
        payload.get("recent_blasting", 0), payload.get("blast_intensity", 0), payload.get("equipment_load", 0)
    )
    parameters = [
        ("Slope Angle (deg)", "slope_angle_deg", 0.20),
        ("Pore Pressure (kPa)", "pore_pressure_kpa", 0.20),
        ("Cohesion (kPa)", "cohesion_kpa", 0.20),
        ("Friction Angle (deg)", "friction_angle_deg", 0.20),
        ("Slope Height (m)", "slope_height_m", 0.20),
        ("Equipment Load (kPa)", "equipment_load", 0.20),
    ]
    results = []
    for label, key, delta_pct in parameters:
        val = float(payload.get(key, 10.0))
        low_val = max(1.0, val * (1.0 - delta_pct))
        high_val = val * (1.0 + delta_pct)
        p_low = dict(payload); p_low[key] = low_val
        p_high = dict(payload); p_high[key] = high_val
        fos_low = calculate_factor_of_safety(
            p_low["slope_height_m"], p_low["slope_angle_deg"], p_low["cohesion_kpa"],
            p_low["friction_angle_deg"], p_low["unit_weight_kn_m3"], p_low["pore_pressure_kpa"],
            p_low.get("recent_blasting", 0), p_low.get("blast_intensity", 0), p_low.get("equipment_load", 0)
        )
        fos_high = calculate_factor_of_safety(
            p_high["slope_height_m"], p_high["slope_angle_deg"], p_high["cohesion_kpa"],
            p_high["friction_angle_deg"], p_high["unit_weight_kn_m3"], p_high["pore_pressure_kpa"],
            p_high.get("recent_blasting", 0), p_high.get("blast_intensity", 0), p_high.get("equipment_load", 0)
        )
        swing = abs(fos_high - fos_low)
        results.append({
            "parameter": label, "key": key, "base_value": val,
            "low_value": low_val, "high_value": high_val,
            "fos_at_minus_20pct": fos_low, "fos_at_plus_20pct": fos_high,
            "swing": round(swing, 3), "base_fos": base_fos
        })
    results.sort(key=lambda x: x["swing"], reverse=True)
    return {"base_fos": base_fos, "sensitivity_parameters": results}


def geotechnical_recommendations(payload: dict[str, Any]) -> dict[str, Any]:
    """Generates ranked geotechnical remediation actions with estimated FoS gains."""
    from backend.physics import calculate_factor_of_safety
    base_fos = calculate_factor_of_safety(
        payload["slope_height_m"], payload["slope_angle_deg"], payload["cohesion_kpa"],
        payload["friction_angle_deg"], payload["unit_weight_kn_m3"], payload["pore_pressure_kpa"],
        payload.get("recent_blasting", 0), payload.get("blast_intensity", 0), payload.get("equipment_load", 0)
    )
    actions = []
    # 1. Drainage: reduce pore pressure by 30 kPa
    if payload.get("pore_pressure_kpa", 0) > 10:
        p_drain = dict(payload); p_drain["pore_pressure_kpa"] = max(0.0, payload["pore_pressure_kpa"] - 30.0)
        f_drain = calculate_factor_of_safety(p_drain["slope_height_m"], p_drain["slope_angle_deg"], p_drain["cohesion_kpa"], p_drain["friction_angle_deg"], p_drain["unit_weight_kn_m3"], p_drain["pore_pressure_kpa"], p_drain.get("recent_blasting", 0), p_drain.get("blast_intensity", 0), p_drain.get("equipment_load", 0))
        actions.append({"priority": 1, "action": "Sub-bench Horizontal Dewatering Drains", "mechanism": "Relieves phreatic pore water pressure (-30 kPa)", "delta_fos": round(f_drain - base_fos, 3), "projected_fos": round(f_drain, 3), "feasibility": "High"})
    # 2. Re-profile slope angle by -4 degrees
    p_flat = dict(payload); p_flat["slope_angle_deg"] = max(10.0, payload["slope_angle_deg"] - 4.0)
    f_flat = calculate_factor_of_safety(p_flat["slope_height_m"], p_flat["slope_angle_deg"], p_flat["cohesion_kpa"], p_flat["friction_angle_deg"], p_flat["unit_weight_kn_m3"], p_flat["pore_pressure_kpa"], p_flat.get("recent_blasting", 0), p_flat.get("blast_intensity", 0), p_flat.get("equipment_load", 0))
    actions.append({"priority": 2, "action": "Pushback Bench Slope Angle by -4°", "mechanism": "Reduces gravitational shear driving stress along failure plane", "delta_fos": round(f_flat - base_fos, 3), "projected_fos": round(f_flat, 3), "feasibility": "Medium"})
    # 3. Crest Equipment Stand-off
    if payload.get("equipment_load", 0) > 0:
        p_load = dict(payload); p_load["equipment_load"] = 0.0
        f_load = calculate_factor_of_safety(p_load["slope_height_m"], p_load["slope_angle_deg"], p_load["cohesion_kpa"], p_load["friction_angle_deg"], p_load["unit_weight_kn_m3"], p_load["pore_pressure_kpa"], p_load.get("recent_blasting", 0), p_load.get("blast_intensity", 0), 0.0)
        actions.append({"priority": 3, "action": "Enforce 25m Crest Stand-off Exclusion Zone", "mechanism": "Removes crest surcharge equipment loading", "delta_fos": round(f_load - base_fos, 3), "projected_fos": round(f_load, 3), "feasibility": "Immediate"})
    # 4. Controlled Presplit Blasting
    if payload.get("recent_blasting", 0) == 1 or payload.get("blast_intensity", 0) > 0:
        p_blast = dict(payload); p_blast["recent_blasting"] = 0; p_blast["blast_intensity"] = 0.0
        f_blast = calculate_factor_of_safety(p_blast["slope_height_m"], p_blast["slope_angle_deg"], p_blast["cohesion_kpa"], p_blast["friction_angle_deg"], p_blast["unit_weight_kn_m3"], p_blast["pore_pressure_kpa"], 0, 0.0, p_blast.get("equipment_load", 0))
        actions.append({"priority": 4, "action": "Switch to Low-Energy Electronic Delay Trim Blasting", "mechanism": "Minimizes dynamic seismic acceleration pulses (k_c = 0)", "delta_fos": round(f_blast - base_fos, 3), "projected_fos": round(f_blast, 3), "feasibility": "High"})
    actions.sort(key=lambda x: x["delta_fos"], reverse=True)
    return {"base_fos": base_fos, "ranked_recommendations": actions}


def model_drift_analysis() -> dict[str, Any]:
    """Evaluates Population Stability Index (PSI) and KS test for data drift."""
    try:
        from backend.database import _load_processed_data
        train_df = _load_processed_data()
        if train_df.empty:
            return {"available": False, "message": "Baseline telemetry dataset unavailable."}
        with connection() as c:
            rows = c.execute("SELECT * FROM prediction_history ORDER BY id DESC LIMIT 200").fetchall()
        if len(rows) < 8:
            return {"available": False, "message": "At least 8 historical prediction records required for drift evaluation."}
        recent_df = pd.DataFrame([dict(r) for r in rows])
        features = ["factor_of_safety", "pore_pressure_kpa", "deformation_rate_mm_day", "slope_angle_deg"]
        drift_metrics = {}
        from scipy.stats import ks_2samp
        for feat in features:
            if feat in train_df.columns and feat in recent_df.columns:
                s_base = train_df[feat].dropna().astype(float)
                s_rec = recent_df[feat].dropna().astype(float)
                if len(s_base) > 0 and len(s_rec) > 0:
                    ks_stat, p_val = ks_2samp(s_base, s_rec)
                    b_cuts = np.quantile(s_base, np.linspace(0, 1, 6))
                    b_cuts[0] -= 1e-5; b_cuts[-1] += 1e-5
                    b_counts = np.histogram(s_base, bins=b_cuts)[0] / max(1, len(s_base))
                    r_counts = np.histogram(s_rec, bins=b_cuts)[0] / max(1, len(s_rec))
                    psi = float(np.sum((r_counts - b_counts) * np.log((r_counts + 1e-4) / (b_counts + 1e-4))))
                    status = "Stable (No Drift)" if psi < 0.10 else ("Moderate Shift" if psi < 0.25 else "Significant Drift")
                    drift_metrics[feat] = {
                        "psi": round(psi, 4), "ks_statistic": round(float(ks_stat), 4),
                        "p_value": round(float(p_val), 4), "status": status,
                        "baseline_mean": round(float(s_base.mean()), 2),
                        "recent_mean": round(float(s_rec.mean()), 2)
                    }
        return {"available": True, "monitored_samples": len(recent_df), "feature_drift": drift_metrics}
    except Exception as exc:
        return {"available": False, "message": f"Drift computation error: {exc}"}


def monte_carlo_pof(payload: dict[str, Any], n: int = 5000):
    from backend.physics import calculate_factor_of_safety
    rng = np.random.default_rng(42)
    samples = []
    # Assumed variation is explicitly illustrative, not estimated field uncertainty.
    for _ in range(n):
        c = max(0, rng.normal(payload["cohesion_kpa"], max(1.0, payload["cohesion_kpa"] * .10)))
        phi = float(np.clip(rng.normal(payload["friction_angle_deg"], 2.0), 1, 75))
        u = max(0, rng.normal(payload["pore_pressure_kpa"], max(1.0, payload["pore_pressure_kpa"] * .15)))
        samples.append(calculate_factor_of_safety(payload["slope_height_m"], payload["slope_angle_deg"], c, phi, payload["unit_weight_kn_m3"], u, payload.get("recent_blasting", 0), payload.get("blast_intensity", 0), payload.get("equipment_load", 0)))
    a = np.asarray(samples)
    pof = float(np.mean(a < 1.0))
    from scipy.stats import norm
    beta = float(-norm.ppf(np.clip(pof, 1e-6, 1 - 1e-6)))
    return {"simulations": n, "probability_of_failure": pof, "reliability_index_beta": beta, "fos_mean": float(a.mean()), "fos_std": float(a.std(ddof=1)), "histogram": {"counts": np.histogram(a, bins=30)[0].tolist(), "edges": np.histogram(a, bins=30)[1].tolist()}, "assumption": "Illustrative Monte Carlo: cohesion ±10%, friction angle SD 2 degrees, pore pressure ±15%; distributions are assumptions, not field-calibrated uncertainty."}


def time_to_failure(sector_id: str):
    from backend.database import get_sector_history
    history = get_sector_history(sector_id, 500)
    if len(history) < 5:
        return {"sector_id": sector_id, "available": False, "message": "At least five historical observations are required."}
    df = pd.DataFrame(history)
    y = pd.to_numeric(df["deformation_rate_mm_day"], errors="coerce").dropna().to_numpy()
    x = np.arange(len(y), dtype=float)
    if len(y) < 5 or np.any(y <= 0):
        return {"sector_id": sector_id, "available": False, "message": "Inverse-velocity estimate requires at least five positive deformation-rate values."}
    inv = 1.0 / y
    slope, intercept = np.polyfit(x, inv, 1)
    if slope >= 0:
        return {"sector_id": sector_id, "available": False, "message": "Inverse velocity is not decreasing in this window; no finite extrapolated window is reported."}
    crossing = -intercept / slope
    steps = crossing - (len(y) - 1)
    return {"sector_id": sector_id, "available": bool(steps > 0), "estimated_observation_steps": float(steps) if steps > 0 else None, "slope": float(slope), "intercept": float(intercept), "method": "Illustrative inverse-velocity linear extrapolation", "message": "This is not a validated time-to-failure forecast. Synthetic history and sparse observations make the estimate indicative only."}


def train_material_classifier():
    if MATERIAL_MODEL.exists():
        try:
            return joblib.load(MATERIAL_MODEL)
        except Exception:
            pass
    df = pd.read_csv(IMAGE_FEATURES)
    features = [c for c in MATERIAL_FEATURES if c in df.columns]
    data = df.dropna(subset=features + ["material"])
    model = RandomForestClassifier(n_estimators=120, class_weight="balanced", random_state=42, n_jobs=1)
    model.fit(data[features], data["material"].astype(str))
    joblib.dump({"model": model, "features": features, "classes": sorted(data["material"].astype(str).unique())}, MATERIAL_MODEL)
    return joblib.load(MATERIAL_MODEL)


def image_features_from_bytes(raw: bytes):
    im = Image.open(io.BytesIO(raw)).convert("RGB").resize((128, 128))
    rgb = np.asarray(im, dtype=np.float32) / 255.0
    gray = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    from skimage.feature import graycomatrix, graycoprops
    gray8 = np.array(im.convert("L"), dtype=np.uint8, copy=True)
    gray_levels = (gray8 // 32).astype(np.uint8)
    glcm = graycomatrix(gray_levels, distances=[1], angles=[0], levels=8, symmetric=True, normed=True)
    gx, gy = np.gradient(gray)
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0
    hist, _ = np.histogram(gray, bins=256, range=(0, 1), density=False)
    hist = hist.astype(float) / max(1, hist.sum())
    hist = hist[hist > 0]
    return {"mean_r": float(rgb[:,:,0].mean()), "mean_g": float(rgb[:,:,1].mean()), "mean_b": float(rgb[:,:,2].mean()), "brightness": float(gray.mean()), "contrast": float(gray.std()), "saturation": float(hsv[:,:,1].mean()), "hue": float(hsv[:,:,0].mean()), "entropy": float(-(hist * np.log2(hist + 1e-12)).sum()), "edge_density": float(np.mean(np.sqrt(gx*gx+gy*gy) > .08)), "sharpness": float(np.var(np.gradient(gray)[0]) + np.var(np.gradient(gray)[1])), "glcm_contrast": float(graycoprops(glcm, "contrast")[0,0]), "glcm_homogeneity": float(graycoprops(glcm, "homogeneity")[0,0]), "glcm_energy": float(graycoprops(glcm, "energy")[0,0]), "glcm_correlation": float(graycoprops(glcm, "correlation")[0,0])}


def classify_material(raw: bytes):
    pack = train_material_classifier()
    feat = image_features_from_bytes(raw)
    x = pd.DataFrame([[feat[c] for c in pack["features"]]], columns=pack["features"])
    model = pack["model"]
    probs = model.predict_proba(x)[0]
    label = str(model.classes_[int(np.argmax(probs))])
    material_norm = label.lower().replace(" ", "_")
    # These are illustrative material ranges for UI initialization only; confirm with lab tests.
    ranges = {
        "basalt": {"cohesion_kpa": [100, 400], "friction_angle_deg": [35, 50]},
        "granite": {"cohesion_kpa": [150, 500], "friction_angle_deg": [35, 55]},
        "limestone": {"cohesion_kpa": [80, 350], "friction_angle_deg": [30, 45]},
        "sandstone": {"cohesion_kpa": [30, 200], "friction_angle_deg": [28, 42]},
        "shale": {"cohesion_kpa": [15, 150], "friction_angle_deg": [20, 35]},
        "clay": {"cohesion_kpa": [10, 100], "friction_angle_deg": [10, 28]},
        "sandy_soil": {"cohesion_kpa": [0, 30], "friction_angle_deg": [28, 38]},
        "gravel_colluvium": {"cohesion_kpa": [0, 40], "friction_angle_deg": [30, 42]},
    }
    return {"predicted_material": label, "confidence": float(np.max(probs)), "class_probabilities": {str(k): float(v) for k, v in zip(model.classes_, probs)}, "texture_features": feat, "illustrative_strength_ranges": ranges.get(material_norm, {"cohesion_kpa": [10, 100], "friction_angle_deg": [20, 40]}), "disclaimer": "Image classification is a prototype aid. Image-derived material labels cannot replace geological logging, sampling, or laboratory testing."}
