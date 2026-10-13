"""TALUS persistence layer.

The database is intentionally data-driven: mine sectors and historical observations
are derived from the processed telemetry dataset rather than being hard-coded.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import urlparse

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.getenv("TALUS_DB_PATH", os.path.join(ROOT_DIR, "database", "talus.db"))
DATABASE_URL = os.getenv("TALUS_DATABASE_URL") or os.getenv("DATABASE_URL")
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "cleaned_features.csv")
DB_SCHEMA_VERSION = 2


class DatabaseConnection:
    """Adapt existing qmark SQL to PostgreSQL while preserving SQLite behavior."""

    def __init__(self, connection: Any, postgres: bool) -> None:
        self.connection = connection
        self.postgres = postgres

    def _sql(self, statement: str) -> str:
        if not self.postgres:
            return statement
        return statement.replace(
            "INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY"
        ).replace("?", "%s")

    def execute(self, statement: str, parameters: tuple = ()) -> Any:
        return self.connection.execute(self._sql(statement), parameters)

    def executemany(self, statement: str, parameters: Any) -> Any:
        return self.connection.executemany(self._sql(statement), parameters)

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "DatabaseConnection":
        self.connection.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return self.connection.__exit__(exc_type, exc_value, traceback)


def get_db_connection() -> DatabaseConnection:
    if DATABASE_URL and urlparse(DATABASE_URL).scheme in {"postgres", "postgresql"}:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL is configured but psycopg is not installed. "
                "Install the pinned project requirements."
            ) from exc
        return DatabaseConnection(
            psycopg.connect(DATABASE_URL, row_factory=dict_row), postgres=True
        )
    if DATABASE_URL and urlparse(DATABASE_URL).scheme not in {"sqlite", "file"}:
        raise ValueError("TALUS_DATABASE_URL must use PostgreSQL or SQLite.")
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return DatabaseConnection(conn, postgres=False)


def _load_processed_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def _table_count(conn: Any, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) AS row_count FROM {table}").fetchone()["row_count"])


def _schema_version(conn: Any) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS app_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    row = conn.execute("SELECT value FROM app_meta WHERE key='schema_version'").fetchone()
    return int(row["value"]) if row else 0


def _set_schema_version(conn: Any) -> None:
    conn.execute(
        "INSERT INTO app_meta(key,value) VALUES('schema_version',?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(DB_SCHEMA_VERSION),),
    )


def _create_tables(conn: Any) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mine_sectors (
            sector_id TEXT PRIMARY KEY,
            sector_name TEXT NOT NULL,
            pit_zone TEXT NOT NULL,
            slope_height_m REAL,
            slope_angle_deg REAL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sector_id TEXT NOT NULL,
            slope_height_m REAL,
            slope_angle_deg REAL,
            cohesion_kpa REAL,
            friction_angle_deg REAL,
            unit_weight_kn_m3 REAL,
            rainfall_mm REAL,
            groundwater_level REAL,
            pore_pressure_kpa REAL,
            deformation_rate_mm_day REAL,
            displacement_mm REAL,
            recent_blasting INTEGER,
            blast_intensity REAL,
            equipment_load REAL,
            factor_of_safety REAL,
            physics_category TEXT,
            ml_risk_label TEXT,
            ml_risk_probability REAL,
            top_contributors_json TEXT,
            scenario_name TEXT DEFAULT 'Historical Telemetry'
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_sector_time ON prediction_history(sector_id, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_risk ON prediction_history(ml_risk_label)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS talus_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('Viewer','Engineer','Admin')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            action TEXT NOT NULL,
            resource TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sector_id TEXT NOT NULL,
            sensor_name TEXT NOT NULL,
            sample_index INTEGER NOT NULL,
            value REAL,
            quality_flag TEXT NOT NULL,
            dataset_type TEXT NOT NULL
        )
    """)
    conn.commit()


def _reset_data_tables(conn: Any) -> None:
    conn.execute("DELETE FROM prediction_history")
    conn.execute("DELETE FROM mine_sectors")
    conn.commit()


def _seed_from_processed_data(conn: Any) -> None:
    df = _load_processed_data()
    if df.empty or "sector_id" not in df.columns:
        return

    df = df.sort_values("timestamp")
    sectors = sorted(df["sector_id"].dropna().astype(str).unique(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)

    sector_rows = []
    for sid in sectors:
        latest = df[df["sector_id"].astype(str) == sid].iloc[-1]
        sector_rows.append((
            sid,
            f"TALUS Monitoring Sector {sid}",
            "Telemetry Grid",
            float(latest.get("slope_height_m", 0)),
            float(latest.get("slope_angle_deg", 0)),
        ))
    conn.executemany(
        "INSERT INTO mine_sectors(sector_id,sector_name,pit_zone,slope_height_m,slope_angle_deg) VALUES(?,?,?,?,?)",
        sector_rows,
    )

    # Store a representative historical window per sector. This keeps the demo DB
    # useful while avoiding fabricated observations and hard-coded sector profiles.
    records: List[tuple] = []
    for sid in sectors:
        sdf = df[df["sector_id"].astype(str) == sid].tail(60)
        for _, r in sdf.iterrows():
            contributors = [
                {"feature": "factor_of_safety", "value": float(r.get("factor_of_safety", 0)), "impact_score": 0.0, "direction": "physics_indicator"},
                {"feature": "deformation_rate_mm_day", "value": float(r.get("deformation_rate_mm_day", 0)), "impact_score": 0.0, "direction": "telemetry"},
                {"feature": "pore_pressure_kpa", "value": float(r.get("pore_pressure_kpa", 0)), "impact_score": 0.0, "direction": "telemetry"},
            ]
            label = str(r.get("risk_label", "Low"))
            prob_map = {"Low": 0.15, "Moderate": 0.40, "High": 0.72, "Critical": 0.94}
            records.append((
                str(r.get("timestamp")), sid, float(r.get("slope_height_m", 0)), float(r.get("slope_angle_deg", 0)),
                float(r.get("cohesion_kpa", 0)), float(r.get("friction_angle_deg", 0)), float(r.get("unit_weight_kn_m3", 0)),
                float(r.get("rainfall_mm", 0)), float(r.get("groundwater_level", 0)), float(r.get("pore_pressure_kpa", 0)),
                float(r.get("deformation_rate_mm_day", 0)), float(r.get("displacement_mm", 0)), int(r.get("recent_blasting", 0)),
                float(r.get("blast_intensity", 0)), float(r.get("equipment_load", 0)), float(r.get("factor_of_safety", 0)),
                label if label in {"Low", "Moderate", "High", "Critical"} else "Low",
                label, prob_map.get(label, 0.15), json.dumps(contributors), "Historical Telemetry",
            ))

    conn.executemany("""
        INSERT INTO prediction_history(
            timestamp,sector_id,slope_height_m,slope_angle_deg,cohesion_kpa,friction_angle_deg,
            unit_weight_kn_m3,rainfall_mm,groundwater_level,pore_pressure_kpa,deformation_rate_mm_day,
            displacement_mm,recent_blasting,blast_intensity,equipment_load,factor_of_safety,
            physics_category,ml_risk_label,ml_risk_probability,top_contributors_json,scenario_name
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, records)
    conn.commit()


def init_db() -> None:
    conn = get_db_connection()
    _create_tables(conn)
    if _schema_version(conn) != DB_SCHEMA_VERSION or _table_count(conn, "mine_sectors") == 0 or _table_count(conn, "prediction_history") == 0:
        _reset_data_tables(conn)
        _seed_from_processed_data(conn)
        _set_schema_version(conn)
        conn.commit()
    conn.close()


def save_prediction(rec: Dict[str, Any]) -> int:
    init_db()
    conn = get_db_connection()
    contributors = json.dumps(rec.get("top_contributors", []))
    cur = conn.execute("""
        INSERT INTO prediction_history(
            timestamp,sector_id,slope_height_m,slope_angle_deg,cohesion_kpa,friction_angle_deg,
            unit_weight_kn_m3,rainfall_mm,groundwater_level,pore_pressure_kpa,deformation_rate_mm_day,
            displacement_mm,recent_blasting,blast_intensity,equipment_load,factor_of_safety,
            physics_category,ml_risk_label,ml_risk_probability,top_contributors_json,scenario_name
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) RETURNING id
    """, (
        rec.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")), rec.get("sector_id"),
        rec.get("slope_height_m"), rec.get("slope_angle_deg"), rec.get("cohesion_kpa"), rec.get("friction_angle_deg"),
        rec.get("unit_weight_kn_m3"), rec.get("rainfall_mm"), rec.get("groundwater_level"), rec.get("pore_pressure_kpa"),
        rec.get("deformation_rate_mm_day"), rec.get("displacement_mm"), rec.get("recent_blasting", 0),
        rec.get("blast_intensity", 0), rec.get("equipment_load", 0), rec.get("factor_of_safety"),
        rec.get("physics_category"), rec.get("ml_risk_label"), rec.get("ml_risk_probability"), contributors,
        rec.get("scenario_name", "User Prediction"),
    ))
    row_id = cur.fetchone()["id"]
    conn.commit()
    conn.close()
    return int(row_id)


def save_sensor_readings(sector_id: str, readings: List[Dict[str, Any]]) -> int:
    """Persist simulator readings with their explicit quality and synthetic flags."""
    if not readings:
        return 0
    conn = get_db_connection()
    try:
        conn.executemany(
            """
            INSERT INTO sensor_readings(
                sector_id,sensor_name,sample_index,value,quality_flag,dataset_type
            ) VALUES(?,?,?,?,?,?)
            """,
            [
                (
                    sector_id,
                    "synthetic_displacement",
                    int(item["sample_index"]),
                    item["displacement_mm"],
                    "DROPOUT" if item["dropout"] else ("OUTLIER_INJECTED" if item["outlier_injected"] else "OK"),
                    "DEMO/SYNTHETIC",
                )
                for item in readings
            ],
        )
        conn.commit()
        return len(readings)
    finally:
        conn.close()


def _decode(row: sqlite3.Row) -> Dict[str, Any]:
    item = dict(row)
    try:
        item["top_contributors"] = json.loads(item.pop("top_contributors_json") or "[]")
    except Exception:
        item["top_contributors"] = []
        item.pop("top_contributors_json", None)
    return item


def get_latest_mine_overview() -> List[Dict[str, Any]]:
    init_db()
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT h.*, s.sector_name, s.pit_zone
        FROM prediction_history h
        JOIN (SELECT sector_id, MAX(id) max_id FROM prediction_history GROUP BY sector_id) latest
          ON h.id = latest.max_id
        JOIN mine_sectors s ON h.sector_id = s.sector_id
        ORDER BY CASE WHEN h.ml_risk_label='Critical' THEN 1 WHEN h.ml_risk_label='High' THEN 2 WHEN h.ml_risk_label='Moderate' THEN 3 ELSE 4 END,
                 h.sector_id
    """).fetchall()
    conn.close()
    return [_decode(r) for r in rows]


def get_sector_history(sector_id: str, limit: int = 60) -> List[Dict[str, Any]]:
    init_db()
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM prediction_history WHERE sector_id=? ORDER BY timestamp ASC LIMIT ?", (sector_id.upper(), limit)).fetchall()
    conn.close()
    return [_decode(r) for r in rows]


def get_data_summary() -> Dict[str, Any]:
    df = _load_processed_data()
    if df.empty:
        return {"records": 0, "sectors": [], "date_min": None, "date_max": None}
    return {
        "records": int(len(df)),
        "sectors": sorted(df["sector_id"].dropna().astype(str).unique().tolist(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x),
        "date_min": str(df["timestamp"].min()),
        "date_max": str(df["timestamp"].max()),
        "features": [c for c in df.columns if c not in {"record_id", "sector_id", "timestamp", "risk_class", "risk_label", "dataset_type"}],
        "risk_distribution": df["risk_label"].value_counts().to_dict(),
    }
