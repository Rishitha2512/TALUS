"""MQTT telemetry ingestion worker for Docker Compose deployments."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any
from urllib.parse import urlparse

from backend.database import init_db, save_sensor_readings

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("talus.ingestor")


def run_ingestor() -> None:
    """Subscribe to the broker and persist each validated sensor event."""
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise RuntimeError("Install pinned paho-mqtt to run the ingestion worker.") from exc
    broker = os.getenv("TALUS_MQTT_URL", "mqtt://broker:1883")
    parsed = urlparse(broker)
    if parsed.scheme not in {"mqtt", "tcp"} or not parsed.hostname:
        raise ValueError("TALUS_MQTT_URL must be an mqtt:// or tcp:// broker address.")
    init_db()

    def on_connect(client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any) -> None:
        if int(reason_code) != 0:
            logger.error("MQTT ingestion connection rejected: %s", reason_code)
            return
        client.subscribe("talus/telemetry", qos=1)
        logger.info("subscribed to talus/telemetry")

    def on_message(client: Any, userdata: Any, message: Any) -> None:
        try:
            event = json.loads(message.payload.decode("utf-8"))
            sector_id = str(event["sector_id"])
            if not sector_id.startswith("S"):
                raise ValueError("sector_id must begin with S")
            save_sensor_readings(sector_id, [event])
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.exception("discarded invalid MQTT telemetry payload")
        except Exception:
            logger.exception("failed to persist MQTT telemetry")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="talus-ingestor")
    client.on_connect = on_connect
    client.on_message = on_message
    delay_seconds = 1
    while True:
        try:
            client.connect(parsed.hostname, parsed.port or 1883, keepalive=30)
            client.loop_forever()
        except (OSError, RuntimeError):
            logger.exception("MQTT ingestion connection failed; retrying")
            time.sleep(delay_seconds)
            delay_seconds = min(30, delay_seconds * 2)


if __name__ == "__main__":
    run_ingestor()
