"""Optional MQTT publishing for the synthetic sensor simulator."""
from __future__ import annotations

import json
import logging
import os
from urllib.parse import urlparse
from typing import Any

logger = logging.getLogger("talus.mqtt")


def publish_sensor_events(events: list[dict[str, Any]]) -> bool:
    """Publish events when TALUS_MQTT_URL is configured; return False when disabled."""
    broker_url = os.getenv("TALUS_MQTT_URL", "").strip()
    if not broker_url:
        return False
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise RuntimeError("MQTT is configured but pinned paho-mqtt is not installed.") from exc

    parsed = urlparse(broker_url)
    if parsed.scheme not in {"mqtt", "tcp"} or not parsed.hostname:
        raise ValueError("TALUS_MQTT_URL must be an mqtt:// or tcp:// broker address.")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="talus-sensor-simulator")
    client.connect(parsed.hostname, parsed.port or 1883, keepalive=20)
    client.loop_start()
    try:
        for event in events:
            info = client.publish(
                "talus/telemetry",
                json.dumps(event, separators=(",", ":")),
                qos=1,
            )
            info.wait_for_publish(timeout=5)
            if not info.is_published():
                raise RuntimeError("MQTT broker did not acknowledge a telemetry event.")
    finally:
        client.loop_stop()
        client.disconnect()
    logger.info("published %s synthetic sensor events to MQTT", len(events))
    return True
