"""In-process WebSocket fan-out for live API telemetry updates."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from fastapi import WebSocket


@dataclass
class Subscriber:
    queue: asyncio.Queue[dict[str, Any]]
    loop: asyncio.AbstractEventLoop


class TelemetryHub:
    """Bounded per-client event queues; slow clients drop oldest messages."""

    def __init__(self) -> None:
        self._clients: dict[int, Subscriber] = {}

    async def connect(self, websocket: WebSocket) -> tuple[int, asyncio.Queue[dict[str, Any]]]:
        await websocket.accept()
        client_id = id(websocket)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._clients[client_id] = Subscriber(queue, asyncio.get_running_loop())
        return client_id, queue

    def disconnect(self, client_id: int) -> None:
        self._clients.pop(client_id, None)

    @staticmethod
    def _enqueue(queue: asyncio.Queue[dict[str, Any]], event: dict[str, Any]) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(event)

    def publish(self, event: dict[str, Any]) -> None:
        for client in tuple(self._clients.values()):
            client.loop.call_soon_threadsafe(self._enqueue, client.queue, event)


telemetry_hub = TelemetryHub()
