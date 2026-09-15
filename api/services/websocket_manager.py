"""WebSocket connection manager: fans out predictions to dashboard clients."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import List, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Broadcasts JSON prediction events to all connected dashboard clients."""

    def __init__(self) -> None:
        self._connections: List[WebSocket] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Remember the server's event loop so worker threads can broadcast."""
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.append(websocket)
        logger.info("WebSocket client connected (%d active)", len(self._connections))

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)
        logger.info("WebSocket client disconnected (%d active)", len(self._connections))

    @property
    def active_connections(self) -> int:
        return len(self._connections)

    async def broadcast(self, payload: dict) -> None:
        """Send a dict to every connected client; drops dead sockets."""
        message = json.dumps(payload, default=str)
        dead = []
        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception:  # noqa: BLE001 - any send failure means the socket is gone
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def broadcast_sync(self, payload: dict) -> None:
        """Thread-safe broadcast from the Kafka consumer thread.

        If no loop is registered yet, the payload is logged and dropped.
        """
        if self._loop is None or self._loop.is_closed():
            logger.debug("No event loop registered; prediction not broadcast: %s", payload.get("event_id"))
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(payload), self._loop)


ws_manager = WebSocketManager()
