"""Async WebSocket client for stepping through SimOS live simulations.

Communicates with SimOS via:
- REST API (httpx) for simulation creation
- WebSocket (websockets) for real-time stepping and state retrieval

Both ``httpx`` and ``websockets`` are lazy-imported so this module can
be imported without those packages installed.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy imports -- httpx and websockets are imported inside methods
import httpx  # noqa: E402
import websockets  # noqa: E402


class SimOSWebSocketClient:
    """Async WebSocket client for SimOS live simulation control.

    Args:
        base_url: SimOS WebSocket base URL (e.g. ``ws://localhost:8000``).
        api_key: SimOS API key for authentication.
    """

    def __init__(
        self,
        base_url: str = "ws://localhost:8000",
        api_key: str = "sk-premium-test-003",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._rest_url = (
            self._base_url
            .replace("ws://", "http://")
            .replace("wss://", "https://")
        )
        self._ws: Any = None
        self._connected: bool = False
        self._topology: dict[str, Any] = {}

    async def create_simulation(self, template: str) -> str:
        """Create a new simulation from a template via the REST API.

        Args:
            template: SimOS template name (e.g. ``healthcare_er``).

        Returns:
            The ``job_id`` of the created simulation.
        """
        headers = {"X-API-Key": self._api_key, "X-SimOS-Client": "web"}

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self._rest_url}/api/v1/templates/{template}",
                headers=headers,
            )
            resp.raise_for_status()
            config = resp.json()

            resp = await client.post(
                f"{self._rest_url}/api/v1/simulations",
                headers=headers,
                json=config,
            )
            resp.raise_for_status()
            job_id: str = resp.json()["job_id"]
            logger.info("Created simulation %s from template %s", job_id, template)
            return job_id

    async def connect(self, job_id: str) -> None:
        """Connect to a live simulation WebSocket.

        Reads the initial topology message sent by SimOS on connect.

        Args:
            job_id: The simulation job ID to connect to.
        """
        ws_url = f"{self._base_url}/simulations/live/{job_id}"
        self._ws = await websockets.connect(ws_url)
        self._connected = True

        msg = await self._ws.recv()
        self._topology = json.loads(msg)
        logger.info("Connected to simulation %s", job_id)

    async def pause(self) -> None:
        """Pause the simulation."""
        await self._send_command({"command": "pause"})

    async def resume(self) -> None:
        """Resume the simulation."""
        await self._send_command({"command": "resume"})

    async def step(self, count: int = 1) -> dict[str, Any]:
        """Step the simulation by *count* events and return the latest state.

        Skips non-actionable messages (heartbeats, etc.) and returns
        the first ``snapshot``, ``event_batch``, or ``simulation_end``
        message received.

        Args:
            count: Number of events to step.

        Returns:
            The state message dictionary.
        """
        await self._send_command({"command": "step", "count": count})
        return await self._receive_state()

    async def set_speed(self, multiplier: float) -> None:
        """Set the simulation speed multiplier.

        Args:
            multiplier: Speed multiplier (e.g. 2.0 for double speed).
        """
        await self._send_command({"command": "set_speed", "value": multiplier})

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self._ws:
            await self._ws.close()
            self._connected = False

    async def _send_command(self, cmd: dict[str, Any]) -> None:
        """Send a JSON command over the WebSocket.

        Raises:
            RuntimeError: If not connected.
        """
        if not self._connected or not self._ws:
            msg = "Not connected to SimOS WebSocket"
            raise RuntimeError(msg)
        await self._ws.send(json.dumps(cmd))

    async def _receive_state(self) -> dict[str, Any]:
        """Receive messages until an actionable state message arrives.

        Actionable types: ``snapshot``, ``event_batch``, ``simulation_end``.

        Returns:
            The parsed message dictionary.
        """
        actionable_types = {"snapshot", "event_batch", "simulation_end"}
        while True:
            raw = await self._ws.recv()
            msg: dict[str, Any] = json.loads(raw)
            if msg.get("type") in actionable_types:
                return msg
