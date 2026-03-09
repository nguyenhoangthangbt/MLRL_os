"""Tests for mlrl_os.rl.simos_client — SimOS WebSocket client.

All WebSocket and HTTP calls are mocked. No real SimOS is required.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlrl_os.rl.simos_client import SimOSWebSocketClient


class TestSimOSWebSocketClientInit:
    """Tests for client initialization."""

    def test_defaults(self) -> None:
        client = SimOSWebSocketClient()
        assert client._base_url == "ws://localhost:8000"
        assert client._api_key == "sk-premium-test-003"
        assert client._rest_url == "http://localhost:8000"
        assert client._connected is False

    def test_custom_url(self) -> None:
        client = SimOSWebSocketClient(
            base_url="ws://example.com:9000/", api_key="sk-test"
        )
        assert client._base_url == "ws://example.com:9000"
        assert client._api_key == "sk-test"
        assert client._rest_url == "http://example.com:9000"

    def test_wss_to_https_conversion(self) -> None:
        client = SimOSWebSocketClient(base_url="wss://secure.example.com")
        assert client._rest_url == "https://secure.example.com"


class TestCreateSimulation:
    """Tests for create_simulation via REST API."""

    @pytest.mark.asyncio()
    async def test_create_simulation_returns_job_id(self) -> None:
        client = SimOSWebSocketClient()

        mock_template_resp = MagicMock()
        mock_template_resp.status_code = 200
        mock_template_resp.json.return_value = {"template": "healthcare_er", "config": {}}
        mock_template_resp.raise_for_status = MagicMock()

        mock_submit_resp = MagicMock()
        mock_submit_resp.status_code = 200
        mock_submit_resp.json.return_value = {"job_id": "job-abc123"}
        mock_submit_resp.raise_for_status = MagicMock()

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_template_resp
        mock_client_instance.post.return_value = mock_submit_resp
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)

        with patch("mlrl_os.rl.simos_client.httpx.AsyncClient", return_value=mock_client_instance):
            job_id = await client.create_simulation("healthcare_er")

        assert job_id == "job-abc123"
        mock_client_instance.get.assert_called_once()
        mock_client_instance.post.assert_called_once()


class TestWebSocketConnect:
    """Tests for WebSocket connection and command sending."""

    @pytest.mark.asyncio()
    async def test_connect_reads_topology(self) -> None:
        client = SimOSWebSocketClient()

        topology_msg = json.dumps({
            "type": "topology",
            "metadata": {"node_names": ["triage", "treatment"]},
        })

        mock_ws = AsyncMock()
        mock_ws.recv.return_value = topology_msg

        with patch("mlrl_os.rl.simos_client.websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = mock_ws
            await client.connect("job-abc123")

        assert client._connected is True
        assert client._topology["type"] == "topology"
        mock_ws.recv.assert_called_once()

    @pytest.mark.asyncio()
    async def test_pause_sends_command(self) -> None:
        client = SimOSWebSocketClient()
        client._ws = AsyncMock()
        client._connected = True

        await client.pause()
        client._ws.send.assert_called_once_with(json.dumps({"command": "pause"}))

    @pytest.mark.asyncio()
    async def test_resume_sends_command(self) -> None:
        client = SimOSWebSocketClient()
        client._ws = AsyncMock()
        client._connected = True

        await client.resume()
        client._ws.send.assert_called_once_with(json.dumps({"command": "resume"}))

    @pytest.mark.asyncio()
    async def test_set_speed_sends_command(self) -> None:
        client = SimOSWebSocketClient()
        client._ws = AsyncMock()
        client._connected = True

        await client.set_speed(2.0)
        client._ws.send.assert_called_once_with(
            json.dumps({"command": "set_speed", "value": 2.0})
        )

    @pytest.mark.asyncio()
    async def test_step_sends_and_receives(self) -> None:
        client = SimOSWebSocketClient()
        client._ws = AsyncMock()
        client._connected = True

        snapshot = {"type": "snapshot", "data": {"throughput": 10}}
        client._ws.recv.return_value = json.dumps(snapshot)

        result = await client.step(count=5)

        client._ws.send.assert_called_once_with(
            json.dumps({"command": "step", "count": 5})
        )
        assert result["type"] == "snapshot"

    @pytest.mark.asyncio()
    async def test_step_skips_non_actionable_messages(self) -> None:
        client = SimOSWebSocketClient()
        client._ws = AsyncMock()
        client._connected = True

        heartbeat = json.dumps({"type": "heartbeat"})
        snapshot = json.dumps({"type": "snapshot", "data": {"throughput": 5}})
        client._ws.recv.side_effect = [heartbeat, snapshot]

        result = await client.step(count=1)
        assert result["type"] == "snapshot"
        assert client._ws.recv.call_count == 2

    @pytest.mark.asyncio()
    async def test_step_returns_event_batch(self) -> None:
        client = SimOSWebSocketClient()
        client._ws = AsyncMock()
        client._connected = True

        event_batch = json.dumps({"type": "event_batch", "events": [1, 2, 3]})
        client._ws.recv.return_value = event_batch

        result = await client.step(count=1)
        assert result["type"] == "event_batch"

    @pytest.mark.asyncio()
    async def test_step_returns_simulation_end(self) -> None:
        client = SimOSWebSocketClient()
        client._ws = AsyncMock()
        client._connected = True

        end_msg = json.dumps({"type": "simulation_end"})
        client._ws.recv.return_value = end_msg

        result = await client.step(count=1)
        assert result["type"] == "simulation_end"


class TestClose:
    """Tests for closing the client."""

    @pytest.mark.asyncio()
    async def test_close_disconnects(self) -> None:
        client = SimOSWebSocketClient()
        client._ws = AsyncMock()
        client._connected = True

        await client.close()
        client._ws.close.assert_called_once()
        assert client._connected is False

    @pytest.mark.asyncio()
    async def test_close_when_not_connected(self) -> None:
        client = SimOSWebSocketClient()
        await client.close()  # Should not raise


class TestSendCommandError:
    """Tests for error handling on send."""

    @pytest.mark.asyncio()
    async def test_send_when_not_connected_raises(self) -> None:
        client = SimOSWebSocketClient()
        with pytest.raises(RuntimeError, match="Not connected"):
            await client.pause()
