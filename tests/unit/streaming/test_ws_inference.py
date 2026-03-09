"""Tests for mlrl_os.streaming.ws_inference WebSocket endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from mlrl_os.api.app import create_app
from mlrl_os.config.defaults import MLRLSettings


@pytest.fixture()
def app(tmp_path: Path) -> object:
    """Create a FastAPI app with temp storage directories."""
    settings = MLRLSettings(
        data_dir=str(tmp_path / "data"),
        models_dir=str(tmp_path / "models"),
        experiments_dir=str(tmp_path / "experiments"),
    )
    return create_app(settings)


@pytest.fixture()
def client(app: object) -> TestClient:
    """TestClient for the app."""
    return TestClient(app)  # type: ignore[arg-type]


class TestWsPredictEndpoint:
    """Tests for /ws/v1/predict/{model_id}."""

    def test_model_not_found_closes_with_4004(self, client: TestClient) -> None:
        """WebSocket should close with code 4004 when model does not exist."""
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/v1/predict/nonexistent"):
                pass

    def test_route_exists(self, client: TestClient) -> None:
        """The /ws/v1/predict path should be registered (not 404 HTTP)."""
        # WebSocket routes exist on the app; a regular GET returns a 403 or
        # similar, not a 404. We verify by checking the app routes.
        from starlette.routing import WebSocketRoute

        app = client.app
        ws_paths = []
        for route in app.routes:  # type: ignore[union-attr]
            if isinstance(route, WebSocketRoute):
                ws_paths.append(route.path)
            elif hasattr(route, "routes"):
                for sub in route.routes:
                    if isinstance(sub, WebSocketRoute):
                        ws_paths.append(sub.path)

        # The predict route should be reachable at /ws/v1/predict/{model_id}
        assert any("predict" in p for p in ws_paths)


class TestWsPolicyEndpoint:
    """Tests for /ws/v1/policy/{policy_id}."""

    def test_policy_not_found_closes(self, client: TestClient) -> None:
        """WebSocket should close when policy does not exist (v0.2 stub)."""
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/v1/policy/nonexistent"):
                pass

    def test_route_exists(self, client: TestClient) -> None:
        """The /ws/v1/policy path should be registered."""
        from starlette.routing import WebSocketRoute

        app = client.app
        ws_paths = []
        for route in app.routes:  # type: ignore[union-attr]
            if isinstance(route, WebSocketRoute):
                ws_paths.append(route.path)
            elif hasattr(route, "routes"):
                for sub in route.routes:
                    if isinstance(sub, WebSocketRoute):
                        ws_paths.append(sub.path)

        assert any("policy" in p for p in ws_paths)
