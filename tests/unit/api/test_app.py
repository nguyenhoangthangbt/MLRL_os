"""Tests for the FastAPI application factory and health endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mlrl_os.api.app import create_app
from mlrl_os.config.defaults import MLRLSettings


@pytest.fixture()
def client(tmp_path) -> TestClient:
    settings = MLRLSettings(
        data_dir=str(tmp_path / "data"),
        models_dir=str(tmp_path / "models"),
        experiments_dir=str(tmp_path / "experiments"),
    )
    app = create_app(settings)
    return TestClient(app)


class TestAppFactory:
    def test_creates_fastapi_app(self, tmp_path) -> None:
        settings = MLRLSettings(data_dir=str(tmp_path / "data"))
        app = create_app(settings)
        assert app.title == "ML/RL OS"

    def test_has_state_dependencies(self, tmp_path) -> None:
        settings = MLRLSettings(
            data_dir=str(tmp_path / "data"),
            models_dir=str(tmp_path / "models"),
            experiments_dir=str(tmp_path / "experiments"),
        )
        app = create_app(settings)
        assert hasattr(app.state, "dataset_registry")
        assert hasattr(app.state, "model_registry")
        assert hasattr(app.state, "experiment_tracker")
        assert hasattr(app.state, "experiment_runner")


class TestHealthEndpoint:
    def test_health_returns_ok(self, client) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert data["storage_backend"] == "postgresql"
        assert "updated_at" in data
