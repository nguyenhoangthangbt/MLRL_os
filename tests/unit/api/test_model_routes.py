"""Tests for model API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mlrl_os.api.app import create_app
from mlrl_os.config.defaults import MLRLSettings


@pytest.fixture()
def settings(tmp_path) -> MLRLSettings:
    return MLRLSettings(
        data_dir=str(tmp_path / "data"),
        models_dir=str(tmp_path / "models"),
        experiments_dir=str(tmp_path / "experiments"),
    )


@pytest.fixture()
def client(settings) -> TestClient:
    app = create_app(settings)
    return TestClient(app)


class TestListModels:
    def test_empty_list(self, client) -> None:
        resp = client.get("/api/v1/models/")
        assert resp.status_code == 200
        assert resp.json() == []


class TestGetModel:
    def test_get_missing_returns_404(self, client) -> None:
        resp = client.get("/api/v1/models/nonexistent")
        assert resp.status_code == 404


class TestPredict:
    def test_predict_missing_model_returns_404(self, client) -> None:
        resp = client.post(
            "/api/v1/models/nonexistent/predict",
            json={"data": [{"f0": 1.0}]},
        )
        assert resp.status_code == 404


class TestFeatureImportance:
    def test_feature_importance_missing_model_returns_404(self, client) -> None:
        resp = client.get("/api/v1/models/nonexistent/feature-importance")
        assert resp.status_code == 404
