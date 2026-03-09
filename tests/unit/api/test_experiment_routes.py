"""Tests for experiment API endpoints."""

from __future__ import annotations

import polars as pl
import pytest
from fastapi.testclient import TestClient

from mlrl_os.api.app import create_app
from mlrl_os.config.defaults import MLRLSettings
from mlrl_os.core.dataset import RawDataset
from mlrl_os.data.registry import DatasetRegistry


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


@pytest.fixture()
def ts_dataset_id(settings) -> str:
    """Register a time-series dataset and return its ID."""
    import numpy as np

    registry = DatasetRegistry(settings)
    rng = np.random.RandomState(42)
    n = 200
    df = pl.DataFrame({
        "ts": list(range(n)),
        "avg_wait": rng.rand(n).tolist(),
        "throughput": rng.rand(n).tolist(),
        "queue_depth": rng.rand(n).tolist(),
    })
    raw = RawDataset(source_type="test", source_path="test", snapshots=df)
    meta = registry.register(raw, name="ts_test")
    return meta.id


class TestGetDefaults:
    def test_ts_defaults(self, client) -> None:
        resp = client.get("/api/v1/experiments/defaults/time_series")
        assert resp.status_code == 200
        data = resp.json()
        assert data["problem_type"] == "time_series"
        assert "target" in data["defaults"]
        assert data["defaults"]["target"] == "avg_wait"

    def test_entity_defaults(self, client) -> None:
        resp = client.get("/api/v1/experiments/defaults/entity_classification")
        assert resp.status_code == 200
        data = resp.json()
        assert data["defaults"]["target"] == "status"

    def test_invalid_problem_type(self, client) -> None:
        resp = client.get("/api/v1/experiments/defaults/invalid")
        assert resp.status_code == 422


class TestValidateExperiment:
    def test_validate_valid_config(self, client, ts_dataset_id) -> None:
        resp = client.post(
            "/api/v1/experiments/validate",
            json={
                "dataset_id": ts_dataset_id,
                "experiment_type": "time_series",
                "target": "avg_wait",
                "algorithms": ["linear"],
                "lookback": "30",
                "horizon": "5",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert data["resolved_config"] is not None

    def test_validate_missing_dataset(self, client) -> None:
        resp = client.post(
            "/api/v1/experiments/validate",
            json={"dataset_id": "nonexistent"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0


class TestSubmitExperiment:
    def test_submit_ts_experiment(self, client, ts_dataset_id) -> None:
        resp = client.post(
            "/api/v1/experiments/",
            json={
                "dataset_id": ts_dataset_id,
                "experiment_type": "time_series",
                "target": "avg_wait",
                "algorithms": ["linear"],
                "lookback": "30",
                "horizon": "5",
                "metrics": ["rmse"],
                "cv_folds": 3,
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["experiment_id"]
        assert data["status"] in ("completed", "failed")

    def test_submit_missing_dataset_returns_error(self, client) -> None:
        resp = client.post(
            "/api/v1/experiments/",
            json={"dataset_id": "nonexistent"},
        )
        assert resp.status_code == 404


class TestListExperiments:
    def test_empty_list(self, client) -> None:
        resp = client.get("/api/v1/experiments/")
        assert resp.status_code == 200
        assert resp.json() == []


class TestGetExperiment:
    def test_get_missing_returns_404(self, client) -> None:
        resp = client.get("/api/v1/experiments/nonexistent")
        assert resp.status_code == 404
