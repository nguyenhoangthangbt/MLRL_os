"""Tests for config resolution endpoints."""

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
    import numpy as np

    registry = DatasetRegistry(settings)
    rng = np.random.RandomState(42)
    n = 200
    df = pl.DataFrame({
        "ts": list(range(n)),
        "avg_wait": rng.rand(n).tolist(),
        "throughput": rng.rand(n).tolist(),
    })
    raw = RawDataset(source_type="test", source_path="test", snapshots=df)
    meta = registry.register(raw, name="ts_resolve_test")
    return meta.id


class TestResolveConfig:
    def test_resolve_returns_full_config(self, client, ts_dataset_id) -> None:
        resp = client.post(
            "/api/v1/config/resolve",
            json={
                "dataset_id": ts_dataset_id,
                "experiment_type": "time_series",
                "target": "avg_wait",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "resolved_config" in data
        config = data["resolved_config"]
        assert config["name"]
        assert config["experiment_type"] == "time_series"

    def test_resolve_missing_dataset_returns_404(self, client) -> None:
        resp = client.post(
            "/api/v1/config/resolve",
            json={"dataset_id": "nonexistent"},
        )
        assert resp.status_code == 404
