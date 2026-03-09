"""Tests for dataset API endpoints."""

from __future__ import annotations

import io
import json

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
def registered_dataset(settings) -> str:
    """Register a dataset and return its ID."""
    registry = DatasetRegistry(settings)
    df = pl.DataFrame({
        "ts": list(range(50)),
        "value_a": [float(i) for i in range(50)],
        "value_b": [float(i * 2) for i in range(50)],
    })
    raw = RawDataset(source_type="test", source_path="test", snapshots=df)
    meta = registry.register(raw, name="test_dataset")
    return meta.id


class TestUploadDataset:
    def test_upload_csv(self, client) -> None:
        csv_content = "ts,value\n1,10.0\n2,20.0\n3,30.0\n"
        resp = client.post(
            "/api/v1/datasets/",
            files={"file": ("data.csv", io.BytesIO(csv_content.encode()), "text/csv")},
            data={"name": "my_dataset"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "my_dataset"
        assert data["dataset_id"]
        assert data["has_snapshots"] is True  # ts column → snapshots

    def test_upload_without_name_uses_filename(self, client) -> None:
        csv_content = "x,y\n1,2\n3,4\n"
        resp = client.post(
            "/api/v1/datasets/",
            files={"file": ("myfile.csv", io.BytesIO(csv_content.encode()), "text/csv")},
        )
        assert resp.status_code == 201
        assert resp.json()["name"] == "myfile.csv"


class TestListDatasets:
    def test_empty_list(self, client) -> None:
        resp = client.get("/api/v1/datasets/")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_lists_registered(self, client, registered_dataset) -> None:
        resp = client.get("/api/v1/datasets/")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        # The field is serialized as "id" (the alias)
        assert data[0]["id"] == registered_dataset


class TestGetDataset:
    def test_get_existing(self, client, registered_dataset) -> None:
        resp = client.get(f"/api/v1/datasets/{registered_dataset}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == registered_dataset
        assert data["name"] == "test_dataset"

    def test_get_missing_returns_404(self, client) -> None:
        resp = client.get("/api/v1/datasets/nonexistent")
        assert resp.status_code == 404


class TestGetSchema:
    def test_get_snapshot_schema(self, client, registered_dataset) -> None:
        resp = client.get(
            f"/api/v1/datasets/{registered_dataset}/schema",
            params={"layer": "snapshots"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["dataset_id"] == registered_dataset
        assert data["layer"] == "snapshots"
        assert len(data["columns"]) > 0


class TestPreviewDataset:
    def test_preview_returns_rows(self, client, registered_dataset) -> None:
        resp = client.get(
            f"/api/v1/datasets/{registered_dataset}/preview",
            params={"layer": "snapshots", "rows": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["rows"]) == 5
        assert data["total_rows"] == 50
        assert "ts" in data["columns"]

    def test_preview_missing_dataset_returns_404(self, client) -> None:
        resp = client.get("/api/v1/datasets/nonexistent/preview")
        assert resp.status_code == 404


class TestAvailableTargets:
    def test_discover_targets(self, client, registered_dataset) -> None:
        resp = client.get(f"/api/v1/datasets/{registered_dataset}/available-targets")
        assert resp.status_code == 200
        data = resp.json()
        assert data["dataset_id"] == registered_dataset
        # Should have time-series targets (numeric columns except ts)
        assert len(data["time_series_targets"]) > 0
