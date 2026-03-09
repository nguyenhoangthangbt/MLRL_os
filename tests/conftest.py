"""Shared test fixtures for ML/RL OS test suite."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from mlrl_os.config.defaults import MLRLSettings
from mlrl_os.core.dataset import DatasetMeta, RawDataset, compute_column_info
from mlrl_os.core.types import (
    ColumnInfo,
    CVStrategy,
    FeatureMatrix,
    ObservationPoint,
    ProblemType,
    TaskType,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SIMOS_EXPORT_SMALL = FIXTURES_DIR / "simos_export_small.json"


# ---------------------------------------------------------------------------
# Temp directory for test storage
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test file storage."""
    return tmp_path


@pytest.fixture()
def test_settings(tmp_path: Path) -> MLRLSettings:
    """MLRLSettings pointing to temp directories."""
    return MLRLSettings(
        data_dir=str(tmp_path / "data"),
        models_dir=str(tmp_path / "models"),
        experiments_dir=str(tmp_path / "experiments"),
        seed_default=42,
        cv_folds_default=3,
    )


# ---------------------------------------------------------------------------
# Raw JSON fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simos_export_small_path() -> Path:
    """Path to the small SimOS export fixture."""
    return SIMOS_EXPORT_SMALL


@pytest.fixture()
def simos_export_small_dict() -> dict[str, Any]:
    """Parsed JSON of the small SimOS export fixture."""
    with open(SIMOS_EXPORT_SMALL, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Synthetic snapshot DataFrame (canonical names, 100+ rows for validation)
# ---------------------------------------------------------------------------


@pytest.fixture()
def snapshot_df() -> pl.DataFrame:
    """Synthetic snapshot DataFrame with canonical column names.

    100 rows at 300s intervals (bucket_size=300, ~8.3 hours).
    """
    rng = np.random.RandomState(42)
    n = 100
    ts = np.arange(300, 300 * (n + 1), 300, dtype=np.float64)

    return pl.DataFrame({
        "ts": ts,
        "bucket_idx": list(range(1, n + 1)),
        "wip": rng.randint(1, 20, size=n).tolist(),
        "in_queue": rng.randint(0, 10, size=n).tolist(),
        "busy": rng.randint(1, 8, size=n).tolist(),
        "throughput": (rng.uniform(2, 12, size=n)).round(2).tolist(),
        "avg_wait": (rng.uniform(10, 120, size=n)).round(2).tolist(),
        "avg_processing": (rng.uniform(30, 200, size=n)).round(2).tolist(),
        "arrival_rate": (rng.uniform(5, 15, size=n)).round(2).tolist(),
        "wip_ratio": (rng.uniform(0.1, 0.8, size=n)).round(3).tolist(),
        "n_intake_queue": rng.randint(0, 5, size=n).tolist(),
        "n_intake_util": (rng.uniform(0.3, 0.95, size=n)).round(3).tolist(),
        "n_assembly_queue": rng.randint(0, 8, size=n).tolist(),
        "n_assembly_util": (rng.uniform(0.4, 0.98, size=n)).round(3).tolist(),
        "n_qc_queue": rng.randint(0, 4, size=n).tolist(),
        "n_qc_util": (rng.uniform(0.2, 0.85, size=n)).round(3).tolist(),
        "r_workers_capacity": [5] * n,
        "r_workers_util": (rng.uniform(0.3, 0.9, size=n)).round(3).tolist(),
    })


# ---------------------------------------------------------------------------
# Synthetic trajectory DataFrame (canonical names, 200+ rows)
# ---------------------------------------------------------------------------


@pytest.fixture()
def trajectory_df() -> pl.DataFrame:
    """Synthetic trajectory DataFrame with canonical column names.

    20 entities, up to 3 steps each → ~60 rows, plus extras → 200 rows.
    """
    rng = np.random.RandomState(42)
    rows: list[dict[str, Any]] = []
    statuses = ["completed", "completed", "completed", "delayed", "failed"]
    nodes = ["intake", "assembly", "qc"]

    for entity_idx in range(67):
        eid = f"e_{entity_idx:03d}"
        n_steps = rng.randint(1, 4)
        status_final = statuses[rng.randint(0, len(statuses))]

        for step in range(n_steps):
            is_done = step == n_steps - 1
            elapsed = float(rng.uniform(50, 500))
            cum_wait = float(rng.uniform(10, 150))
            cum_proc = float(rng.uniform(30, 300))

            rows.append({
                "eid": eid,
                "etype": "order",
                "epriority": int(rng.randint(1, 4)),
                "source_idx": 0,
                "step": step,
                "node": nodes[step % len(nodes)],
                "t_enter": float(rng.uniform(0, 1000)),
                "t_complete": float(rng.uniform(100, 2000)),
                "wait": float(rng.uniform(5, 80)),
                "processing": float(rng.uniform(20, 200)),
                "setup": float(rng.uniform(0, 10)),
                "transit": float(rng.uniform(1, 15)),
                "transit_cost": float(rng.uniform(0.5, 5)),
                "done": is_done,
                "status": status_final if is_done else "in_progress",
                "total_time": float(rng.uniform(200, 600)) if is_done else 0.0,
                "s_priority": int(rng.randint(1, 4)),
                "s_elapsed": elapsed,
                "s_cum_wait": cum_wait,
                "s_cum_processing": cum_proc,
                "s_wait_ratio": round(cum_wait / max(elapsed, 1), 3),
                "s_steps_done": step,
                "s_cum_cost": float(rng.uniform(10, 100)),
                "s_cum_transit_cost": float(rng.uniform(1, 10)),
                "s_cum_setup": float(rng.uniform(0, 15)),
                "s_source_idx": 0,
                "s_node_util": float(rng.uniform(0.3, 0.95)),
                "s_node_avg_queue": float(rng.uniform(0.5, 4)),
                "s_node_max_queue": int(rng.randint(1, 8)),
                "s_node_avg_wait": float(rng.uniform(10, 80)),
                "s_node_avg_processing": float(rng.uniform(30, 200)),
                "s_node_throughput": float(rng.uniform(2, 10)),
                "s_node_concurrency": int(rng.randint(1, 3)),
                "s_node_setup_count": int(rng.randint(0, 5)),
                "s_node_mutation_count": int(rng.randint(0, 2)),
                "s_sys_util": float(rng.uniform(0.4, 0.85)),
                "s_sys_throughput": float(rng.uniform(3, 8)),
                "s_sys_bottleneck_util": float(rng.uniform(0.6, 0.98)),
                "s_sys_node_count": 3,
                "s_sys_total_capacity": 5,
            })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# FeatureMatrix fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def regression_feature_matrix() -> FeatureMatrix:
    """Simple regression FeatureMatrix for testing."""
    rng = np.random.RandomState(42)
    n, p = 200, 5
    X = rng.randn(n, p)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + rng.randn(n) * 0.1
    return FeatureMatrix(
        X=X,
        y=y,
        feature_names=[f"f{i}" for i in range(p)],
        problem_type=ProblemType.TIME_SERIES,
        target_name="target",
        task_type=TaskType.REGRESSION,
        temporal_index=np.arange(n, dtype=np.float64),
    )


@pytest.fixture()
def classification_feature_matrix() -> FeatureMatrix:
    """Simple classification FeatureMatrix for testing."""
    rng = np.random.RandomState(42)
    n, p = 200, 5
    X = rng.randn(n, p)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    return FeatureMatrix(
        X=X,
        y=y,
        feature_names=[f"f{i}" for i in range(p)],
        problem_type=ProblemType.ENTITY_CLASSIFICATION,
        target_name="status",
        task_type=TaskType.CLASSIFICATION,
        class_names=["0", "1"],
    )


# ---------------------------------------------------------------------------
# DatasetMeta fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def snapshot_dataset_meta(snapshot_df: pl.DataFrame) -> DatasetMeta:
    """DatasetMeta for a snapshot-only dataset."""
    cols = compute_column_info(snapshot_df)
    return DatasetMeta(
        id="ds_test_snap_001",
        name="Test Snapshots",
        content_hash="fakehash_snap",
        source_type="simos",
        source_path="/fake/path/export.json",
        snapshot_row_count=len(snapshot_df),
        snapshot_column_count=len(snapshot_df.columns),
        snapshot_columns=cols,
        has_snapshots=True,
        has_trajectories=False,
        registered_at="2025-01-15T10:00:00Z",
    )


@pytest.fixture()
def trajectory_dataset_meta(trajectory_df: pl.DataFrame) -> DatasetMeta:
    """DatasetMeta for a trajectory-only dataset."""
    cols = compute_column_info(trajectory_df)
    return DatasetMeta(
        id="ds_test_traj_001",
        name="Test Trajectories",
        content_hash="fakehash_traj",
        source_type="simos",
        source_path="/fake/path/export.json",
        trajectory_row_count=len(trajectory_df),
        trajectory_column_count=len(trajectory_df.columns),
        trajectory_columns=cols,
        has_snapshots=False,
        has_trajectories=True,
        registered_at="2025-01-15T10:00:00Z",
    )


@pytest.fixture()
def raw_snapshot_dataset(snapshot_df: pl.DataFrame) -> RawDataset:
    """RawDataset with snapshot data."""
    return RawDataset(
        source_type="simos",
        source_path="/fake/path",
        snapshots=snapshot_df,
        column_info_snapshots=compute_column_info(snapshot_df),
    )


@pytest.fixture()
def raw_trajectory_dataset(trajectory_df: pl.DataFrame) -> RawDataset:
    """RawDataset with trajectory data."""
    return RawDataset(
        source_type="simos",
        source_path="/fake/path",
        trajectories=trajectory_df,
        column_info_trajectories=compute_column_info(trajectory_df),
    )
