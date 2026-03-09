# ML/RL OS v0.2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add RL training (PPO/DQN against SimOS), hyperparameter tuning, PostgreSQL optional backend, streaming inference, LSTM model, and HTML reports to the complete v0.1 foundation.

**Architecture:** v0.2 extends the existing pipeline with 6 new modules while preserving all v0.1 contracts. StorageBackend protocol abstracts file vs. postgres. RL engine uses custom PPO/DQN (no Gymnasium) with SimOS WebSocket stepping. All new algorithms lazy-loaded via existing registry pattern.

**Tech Stack:** Python 3.13, PyTorch (RL networks + LSTM), websockets (SimOS client), asyncpg (PostgreSQL), Optuna (tuning already in deps)

**Smoke test requirement:** At least 2 SimOS templates (healthcare_er + logistics_otd). Document findings and enhancement proposals.

---

## Phase 1: Storage Backend Abstraction (Foundation)

Everything else builds on pluggable storage. Refactor existing file-based registries behind a protocol, then add PostgreSQL.

### Task 1.1: Storage Protocol

**Files:**
- Create: `src/mlrl_os/storage/__init__.py`
- Create: `src/mlrl_os/storage/protocol.py`
- Test: `tests/unit/storage/test_protocol.py`

**Step 1: Write the failing test**

```python
# tests/unit/storage/__init__.py
# (empty)

# tests/unit/storage/test_protocol.py
"""Tests for StorageBackend protocol compliance."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mlrl_os.storage.protocol import StorageBackend


def test_protocol_exists():
    """StorageBackend protocol is importable."""
    assert hasattr(StorageBackend, "save_dataset")
    assert hasattr(StorageBackend, "load_dataset")
    assert hasattr(StorageBackend, "list_datasets")
    assert hasattr(StorageBackend, "save_model")
    assert hasattr(StorageBackend, "load_model")
    assert hasattr(StorageBackend, "list_models")
    assert hasattr(StorageBackend, "save_experiment")
    assert hasattr(StorageBackend, "load_experiment")
    assert hasattr(StorageBackend, "list_experiments")
    assert hasattr(StorageBackend, "delete_dataset")
    assert hasattr(StorageBackend, "delete_model")
    assert hasattr(StorageBackend, "delete_experiment")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/storage/test_protocol.py -v`
Expected: FAIL — cannot import StorageBackend

**Step 3: Write implementation**

```python
# src/mlrl_os/storage/__init__.py
"""Pluggable storage backends for ML/RL OS."""

# src/mlrl_os/storage/protocol.py
"""StorageBackend protocol — all backends implement this contract."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import polars as pl

from mlrl_os.core.dataset import DatasetMeta, RawDataset
from mlrl_os.core.experiment import ExperimentResult
from mlrl_os.models.algorithms.protocol import TrainedModel


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for dataset, model, and experiment storage."""

    # --- Datasets ---
    def save_dataset(self, raw: RawDataset, name: str) -> DatasetMeta: ...
    def load_dataset(self, dataset_id: str, layer: str) -> pl.DataFrame: ...
    def get_dataset_meta(self, dataset_id: str) -> DatasetMeta: ...
    def list_datasets(self) -> list[DatasetMeta]: ...
    def has_dataset(self, dataset_id: str) -> bool: ...
    def delete_dataset(self, dataset_id: str) -> None: ...

    # --- Models ---
    def save_model(
        self, model: TrainedModel, experiment_id: str,
        metrics: dict[str, float] | None = None,
    ) -> str: ...
    def load_model(self, model_id: str) -> TrainedModel: ...
    def get_model_meta(self, model_id: str) -> dict[str, Any]: ...
    def list_models(self) -> list[dict[str, Any]]: ...
    def has_model(self, model_id: str) -> bool: ...
    def delete_model(self, model_id: str) -> None: ...

    # --- Experiments ---
    def save_experiment(self, result: ExperimentResult, config: dict[str, Any]) -> None: ...
    def load_experiment(self, experiment_id: str) -> ExperimentResult: ...
    def list_experiments(self) -> list[ExperimentResult]: ...
    def has_experiment(self, experiment_id: str) -> bool: ...
    def delete_experiment(self, experiment_id: str) -> None: ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/storage/test_protocol.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mlrl_os/storage/ tests/unit/storage/
git commit -m "feat(storage): add StorageBackend protocol"
```

---

### Task 1.2: File Storage Backend

**Files:**
- Create: `src/mlrl_os/storage/file_backend.py`
- Test: `tests/unit/storage/test_file_backend.py`

**Step 1: Write the failing test**

```python
# tests/unit/storage/test_file_backend.py
"""Tests for FileStorageBackend — wraps existing registry behavior."""
from __future__ import annotations

import polars as pl
import pytest

from mlrl_os.storage.file_backend import FileStorageBackend
from mlrl_os.storage.protocol import StorageBackend
from mlrl_os.core.dataset import RawDataset
from mlrl_os.config.defaults import MLRLSettings


@pytest.fixture
def backend(tmp_path):
    settings = MLRLSettings(
        data_dir=str(tmp_path / "data"),
        models_dir=str(tmp_path / "models"),
        experiments_dir=str(tmp_path / "experiments"),
    )
    return FileStorageBackend(settings)


def test_implements_protocol(backend):
    assert isinstance(backend, StorageBackend)


def test_dataset_roundtrip(backend):
    df = pl.DataFrame({"ts": [1, 2, 3], "value": [10.0, 20.0, 30.0]})
    raw = RawDataset(source_type="external", source_path="test.csv", snapshots=df)
    meta = backend.save_dataset(raw, "test_dataset")
    assert meta.name == "test_dataset"
    assert backend.has_dataset(meta.id)
    loaded = backend.load_dataset(meta.id, "snapshots")
    assert loaded.shape == (3, 2)
    datasets = backend.list_datasets()
    assert len(datasets) == 1


def test_dataset_not_found(backend):
    with pytest.raises(KeyError):
        backend.get_dataset_meta("nonexistent")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/storage/test_file_backend.py -v`
Expected: FAIL — cannot import FileStorageBackend

**Step 3: Write implementation**

FileStorageBackend delegates to the existing `DatasetRegistry`, `ModelRegistry`, and `ExperimentTracker`. This is a thin adapter that unifies them behind the StorageBackend protocol.

```python
# src/mlrl_os/storage/file_backend.py
"""File-system storage backend — wraps existing v0.1 registries."""
from __future__ import annotations

from typing import Any

import polars as pl

from mlrl_os.config.defaults import MLRLSettings
from mlrl_os.core.dataset import DatasetMeta, RawDataset
from mlrl_os.core.experiment import ExperimentResult
from mlrl_os.data.registry import DatasetRegistry
from mlrl_os.experiment.tracker import ExperimentTracker
from mlrl_os.models.algorithms.protocol import TrainedModel
from mlrl_os.models.registry import ModelRegistry


class FileStorageBackend:
    """File-system storage using existing v0.1 registries."""

    def __init__(self, settings: MLRLSettings | None = None) -> None:
        self._settings = settings or MLRLSettings()
        self._datasets = DatasetRegistry(self._settings)
        self._models = ModelRegistry(self._settings)
        self._experiments = ExperimentTracker(self._settings)

    # --- Datasets ---
    def save_dataset(self, raw: RawDataset, name: str) -> DatasetMeta:
        return self._datasets.register(raw, name)

    def load_dataset(self, dataset_id: str, layer: str) -> pl.DataFrame:
        return self._datasets.get_data(dataset_id, layer)

    def get_dataset_meta(self, dataset_id: str) -> DatasetMeta:
        return self._datasets.get_meta(dataset_id)

    def list_datasets(self) -> list[DatasetMeta]:
        return self._datasets.list_datasets()

    def has_dataset(self, dataset_id: str) -> bool:
        return self._datasets.has(dataset_id)

    def delete_dataset(self, dataset_id: str) -> None:
        return self._datasets.delete(dataset_id)

    # --- Models ---
    def save_model(
        self, model: TrainedModel, experiment_id: str,
        metrics: dict[str, float] | None = None,
    ) -> str:
        meta = self._models.register(model, experiment_id, metrics)
        return meta.id

    def load_model(self, model_id: str) -> TrainedModel:
        return self._models.load_model(model_id)

    def get_model_meta(self, model_id: str) -> dict[str, Any]:
        return self._models.get_meta(model_id).model_dump()

    def list_models(self) -> list[dict[str, Any]]:
        return [m.model_dump() for m in self._models.list_models()]

    def has_model(self, model_id: str) -> bool:
        return self._models.has(model_id)

    def delete_model(self, model_id: str) -> None:
        return self._models.delete(model_id)

    # --- Experiments ---
    def save_experiment(self, result: ExperimentResult, config: dict[str, Any]) -> None:
        self._experiments.record(result.experiment_id, config, result)

    def load_experiment(self, experiment_id: str) -> ExperimentResult:
        return self._experiments.get_result(experiment_id)

    def list_experiments(self) -> list[ExperimentResult]:
        return self._experiments.list_experiments()

    def has_experiment(self, experiment_id: str) -> bool:
        return self._experiments.has(experiment_id)

    def delete_experiment(self, experiment_id: str) -> None:
        import shutil
        exp_dir = self._experiments._base_dir / experiment_id
        if not exp_dir.exists():
            raise KeyError(f"Experiment {experiment_id} not found")
        shutil.rmtree(exp_dir)

    # --- Access underlying registries (for backward compat) ---
    @property
    def dataset_registry(self) -> DatasetRegistry:
        return self._datasets

    @property
    def model_registry(self) -> ModelRegistry:
        return self._models

    @property
    def experiment_tracker(self) -> ExperimentTracker:
        return self._experiments
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/storage/test_file_backend.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mlrl_os/storage/file_backend.py tests/unit/storage/test_file_backend.py
git commit -m "feat(storage): add FileStorageBackend wrapping v0.1 registries"
```

---

### Task 1.3: PostgreSQL Backend + DB Setup Script

**Files:**
- Create: `src/mlrl_os/storage/postgres_backend.py`
- Create: `scripts/create_db.py`
- Create: `.env.development` (copy from main repo)
- Test: `tests/unit/storage/test_postgres_backend.py`

**Step 1: Write the failing test**

```python
# tests/unit/storage/test_postgres_backend.py
"""Tests for PostgresStorageBackend — requires running PostgreSQL.

Skip if DATABASE_URL not configured or DB unreachable.
"""
from __future__ import annotations

import os
import polars as pl
import pytest

from mlrl_os.core.dataset import RawDataset

# Skip entire module if no database URL
DATABASE_URL = os.environ.get(
    "MLRLOS_DATABASE_URL",
    "postgresql://simos:simos_dev@localhost:5432/MLRL_os",
)

try:
    import asyncpg  # noqa: F401
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

pytestmark = pytest.mark.skipif(
    not HAS_ASYNCPG, reason="asyncpg not installed"
)


@pytest.fixture
async def backend():
    from mlrl_os.storage.postgres_backend import PostgresStorageBackend
    b = PostgresStorageBackend(DATABASE_URL)
    try:
        await b.connect()
    except Exception:
        pytest.skip("PostgreSQL not reachable")
    await b.ensure_tables()
    yield b
    # Cleanup test data
    await b.close()


@pytest.mark.asyncio
async def test_implements_protocol(backend):
    from mlrl_os.storage.protocol import StorageBackend
    # Postgres backend uses async, so protocol check is structural
    assert hasattr(backend, "save_dataset")
    assert hasattr(backend, "load_dataset")


@pytest.mark.asyncio
async def test_dataset_roundtrip(backend):
    df = pl.DataFrame({"ts": [1, 2, 3], "value": [10.0, 20.0, 30.0]})
    raw = RawDataset(source_type="external", source_path="test.csv", snapshots=df)
    meta = await backend.save_dataset(raw, "pg_test")
    assert meta.name == "pg_test"
    loaded = await backend.load_dataset(meta.id, "snapshots")
    assert loaded.shape == (3, 2)
    # Cleanup
    await backend.delete_dataset(meta.id)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/storage/test_postgres_backend.py -v`
Expected: FAIL — cannot import PostgresStorageBackend

**Step 3: Write DB creation script**

```python
# scripts/create_db.py
"""Create the MLRL_os database and tables.

Usage: python scripts/create_db.py
Reads MLRLOS_DATABASE_URL from environment or .env.development.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


TABLES_SQL = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    content_hash TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_path TEXT NOT NULL,
    meta JSONB NOT NULL DEFAULT '{}',
    data_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT,
    problem_type TEXT NOT NULL,
    dataset_id TEXT REFERENCES datasets(id),
    config JSONB NOT NULL DEFAULT '{}',
    result JSONB,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    experiment_id TEXT REFERENCES experiments(id),
    algorithm TEXT NOT NULL,
    task TEXT NOT NULL,
    feature_names JSONB DEFAULT '[]',
    metrics JSONB NOT NULL DEFAULT '{}',
    artifact_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rl_policies (
    id TEXT PRIMARY KEY,
    experiment_id TEXT REFERENCES experiments(id),
    algorithm TEXT NOT NULL,
    template TEXT NOT NULL,
    reward_function TEXT NOT NULL,
    training_curves JSONB,
    artifact_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""


def _load_env():
    """Load DATABASE_URL from env or .env.development."""
    url = os.environ.get("MLRLOS_DATABASE_URL")
    if url:
        return url
    env_file = Path(__file__).parent.parent / ".env.development"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("MLRLOS_DATABASE_URL="):
                return line.split("=", 1)[1]
    return "postgresql://simos:simos_dev@localhost:5432/MLRL_os"


async def main():
    import asyncpg

    url = _load_env()
    print(f"Connecting to: {url}")

    # Try to create the database if it doesn't exist
    # Connect to 'postgres' default db first
    server_url = url.rsplit("/", 1)[0] + "/postgres"
    db_name = url.rsplit("/", 1)[1]

    try:
        conn = await asyncpg.connect(server_url)
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", db_name
        )
        if not exists:
            await conn.execute(f'CREATE DATABASE "{db_name}"')
            print(f"Created database: {db_name}")
        else:
            print(f"Database already exists: {db_name}")
        await conn.close()
    except Exception as e:
        print(f"Warning: Could not check/create database: {e}")

    # Now connect to the target database and create tables
    conn = await asyncpg.connect(url)
    await conn.execute(TABLES_SQL)
    print("Tables created successfully.")
    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 4: Write PostgresStorageBackend**

```python
# src/mlrl_os/storage/postgres_backend.py
"""PostgreSQL storage backend using asyncpg."""
from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from mlrl_os.core.dataset import DatasetMeta, RawDataset, compute_column_info
from mlrl_os.core.experiment import ExperimentResult
from mlrl_os.models.algorithms.protocol import TrainedModel


class PostgresStorageBackend:
    """PostgreSQL storage backend.

    Dataset Parquet files are stored on disk (referenced by data_path column).
    Model artifacts stored on disk (referenced by artifact_path column).
    Metadata and results stored in PostgreSQL as JSONB.
    """

    def __init__(self, database_url: str, data_dir: str = "./data") -> None:
        self._database_url = database_url
        self._data_dir = Path(data_dir)
        self._pool = None

    async def connect(self) -> None:
        import asyncpg
        self._pool = await asyncpg.create_pool(self._database_url, min_size=1, max_size=5)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def ensure_tables(self) -> None:
        """Create tables if they don't exist."""
        from scripts.create_db import TABLES_SQL
        async with self._pool.acquire() as conn:
            await conn.execute(TABLES_SQL)

    # --- Datasets ---
    async def save_dataset(self, raw: RawDataset, name: str) -> DatasetMeta:
        ts = datetime.now(timezone.utc).isoformat()
        content_hash = self._hash_dataset(raw)
        dataset_id = hashlib.sha256(
            f"{name}:{ts}:{content_hash}".encode()
        ).hexdigest()[:12]

        # Save parquet to disk
        ds_dir = self._data_dir / "datasets" / dataset_id
        ds_dir.mkdir(parents=True, exist_ok=True)
        if raw.snapshots is not None:
            raw.snapshots.write_parquet(ds_dir / "snapshots.parquet")
        if raw.trajectories is not None:
            raw.trajectories.write_parquet(ds_dir / "trajectories.parquet")

        col_snap = compute_column_info(raw.snapshots) if raw.snapshots is not None else None
        col_traj = compute_column_info(raw.trajectories) if raw.trajectories is not None else None

        meta = DatasetMeta(
            id=dataset_id,
            name=name or raw.source_path,
            content_hash=content_hash,
            source_type=raw.source_type,
            source_path=raw.source_path,
            snapshot_row_count=raw.snapshots.height if raw.snapshots is not None else None,
            trajectory_row_count=raw.trajectories.height if raw.trajectories is not None else None,
            snapshot_column_count=raw.snapshots.width if raw.snapshots is not None else None,
            trajectory_column_count=raw.trajectories.width if raw.trajectories is not None else None,
            snapshot_columns=col_snap,
            trajectory_columns=col_traj,
            has_snapshots=raw.has_snapshots,
            has_trajectories=raw.has_trajectories,
            simos_metadata=raw.metadata_dict,
            simos_summary=raw.summary_dict,
            registered_at=ts,
        )

        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO datasets (id, name, version, content_hash, source_type,
                   source_path, meta, data_path, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)""",
                meta.id, meta.name, meta.version, meta.content_hash,
                meta.source_type, meta.source_path,
                json.dumps(meta.model_dump(), default=str),
                str(ds_dir), ts,
            )
        return meta

    async def load_dataset(self, dataset_id: str, layer: str = "snapshots") -> pl.DataFrame:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_path FROM datasets WHERE id = $1", dataset_id
            )
        if not row:
            raise KeyError(f"Dataset {dataset_id} not found")
        path = Path(row["data_path"]) / f"{layer}.parquet"
        if not path.exists():
            raise ValueError(f"Layer {layer} not found for dataset {dataset_id}")
        return pl.read_parquet(path)

    async def get_dataset_meta(self, dataset_id: str) -> DatasetMeta:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT meta FROM datasets WHERE id = $1", dataset_id
            )
        if not row:
            raise KeyError(f"Dataset {dataset_id} not found")
        return DatasetMeta.model_validate(json.loads(row["meta"]))

    async def list_datasets(self) -> list[DatasetMeta]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT meta FROM datasets ORDER BY created_at")
        return [DatasetMeta.model_validate(json.loads(r["meta"])) for r in rows]

    async def has_dataset(self, dataset_id: str) -> bool:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM datasets WHERE id = $1)", dataset_id
            )

    async def delete_dataset(self, dataset_id: str) -> None:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM datasets WHERE id = $1", dataset_id
            )
        if result == "DELETE 0":
            raise KeyError(f"Dataset {dataset_id} not found")

    # --- Models ---
    async def save_model(
        self, model: TrainedModel, experiment_id: str,
        metrics: dict[str, float] | None = None,
    ) -> str:
        import joblib
        ts = datetime.now(timezone.utc).isoformat()
        model_id = hashlib.sha256(
            f"{experiment_id}:{model.algorithm_name}:{ts}".encode()
        ).hexdigest()[:12]

        model_dir = self._data_dir / "models" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")

        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO models (id, experiment_id, algorithm, task,
                   feature_names, metrics, artifact_path, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
                model_id, experiment_id, model.algorithm_name, model.task,
                json.dumps(model.feature_names),
                json.dumps(metrics or {}),
                str(model_dir / "model.joblib"), ts,
            )
        return model_id

    async def load_model(self, model_id: str) -> TrainedModel:
        import joblib
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT artifact_path FROM models WHERE id = $1", model_id
            )
        if not row:
            raise KeyError(f"Model {model_id} not found")
        return joblib.load(row["artifact_path"])

    async def get_model_meta(self, model_id: str) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM models WHERE id = $1", model_id)
        if not row:
            raise KeyError(f"Model {model_id} not found")
        return dict(row)

    async def list_models(self) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM models ORDER BY created_at DESC")
        return [dict(r) for r in rows]

    async def has_model(self, model_id: str) -> bool:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM models WHERE id = $1)", model_id
            )

    async def delete_model(self, model_id: str) -> None:
        async with self._pool.acquire() as conn:
            result = await conn.execute("DELETE FROM models WHERE id = $1", model_id)
        if result == "DELETE 0":
            raise KeyError(f"Model {model_id} not found")

    # --- Experiments ---
    async def save_experiment(self, result: ExperimentResult, config: dict[str, Any]) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO experiments (id, name, problem_type, config, result, status, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7)
                   ON CONFLICT (id) DO UPDATE SET result = $5, status = $6""",
                result.experiment_id, result.name,
                result.experiment_type.value if hasattr(result.experiment_type, 'value') else str(result.experiment_type),
                json.dumps(config, default=str),
                json.dumps(result.model_dump(), default=str),
                result.status.value if hasattr(result.status, 'value') else str(result.status),
                result.created_at,
            )

    async def load_experiment(self, experiment_id: str) -> ExperimentResult:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT result FROM experiments WHERE id = $1", experiment_id
            )
        if not row:
            raise KeyError(f"Experiment {experiment_id} not found")
        return ExperimentResult.model_validate(json.loads(row["result"]))

    async def list_experiments(self) -> list[ExperimentResult]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT result FROM experiments ORDER BY created_at DESC"
            )
        return [ExperimentResult.model_validate(json.loads(r["result"])) for r in rows]

    async def has_experiment(self, experiment_id: str) -> bool:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM experiments WHERE id = $1)", experiment_id
            )

    async def delete_experiment(self, experiment_id: str) -> None:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM experiments WHERE id = $1", experiment_id
            )
        if result == "DELETE 0":
            raise KeyError(f"Experiment {experiment_id} not found")

    # --- Helpers ---
    @staticmethod
    def _hash_dataset(raw: RawDataset) -> str:
        import hashlib
        h = hashlib.sha256()
        if raw.snapshots is not None:
            h.update(raw.snapshots.write_ipc(None).getvalue())
        if raw.trajectories is not None:
            h.update(raw.trajectories.write_ipc(None).getvalue())
        return h.hexdigest()[:16]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/storage/test_postgres_backend.py -v`
Expected: PASS (or skip if no PostgreSQL)

**Step 5: Add .env.development, update pyproject.toml**

Add `asyncpg>=0.29` to `[project.optional-dependencies].postgres`.

**Step 6: Commit**

```bash
git add src/mlrl_os/storage/postgres_backend.py scripts/create_db.py tests/unit/storage/
git commit -m "feat(storage): add PostgreSQL backend + DB creation script"
```

---

### Task 1.4: Wire Storage Backend into Settings and App Factory

**Files:**
- Modify: `src/mlrl_os/config/defaults.py` — add storage_backend + database_url settings
- Modify: `src/mlrl_os/api/app.py` — use storage backend based on settings
- Test: `tests/unit/config/test_defaults.py` — add storage setting tests

**Step 1: Add settings fields**

In `config/defaults.py`, add to `MLRLSettings`:
```python
storage_backend: str = "file"  # "file" | "postgres"
database_url: str = ""
```

With `model_config = SettingsConfigDict(env_prefix="MLRLOS_")` or add secondary prefix.

**Step 2: Update app.py to construct backend**

In `create_app()`, after creating settings:
```python
if settings.storage_backend == "postgres":
    from mlrl_os.storage.postgres_backend import PostgresStorageBackend
    backend = PostgresStorageBackend(settings.database_url, settings.data_dir)
    # async init handled via lifespan
else:
    from mlrl_os.storage.file_backend import FileStorageBackend
    backend = FileStorageBackend(settings)
app.state.storage = backend
```

Keep backward compatibility: `app.state.dataset_registry`, `app.state.model_registry`, etc. still available (from FileStorageBackend properties or wrapped).

**Step 3: Tests and commit**

```bash
git commit -m "feat(storage): wire backend selection into settings and app factory"
```

---

## Phase 2: Hyperparameter Tuning Integration

### Task 2.1: Wire Optuna Tuning into Experiment Runner

**Files:**
- Modify: `src/mlrl_os/models/engine.py` — already has `_tune_and_validate`, verify wiring
- Modify: `src/mlrl_os/config/schemas.py` — add `n_trials` field if missing
- Test: `tests/unit/models/test_engine.py` — add tuning test

**Step 1: Write the failing test**

```python
# Add to tests/unit/models/test_engine.py
def test_train_with_tuning(small_feature_matrix):
    """Optuna tuning produces different params than defaults."""
    engine = ModelEngine()
    result = engine.train_and_evaluate(
        feature_matrix=small_feature_matrix,
        algorithm_names=["random_forest"],
        metric_names=["f1_weighted"],
        cv_strategy=CVStrategy.STRATIFIED_KFOLD,
        cv_folds=3,
        seed=42,
        hyperparameter_tuning=True,
    )
    assert result.best_algorithm == "random_forest"
    assert result.best_metrics["f1_weighted"] >= 0.0
```

**Step 2: Verify the existing tuning path works end-to-end**

Read `engine.py` to check if `hyperparameter_tuning=True` already calls `_tune_and_validate`. If yes, this task is just adding test coverage + `n_trials` config.

**Step 3: Add n_trials to ResolvedModelConfig**

```python
# In config/schemas.py ResolvedModelConfig:
n_trials: int = 20
```

**Step 4: Update config resolver to pass n_trials through**

**Step 5: Tests and commit**

```bash
git commit -m "feat(tuning): wire Optuna tuning into experiment pipeline with n_trials config"
```

---

## Phase 3: RL Engine (Largest Module)

### Task 3.1: RL Core Types — Spaces and Rewards Protocol

**Files:**
- Create: `src/mlrl_os/rl/__init__.py`
- Create: `src/mlrl_os/rl/spaces.py`
- Create: `src/mlrl_os/rl/rewards.py`
- Test: `tests/unit/rl/__init__.py`
- Test: `tests/unit/rl/test_spaces.py`
- Test: `tests/unit/rl/test_rewards.py`

**Step 1: Write the failing tests**

```python
# tests/unit/rl/test_spaces.py
import numpy as np
from mlrl_os.rl.spaces import ObservationSpec, ActionSpec

def test_observation_spec():
    spec = ObservationSpec(
        dim=24,
        names=["f" + str(i) for i in range(24)],
        low=np.zeros(24),
        high=np.ones(24),
    )
    assert spec.dim == 24
    assert len(spec.names) == 24

def test_action_spec_discrete():
    spec = ActionSpec(type="discrete", n=4, labels=["a", "b", "c", "d"])
    assert spec.n == 4

def test_action_spec_continuous():
    spec = ActionSpec(type="continuous", shape=(2,), low=np.array([-1, -1]), high=np.array([1, 1]))
    assert spec.shape == (2,)
```

```python
# tests/unit/rl/test_rewards.py
from mlrl_os.rl.rewards import (
    ThroughputReward, SLAComplianceReward,
    CostMinimizationReward, CompositeReward,
)

def test_throughput_reward():
    r = ThroughputReward()
    reward = r.compute(
        prev_state={"completed_count": 5},
        action=0,
        curr_state={"completed_count": 6},
        done=False,
    )
    assert reward == 1.0  # one completion

def test_composite_reward():
    r = CompositeReward([
        (ThroughputReward(), 1.0),
        (CostMinimizationReward(), 0.5),
    ])
    reward = r.compute(
        prev_state={"completed_count": 5, "total_cost": 100},
        action=0,
        curr_state={"completed_count": 6, "total_cost": 110},
        done=False,
    )
    assert isinstance(reward, float)
```

**Step 2: Write implementations**

See design doc section 2.5 and 2.11 for full specs.

**Step 3: Commit**

```bash
git commit -m "feat(rl): add observation/action specs and reward function protocol"
```

---

### Task 3.2: Neural Networks (PyTorch)

**Files:**
- Create: `src/mlrl_os/rl/networks.py`
- Test: `tests/unit/rl/test_networks.py`

**Step 1: Write failing tests**

```python
# tests/unit/rl/test_networks.py
import pytest
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

from mlrl_os.rl.networks import MLP, ActorCritic

def test_mlp_forward():
    net = MLP(input_dim=24, output_dim=4, hidden_dims=[64, 64])
    x = torch.randn(8, 24)
    out = net(x)
    assert out.shape == (8, 4)

def test_actor_critic_forward():
    ac = ActorCritic(obs_dim=24, act_dim=4, hidden_dims=[64, 64])
    x = torch.randn(8, 24)
    dist, value = ac(x)
    assert value.shape == (8, 1)
    action = dist.sample()
    assert action.shape == (8,)
```

**Step 2: Implement MLP and ActorCritic**

```python
# src/mlrl_os/rl/networks.py
"""Neural network architectures for RL algorithms."""
from __future__ import annotations
# PyTorch lazy-imported at function/class level
```

**Step 3: Commit**

```bash
git commit -m "feat(rl): add MLP and ActorCritic network architectures"
```

---

### Task 3.3: Replay Buffer

**Files:**
- Create: `src/mlrl_os/rl/replay_buffer.py`
- Test: `tests/unit/rl/test_replay_buffer.py`

**Step 1: Write failing tests**

```python
# tests/unit/rl/test_replay_buffer.py
import numpy as np
from mlrl_os.rl.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

def test_replay_buffer_push_sample():
    buf = ReplayBuffer(capacity=100, seed=42)
    for i in range(50):
        buf.push(
            state=np.random.randn(24),
            action=0,
            reward=1.0,
            next_state=np.random.randn(24),
            done=False,
        )
    assert len(buf) == 50
    batch = buf.sample(16)
    assert batch.states.shape == (16, 24)
    assert batch.actions.shape == (16,)

def test_replay_buffer_overflow():
    buf = ReplayBuffer(capacity=10, seed=42)
    for i in range(20):
        buf.push(np.zeros(4), 0, 0.0, np.zeros(4), False)
    assert len(buf) == 10

def test_prioritized_buffer():
    buf = PrioritizedReplayBuffer(capacity=100, seed=42)
    for i in range(50):
        buf.push(np.random.randn(4), 0, 1.0, np.random.randn(4), False)
    batch = buf.sample(8)
    assert batch.states.shape[0] == 8
```

**Step 2: Implement**

**Step 3: Commit**

```bash
git commit -m "feat(rl): add uniform and prioritized replay buffers"
```

---

### Task 3.4: DQN Algorithm

**Files:**
- Create: `src/mlrl_os/rl/algorithms/__init__.py`
- Create: `src/mlrl_os/rl/algorithms/protocol.py`
- Create: `src/mlrl_os/rl/algorithms/registry.py`
- Create: `src/mlrl_os/rl/algorithms/dqn.py`
- Test: `tests/unit/rl/test_dqn.py`

**Step 1: Write failing tests**

```python
# tests/unit/rl/test_dqn.py
import pytest
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

import numpy as np
from mlrl_os.rl.algorithms.dqn import DQN

def test_dqn_select_action():
    dqn = DQN(obs_dim=4, act_dim=2, seed=42)
    obs = np.array([0.1, 0.2, 0.3, 0.4])
    action = dqn.select_action(obs, explore=False)
    assert action in (0, 1)

def test_dqn_train_step():
    dqn = DQN(obs_dim=4, act_dim=2, seed=42)
    from mlrl_os.rl.replay_buffer import ReplayBuffer
    buf = ReplayBuffer(capacity=100, seed=42)
    for _ in range(64):
        buf.push(np.random.randn(4), 0, 1.0, np.random.randn(4), False)
    losses = dqn.train_step(buf.sample(32))
    assert "loss" in losses

def test_dqn_save_load(tmp_path):
    dqn = DQN(obs_dim=4, act_dim=2, seed=42)
    dqn.save(tmp_path / "dqn.pt")
    dqn2 = DQN(obs_dim=4, act_dim=2, seed=42)
    dqn2.load(tmp_path / "dqn.pt")
    obs = np.array([0.1, 0.2, 0.3, 0.4])
    assert dqn.select_action(obs, explore=False) == dqn2.select_action(obs, explore=False)
```

**Step 2: Implement DQN**

Key: epsilon-greedy, target network, Huber loss, soft updates. Seeded via `seed_hash("dqn", seed)`.

**Step 3: Commit**

```bash
git commit -m "feat(rl): add DQN algorithm with target network and epsilon-greedy"
```

---

### Task 3.5: PPO Algorithm

**Files:**
- Create: `src/mlrl_os/rl/algorithms/ppo.py`
- Test: `tests/unit/rl/test_ppo.py`

**Step 1: Write failing tests**

```python
# tests/unit/rl/test_ppo.py
import pytest
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

import numpy as np
from mlrl_os.rl.algorithms.ppo import PPO

def test_ppo_select_action():
    ppo = PPO(obs_dim=4, act_dim=2, seed=42)
    obs = np.array([0.1, 0.2, 0.3, 0.4])
    action = ppo.select_action(obs, explore=True)
    assert action in (0, 1)

def test_ppo_train_step():
    ppo = PPO(obs_dim=4, act_dim=2, seed=42)
    # PPO uses rollout buffer, not replay buffer
    rollout = ppo.collect_rollout_batch(
        states=np.random.randn(64, 4),
        actions=np.random.randint(0, 2, 64),
        rewards=np.random.randn(64),
        dones=np.zeros(64, dtype=bool),
        values=np.random.randn(64),
        log_probs=np.random.randn(64),
    )
    losses = ppo.train_step(rollout)
    assert "policy_loss" in losses
    assert "value_loss" in losses
```

**Step 2: Implement PPO**

Key: clipped surrogate objective, GAE, mini-batch updates, entropy bonus. Seeded via `seed_hash("ppo", seed)`.

**Step 3: Commit**

```bash
git commit -m "feat(rl): add PPO algorithm with clipped objective and GAE"
```

---

### Task 3.6: SimOS WebSocket Client

**Files:**
- Create: `src/mlrl_os/rl/simos_client.py`
- Test: `tests/unit/rl/test_simos_client.py`

**Step 1: Write failing tests (with mock WebSocket)**

```python
# tests/unit/rl/test_simos_client.py
import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from mlrl_os.rl.simos_client import SimOSWebSocketClient

@pytest.mark.asyncio
async def test_step_sends_command():
    client = SimOSWebSocketClient(
        base_url="ws://localhost:8000",
        api_key="sk-premium-test-003",
    )
    # Mock the websocket connection
    mock_ws = AsyncMock()
    mock_ws.recv = AsyncMock(return_value=json.dumps({
        "type": "event_batch",
        "events": [],
        "snapshot": {"sys_wip": 5},
    }))
    client._ws = mock_ws
    client._connected = True

    events = await client.step(count=1)
    mock_ws.send.assert_called_once()
    sent = json.loads(mock_ws.send.call_args[0][0])
    assert sent["command"] == "step"
    assert sent["count"] == 1
```

**Step 2: Implement**

Uses `websockets` library. Methods: connect(job_id), pause(), resume(), step(count), receive_snapshot(), close().

**Step 3: Commit**

```bash
git commit -m "feat(rl): add SimOS WebSocket client for live simulation control"
```

---

### Task 3.7: SimOS Environment

**Files:**
- Create: `src/mlrl_os/rl/environment.py`
- Test: `tests/unit/rl/test_environment.py`

**Step 1: Write failing tests (with mocked client)**

```python
# tests/unit/rl/test_environment.py
import pytest
import numpy as np
from unittest.mock import AsyncMock, patch

from mlrl_os.rl.environment import SimOSEnvironment
from mlrl_os.rl.rewards import ThroughputReward

@pytest.mark.asyncio
async def test_environment_reset():
    env = SimOSEnvironment(
        simos_url="ws://localhost:8000",
        template="healthcare_er",
        reward_fn=ThroughputReward(),
        seed=42,
    )
    # Mock the REST + WebSocket calls
    with patch.object(env, '_start_simulation', new_callable=AsyncMock) as mock_start:
        mock_start.return_value = {"sys_wip": 0, "sys_utilization": 0.0}
        obs = await env.reset()
        assert isinstance(obs, np.ndarray)

@pytest.mark.asyncio
async def test_environment_step():
    env = SimOSEnvironment(
        simos_url="ws://localhost:8000",
        template="healthcare_er",
        reward_fn=ThroughputReward(),
        seed=42,
    )
    with patch.object(env, '_step_simulation', new_callable=AsyncMock) as mock_step:
        mock_step.return_value = (
            {"sys_wip": 3, "completed_count": 1},
            False,  # done
        )
        env._prev_state = {"sys_wip": 0, "completed_count": 0}
        obs, reward, done, info = await env.step(action=0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
```

**Step 2: Implement**

SimOSEnvironment orchestrates: REST API for simulation creation, WebSocket client for stepping, obs_builder for state→observation, reward_fn for reward computation.

**Step 3: Commit**

```bash
git commit -m "feat(rl): add SimOSEnvironment with reset/step/close lifecycle"
```

---

### Task 3.8: Curriculum Manager

**Files:**
- Create: `src/mlrl_os/rl/curriculum.py`
- Test: `tests/unit/rl/test_curriculum.py`

**Step 1: Write failing tests**

```python
# tests/unit/rl/test_curriculum.py
from mlrl_os.rl.curriculum import CurriculumManager, StressScenario

def test_curriculum_advancement():
    scenarios = [
        StressScenario(id=0, description="baseline", difficulty=0, config_change={}),
        StressScenario(id=1, description="+10% load", difficulty=3, config_change={"rate": 1.1}),
        StressScenario(id=2, description="+20% load", difficulty=6, config_change={"rate": 1.2}),
    ]
    cm = CurriculumManager(scenarios, threshold=0.8)
    assert cm.current_scenario().id == 0
    assert not cm.should_advance(0.5)
    assert cm.should_advance(0.85)
    cm.advance()
    assert cm.current_scenario().id == 1

def test_curriculum_completion():
    scenarios = [StressScenario(id=0, description="only", difficulty=0, config_change={})]
    cm = CurriculumManager(scenarios, threshold=0.8)
    cm.advance()
    assert cm.advance() is None  # curriculum complete
```

**Step 2: Implement**

**Step 3: Commit**

```bash
git commit -m "feat(rl): add curriculum manager for Layer 5 stress scenarios"
```

---

### Task 3.9: RL Config Schemas and Validation Rules

**Files:**
- Modify: `src/mlrl_os/core/types.py` — add `REINFORCEMENT_LEARNING` to ProblemType
- Create: `src/mlrl_os/config/rl_schemas.py` — RL-specific config models
- Modify: `src/mlrl_os/validation/gate.py` — add RL validation rules
- Test: `tests/unit/config/test_rl_schemas.py`
- Test: `tests/unit/validation/test_gate.py` — add RL validation tests

**Step 1: Add ProblemType.REINFORCEMENT_LEARNING**

```python
# In core/types.py ProblemType enum:
REINFORCEMENT_LEARNING = "reinforcement_learning"
```

**Step 2: Create RL config schemas**

```python
# src/mlrl_os/config/rl_schemas.py
class ResolvedRewardConfig(BaseModel):
    function: str  # "throughput", "sla_compliance", "cost_minimization", "composite"
    components: list[dict[str, Any]] = []  # for composite: [{name, weight}, ...]

class ResolvedCurriculumConfig(BaseModel):
    enabled: bool = False
    threshold: float = 0.8

class ResolvedRLConfig(BaseModel):
    name: str
    experiment_type: Literal["reinforcement_learning"]
    seed: int
    simos_url: str
    template: str
    max_steps_per_episode: int = 10_000
    algorithm: str  # "ppo" | "dqn"
    max_episodes: int = 1000
    batch_size: int = 64
    learning_rate: float = 0.0003
    gamma: float = 0.99
    # PPO-specific
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    # DQN-specific
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: int = 10_000
    target_update_freq: int = 100
    replay_buffer_size: int = 10_000
    # Reward
    reward: ResolvedRewardConfig
    # Curriculum
    curriculum: ResolvedCurriculumConfig = ResolvedCurriculumConfig()
    # Evaluation
    eval_episodes: int = 50
    eval_metrics: list[str] = ["mean_reward", "success_rate"]
```

**Step 3: Add RL validation rules to gate.py**

```python
# VRL-01: algorithm must be "ppo" or "dqn"
# VRL-02: max_episodes >= 1
# VRL-03: learning_rate > 0
# VRL-04: gamma in [0, 1]
# VRL-05: batch_size >= 1
# VRL-06: simos_url must be ws:// or wss://
```

**Step 4: Tests and commit**

```bash
git commit -m "feat(rl): add RL config schemas, ProblemType enum, and validation rules"
```

---

### Task 3.10: RL Experiment Runner

**Files:**
- Create: `src/mlrl_os/rl/runner.py`
- Test: `tests/unit/rl/test_runner.py`

**Step 1: Write failing tests (mocked environment)**

```python
# tests/unit/rl/test_runner.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import numpy as np

from mlrl_os.rl.runner import RLExperimentRunner

@pytest.mark.asyncio
async def test_rl_runner_completes(tmp_path):
    """RL runner executes training loop with mocked environment."""
    config = {
        "name": "test_rl",
        "experiment_type": "reinforcement_learning",
        "seed": 42,
        "simos_url": "ws://localhost:8000",
        "template": "healthcare_er",
        "algorithm": "dqn",
        "max_episodes": 3,
        "batch_size": 8,
        "eval_episodes": 2,
        "replay_buffer_size": 100,
        "reward": {"function": "throughput"},
    }
    runner = RLExperimentRunner(artifacts_dir=tmp_path)

    # Mock environment
    mock_env = AsyncMock()
    mock_env.reset = AsyncMock(return_value=np.zeros(4))
    mock_env.step = AsyncMock(return_value=(np.zeros(4), 1.0, True, {}))
    mock_env.obs_dim = 4
    mock_env.act_dim = 2

    with patch.object(runner, '_build_environment', return_value=mock_env):
        result = await runner.run(config)

    assert result.status.value == "completed"
    assert result.experiment_type.value == "reinforcement_learning"
```

**Step 2: Implement RL runner**

See design doc section 2.13. Training loop: reset → step → collect → train → evaluate → store.

**Step 3: Commit**

```bash
git commit -m "feat(rl): add RLExperimentRunner with full training loop"
```

---

### Task 3.11: RL API Routes

**Files:**
- Create: `src/mlrl_os/api/rl_routes.py`
- Modify: `src/mlrl_os/api/app.py` — register RL router
- Modify: `src/mlrl_os/api/schemas.py` — add RL request/response models
- Test: `tests/unit/api/test_rl_routes.py`

**Step 1: Write failing tests**

```python
# tests/unit/api/test_rl_routes.py
import pytest
from httpx import AsyncClient, ASGITransport
from mlrl_os.api.app import create_app

@pytest.fixture
def app(tmp_path):
    from mlrl_os.config.defaults import MLRLSettings
    settings = MLRLSettings(
        data_dir=str(tmp_path / "data"),
        models_dir=str(tmp_path / "models"),
        experiments_dir=str(tmp_path / "experiments"),
    )
    return create_app(settings)

@pytest.mark.asyncio
async def test_list_rl_experiments(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/v1/rl/experiments")
        assert resp.status_code == 200
        assert resp.json() == []

@pytest.mark.asyncio
async def test_list_rl_policies(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/v1/rl/policies")
        assert resp.status_code == 200
        assert resp.json() == []
```

**Step 2: Implement routes**

POST `/api/v1/rl/experiments` — submit RL experiment
GET `/api/v1/rl/experiments` — list RL experiments
GET `/api/v1/rl/experiments/{id}` — get RL experiment result
GET `/api/v1/rl/policies` — list trained policies
GET `/api/v1/rl/policies/{id}` — get policy metadata

**Step 3: Commit**

```bash
git commit -m "feat(rl): add RL API routes for experiments and policies"
```

---

## Phase 4: LSTM Sequence Model

### Task 4.1: LSTM Algorithm via Registry

**Files:**
- Create: `src/mlrl_os/models/algorithms/lstm.py`
- Modify: `src/mlrl_os/models/algorithms/registry.py` — register lstm
- Modify: `src/mlrl_os/validation/gate.py` — add "lstm" to KNOWN_ALGORITHMS
- Test: `tests/unit/models/test_lstm.py`

**Step 1: Write the failing test**

```python
# tests/unit/models/test_lstm.py
import pytest
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

import numpy as np
from mlrl_os.models.algorithms.lstm import LSTMAlgorithm

def test_lstm_train_regression():
    algo = LSTMAlgorithm()
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    model = algo.train(X, y, task="regression", seed=42, epochs=5)
    assert model.algorithm_name == "lstm"
    preds = algo.predict(model, X)
    assert preds.shape == (100,)

def test_lstm_train_classification():
    algo = LSTMAlgorithm()
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randint(0, 3, 100)
    model = algo.train(X, y, task="classification", seed=42, epochs=5)
    preds = algo.predict(model, X)
    assert preds.shape == (100,)

def test_lstm_in_registry():
    from mlrl_os.models.algorithms.registry import default_registry
    reg = default_registry()
    assert reg.has("lstm")
```

**Step 2: Implement LSTM algorithm**

PyTorch LSTM with configurable hidden size, num layers, dropout. Lazy-loaded. Early stopping on validation loss.

**Step 3: Register in default_registry()**

Add to `registry.py`:
```python
from mlrl_os.models.algorithms.lstm import LSTMAlgorithm
registry.register("lstm", LSTMAlgorithm)
```

Add `"lstm"` to `KNOWN_ALGORITHMS` in `gate.py`.

**Step 4: Commit**

```bash
git commit -m "feat(models): add LSTM algorithm for time-series via registry"
```

---

## Phase 5: Streaming Inference

### Task 5.1: WebSocket Prediction Endpoints

**Files:**
- Create: `src/mlrl_os/streaming/__init__.py`
- Create: `src/mlrl_os/streaming/ws_inference.py`
- Modify: `src/mlrl_os/api/app.py` — mount WebSocket routes
- Test: `tests/unit/streaming/test_ws_inference.py`

**Step 1: Write the failing test**

```python
# tests/unit/streaming/test_ws_inference.py
import pytest
from httpx import AsyncClient, ASGITransport
from starlette.testclient import TestClient
from mlrl_os.api.app import create_app

@pytest.fixture
def app(tmp_path):
    from mlrl_os.config.defaults import MLRLSettings
    settings = MLRLSettings(
        data_dir=str(tmp_path / "data"),
        models_dir=str(tmp_path / "models"),
        experiments_dir=str(tmp_path / "experiments"),
    )
    return create_app(settings)

def test_ws_predict_model_not_found(app):
    client = TestClient(app)
    with client.websocket_connect("/ws/v1/predict/nonexistent") as ws:
        # Should receive error or close
        pass  # Expect WebSocket close with error
```

**Step 2: Implement WebSocket endpoints**

Two endpoints:
- `/ws/v1/predict/{model_id}` — ML model batch predictions
- `/ws/v1/policy/{policy_id}` — RL policy action selection

**Step 3: Commit**

```bash
git commit -m "feat(streaming): add WebSocket endpoints for ML predictions and RL policy actions"
```

---

## Phase 6: HTML Report Export

### Task 6.1: Standalone HTML Report Generator

**Files:**
- Create: `src/mlrl_os/evaluation/html_export.py`
- Test: `tests/unit/evaluation/test_html_export.py`

**Step 1: Write the failing test**

```python
# tests/unit/evaluation/test_html_export.py
from pathlib import Path
from mlrl_os.evaluation.html_export import export_html_report
from mlrl_os.evaluation.reports import EvaluationReport
from mlrl_os.core.experiment import ExperimentResult, AlgorithmScore
from mlrl_os.core.types import ExperimentStatus, ProblemType

def test_html_export_creates_file(tmp_path):
    result = ExperimentResult(
        experiment_id="test123",
        name="test_experiment",
        status=ExperimentStatus.COMPLETED,
        experiment_type=ProblemType.ENTITY_CLASSIFICATION,
        created_at="2026-03-10T00:00:00",
        best_algorithm="lightgbm",
        metrics={"f1_weighted": 0.95, "auc_roc": 0.98},
        all_algorithm_scores=[
            AlgorithmScore(algorithm="lightgbm", metrics={"f1_weighted": 0.95}),
        ],
    )
    output = tmp_path / "report.html"
    path = export_html_report(result, output)
    assert path.exists()
    html = path.read_text()
    assert "<html" in html
    assert "test_experiment" in html
    assert "0.95" in html
```

**Step 2: Implement HTML export**

Single self-contained HTML file with inline CSS. Sections: summary, metrics table, algorithm comparison, feature importance (if available). Plotly.js CDN for charts (lazy script tag).

**Step 3: Commit**

```bash
git commit -m "feat(evaluation): add standalone HTML report export"
```

---

## Phase 7: Integration & Smoke Tests

### Task 7.1: Update pyproject.toml Dependencies

**Files:**
- Modify: `pyproject.toml` — add torch, websockets, asyncpg to optional deps

```toml
[project.optional-dependencies]
rl = ["torch>=2.0", "websockets>=12.0"]
postgres = ["asyncpg>=0.29"]
all = ["lightgbm>=4.0", "xgboost>=2.0", "torch>=2.0", "websockets>=12.0", "asyncpg>=0.29"]
dev = [
    # existing dev deps...
    "torch>=2.0",
    "websockets>=12.0",
    "asyncpg>=0.29",
]
```

**Step 1: Update and install**

```bash
pip install -e ".[dev,all]"
```

**Step 2: Commit**

```bash
git commit -m "chore: update dependencies for v0.2 (torch, websockets, asyncpg)"
```

---

### Task 7.2: Run All Existing Tests (Regression Check)

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: All 416+ existing tests PASS. No regressions.

**Step 2: Run type checks**

```bash
mypy src/mlrl_os/ --ignore-missing-imports
ruff check src/ tests/
```

Expected: Clean (or document known issues with new torch types).

---

### Task 7.3: Smoke Test — healthcare_er Template (Entity Classification + RL)

**Requires: SimOS running on localhost:8000**

**Step 1: Run SimOS healthcare_er simulation and export**

```bash
# Get template
curl -s http://localhost:8000/api/v1/templates/healthcare_er \
  -H "X-API-Key: sk-premium-test-003" -H "X-SimOS-Client: web" > /tmp/healthcare_config.json

# Submit simulation
JOB_ID=$(curl -s -X POST http://localhost:8000/api/v1/simulations \
  -H "X-API-Key: sk-premium-test-003" -H "X-SimOS-Client: web" \
  -H "Content-Type: application/json" -d @/tmp/healthcare_config.json | python -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

# Poll until completed
curl -s http://localhost:8000/api/v1/simulations/$JOB_ID \
  -H "X-API-Key: sk-premium-test-003" -H "X-SimOS-Client: web"

# Export ML data
curl -s -X POST "http://localhost:8000/api/v1/simulations/$JOB_ID/export-ml?bucket_seconds=60" \
  -H "X-API-Key: sk-premium-test-003" -H "X-SimOS-Client: web" > tests/fixtures/healthcare_er_export.json
```

**Step 2: Run supervised learning experiment via CLI**

```bash
mlrl-os datasets import tests/fixtures/healthcare_er_export.json --name healthcare_er
mlrl-os run - <<'YAML'
dataset_id: <dataset_id from import>
target: delay_severity
algorithms: [lightgbm, random_forest, lstm]
hyperparameter_tuning: true
n_trials: 5
YAML
```

**Step 3: Run RL experiment (if SimOS WebSocket available)**

```bash
# Via Python script or API call
python -c "
import asyncio
from mlrl_os.rl.runner import RLExperimentRunner
config = {
    'name': 'healthcare_er_dqn',
    'experiment_type': 'reinforcement_learning',
    'seed': 42,
    'simos_url': 'ws://localhost:8000',
    'template': 'healthcare_er',
    'algorithm': 'dqn',
    'max_episodes': 10,
    'batch_size': 32,
    'eval_episodes': 5,
    'reward': {'function': 'sla_compliance'},
}
runner = RLExperimentRunner()
result = asyncio.run(runner.run(config))
print(f'Status: {result.status}')
print(f'Metrics: {result.metrics}')
"
```

**Step 4: Document findings in smoke test report**

---

### Task 7.4: Smoke Test — logistics_otd Template

**Step 1: Same flow as 7.3 but with logistics_otd template**

**Step 2: Test with both `delay_severity` and `sla_breach` targets**

**Step 3: Compare results across templates**

---

### Task 7.5: Write Smoke Test Report

**Files:**
- Create: `docs/reports/2026-03-10-v02-smoke-test-report.md`

**Content:**
1. Test environment (SimOS version, ML/RL OS version, hardware)
2. Template results table (healthcare_er, logistics_otd)
3. Supervised learning metrics (with and without tuning)
4. RL training curves and metrics
5. LSTM vs. gradient boosting comparison
6. PostgreSQL backend test results
7. Streaming inference latency measurements
8. Findings and issues discovered
9. Enhancement proposals (including SimOS data extraction improvements)
10. Recommendations for v0.3

**Step 1: Write report with all findings**

**Step 2: Commit report**

```bash
git commit -m "docs: v0.2 smoke test report with findings and enhancement proposals"
```

---

## Summary: Task Count

| Phase | Tasks | Estimated Tests |
|---|---|---|
| Phase 1: Storage Backend | 4 tasks | ~20 tests |
| Phase 2: Hyperparameter Tuning | 1 task | ~5 tests |
| Phase 3: RL Engine | 11 tasks | ~50 tests |
| Phase 4: LSTM Model | 1 task | ~5 tests |
| Phase 5: Streaming Inference | 1 task | ~10 tests |
| Phase 6: HTML Reports | 1 task | ~5 tests |
| Phase 7: Integration & Smoke | 5 tasks | integration tests |
| **Total** | **24 tasks** | **~95 new tests** |

Target: v0.1's 416 tests + ~95 new = **~510+ total tests**, all green.

## Build Order (Dependency-Safe)

```
Phase 1 (Storage) ──────┐
Phase 2 (Tuning) ───────┤
Phase 4 (LSTM) ─────────┤── Phase 7 (Smoke Tests)
Phase 6 (HTML Reports) ─┤
Phase 3 (RL Engine):     │
  3.1 Spaces/Rewards ────┤
  3.2 Networks ──────────┤
  3.3 Replay Buffer ─────┤
  3.4 DQN ───────────────┤
  3.5 PPO ───────────────┤
  3.6 SimOS Client ──────┤
  3.7 Environment ───────┤
  3.8 Curriculum ────────┤
  3.9 RL Config ─────────┤
  3.10 RL Runner ────────┤
  3.11 RL Routes ────────┤
Phase 5 (Streaming) ────┘
```

Phases 1, 2, 4, 6 are independent of each other. Phase 3 tasks are sequential (each builds on previous). Phase 5 depends on model registry (Phase 1) and RL policy registry (Phase 3). Phase 7 depends on everything.
