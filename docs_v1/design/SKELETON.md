# SKELETON

> File-by-file implementation guide with dependencies, responsibilities, and key functions.
>
> Implementation-ready: a new Claude Code session reads this and knows exactly what to build.
>
> Last updated: 2026-03-09.

---

## Build Order (dependency-safe)

Files are numbered. A file may only depend on files with lower numbers.

---

### 1. `src/mlrl_os/__init__.py`

```python
"""ML/RL OS — Predictive Intelligence Instrument."""
__version__ = "0.1.0"
```

Dependencies: none.

---

### 2. `src/mlrl_os/core/types.py`

**Purpose:** All shared types, enums, and base models used across modules.

```python
from enum import Enum
from pydantic import BaseModel
import numpy as np

class ProblemType(str, Enum):
    TIME_SERIES = "time_series"
    ENTITY_CLASSIFICATION = "entity_classification"

class TaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"

class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ObservationPoint(str, Enum):
    ALL_STEPS = "all_steps"
    ENTRY_ONLY = "entry_only"
    MIDPOINT = "midpoint"

class CVStrategy(str, Enum):
    TEMPORAL = "temporal"
    STRATIFIED_KFOLD = "stratified_kfold"
    KFOLD = "kfold"

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    null_count: int
    null_rate: float
    unique_count: int
    is_numeric: bool
    is_categorical: bool
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    categories: list[str] | None = None
    category_counts: dict[str, int] | None = None

class TargetInfo(BaseModel):
    column: str
    task_type: TaskType
    is_default: bool
    null_rate: float
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    unique_count: int | None = None
    classes: list[str] | None = None
    class_balance: dict[str, float] | None = None

class AvailableTargets(BaseModel):
    dataset_id: str
    time_series_targets: list[TargetInfo]
    entity_targets: list[TargetInfo]

class FeatureMatrix:
    """Model-ready data. Not a Pydantic model (contains numpy arrays)."""
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        problem_type: ProblemType,
        target_name: str,
        task_type: TaskType,
        temporal_index: np.ndarray | None = None,
        entity_ids: np.ndarray | None = None,
        class_names: list[str] | None = None,
    ): ...

    @property
    def sample_count(self) -> int: return self.X.shape[0]

    @property
    def feature_count(self) -> int: return self.X.shape[1]
```

Dependencies: none.
Tests: `tests/unit/core/test_types.py` — enum values, ColumnInfo construction, FeatureMatrix shapes.

---

### 3. `src/mlrl_os/experiment/seed.py`

**Purpose:** Deterministic seeded RNG. Same pattern as SimOS.

```python
import hashlib
import random

def seed_hash(name: str, global_seed: int) -> int:
    """Deterministic seed for a named component."""
    return int.from_bytes(
        hashlib.sha256(f"{name}:{global_seed}".encode()).digest()[:8],
        byteorder="big",
    )

def make_rng(name: str, global_seed: int) -> random.Random:
    """Create a seeded Random instance for a named component."""
    return random.Random(seed_hash(name, global_seed))
```

Dependencies: none.
Tests: `tests/unit/experiment/test_seed.py` — determinism, independence, collision resistance.

---

### 4. `src/mlrl_os/core/dataset.py`

**Purpose:** Dataset-related models.

```python
from pydantic import BaseModel
from mlrl_os.core.types import ColumnInfo

class DatasetMeta(BaseModel):
    id: str
    name: str
    version: int
    content_hash: str
    source_type: str                       # "simos" | "csv" | "parquet"
    source_path: str
    snapshot_row_count: int | None = None
    trajectory_row_count: int | None = None
    snapshot_column_count: int | None = None
    trajectory_column_count: int | None = None
    snapshot_columns: list[ColumnInfo] | None = None
    trajectory_columns: list[ColumnInfo] | None = None
    has_snapshots: bool
    has_trajectories: bool
    registered_at: str

class RawDataset:
    """Loaded data before registration. Holds Polars DataFrames."""
    def __init__(
        self,
        source_type: str,
        source_path: str,
        snapshots: "pl.DataFrame | None" = None,
        trajectories: "pl.DataFrame | None" = None,
        metadata_dict: dict | None = None,
        summary_dict: dict | None = None,
    ): ...
```

Dependencies: types.py (#2).
Tests: `tests/unit/core/test_dataset.py` — construction, validation.

---

### 5. `src/mlrl_os/core/experiment.py`

**Purpose:** Experiment result models.

```python
class AlgorithmScore(BaseModel):
    algorithm: str
    metrics: dict[str, float]
    metrics_std: dict[str, float]
    rank: int

class FeatureImportanceEntry(BaseModel):
    feature: str
    importance: float
    rank: int

class ExperimentResult(BaseModel):
    experiment_id: str
    name: str
    status: ExperimentStatus
    experiment_type: ProblemType
    created_at: str
    completed_at: str | None = None
    duration_seconds: float | None = None
    best_algorithm: str | None = None
    metrics: dict[str, float] | None = None
    all_algorithm_scores: list[AlgorithmScore] | None = None
    feature_importance: list[FeatureImportanceEntry] | None = None
    model_id: str | None = None
    sample_count: int | None = None
    feature_count: int | None = None
    resolved_config: dict | None = None
```

Dependencies: types.py (#2).

---

### 6. `src/mlrl_os/data/simos_loader.py`

**Purpose:** Load SimOS 5-layer ML export JSON. Extracts Layer 2 and Layer 3 into Polars DataFrames.

**Key design: Schema Adapter pattern for decoupling (see §6.1 below).**

```python
class SimosSchemaAdapter:
    """Maps SimOS export field names to ML/RL OS canonical names.

    This is the ONLY place that knows SimOS field names.
    If SimOS changes its export schema, update ONLY this class.
    """

    # Canonical snapshot columns (ML/RL OS internal names)
    # mapped from SimOS export names
    SNAPSHOT_MAPPING: dict[str, str]       # simos_name → canonical_name
    TRAJECTORY_STATE_MAPPING: dict[str, str]
    METADATA_MAPPING: dict[str, str]

    def map_snapshot(self, raw: dict) -> dict: ...
    def map_trajectory(self, raw: dict) -> dict: ...
    def map_metadata(self, raw: dict) -> dict: ...
    def detect_dynamic_columns(self, raw: dict) -> DynamicColumns: ...

class SimosLoader:
    """Load SimOS export JSON. Uses SchemaAdapter for field mapping."""

    def __init__(self, adapter: SimosSchemaAdapter | None = None): ...

    def load(self, path: Path) -> RawDataset:
        """Load SimOS JSON, extract snapshots and trajectories."""

    def extract_snapshots(self, raw_snapshots: list[dict]) -> pl.DataFrame:
        """Extract Layer 3 snapshots into Polars DataFrame.
        Maps SimOS field names to canonical names via adapter.
        """

    def extract_trajectories(self, raw_trajectories: list[dict]) -> pl.DataFrame:
        """Extract Layer 2 trajectories into Polars DataFrame.
        Flattens nested state/reward/next_state dicts.
        Maps field names to canonical names via adapter.
        """

    def _parse_metadata(self, raw: dict) -> dict: ...
    def _parse_summary(self, raw: dict) -> dict: ...
```

Dependencies: dataset.py (#4), types.py (#2).
Tests: `tests/unit/data/test_simos_loader.py` — load fixture, verify columns, verify values.

---

### 7. `src/mlrl_os/data/external_loader.py`

**Purpose:** Load CSV and Parquet files.

```python
class ExternalLoader:
    def load_csv(self, path: Path, **kwargs) -> RawDataset: ...
    def load_parquet(self, path: Path) -> RawDataset: ...
    def _detect_data_type(self, df: pl.DataFrame) -> str:
        """Detect if data is time-series or entity-like."""
```

Dependencies: dataset.py (#4).
Tests: `tests/unit/data/test_external_loader.py`.

---

### 8. `src/mlrl_os/data/validation.py`

**Purpose:** Data quality checks on loaded datasets.

```python
class DataQualityReport(BaseModel):
    row_count: int
    column_count: int
    missing_rates: dict[str, float]
    constant_columns: list[str]
    duplicate_count: int
    quality: str                           # "insufficient" | "minimal" | "good" | "excellent"
    issues: list[str]
    warnings: list[str]

def validate_data_quality(df: pl.DataFrame) -> DataQualityReport: ...
```

Dependencies: types.py (#2).
Tests: `tests/unit/data/test_validation.py`.

---

### 9. `src/mlrl_os/data/registry.py`

**Purpose:** Dataset versioning and storage.

```python
class DatasetRegistry:
    def __init__(self, data_dir: Path): ...
    def register(self, raw: RawDataset, name: str | None = None) -> DatasetMeta: ...
    def get(self, dataset_id: str) -> DatasetMeta: ...
    def list(self) -> list[DatasetMeta]: ...
    def load_snapshots(self, dataset_id: str) -> pl.DataFrame: ...
    def load_trajectories(self, dataset_id: str) -> pl.DataFrame: ...
    def _compute_content_hash(self, raw: RawDataset) -> str: ...
    def _generate_id(self) -> str: ...
```

Dependencies: dataset.py (#4), validation.py (#8).
Tests: `tests/unit/data/test_registry.py` — register, retrieve, list, content hash.

---

### 10. `src/mlrl_os/data/discovery.py`

**Purpose:** Auto-discover available features and targets from dataset.

```python
class TargetDiscovery:
    def discover(self, dataset_meta: DatasetMeta, df: pl.DataFrame | None = None) -> AvailableTargets: ...
    def _discover_ts_targets(self, columns: list[ColumnInfo]) -> list[TargetInfo]: ...
    def _discover_entity_targets(self, columns: list[ColumnInfo], df: pl.DataFrame) -> list[TargetInfo]: ...
    def _derive_sla_breach(self, df: pl.DataFrame) -> TargetInfo | None: ...
    def _derive_delay_severity(self, df: pl.DataFrame) -> TargetInfo | None: ...
```

Dependencies: types.py (#2), dataset.py (#4).
Tests: `tests/unit/data/test_discovery.py` — discover from SimOS data, discover from external.

---

### 11. `src/mlrl_os/features/detection.py`

**Purpose:** Auto-detect problem type.

```python
class ProblemTypeDetector:
    def detect(self, dataset_meta: DatasetMeta, user_config: dict | None = None) -> ProblemType: ...
```

Dependencies: types.py (#2), dataset.py (#4).
Tests: `tests/unit/features/test_detection.py`.

---

### 12. `src/mlrl_os/config/schemas.py`

**Purpose:** Pydantic models for all config types. See CONTRACTS.md §4 for full schema.

Dependencies: types.py (#2).
Tests: `tests/unit/config/test_schemas.py` — serialization, validation.

---

### 13. `src/mlrl_os/config/defaults.py`

**Purpose:** Default configs per problem type + BaseSettings for env vars.

```python
from pydantic_settings import BaseSettings

class MLRLSettings(BaseSettings):
    model_config = {"env_prefix": "MLRL_"}
    env: str = "development"
    data_dir: str = "./data"
    models_dir: str = "./models"
    experiments_dir: str = "./experiments"
    log_level: str = "INFO"
    api_port: int = 8001
    cors_allow_origins: str = "http://localhost:5175"
    max_training_rows: int = 1_000_000
    cv_folds_default: int = 5
    seed_default: int = 42

TS_DEFAULTS: dict = { ... }             # from CONSTITUTION.md §4
ENTITY_DEFAULTS: dict = { ... }         # from CONSTITUTION.md §4
```

Dependencies: none.
Tests: `tests/unit/config/test_defaults.py`.

---

### 14. `src/mlrl_os/config/resolver.py`

**Purpose:** Merge user config + defaults → resolved config.

```python
class ConfigResolver:
    def resolve(self, user_config: dict, dataset_meta: DatasetMeta) -> ResolvedExperimentConfig: ...
    def _detect_or_use_type(self, user_config: dict, dataset_meta: DatasetMeta) -> ProblemType: ...
    def _merge(self, defaults: dict, overrides: dict) -> dict: ...
    def _resolve_feature_columns(self, dataset_meta: DatasetMeta, config: dict) -> list[str]: ...
```

Dependencies: schemas.py (#12), defaults.py (#13), detection.py (#11), discovery.py (#10).
Tests: `tests/unit/config/test_resolver.py` — zero config, partial override, full override, conflict.

---

### 15. `src/mlrl_os/validation/gate.py`

**Purpose:** All validation rules. See CONSTITUTION.md §5.

```python
class ValidationError(BaseModel):
    code: str
    field: str
    message: str
    suggestion: str | None = None

class ValidationResult(BaseModel):
    valid: bool
    errors: list[ValidationError]
    warnings: list[str]

class ValidationGate:
    def validate(self, config: ResolvedExperimentConfig, df: pl.DataFrame) -> ValidationResult: ...
    def _universal_rules(self, config, df) -> list[ValidationError]: ...
    def _time_series_rules(self, config, df) -> list[ValidationError]: ...
    def _entity_rules(self, config, df) -> list[ValidationError]: ...
```

Dependencies: schemas.py (#12), types.py (#2).
Tests: `tests/unit/validation/test_gate.py` — one test per rule (V-01..VE-04).

---

### 16. `src/mlrl_os/features/time_series.py`

**Purpose:** Time-series feature engineering. See CONTRACTS.md §3.1.

Dependencies: types.py (#2), schemas.py (#12).
Tests: `tests/unit/features/test_time_series.py` — lag, rolling, trend, ratio, windowing.

---

### 17. `src/mlrl_os/features/entity.py`

**Purpose:** Entity feature engineering. See CONTRACTS.md §3.2.

Dependencies: types.py (#2), schemas.py (#12).
Tests: `tests/unit/features/test_entity.py` — state flattening, derived features, observation points.

---

### 18. `src/mlrl_os/features/store.py`

**Purpose:** Feature definition registry for reuse.

Dependencies: types.py (#2).
Tests: `tests/unit/features/test_store.py`.

---

### 19. `src/mlrl_os/models/algorithms/protocol.py`

**Purpose:** Algorithm protocol (interface). See BLUEPRINT.md §2.5.1.

Dependencies: types.py (#2).
Tests: `tests/unit/models/test_protocol.py` — protocol compliance.

---

### 20. `src/mlrl_os/models/algorithms/registry.py`

**Purpose:** Algorithm registry. Lazy-loads algorithm implementations.

Dependencies: protocol.py (#19).
Tests: `tests/unit/models/test_algorithm_registry.py`.

---

### 21-23. Algorithm wrappers

- `#21 lightgbm.py` — LightGBM wrapper (see CONTRACTS.md §6.1)
- `#22 xgboost.py` — XGBoost wrapper (see CONTRACTS.md §6.2)
- `#23 random_forest.py` — sklearn RF wrapper (see CONTRACTS.md §6.3)

Dependencies: protocol.py (#19).
Tests: `tests/unit/models/test_algorithms/test_lightgbm.py`, etc.

---

### 24. `src/mlrl_os/models/engine.py`

**Purpose:** Train, cross-validate, select best model. See BLUEPRINT.md §2.5.3.

Dependencies: protocol.py (#19), registry.py (#20), types.py (#2), seed.py (#3).
Tests: `tests/unit/models/test_engine.py` — training, CV splits, model selection.

---

### 25. `src/mlrl_os/models/registry.py`

**Purpose:** Versioned model storage.

Dependencies: types.py (#2), experiment.py (#5).
Tests: `tests/unit/models/test_model_registry.py`.

---

### 26. `src/mlrl_os/evaluation/metrics.py`

**Purpose:** Metric registry and computation. See BLUEPRINT.md §2.6.

Dependencies: none (uses sklearn.metrics).
Tests: `tests/unit/evaluation/test_metrics.py`.

---

### 27. `src/mlrl_os/evaluation/reports.py`

**Purpose:** Generate evaluation reports. See CONTRACTS.md §7.

Dependencies: metrics.py (#26), experiment.py (#5).
Tests: `tests/unit/evaluation/test_reports.py`.

---

### 28. `src/mlrl_os/evaluation/comparison.py`

**Purpose:** Compare multiple experiment results.

Dependencies: experiment.py (#5).
Tests: `tests/unit/evaluation/test_comparison.py`.

---

### 29. `src/mlrl_os/experiment/tracker.py`

**Purpose:** Experiment history storage.

Dependencies: experiment.py (#5).
Tests: `tests/unit/experiment/test_tracker.py`.

---

### 30. `src/mlrl_os/experiment/runner.py`

**Purpose:** Full pipeline orchestrator. The main entry point.

```python
class ExperimentRunner:
    def __init__(
        self,
        dataset_registry: DatasetRegistry,
        model_registry: ModelRegistry,
        experiment_tracker: ExperimentTracker,
        settings: MLRLSettings | None = None,
    ): ...

    def run(self, user_config: dict) -> ExperimentResult:
        """Full pipeline:
        1. Load dataset from registry
        2. Resolve config (merge defaults)
        3. Validate (gate check) → raise if invalid
        4. Build feature matrix (TS or Entity engine)
        5. Train and evaluate (Model Engine)
        6. Generate report
        7. Register model
        8. Store experiment
        9. Return result
        """

    def validate_only(self, user_config: dict) -> ValidationResult:
        """Validate without running."""

    def run_from_yaml(self, path: Path) -> ExperimentResult: ...
```

Dependencies: ALL previous modules.
Tests: `tests/unit/experiment/test_runner.py`, `tests/integration/test_end_to_end_*.py`.

---

### 31+. API and Web UI (Phase 2 & 3)

Built on top of the core engine. See BLUEPRINT.md §2.8 for API routes.

---

## Test Fixture Specification

### `tests/fixtures/simos_export_small.json`

Minimal SimOS export for unit tests. Hand-crafted with known values.

```
- 10 entities (5 completed, 2 rejected, 3 timed_out)
- 3 nodes: reception, processing, dispatch
- 1 resource: workers (capacity 3)
- Duration: 3600s (1 hour)
- Bucket: 60s → 60 snapshots
- ~30 trajectory records (10 entities × 3 steps avg)
- Domain: generic
```

### `tests/fixtures/simos_export_supply_chain.json`

Medium SimOS export for integration tests.

```
- 100 entities (85 completed, 5 rejected, 10 timed_out)
- 5 nodes: sourcing, manufacturing, assembly, quality, shipping
- 2 resources: machines, operators
- Duration: 86400s (24 hours)
- Bucket: 60s → 1440 snapshots
- ~400 trajectory records
- Domain: supply_chain
```

Both fixtures must be deterministic (seed=42) with hand-verified values for key metrics.
