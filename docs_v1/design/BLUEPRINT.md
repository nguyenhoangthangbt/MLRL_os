# BLUEPRINT

> Full-stack architecture, module specifications, and interfaces for ML/RL OS v0.1.
>
> Inherits all terminology and principles from CONSTITUTION.md.
>
> Last updated: 2026-03-09.

---

## 1. System Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           ML/RL OS v0.1                                  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │  Web UI (React 19 + Vite)                              Port 5175  │   │
│  │                                                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐  │   │
│  │  │ Experiment   │  │ Data         │  │ Results                 │  │   │
│  │  │ Builder      │  │ Explorer     │  │ Dashboard               │  │   │
│  │  │              │  │              │  │                         │  │   │
│  │  │ 7-step flow  │  │ columns      │  │ metrics, charts,        │  │   │
│  │  │ → YAML       │  │ stats        │  │ feature importance,     │  │   │
│  │  │ → validate   │  │ quality      │  │ model comparison        │  │   │
│  │  │ → submit     │  │ preview      │  │                         │  │   │
│  │  └──────────────┘  └──────────────┘  └─────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                              │ HTTP (Axios)                              │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │  Backend API (FastAPI + Pydantic v2)                    Port 8001  │   │
│  │                                                                   │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │   │
│  │  │ Data     │  │Experiment│  │ Model    │  │ Evaluation       │  │   │
│  │  │ Routes   │  │ Routes   │  │ Routes   │  │ Routes           │  │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────────────┘  │   │
│  │       │              │             │              │               │   │
│  │  ┌────┴──────────────┴─────────────┴──────────────┴────────────┐  │   │
│  │  │                    Core Engine                               │  │   │
│  │  │                                                              │  │   │
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │  │   │
│  │  │  │ Dataset  │ │ Feature  │ │ Model    │ │ Evaluation    │  │  │   │
│  │  │  │ Engine   │ │ Engine   │ │ Engine   │ │ Engine        │  │  │   │
│  │  │  └──────────┘ └──────────┘ └──────────┘ └───────────────┘  │  │   │
│  │  │                                                              │  │   │
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │  │   │
│  │  │  │ Config   │ │ Dataset  │ │ Model    │ │ Experiment    │  │  │   │
│  │  │  │ Resolver │ │ Registry │ │ Registry │ │ Tracker       │  │  │   │
│  │  │  └──────────┘ └──────────┘ └──────────┘ └───────────────┘  │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │  Storage: File System (v0.1) │ PostgreSQL (v0.2)                   │   │
│  │  Datasets: ./data/           │ Models: ./models/                   │   │
│  │  Experiments: ./experiments/ │ Reports: ./reports/                 │   │
│  └────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Specifications

### 2.1 Data Ingestion Module (`src/mlrl_os/data/`)

Responsible for loading, validating, and registering datasets.

#### 2.1.1 SimOS Loader

```python
class SimosLoader:
    """Load SimOS 5-layer ML export format."""

    def load(self, path: Path) -> RawDataset:
        """Load SimOS JSON export.

        Extracts:
          - metadata (seed, duration, domain, entity types, node names)
          - Layer 2 trajectories → entity records
          - Layer 3 snapshots → time-series records
          - Layer 4 domain enrichment → additional entity features

        Returns:
            RawDataset with detected layers and metadata.
        """

    def extract_snapshots(self, raw: dict) -> pl.DataFrame:
        """Extract Layer 3 snapshots into a Polars DataFrame.

        Columns: timestamp, sys_utilization, sys_wip, sys_throughput_per_hour,
                 sys_avg_wait_time, sys_avg_lead_time, sys_sla_compliance,
                 node_*_queue, node_*_utilization, resource_*_utilization, ...
        """

    def extract_trajectories(self, raw: dict) -> pl.DataFrame:
        """Extract Layer 2 trajectories into a Polars DataFrame.

        Columns: entity_id, entity_type, step_index, node_name,
                 state.* (flattened), reward.* (flattened),
                 episode_done, episode_status, episode_total_time, ...
        """
```

#### 2.1.2 External Loader

```python
class ExternalLoader:
    """Load CSV and Parquet files."""

    def load_csv(self, path: Path, **kwargs) -> RawDataset:
        """Load CSV with auto-type detection."""

    def load_parquet(self, path: Path) -> RawDataset:
        """Load Parquet file."""
```

#### 2.1.3 Dataset Registry

```python
class DatasetRegistry:
    """Versioned dataset storage."""

    def register(self, raw: RawDataset, name: str) -> DatasetMeta:
        """Register dataset. Computes content hash for versioning.

        Returns:
            DatasetMeta with id, name, version, content_hash,
            row_count, column_count, schema, registered_at.
        """

    def get(self, dataset_id: str) -> DatasetMeta: ...
    def list(self) -> list[DatasetMeta]: ...
    def load_data(self, dataset_id: str) -> pl.DataFrame: ...
```

#### 2.1.4 Target Discovery

```python
class TargetDiscovery:
    """Auto-discover available targets from dataset."""

    def discover(self, df: pl.DataFrame, dataset_meta: DatasetMeta) -> AvailableTargets:
        """Inspect dataset and return available targets.

        Returns:
            AvailableTargets with:
              - time_series_targets: list of numeric columns suitable for TS forecasting
              - entity_targets: list of categorical columns suitable for classification
              - metadata per target: type, unique_values (if categorical),
                distribution summary (if numeric), null_rate
        """
```

---

### 2.2 Feature Engine (`src/mlrl_os/features/`)

Responsible for transforming raw data into model-ready feature matrices.

#### 2.2.1 Problem Type Detection

```python
class ProblemTypeDetector:
    """Auto-detect problem type from dataset and user config."""

    def detect(self, dataset: DatasetMeta, user_config: dict | None) -> ProblemType:
        """Detection logic:

        1. If user explicitly specifies type → use it
        2. If dataset has 'snapshots' layer → TIME_SERIES
        3. If dataset has 'trajectories' layer → ENTITY_CLASSIFICATION
        4. If external data with timestamp column → TIME_SERIES
        5. If external data with categorical target → ENTITY_CLASSIFICATION
        6. Else → raise ambiguous error, ask user to specify
        """
```

#### 2.2.2 Time-Series Feature Engineering

```python
class TimeSeriesFeatureEngine:
    """Transform snapshot time-series into supervised learning features."""

    def build_features(
        self,
        df: pl.DataFrame,
        config: ResolvedTimeSeriesConfig,
    ) -> FeatureMatrix:
        """Apply windowing and feature engineering.

        Pipeline:
          1. Sort by timestamp
          2. Generate lag features for each target-relevant column
          3. Generate rolling statistics (mean, std) over specified windows
          4. Generate trend features (linear slope over window)
          5. Generate ratio features (e.g., wip / capacity)
          6. Generate cross-node features (max_queue / mean_queue)
          7. Create target column (value at t + horizon)
          8. Drop rows with insufficient history (first lookback rows)

        Returns:
            FeatureMatrix with X (features), y (target), feature_names,
            temporal_index (for temporal CV).
        """

    def _lag_features(self, df: pl.DataFrame, columns: list[str],
                      intervals: list[int]) -> pl.DataFrame: ...
    def _rolling_features(self, df: pl.DataFrame, columns: list[str],
                          windows: list[int]) -> pl.DataFrame: ...
    def _trend_features(self, df: pl.DataFrame, columns: list[str],
                        window: int) -> pl.DataFrame: ...
    def _ratio_features(self, df: pl.DataFrame) -> pl.DataFrame: ...
```

#### 2.2.3 Entity Feature Engineering

```python
class EntityFeatureEngine:
    """Transform entity trajectory records into classification features."""

    def build_features(
        self,
        df: pl.DataFrame,
        config: ResolvedEntityConfig,
    ) -> FeatureMatrix:
        """Extract features from entity trajectory state vectors.

        Pipeline:
          1. Flatten state dict columns into individual feature columns
          2. Include/exclude feature groups per config:
             - entity_state (10 features)
             - node_state (9 features)
             - system_state (5 features)
             - resource_state (2 per resource)
             - domain_enrichment (variable)
          3. Engineer derived features:
             - progress_ratio = steps_completed / total_steps (if known)
             - wait_trend = cumulative_wait / elapsed_time slope
             - queue_relative = node_queue / mean_queue_across_nodes
          4. Filter by observation_point:
             - all_steps: every trajectory row
             - entry_only: first step per entity
             - midpoint: middle step per entity
          5. Extract target column

        Returns:
            FeatureMatrix with X, y, feature_names, entity_ids.
        """
```

#### 2.2.4 Feature Store

```python
class FeatureStore:
    """Registry of reusable feature definitions."""

    def register_definition(self, name: str, definition: FeatureDefinition) -> str:
        """Register a feature transformation for reuse.

        A FeatureDefinition includes:
          - source_column: str
          - transform_type: lag | rolling_mean | rolling_std | trend | ratio | derived
          - parameters: dict (window, interval, etc.)
        """

    def get_definition(self, name: str) -> FeatureDefinition: ...
    def list_definitions(self) -> list[FeatureDefinition]: ...
    def apply(self, df: pl.DataFrame, names: list[str]) -> pl.DataFrame: ...
```

---

### 2.3 Config Resolution Module (`src/mlrl_os/config/`)

Responsible for merging user config with defaults into a fully resolved config.

```python
class ConfigResolver:
    """Merge user config + defaults → resolved config."""

    def resolve(
        self,
        user_config: dict,
        dataset: DatasetMeta,
    ) -> ResolvedExperimentConfig:
        """Resolution pipeline:

        1. Detect problem type (from data or user config)
        2. Load defaults for detected problem type
        3. Auto-discover available targets and features
        4. Merge user overrides onto defaults (user wins)
        5. Return fully resolved config

        The resolved config has NO optional fields —
        every setting has a concrete value.
        """

    def _merge(self, defaults: dict, overrides: dict) -> dict:
        """Deep merge. User values win. Unknown keys raise error."""
```

#### Resolved Config Schema

```python
class ResolvedExperimentConfig(BaseModel):
    """Fully resolved experiment configuration. No optional fields."""

    # Experiment identity
    name: str
    experiment_type: ProblemType        # time_series | entity_classification
    seed: int

    # Dataset reference
    dataset_id: str
    dataset_layer: str                  # snapshots | trajectories

    # Feature configuration (type-specific)
    features: ResolvedTimeSeriesFeatures | ResolvedEntityFeatures

    # Model configuration
    model: ResolvedModelConfig

    # Evaluation configuration
    evaluation: ResolvedEvaluationConfig


class ResolvedTimeSeriesFeatures(BaseModel):
    target: str
    lookback: str                       # e.g., "8h"
    horizon: str                        # e.g., "1h"
    lag_intervals: list[str]
    rolling_windows: list[str]
    include_trend: bool
    include_ratios: bool
    include_cross_node: bool
    feature_columns: list[str]          # resolved list of input columns


class ResolvedEntityFeatures(BaseModel):
    target: str
    observation_point: str              # all_steps | entry_only | midpoint
    include_entity_state: bool
    include_node_state: bool
    include_system_state: bool
    add_progress_ratio: bool
    add_wait_trend: bool
    feature_columns: list[str]          # resolved list of input columns


class ResolvedModelConfig(BaseModel):
    algorithms: list[str]
    selection: str                      # best_cv
    cross_validation: CVConfig
    handle_imbalance: bool
    hyperparameter_tuning: bool


class CVConfig(BaseModel):
    strategy: str                       # temporal | stratified_kfold | kfold
    folds: int


class ResolvedEvaluationConfig(BaseModel):
    metrics: list[str]
    generate_report: bool
    plot_predictions: bool
    plot_feature_importance: bool
    plot_confusion_matrix: bool
    plot_roc_curve: bool
```

---

### 2.4 Validation Gate (`src/mlrl_os/validation/`)

```python
class ValidationGate:
    """Validate resolved experiment config before training."""

    def validate(
        self,
        config: ResolvedExperimentConfig,
        dataset: pl.DataFrame,
    ) -> ValidationResult:
        """Run all validation rules.

        Returns:
            ValidationResult with:
              - valid: bool
              - errors: list[ValidationError]  (code, message, field)
              - warnings: list[ValidationWarning]

        All errors are collected — not just the first.
        """

    def _run_universal_rules(self, config, dataset) -> list[ValidationError]: ...
    def _run_time_series_rules(self, config, dataset) -> list[ValidationError]: ...
    def _run_entity_rules(self, config, dataset) -> list[ValidationError]: ...
```

---

### 2.5 Model Engine (`src/mlrl_os/models/`)

#### 2.5.1 Algorithm Protocol

```python
class Algorithm(Protocol):
    """Interface all ML algorithms must implement."""

    @property
    def name(self) -> str: ...

    @property
    def supports_regression(self) -> bool: ...

    @property
    def supports_classification(self) -> bool: ...

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str,              # "regression" | "classification"
        seed: int,
        **kwargs,
    ) -> TrainedModel: ...

    def predict(self, model: TrainedModel, X: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, model: TrainedModel, X: np.ndarray) -> np.ndarray | None:
        """Return class probabilities for classification. None if not supported."""

    def feature_importance(self, model: TrainedModel) -> dict[str, float] | None:
        """Return feature name → importance score. None if not supported."""
```

#### 2.5.2 Algorithm Registry

```python
class AlgorithmRegistry:
    """Registry of available ML algorithms. Lazy-loaded."""

    _algorithms: dict[str, type[Algorithm]]

    def register(self, name: str, algorithm_cls: type[Algorithm]) -> None: ...
    def get(self, name: str) -> Algorithm: ...
    def list_available(self) -> list[str]: ...

# Built-in registrations (v0.1):
# "lightgbm"      → LightGBMAlgorithm
# "xgboost"       → XGBoostAlgorithm
# "random_forest"  → RandomForestAlgorithm
# "extra_trees"    → ExtraTreesAlgorithm
# "linear"         → LinearAlgorithm (Ridge/LogisticRegression)
```

#### 2.5.3 Model Engine

```python
class ModelEngine:
    """Train and evaluate models."""

    def train_and_evaluate(
        self,
        feature_matrix: FeatureMatrix,
        config: ResolvedExperimentConfig,
    ) -> ExperimentResult:
        """Full training pipeline:

        1. For each algorithm in config.model.algorithms:
           a. Create cross-validation splits (temporal or stratified)
           b. For each fold:
              - Train on train split
              - Predict on test split
              - Compute metrics
           c. Aggregate fold metrics (mean ± std)
        2. Select best algorithm by primary metric
        3. Retrain best algorithm on full dataset
        4. Compute feature importance
        5. Generate evaluation report

        Returns:
            ExperimentResult with per-algorithm CV scores,
            best model, feature importance, evaluation report.
        """

    def _temporal_cv_split(
        self, X: np.ndarray, y: np.ndarray, temporal_index: np.ndarray,
        n_folds: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Time-series CV: train on past, test on future. No leakage."""

    def _stratified_cv_split(
        self, X: np.ndarray, y: np.ndarray, n_folds: int, seed: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Stratified k-fold for classification. Preserves class ratios."""
```

#### 2.5.4 Model Registry

```python
class ModelRegistry:
    """Versioned storage for trained models."""

    def register(
        self,
        model: TrainedModel,
        experiment_id: str,
        config: ResolvedExperimentConfig,
        metrics: dict[str, float],
    ) -> ModelMeta:
        """Register trained model with full provenance.

        Stores:
          - model artifact (serialized)
          - algorithm name and version
          - training dataset reference (id + content hash)
          - resolved config
          - evaluation metrics
          - feature importance
          - training timestamp
          - seed

        Returns:
            ModelMeta with model_id, version, provenance.
        """

    def get(self, model_id: str) -> tuple[TrainedModel, ModelMeta]: ...
    def list(self) -> list[ModelMeta]: ...
    def predict(self, model_id: str, X: np.ndarray) -> np.ndarray: ...
```

---

### 2.6 Evaluation Engine (`src/mlrl_os/evaluation/`)

```python
class EvaluationEngine:
    """Compute metrics and generate reports."""

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None,
        metrics: list[str],
        task: str,
    ) -> EvaluationResult:
        """Compute all requested metrics.

        Regression metrics: rmse, mae, mape, r2, explained_variance
        Classification metrics: f1_weighted, f1_macro, auc_roc, precision,
                                recall, accuracy, confusion_matrix
        """

    def generate_report(
        self,
        result: ExperimentResult,
        config: ResolvedExperimentConfig,
    ) -> Report:
        """Generate structured evaluation report.

        Includes:
          - experiment config summary
          - per-algorithm CV scores (mean ± std)
          - best model selection rationale
          - feature importance ranking
          - predictions vs actuals (sample)
          - visualizations (Plotly JSON):
            - TS: predicted vs actual line chart with confidence band
            - Entity: confusion matrix heatmap, ROC curve
            - Both: feature importance bar chart
        """

    def compare_models(
        self,
        results: list[ExperimentResult],
    ) -> ComparisonReport:
        """Compare multiple experiment results side by side."""
```

#### Metric Registry

```python
METRIC_REGISTRY: dict[str, MetricFn] = {
    # Regression
    "rmse": root_mean_squared_error,
    "mae": mean_absolute_error,
    "mape": mean_absolute_percentage_error,
    "r2": r2_score,

    # Classification
    "f1_weighted": partial(f1_score, average="weighted"),
    "f1_macro": partial(f1_score, average="macro"),
    "auc_roc": roc_auc_score,
    "precision": partial(precision_score, average="weighted"),
    "recall": partial(recall_score, average="weighted"),
    "accuracy": accuracy_score,
}
```

---

### 2.7 Experiment Runner (`src/mlrl_os/experiment/`)

```python
class ExperimentRunner:
    """Orchestrate the full experiment pipeline."""

    def run(self, user_config: dict | str) -> ExperimentResult:
        """Execute experiment from YAML config or dict.

        Pipeline:
          1. Parse user config (YAML string or dict)
          2. Load dataset from registry
          3. Resolve config (merge defaults)
          4. Validate (gate check)
             → if invalid: return ValidationError with all issues
          5. Detect problem type
          6. Build feature matrix (TS or Entity engine)
          7. Train and evaluate models (Model Engine)
          8. Generate evaluation report
          9. Register best model in Model Registry
          10. Store experiment in Experiment Tracker
          11. Return ExperimentResult

        Raises:
            ValidationError: if config fails validation gate.
        """

    def run_from_yaml(self, yaml_path: Path) -> ExperimentResult:
        """Load YAML file and run."""

    def run_from_builder(self, builder_output: dict) -> ExperimentResult:
        """Run from Builder UI submission. Same pipeline."""
```

#### Experiment Tracker

```python
class ExperimentTracker:
    """Persistent experiment history."""

    def record(
        self,
        experiment_id: str,
        config: ResolvedExperimentConfig,
        result: ExperimentResult,
    ) -> None:
        """Store experiment with all artifacts.

        Stored:
          - experiment_id
          - resolved config (full)
          - dataset reference (id + hash)
          - model reference (id)
          - evaluation metrics
          - feature importance
          - training duration
          - timestamp
        """

    def get(self, experiment_id: str) -> ExperimentRecord: ...
    def list(self, limit: int = 50) -> list[ExperimentRecord]: ...
    def compare(self, ids: list[str]) -> ComparisonReport: ...
```

---

### 2.8 API Layer (`src/mlrl_os/api/`)

#### Endpoints

| Method | Path | Purpose |
|---|---|---|
| **Data** | | |
| POST | `/api/v1/datasets` | Upload and register dataset |
| GET | `/api/v1/datasets` | List registered datasets |
| GET | `/api/v1/datasets/{id}` | Get dataset metadata |
| GET | `/api/v1/datasets/{id}/schema` | Get column schema with types and stats |
| GET | `/api/v1/datasets/{id}/available-targets` | Discover available targets |
| GET | `/api/v1/datasets/{id}/preview` | Preview first N rows |
| **Experiments** | | |
| POST | `/api/v1/experiments` | Submit experiment (YAML body or JSON) |
| GET | `/api/v1/experiments` | List experiments |
| GET | `/api/v1/experiments/{id}` | Get experiment result |
| GET | `/api/v1/experiments/{id}/report` | Get evaluation report |
| POST | `/api/v1/experiments/validate` | Validate config without running |
| GET | `/api/v1/experiments/defaults/{problem_type}` | Get default config for problem type |
| **Models** | | |
| GET | `/api/v1/models` | List registered models |
| GET | `/api/v1/models/{id}` | Get model metadata |
| POST | `/api/v1/models/{id}/predict` | Run prediction with registered model |
| GET | `/api/v1/models/{id}/feature-importance` | Get feature importance |
| **Config** | | |
| POST | `/api/v1/config/resolve` | Resolve partial config → full config (preview) |
| GET | `/api/v1/health` | Health check |

#### Key Request/Response Schemas

```python
# POST /api/v1/experiments
class ExperimentRequest(BaseModel):
    """User-submitted experiment config. All fields optional except dataset."""
    name: str | None = None
    seed: int | None = None
    dataset_id: str                        # required
    dataset_layer: str | None = None       # auto-detected if omitted
    experiment_type: str | None = None     # auto-detected if omitted

    # Feature config (all optional — defaults apply)
    target: str | None = None
    lookback: str | None = None            # TS only
    horizon: str | None = None             # TS only
    lag_intervals: list[str] | None = None
    rolling_windows: list[str] | None = None
    observation_point: str | None = None   # entity only
    feature_columns: list[str] | None = None

    # Model config (all optional — defaults apply)
    algorithms: list[str] | None = None
    cv_folds: int | None = None

    # Evaluation config (all optional — defaults apply)
    metrics: list[str] | None = None


# Response: GET /api/v1/datasets/{id}/available-targets
class AvailableTargetsResponse(BaseModel):
    dataset_id: str
    time_series_targets: list[TargetInfo]
    entity_targets: list[TargetInfo]

class TargetInfo(BaseModel):
    column: str
    type: str                              # regression | classification
    is_default: bool
    stats: TargetStats                     # distribution summary

class TargetStats(BaseModel):
    null_rate: float
    unique_count: int | None               # for classification
    classes: list[str] | None              # for classification
    class_balance: dict[str, float] | None # for classification
    mean: float | None                     # for regression
    std: float | None                      # for regression
    min: float | None                      # for regression
    max: float | None                      # for regression


# Response: POST /api/v1/experiments/validate
class ValidationResponse(BaseModel):
    valid: bool
    resolved_config: dict | None           # shown if valid
    errors: list[ValidationErrorDetail]
    warnings: list[str]

class ValidationErrorDetail(BaseModel):
    code: str                              # e.g., "V-03"
    field: str                             # e.g., "features.target"
    message: str                           # human-readable
    suggestion: str | None                 # e.g., "Available targets: ..."


# Response: GET /api/v1/experiments/{id}
class ExperimentResponse(BaseModel):
    experiment_id: str
    name: str
    status: str                            # pending | running | completed | failed
    experiment_type: str
    created_at: str
    duration_seconds: float | None

    # Config
    resolved_config: dict

    # Results (if completed)
    best_algorithm: str | None
    metrics: dict[str, float] | None       # primary metrics
    all_algorithm_scores: list[AlgorithmScore] | None
    feature_importance: dict[str, float] | None
    model_id: str | None

class AlgorithmScore(BaseModel):
    algorithm: str
    metrics: dict[str, float]              # mean across folds
    metrics_std: dict[str, float]          # std across folds
```

---

## 3. Source Layout

### Backend (`src/mlrl_os/`)

```
src/mlrl_os/
├── __init__.py
├── core/                        # Core types and protocols
│   ├── __init__.py
│   ├── types.py                 # ProblemType, FeatureMatrix, TrainedModel, enums
│   ├── experiment.py            # ExperimentResult, ExperimentRecord
│   └── dataset.py               # RawDataset, DatasetMeta, AvailableTargets
│
├── data/                        # Data ingestion & management
│   ├── __init__.py
│   ├── simos_loader.py          # SimOS 5-layer export loader
│   ├── external_loader.py       # CSV / Parquet loader
│   ├── registry.py              # Dataset versioning & registry
│   ├── discovery.py             # Target & feature auto-discovery
│   └── validation.py            # Data quality checks
│
├── features/                    # Feature engineering
│   ├── __init__.py
│   ├── time_series.py           # Windowing, lag, rolling, trend, ratio
│   ├── entity.py                # Entity state, node state, system state, derived
│   ├── detection.py             # Problem type auto-detection
│   └── store.py                 # Feature definition registry
│
├── config/                      # Configuration management
│   ├── __init__.py
│   ├── defaults.py              # BaseSettings (env vars), default configs
│   ├── resolver.py              # Merge user config + defaults → resolved
│   └── schemas.py               # Pydantic models for all config types
│
├── validation/                  # Validation gate
│   ├── __init__.py
│   └── gate.py                  # All validation rules (V-01..VE-04)
│
├── models/                      # Model training & management
│   ├── __init__.py
│   ├── engine.py                # Train, evaluate, select best
│   ├── registry.py              # Model versioning & storage
│   └── algorithms/
│       ├── __init__.py
│       ├── protocol.py          # Algorithm protocol
│       ├── registry.py          # Algorithm registry
│       ├── lightgbm.py          # LightGBM wrapper
│       ├── xgboost.py           # XGBoost wrapper
│       ├── random_forest.py     # sklearn RandomForest wrapper
│       ├── extra_trees.py       # sklearn ExtraTrees wrapper
│       └── linear.py            # Ridge / LogisticRegression
│
├── evaluation/                  # Evaluation & reporting
│   ├── __init__.py
│   ├── metrics.py               # Metric registry & computation
│   ├── reports.py               # Report generation (JSON + HTML)
│   └── comparison.py            # Multi-experiment comparison
│
├── experiment/                  # Experiment orchestration
│   ├── __init__.py
│   ├── runner.py                # Full pipeline: config → result
│   ├── tracker.py               # Experiment history & artifacts
│   └── seed.py                  # Seeded RNG utilities (seed_hash)
│
├── api/                         # FastAPI application
│   ├── __init__.py
│   ├── app.py                   # App factory (create_app)
│   ├── data_routes.py           # Dataset endpoints
│   ├── experiment_routes.py     # Experiment endpoints
│   ├── model_routes.py          # Model endpoints
│   ├── config_routes.py         # Config resolution endpoints
│   └── schemas.py               # Request/response Pydantic models
│
└── cli.py                       # CLI entry point
```

### Web UI (`web/`)

```
web/
├── src/
│   ├── api/                     # Axios client + endpoint modules
│   │   ├── client.ts            # Base Axios with interceptors
│   │   ├── datasets.ts          # Dataset API calls
│   │   ├── experiments.ts       # Experiment API calls
│   │   ├── models.ts            # Model API calls
│   │   └── types.ts             # TypeScript interfaces (mirror backend)
│   │
│   ├── components/
│   │   ├── builder/             # Experiment Builder (7-step flow)
│   │   │   ├── BuilderLayout.tsx
│   │   │   ├── StepDataset.tsx      # Step 1: select/upload dataset
│   │   │   ├── StepExplore.tsx      # Step 2: explore data
│   │   │   ├── StepProblemType.tsx   # Step 3: select problem type
│   │   │   ├── StepTarget.tsx       # Step 4: select target
│   │   │   ├── StepFeatures.tsx     # Step 5: select features
│   │   │   ├── StepModel.tsx        # Step 6: configure model
│   │   │   ├── StepReview.tsx       # Step 7: review YAML & submit
│   │   │   └── ValidationErrors.tsx # Error display component
│   │   │
│   │   ├── results/             # Experiment results display
│   │   │   ├── MetricsSummary.tsx
│   │   │   ├── FeatureImportance.tsx
│   │   │   ├── PredictionsChart.tsx  # TS: predicted vs actual
│   │   │   ├── ConfusionMatrix.tsx   # Entity: confusion matrix
│   │   │   ├── ROCCurve.tsx          # Entity: ROC curve
│   │   │   └── ModelComparison.tsx
│   │   │
│   │   ├── data/                # Data exploration
│   │   │   ├── DatasetList.tsx
│   │   │   ├── SchemaViewer.tsx
│   │   │   ├── ColumnStats.tsx
│   │   │   └── DataPreview.tsx
│   │   │
│   │   ├── layout/              # App shell
│   │   │   ├── AppShell.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   └── TopBar.tsx
│   │   │
│   │   └── shared/              # Reusable components
│   │       ├── YamlViewer.tsx
│   │       ├── Toast.tsx
│   │       └── LoadingSpinner.tsx
│   │
│   ├── pages/
│   │   ├── DashboardPage.tsx    # Overview: recent experiments, models
│   │   ├── BuilderPage.tsx      # Experiment Builder
│   │   ├── DatasetsPage.tsx     # Dataset management
│   │   ├── ExperimentsPage.tsx  # Experiment history
│   │   ├── ResultsPage.tsx      # Single experiment results
│   │   └── ModelsPage.tsx       # Model registry browser
│   │
│   ├── store/                   # Zustand stores
│   │   ├── useBuilderStore.ts   # Builder state (7 steps)
│   │   └── useAppStore.ts       # Global app state
│   │
│   └── lib/
│       ├── yamlGenerator.ts     # Builder state → YAML
│       └── formatting.ts        # Number/date formatting
│
├── package.json
├── vite.config.ts
├── tsconfig.json
└── tailwind.config.ts
```

### Tests (`tests/`)

```
tests/
├── unit/
│   ├── data/
│   │   ├── test_simos_loader.py
│   │   ├── test_external_loader.py
│   │   ├── test_registry.py
│   │   └── test_discovery.py
│   ├── features/
│   │   ├── test_time_series.py
│   │   ├── test_entity.py
│   │   ├── test_detection.py
│   │   └── test_store.py
│   ├── config/
│   │   ├── test_resolver.py
│   │   └── test_defaults.py
│   ├── validation/
│   │   └── test_gate.py             # every rule V-01..VE-04
│   ├── models/
│   │   ├── test_engine.py
│   │   ├── test_registry.py
│   │   └── test_algorithms/
│   │       ├── test_lightgbm.py
│   │       ├── test_xgboost.py
│   │       └── test_random_forest.py
│   ├── evaluation/
│   │   ├── test_metrics.py
│   │   └── test_reports.py
│   └── experiment/
│       ├── test_runner.py
│       └── test_tracker.py
├── integration/
│   ├── test_end_to_end_ts.py        # full pipeline: SimOS export → TS forecast → report
│   ├── test_end_to_end_entity.py    # full pipeline: SimOS export → entity classify → report
│   ├── test_zero_config.py          # dataset-only config → valid results
│   ├── test_builder_flow.py         # Builder YAML → validate → run
│   └── test_api.py                  # API endpoint integration
└── fixtures/
    ├── simos_export_healthcare.json  # sample SimOS export (healthcare domain)
    ├── simos_export_supply_chain.json # sample SimOS export (supply chain)
    ├── sample_timeseries.csv         # external time-series data
    └── sample_entities.csv           # external entity data
```

---

## 4. Implementation Phases

### Phase 1: Core Engine (Foundation)

Build bottom-up: types → data → features → models → evaluation → runner.

| Order | Module | Depends On | Test Count (est.) |
|---|---|---|---|
| 1.1 | `core/types.py` | — | 15 |
| 1.2 | `experiment/seed.py` | — | 10 |
| 1.3 | `data/simos_loader.py` | core | 25 |
| 1.4 | `data/external_loader.py` | core | 15 |
| 1.5 | `data/registry.py` | core | 20 |
| 1.6 | `data/discovery.py` | core, data | 20 |
| 1.7 | `features/detection.py` | core | 10 |
| 1.8 | `features/time_series.py` | core | 30 |
| 1.9 | `features/entity.py` | core | 25 |
| 1.10 | `config/defaults.py` | — | 10 |
| 1.11 | `config/schemas.py` | core | 15 |
| 1.12 | `config/resolver.py` | config, data | 25 |
| 1.13 | `validation/gate.py` | config, data | 35 |
| 1.14 | `models/algorithms/protocol.py` | core | 5 |
| 1.15 | `models/algorithms/registry.py` | protocol | 10 |
| 1.16 | `models/algorithms/lightgbm.py` | protocol | 15 |
| 1.17 | `models/algorithms/xgboost.py` | protocol | 15 |
| 1.18 | `models/algorithms/random_forest.py` | protocol | 10 |
| 1.19 | `models/engine.py` | algorithms, features | 25 |
| 1.20 | `models/registry.py` | core | 15 |
| 1.21 | `evaluation/metrics.py` | core | 20 |
| 1.22 | `evaluation/reports.py` | metrics | 15 |
| 1.23 | `experiment/runner.py` | all above | 20 |
| 1.24 | `experiment/tracker.py` | core | 15 |
| | **Phase 1 Total** | | **~440** |

### Phase 2: API + Validation

| Order | Module | Depends On |
|---|---|---|
| 2.1 | `api/app.py` | — |
| 2.2 | `api/schemas.py` | core |
| 2.3 | `api/data_routes.py` | data |
| 2.4 | `api/experiment_routes.py` | experiment |
| 2.5 | `api/model_routes.py` | models |
| 2.6 | `api/config_routes.py` | config |
| 2.7 | Integration tests | all |

### Phase 3: Web UI (Builder + Results)

| Order | Module | Depends On |
|---|---|---|
| 3.1 | API client + types | backend API |
| 3.2 | App shell + routing | — |
| 3.3 | Builder (7 steps) | API client |
| 3.4 | Results dashboard | API client |
| 3.5 | Data explorer | API client |
| 3.6 | Experiment history | API client |

### Phase 4: Polish & Documentation

| Order | Module |
|---|---|
| 4.1 | CLI (`mlrl-os run experiment.yaml`) |
| 4.2 | Feature Store persistence |
| 4.3 | HTML report export |
| 4.4 | User documentation |

---

## 5. Environment Variables

All variables prefixed with `MLRL_`.

| Variable | Default | Description |
|---|---|---|
| `MLRL_ENV` | `development` | Environment: development, staging, production |
| `MLRL_DATA_DIR` | `./data` | Dataset storage directory |
| `MLRL_MODELS_DIR` | `./models` | Model artifact directory |
| `MLRL_EXPERIMENTS_DIR` | `./experiments` | Experiment history directory |
| `MLRL_LOG_LEVEL` | `INFO` | Logging level |
| `MLRL_API_PORT` | `8001` | Backend API port |
| `MLRL_CORS_ALLOW_ORIGINS` | `http://localhost:5175` | CORS origins |
| `MLRL_MAX_TRAINING_ROWS` | `1000000` | Maximum training dataset rows |
| `MLRL_CV_FOLDS_DEFAULT` | `5` | Default cross-validation folds |
| `MLRL_SEED_DEFAULT` | `42` | Default experiment seed |
