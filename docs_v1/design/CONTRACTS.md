# CONTRACTS

> Exact data schemas, API contracts, YAML config schema, and concrete data examples at every pipeline stage.
>
> Implementation-ready: every field name, every type, every example value.
>
> Inherits from CONSTITUTION.md and BLUEPRINT.md.
>
> Last updated: 2026-03-09.

---

## 1. Data Decoupling Architecture

### The Problem

SimOS export schema may change over time (new columns, renamed fields, restructured layers). ML/RL OS must NOT break when this happens. Hard-coding SimOS field names throughout the codebase would create tight coupling.

### The Solution: Schema Adapter Pattern

ML/RL OS uses **canonical internal column names** everywhere. The ONLY place that knows SimOS-specific field names is the `SimosSchemaAdapter` class. If SimOS changes its export format, update ONE file.

```
SimOS Export (external schema)
    ↓
SimosSchemaAdapter (maps external → canonical)
    ↓
Canonical DataFrame (internal schema)
    ↓
Feature Engine, Model Engine, etc. (only see canonical names)
```

### Canonical Column Names (ML/RL OS Internal)

These are the column names used INSIDE ML/RL OS. They never change regardless of data source.

**Snapshot canonical columns (time-series):**

```python
# System-level (always present)
CANONICAL_SYS_COLUMNS = [
    "ts",                              # timestamp (seconds)
    "bucket_idx",                      # bucket index
    "wip",                             # work in process
    "in_queue",                        # total entities in queues
    "busy",                            # total busy processing slots
    "cum_arrivals",                    # cumulative arrivals
    "cum_completions",                 # cumulative completions
    "arrival_rate",                    # arrivals per hour
    "throughput",                      # completions per hour
    "wip_ratio",                       # wip / capacity
    "avg_wait",                        # average wait time
    "avg_processing",                  # average processing time
    "wait_cost_bucket",               # wait cost in this bucket
    "idle_cost_bucket",               # idle cost in this bucket
    "revenue_bucket",                 # revenue in this bucket
    "cum_wait_cost",                  # cumulative wait cost
    "cum_idle_cost",                  # cumulative idle cost
    "cum_revenue",                    # cumulative revenue
    "cum_net_cost",                   # cumulative net cost
    "sla_breaches_bucket",            # SLA breaches in this bucket
    "cum_sla_breaches",               # cumulative SLA breaches
    "sla_compliance",                 # SLA compliance rate
]

# Per-node (dynamic, pattern: n_{name}_{metric})
# n_triage_queue, n_triage_busy, n_triage_util, n_triage_throughput, ...
NODE_METRICS = ["queue", "busy", "concurrency", "util", "cum_processed",
                "arrivals_bucket", "completions_bucket", "throughput",
                "avg_wait", "avg_processing"]

# Per-resource (dynamic, pattern: r_{name}_{metric})
# r_doctors_capacity, r_doctors_util, ...
RESOURCE_METRICS = ["capacity", "util"]
```

**Trajectory canonical columns (entity classification):**

```python
# Top-level
CANONICAL_TRAJ_COLUMNS = [
    "eid",                             # entity ID
    "etype",                           # entity type
    "epriority",                       # entity priority
    "source_idx",                      # arrival source index
    "step",                            # step index (0-based)
    "node",                            # node name
    "t_enter",                         # sim time entered queue
    "t_complete",                      # sim time processing done
    "wait",                            # wait time at this step
    "processing",                      # processing time at this step
    "setup",                           # setup time
    "transit",                         # transit time
    "transit_cost",                    # transport cost
    "done",                            # episode complete (bool)
    "status",                          # episode_status
    "total_time",                      # total entity time (if done)
]

# State vector canonical names (prefix: s_)
CANONICAL_STATE_ENTITY = [
    "s_priority", "s_elapsed", "s_cum_wait", "s_cum_processing",
    "s_wait_ratio", "s_steps_done", "s_cum_cost", "s_cum_transit_cost",
    "s_cum_setup", "s_source_idx",
]
CANONICAL_STATE_NODE = [
    "s_node_util", "s_node_avg_queue", "s_node_max_queue",
    "s_node_avg_wait", "s_node_avg_processing", "s_node_throughput",
    "s_node_concurrency", "s_node_setup_count", "s_node_mutation_count",
]
CANONICAL_STATE_SYSTEM = [
    "s_sys_util", "s_sys_throughput", "s_sys_bottleneck_util",
    "s_sys_node_count", "s_sys_total_capacity",
]
# Dynamic: s_r_{name}_util, s_r_{name}_capacity, s_attr_{name}
```

### Schema Adapter Implementation

```python
class SimosSchemaAdapter:
    """Maps SimOS v3.0 export field names → ML/RL OS canonical names.

    THIS IS THE ONLY FILE THAT KNOWS SIMOS FIELD NAMES.
    When SimOS changes its export schema, update ONLY this adapter.
    """

    # SimOS snapshot field → canonical name
    SNAPSHOT_MAP = {
        "timestamp": "ts",
        "bucket_index": "bucket_idx",
        "sys_wip": "wip",
        "sys_total_in_queue": "in_queue",
        "sys_total_busy": "busy",
        "sys_cumulative_arrivals": "cum_arrivals",
        "sys_cumulative_completions": "cum_completions",
        "sys_arrival_rate_per_hour": "arrival_rate",
        "sys_throughput_per_hour": "throughput",
        "sys_wip_ratio": "wip_ratio",
        "sys_avg_wait_time": "avg_wait",
        "sys_avg_processing_time": "avg_processing",
        "sys_wait_cost_in_bucket": "wait_cost_bucket",
        "sys_idle_cost_in_bucket": "idle_cost_bucket",
        "sys_revenue_in_bucket": "revenue_bucket",
        "sys_cumulative_wait_cost": "cum_wait_cost",
        "sys_cumulative_idle_cost": "cum_idle_cost",
        "sys_cumulative_revenue": "cum_revenue",
        "sys_cumulative_net_cost": "cum_net_cost",
        "sys_sla_breaches_in_bucket": "sla_breaches_bucket",
        "sys_cumulative_sla_breaches": "cum_sla_breaches",
        "sys_sla_compliance": "sla_compliance",
    }

    # SimOS node field pattern → canonical pattern
    # "node_{name}_{metric}" → "n_{name}_{canonical_metric}"
    NODE_METRIC_MAP = {
        "queue": "queue",
        "busy": "busy",
        "concurrency": "concurrency",
        "utilization": "util",
        "cumulative_processed": "cum_processed",
        "arrivals_in_bucket": "arrivals_bucket",
        "completions_in_bucket": "completions_bucket",
        "throughput_per_hour": "throughput",
        "avg_wait_time": "avg_wait",
        "avg_processing_time": "avg_processing",
    }

    # SimOS trajectory state key → canonical state key
    STATE_MAP = {
        "entity_priority": "s_priority",
        "entity_elapsed_time": "s_elapsed",
        "entity_cumulative_wait": "s_cum_wait",
        "entity_cumulative_processing": "s_cum_processing",
        "entity_wait_ratio": "s_wait_ratio",
        "entity_steps_completed": "s_steps_done",
        "entity_cumulative_cost": "s_cum_cost",
        "entity_cumulative_transport_cost": "s_cum_transit_cost",
        "entity_cumulative_setup_time": "s_cum_setup",
        "entity_source_index": "s_source_idx",
        "node_utilization": "s_node_util",
        "node_avg_queue_depth": "s_node_avg_queue",
        "node_max_queue_depth": "s_node_max_queue",
        "node_avg_wait_time": "s_node_avg_wait",
        "node_avg_processing_time": "s_node_avg_processing",
        "node_throughput_per_hour": "s_node_throughput",
        "node_concurrency": "s_node_concurrency",
        "node_setup_count": "s_node_setup_count",
        "node_mutation_count": "s_node_mutation_count",
        "sys_utilization": "s_sys_util",
        "sys_throughput_per_hour": "s_sys_throughput",
        "sys_bottleneck_utilization": "s_sys_bottleneck_util",
        "sys_node_count": "s_sys_node_count",
        "sys_total_capacity": "s_sys_total_capacity",
    }

    def map_snapshot_record(self, raw: dict) -> dict:
        """Map one SimOS snapshot record to canonical names."""

    def map_trajectory_record(self, raw: dict) -> dict:
        """Map one SimOS trajectory record to canonical names.
        Flattens state, reward, next_state dicts.
        """

    def map_dynamic_column(self, simos_col: str) -> str:
        """Map dynamic column names (node_X_metric, resource_X_metric)."""
```

### External Data Adapters

External CSV/Parquet data uses canonical names directly (user maps their columns) or provides a mapping file:

```yaml
# Optional: column_mapping.yaml (for external data)
mapping:
  "timestamp_col": "ts"
  "queue_depth": "wip"
  "throughput_rate": "throughput"
  "customer_id": "eid"
  "final_status": "status"
```

If no mapping provided, column names are used as-is (user's responsibility).

### Key Rule

**All code downstream of the data loaders uses ONLY canonical column names.** Feature engines, model engine, evaluation — none of them know about SimOS field names. This guarantees:

1. SimOS schema changes → update `SimosSchemaAdapter` only
2. External data sources → provide a mapping or use canonical names
3. New data sources (future) → write a new adapter, no pipeline changes

---

## 2. SimOS Export Schema (Reference Only)

> This section documents the CURRENT SimOS export schema for reference.
> The actual mapping is in `SimosSchemaAdapter`. If SimOS changes, update the adapter, not this doc.

### 2.1 Top-Level Export Structure

```json
{
  "metadata": { ... },
  "summary": { ... },
  "event_stream": [ ... ],       // Layer 1 — NOT used in v0.1
  "trajectories": [ ... ],       // Layer 2 — entity classification
  "snapshots": [ ... ],          // Layer 3 — time-series forecasting
  "stress_scenarios": [ ... ]    // Layer 5 — NOT used in v0.1
}
```

### 1.2 Metadata Schema

```python
class SimosExportMetadata(BaseModel):
    export_version: str                    # "3.0"
    seed: int                              # e.g., 42
    duration: float                        # simulation duration in seconds
    domain: str                            # "healthcare" | "supply_chain" | "service" | "generic"
    template_name: str                     # e.g., "healthcare_er"
    entity_type: str                       # primary entity type, e.g., "patient"
    entity_types: list[str]                # all entity types
    entity_count: int                      # total entities generated
    node_count: int                        # number of processing nodes
    resource_count: int                    # number of resource pools
    node_names: list[str]                  # e.g., ["triage", "diagnosis", "treatment"]
    resource_names: list[str]              # e.g., ["doctors", "nurses"]
    multi_source: bool                     # True if multiple arrival sources
    source_count: int                      # number of arrival sources
    features: SimosExportFeatureFlags
    event_count: int | None                # if event_stream included
    trajectory_count: int | None           # if trajectories included
    snapshot_count: int | None             # if snapshots included

class SimosExportFeatureFlags(BaseModel):
    transport_cost: bool
    setup_time: bool
    mutations: bool
```

### 1.3 Layer 2 — Trajectory Record Schema

Each record is one step of one entity's journey.

```python
class TrajectoryRecord(BaseModel):
    """One step of one entity — the raw record from SimOS export."""
    # Identity
    entity_id: str                         # e.g., "patient_001"
    entity_type: str                       # e.g., "patient"
    entity_priority: int                   # 0-10
    source_index: int | None               # arrival source index

    # Step info
    step_index: int                        # 0-based step number
    node_name: str                         # e.g., "triage"
    sim_time_enter: float                  # sim time when entity entered node queue
    sim_time_complete: float               # sim time when processing finished

    # Timing
    wait_time: float                       # seconds waiting in queue
    processing_time: float                 # seconds being processed
    setup_time: float                      # seconds for setup (may be 0)
    transit_time: float                    # seconds in transit to this node
    transport_cost: float                  # cost of transit

    # State vector (19 base + dynamic resource + dynamic attr)
    state: dict[str, float]
    # Keys always present:
    #   entity_priority, entity_elapsed_time, entity_cumulative_wait,
    #   entity_cumulative_processing, entity_wait_ratio, entity_steps_completed,
    #   entity_cumulative_cost, entity_cumulative_transport_cost,
    #   entity_cumulative_setup_time, entity_source_index,
    #   node_utilization, node_avg_queue_depth, node_max_queue_depth,
    #   node_avg_wait_time, node_avg_processing_time, node_throughput_per_hour,
    #   node_concurrency, node_setup_count, node_mutation_count,
    #   sys_utilization, sys_throughput_per_hour, sys_bottleneck_utilization,
    #   sys_node_count, sys_total_capacity
    # Dynamic keys:
    #   resource_{name}_utilization, resource_{name}_capacity  (per resource pool)
    #   attr_{name}  (per numeric entity attribute)

    # Action context
    action_context: dict[str, Any]
    # Keys: node_name, node_concurrency, resources_required, setup_time,
    #        transit_time, transport_cost

    # Reward (multi-objective)
    reward: dict[str, float]
    # Keys: time_penalty, cost_penalty, transport_cost_penalty, step_completion,
    #        sla_penalty, processing_efficiency, sla_budget_remaining,
    #        episode_completion (only on last step)

    # Next state (same schema as state)
    next_state: dict[str, float]

    # Episode info
    episode_done: bool                     # True on entity's final step
    episode_status: str                    # "completed" | "rejected" | "timed_out"
    episode_total_time: float | None       # total time if episode_done=True
```

**Concrete example:**

```json
{
  "entity_id": "patient_042",
  "entity_type": "patient",
  "entity_priority": 3,
  "source_index": 0,
  "step_index": 1,
  "node_name": "diagnosis",
  "sim_time_enter": 1245.8,
  "sim_time_complete": 1423.2,
  "wait_time": 120.5,
  "processing_time": 56.9,
  "setup_time": 0.0,
  "transit_time": 0.0,
  "transport_cost": 0.0,
  "state": {
    "entity_priority": 3.0,
    "entity_elapsed_time": 423.2,
    "entity_cumulative_wait": 185.3,
    "entity_cumulative_processing": 121.7,
    "entity_wait_ratio": 0.604,
    "entity_steps_completed": 1.0,
    "entity_cumulative_cost": 0.927,
    "entity_cumulative_transport_cost": 0.0,
    "entity_cumulative_setup_time": 0.0,
    "entity_source_index": 0.0,
    "node_utilization": 0.82,
    "node_avg_queue_depth": 6.3,
    "node_max_queue_depth": 12.0,
    "node_avg_wait_time": 95.4,
    "node_avg_processing_time": 55.2,
    "node_throughput_per_hour": 28.1,
    "node_concurrency": 3.0,
    "node_setup_count": 0.0,
    "node_mutation_count": 0.0,
    "sys_utilization": 0.71,
    "sys_throughput_per_hour": 35.4,
    "sys_bottleneck_utilization": 0.89,
    "sys_node_count": 4.0,
    "sys_total_capacity": 12.0,
    "resource_doctors_utilization": 0.78,
    "resource_doctors_capacity": 3.0,
    "resource_nurses_utilization": 0.65,
    "resource_nurses_capacity": 5.0
  },
  "action_context": {
    "node_name": "diagnosis",
    "node_concurrency": 3,
    "resources_required": {"doctors": 1},
    "setup_time": 0.0,
    "transit_time": 0.0,
    "transport_cost": 0.0
  },
  "reward": {
    "time_penalty": -120.5,
    "cost_penalty": -0.603,
    "transport_cost_penalty": 0.0,
    "step_completion": 1.0,
    "sla_penalty": 0.0,
    "processing_efficiency": 1.031,
    "sla_budget_remaining": 176.8
  },
  "next_state": {
    "entity_priority": 3.0,
    "entity_elapsed_time": 423.2,
    "entity_cumulative_wait": 185.3,
    "...": "same schema as state"
  },
  "episode_done": false,
  "episode_status": "completed",
  "episode_total_time": null,
  "attr_acuity": 3
}
```

### 1.4 Layer 3 — Snapshot Record Schema

Each record is one time bucket of system state.

```python
class SnapshotRecord(BaseModel):
    """One time bucket of system state — raw record from SimOS export."""
    timestamp: float                       # bucket end time (seconds)
    bucket_index: int                      # 0-based bucket number

    # System-level metrics
    sys_wip: int                           # work in process
    sys_total_in_queue: int
    sys_total_busy: int
    sys_cumulative_arrivals: int
    sys_cumulative_completions: int
    sys_arrival_rate_per_hour: float
    sys_throughput_per_hour: float
    sys_wip_ratio: float
    sys_avg_wait_time: float
    sys_avg_processing_time: float

    # Cost metrics
    sys_total_wait_in_bucket: float
    sys_total_processing_in_bucket: float
    sys_wait_cost_in_bucket: float
    sys_idle_cost_in_bucket: float
    sys_revenue_in_bucket: float
    sys_cumulative_wait_cost: float
    sys_cumulative_idle_cost: float
    sys_cumulative_revenue: float
    sys_cumulative_net_cost: float

    # SLA metrics
    sys_sla_breaches_in_bucket: int
    sys_cumulative_sla_breaches: int
    sys_sla_compliance: float

    # Per-node metrics (dynamic, per node name)
    # node_{name}_queue: int
    # node_{name}_busy: int
    # node_{name}_concurrency: int
    # node_{name}_utilization: float
    # node_{name}_cumulative_processed: int
    # node_{name}_arrivals_in_bucket: int
    # node_{name}_completions_in_bucket: int
    # node_{name}_throughput_per_hour: float
    # node_{name}_avg_wait_time: float
    # node_{name}_avg_processing_time: float

    # Per-resource metrics (dynamic, per resource name)
    # resource_{name}_capacity: int
    # resource_{name}_utilization: float
```

**Concrete example:**

```json
{
  "timestamp": 3600.0,
  "bucket_index": 59,
  "sys_wip": 12,
  "sys_total_in_queue": 8,
  "sys_total_busy": 7,
  "sys_cumulative_arrivals": 145,
  "sys_cumulative_completions": 133,
  "sys_arrival_rate_per_hour": 48.3,
  "sys_throughput_per_hour": 44.3,
  "sys_wip_ratio": 0.571,
  "sys_avg_wait_time": 85.2,
  "sys_avg_processing_time": 52.1,
  "sys_total_wait_in_bucket": 342.8,
  "sys_total_processing_in_bucket": 210.4,
  "sys_wait_cost_in_bucket": 1.714,
  "sys_idle_cost_in_bucket": 12.5,
  "sys_revenue_in_bucket": 200.0,
  "sys_cumulative_wait_cost": 98.7,
  "sys_cumulative_idle_cost": 725.0,
  "sys_cumulative_revenue": 6650.0,
  "sys_cumulative_net_cost": -5826.3,
  "sys_sla_breaches_in_bucket": 0,
  "sys_cumulative_sla_breaches": 3,
  "sys_sla_compliance": 0.977,
  "node_triage_queue": 2,
  "node_triage_busy": 1,
  "node_triage_concurrency": 2,
  "node_triage_utilization": 0.5,
  "node_triage_cumulative_processed": 145,
  "node_triage_arrivals_in_bucket": 3,
  "node_triage_completions_in_bucket": 3,
  "node_triage_throughput_per_hour": 48.3,
  "node_triage_avg_wait_time": 15.2,
  "node_triage_avg_processing_time": 8.5,
  "node_diagnosis_queue": 6,
  "node_diagnosis_busy": 3,
  "node_diagnosis_concurrency": 3,
  "node_diagnosis_utilization": 0.82,
  "resource_doctors_capacity": 3,
  "resource_doctors_utilization": 0.78
}
```

### 1.5 Summary Schema

```python
class SimosExportSummary(BaseModel):
    entity_count: int
    status_distribution: dict[str, int]    # {"completed": 120, "rejected": 5, "timed_out": 8}
    source_distribution: dict[str, int]    # {"source_0": 133}
    lead_time: LeadTimeStats
    transport_cost: TransportCostStats
    setup_time: SetupTimeStats
    utilization: UtilizationStats
    bottleneck_node: str
    bottleneck_utilization: float
    resource_summary: dict[str, ResourceSummary]

class LeadTimeStats(BaseModel):
    mean: float
    median: float
    std: float
    min: float
    max: float

class TransportCostStats(BaseModel):
    total: float
    mean: float
    entity_count_with_transport: int

class SetupTimeStats(BaseModel):
    total: float
    mean: float
    entity_count_with_setup: int

class UtilizationStats(BaseModel):
    mean: float
    max: float
    min: float

class ResourceSummary(BaseModel):
    capacity: int
    utilization: float
```

---

## 2. Internal Data Schemas

### 2.1 RawDataset

Produced by `SimosLoader` or `ExternalLoader`. Input to the Dataset Registry.

```python
class RawDataset(BaseModel):
    """Raw loaded data before registration."""
    source_type: str                       # "simos" | "csv" | "parquet"
    source_path: str                       # original file path

    # SimOS-specific (None for external)
    metadata: SimosExportMetadata | None
    summary: SimosExportSummary | None

    # Data frames (at least one must be non-None)
    snapshots: pl.DataFrame | None         # Layer 3 for time-series
    trajectories: pl.DataFrame | None      # Layer 2 for entity classification

    # Schema info (auto-detected)
    snapshot_columns: list[ColumnInfo] | None
    trajectory_columns: list[ColumnInfo] | None

class ColumnInfo(BaseModel):
    name: str
    dtype: str                             # "float64" | "int64" | "str" | "bool"
    null_count: int
    null_rate: float
    unique_count: int
    is_numeric: bool
    is_categorical: bool
    # Numeric stats (None if categorical)
    mean: float | None
    std: float | None
    min: float | None
    max: float | None
    # Categorical stats (None if numeric)
    categories: list[str] | None
    category_counts: dict[str, int] | None
```

### 2.2 DatasetMeta

Stored in Dataset Registry after registration.

```python
class DatasetMeta(BaseModel):
    id: str                                # "ds_20260309_143022_a1b2c3"
    name: str                              # user-provided or auto-generated
    version: int                           # auto-incremented
    content_hash: str                      # SHA-256 of data content
    source_type: str                       # "simos" | "csv" | "parquet"
    source_path: str

    # Shape
    snapshot_row_count: int | None
    trajectory_row_count: int | None
    snapshot_column_count: int | None
    trajectory_column_count: int | None

    # Schema
    snapshot_columns: list[ColumnInfo] | None
    trajectory_columns: list[ColumnInfo] | None

    # SimOS metadata (None for external)
    simos_metadata: SimosExportMetadata | None
    simos_summary: SimosExportSummary | None

    # Available layers
    has_snapshots: bool
    has_trajectories: bool

    # Registration
    registered_at: str                     # ISO 8601 timestamp
```

### 2.3 AvailableTargets

Returned by Target Discovery. Shows user what they can predict.

```python
class AvailableTargets(BaseModel):
    dataset_id: str

    time_series_targets: list[TargetInfo]
    entity_targets: list[TargetInfo]

class TargetInfo(BaseModel):
    column: str                            # e.g., "sys_avg_wait_time"
    task_type: str                         # "regression" | "classification"
    is_default: bool                       # True for default target
    null_rate: float

    # Regression stats (None for classification)
    mean: float | None
    std: float | None
    min: float | None
    max: float | None

    # Classification stats (None for regression)
    unique_count: int | None
    classes: list[str] | None
    class_balance: dict[str, float] | None # class → proportion
```

**Default target selection logic:**

```python
# Time-series defaults (first available wins):
TS_DEFAULT_TARGETS = [
    "sys_avg_lead_time",         # not present in snapshots directly
    "sys_avg_wait_time",         # most common operational KPI
    "sys_throughput_per_hour",
    "sys_wip",
    "sys_sla_compliance",
]

# Entity classification defaults:
ENTITY_DEFAULT_TARGETS = [
    "episode_status",            # completed | rejected | timed_out
    "sla_breach",                # derived: bool (needs SLA threshold)
]
```

**Target discovery rules:**

For time-series (snapshots):
- All `sys_*` numeric columns are candidate targets
- All `node_{name}_*` numeric columns are candidate targets
- All `resource_{name}_*` numeric columns are candidate targets
- Exclude: `timestamp`, `bucket_index` (not predictable)
- Exclude: cumulative columns that only increase (e.g., `sys_cumulative_arrivals`) — flag as warning

For entity classification (trajectories):
- `episode_status` → multi-class classification (always available)
- `sla_breach` → binary classification (derived if SLA info in metadata)
- `delay_severity` → ordinal classification (derived from `episode_total_time` percentiles)
- Any `attr_*` categorical column → classification target

### 2.4 FeatureMatrix

Output of Feature Engine. Input to Model Engine.

```python
class FeatureMatrix(BaseModel):
    """Model-ready data."""
    X: np.ndarray                          # (n_samples, n_features) float64
    y: np.ndarray                          # (n_samples,) float64 or int64
    feature_names: list[str]               # length = n_features
    sample_count: int
    feature_count: int

    # Problem-type specific indices
    temporal_index: np.ndarray | None      # (n_samples,) for temporal CV ordering
    entity_ids: np.ndarray | None          # (n_samples,) for entity grouping

    # Metadata
    problem_type: str                      # "time_series" | "entity_classification"
    target_name: str
    task_type: str                         # "regression" | "classification"
    class_names: list[str] | None          # for classification targets
```

---

## 3. Feature Engineering — Concrete Transformations

### 3.1 Time-Series Feature Pipeline

**Input:** Polars DataFrame of snapshot records, sorted by `timestamp`.

**Step-by-step transformation with concrete example:**

Given 3 raw snapshot rows (simplified):

```
timestamp | sys_wip | sys_throughput_per_hour | node_triage_queue
60        | 5       | 45.2                    | 2
120       | 7       | 44.8                    | 3
180       | 6       | 46.1                    | 1
```

After feature engineering (lookback=2 steps, horizon=1 step):

```
# Row for timestamp=180 (predicting value at timestamp=240):

Features:
  sys_wip_lag_1         = 7     (value at t-1)
  sys_wip_lag_2         = 5     (value at t-2)
  sys_wip_rolling_mean_2 = 6.0  (mean of last 2)
  sys_wip_rolling_std_2  = 1.0  (std of last 2)
  sys_wip_trend_2       = 0.5   (linear slope over last 2)
  sys_throughput_lag_1   = 44.8
  sys_throughput_lag_2   = 45.2
  sys_throughput_rolling_mean_2 = 45.0
  sys_throughput_rolling_std_2  = 0.283
  sys_throughput_trend_2 = -0.2
  node_triage_queue_lag_1 = 3
  node_triage_queue_lag_2 = 2
  wip_capacity_ratio    = 6 / total_capacity  (ratio feature)
  queue_imbalance       = max_queue / mean_queue (cross-node)

Target:
  sys_throughput_per_hour_horizon_1 = <value at timestamp 240>
```

**Feature naming convention:**

```
{column}_lag_{interval}              # lag feature
{column}_rolling_mean_{window}       # rolling mean
{column}_rolling_std_{window}        # rolling std
{column}_trend_{window}              # linear slope
{column}_delta_{interval}            # absolute change
{numerator}_to_{denominator}_ratio   # ratio feature
queue_imbalance                      # max_queue / mean_queue
utilization_imbalance                # max_util / mean_util
```

**Columns used as feature sources (auto-selected from snapshots):**

```python
# System-level (always included)
SYS_FEATURE_COLUMNS = [
    "sys_wip",
    "sys_total_in_queue",
    "sys_total_busy",
    "sys_arrival_rate_per_hour",
    "sys_throughput_per_hour",
    "sys_wip_ratio",
    "sys_avg_wait_time",
    "sys_avg_processing_time",
    "sys_sla_compliance",
]

# Per-node (included if include_cross_node=True)
# Pattern: node_{name}_{metric}
NODE_FEATURE_METRICS = [
    "queue", "busy", "utilization", "throughput_per_hour",
    "avg_wait_time", "avg_processing_time",
]

# Per-resource (always included if present)
# Pattern: resource_{name}_{metric}
RESOURCE_FEATURE_METRICS = ["utilization"]
```

### 3.2 Entity Feature Pipeline

**Input:** Polars DataFrame of trajectory records.

**Step-by-step transformation:**

Given a trajectory record for entity at step 2:

```python
state = {
    "entity_priority": 3.0,
    "entity_elapsed_time": 423.2,
    "entity_cumulative_wait": 185.3,
    "entity_cumulative_processing": 121.7,
    "entity_wait_ratio": 0.604,
    "entity_steps_completed": 2.0,
    "entity_cumulative_cost": 0.927,
    "entity_cumulative_transport_cost": 0.0,
    "entity_cumulative_setup_time": 0.0,
    "entity_source_index": 0.0,
    "node_utilization": 0.82,
    "node_avg_queue_depth": 6.3,
    "node_max_queue_depth": 12.0,
    "node_avg_wait_time": 95.4,
    "node_avg_processing_time": 55.2,
    "node_throughput_per_hour": 28.1,
    "node_concurrency": 3.0,
    "node_setup_count": 0.0,
    "node_mutation_count": 0.0,
    "sys_utilization": 0.71,
    "sys_throughput_per_hour": 35.4,
    "sys_bottleneck_utilization": 0.89,
    "sys_node_count": 4.0,
    "sys_total_capacity": 12.0,
    "resource_doctors_utilization": 0.78,
    "resource_doctors_capacity": 3.0,
}
```

After feature engineering:

```
# All state keys become features directly (flattened)
entity_priority            = 3.0
entity_elapsed_time        = 423.2
entity_cumulative_wait     = 185.3
...all 24+ state keys...

# Derived features (add_progress_ratio=True, add_wait_trend=True):
progress_ratio             = 2.0 / 4.0  (steps_completed / sys_node_count)
wait_trend                 = 185.3 / 423.2  (cumulative_wait / elapsed_time)
queue_relative             = 6.3 / mean_queue_across_all_entities  (relative congestion)
utilization_relative       = 0.82 / 0.71  (node vs system utilization)
cost_rate                  = 0.927 / 423.2  (cost per second elapsed)
```

**Feature group inclusion:**

```python
# include_entity_state=True (default): 10 features
ENTITY_STATE_KEYS = [
    "entity_priority", "entity_elapsed_time", "entity_cumulative_wait",
    "entity_cumulative_processing", "entity_wait_ratio", "entity_steps_completed",
    "entity_cumulative_cost", "entity_cumulative_transport_cost",
    "entity_cumulative_setup_time", "entity_source_index",
]

# include_node_state=True (default): 9 features
NODE_STATE_KEYS = [
    "node_utilization", "node_avg_queue_depth", "node_max_queue_depth",
    "node_avg_wait_time", "node_avg_processing_time", "node_throughput_per_hour",
    "node_concurrency", "node_setup_count", "node_mutation_count",
]

# include_system_state=True (default): 5 features
SYSTEM_STATE_KEYS = [
    "sys_utilization", "sys_throughput_per_hour", "sys_bottleneck_utilization",
    "sys_node_count", "sys_total_capacity",
]

# Resource and attr keys: included if present (dynamic)
# Pattern: resource_{name}_utilization, resource_{name}_capacity, attr_{name}
```

**Observation point filtering:**

```python
# observation_point="all_steps" (default):
#   Use every trajectory row — one sample per entity per step.
#   Most data, progressive prediction capability.

# observation_point="entry_only":
#   Use only step_index=0 rows — one sample per entity at entry.
#   Fewer samples, prediction at arrival time only.

# observation_point="midpoint":
#   Use step closest to steps_completed = total_steps / 2.
#   One sample per entity at journey midpoint.
```

**Entity classification target derivation:**

```python
# "episode_status" — directly from trajectory (always available)
# Values: "completed", "rejected", "timed_out"
# Encoded as integers: completed=0, rejected=1, timed_out=2

# "sla_breach" — derived from episode_total_time + SLA threshold
# If metadata has SLA info: breach = episode_total_time > sla_threshold
# If no SLA info: use p75 of episode_total_time as threshold
# Values: 0 (no breach), 1 (breach)

# "delay_severity" — derived from episode_total_time percentiles
# on_time: <= p50
# mild: p50 < t <= p75
# severe: p75 < t <= p95
# critical: > p95
# Encoded as integers: on_time=0, mild=1, severe=2, critical=3
```

---

## 4. Experiment YAML Schema

### 4.1 Full Schema (all fields, all optional except dataset)

```yaml
# Every field is optional except dataset.path or dataset.id.
# Omitted fields use defaults from CONSTITUTION.md §4.

experiment:
  name: string                          # default: auto-generated from timestamp
  seed: integer                         # default: 42

dataset:
  # One of path or id required
  path: string                          # file path to SimOS JSON, CSV, or Parquet
  id: string                            # registered dataset ID (if already uploaded)
  layer: string                         # "snapshots" | "trajectories" | auto-detected

features:
  target: string                        # column name | default per problem type
  # Time-series specific
  lookback: string                      # duration string: "8h", "4h", "30m" | default "8h"
  horizon: string                       # duration string: "1h", "30m" | default "1h"
  lag_intervals: list[string]           # default ["1h", "2h", "4h", "8h"]
  rolling_windows: list[string]         # default ["2h", "4h"]
  include_trend: boolean                # default true
  include_ratios: boolean               # default true
  include_cross_node: boolean           # default true
  # Entity specific
  observation_point: string             # "all_steps" | "entry_only" | "midpoint" | default "all_steps"
  include_entity_state: boolean         # default true
  include_node_state: boolean           # default true
  include_system_state: boolean         # default true
  add_progress_ratio: boolean           # default true
  add_wait_trend: boolean               # default true
  # Shared
  feature_columns: list[string]         # explicit column list (overrides auto-selection)
  exclude_columns: list[string]         # columns to exclude from auto-selection

model:
  algorithms: list[string]              # default ["lightgbm", "xgboost"] (TS) or ["lightgbm"] (entity)
  selection: string                     # "best_cv" | default "best_cv"
  cross_validation:
    strategy: string                    # "temporal" | "stratified_kfold" | "kfold" | default per type
    folds: integer                      # default 5
  handle_imbalance: boolean             # default true (entity only)
  hyperparameter_tuning: boolean        # default false

evaluation:
  metrics: list[string]                 # default per problem type
  generate_report: boolean              # default true
  plot_predictions: boolean             # default true
  plot_feature_importance: boolean      # default true
  plot_confusion_matrix: boolean        # default true (entity only)
  plot_roc_curve: boolean               # default true (entity only)
```

### 4.2 Minimal Examples

**Absolute minimum (zero config):**

```yaml
dataset:
  path: "./data/simos_export.json"
```

**Time-series with one override:**

```yaml
dataset:
  path: "./data/simos_export.json"
  layer: "snapshots"
features:
  target: "sys_wip"
```

**Entity classification with overrides:**

```yaml
dataset:
  path: "./data/simos_export.json"
  layer: "trajectories"
features:
  target: "sla_breach"
  observation_point: "entry_only"
model:
  algorithms: ["lightgbm", "xgboost", "random_forest"]
```

### 4.3 Duration String Parsing

```python
def parse_duration(s: str) -> int:
    """Parse duration string to seconds.

    Supported formats:
      "30m"  → 1800
      "1h"   → 3600
      "2h"   → 7200
      "8h"   → 28800
      "1d"   → 86400
      "30s"  → 30
      "1.5h" → 5400

    Rules:
      - Suffix required: s (seconds), m (minutes), h (hours), d (days)
      - Numeric part can be int or float
      - No spaces
    """
```

---

## 5. API Request/Response Schemas

### 5.1 Dataset Upload

```
POST /api/v1/datasets
Content-Type: multipart/form-data

Form fields:
  file: <binary>                         # JSON, CSV, or Parquet file
  name: string (optional)                # dataset name

Response 201:
{
  "dataset_id": "ds_20260309_143022_a1b2c3",
  "name": "simos_export_healthcare",
  "version": 1,
  "source_type": "simos",
  "has_snapshots": true,
  "has_trajectories": true,
  "snapshot_row_count": 1440,
  "trajectory_row_count": 3200,
  "registered_at": "2026-03-09T14:30:22Z"
}
```

### 5.2 Available Targets

```
GET /api/v1/datasets/{id}/available-targets

Response 200:
{
  "dataset_id": "ds_20260309_143022_a1b2c3",
  "time_series_targets": [
    {
      "column": "sys_avg_wait_time",
      "task_type": "regression",
      "is_default": true,
      "null_rate": 0.0,
      "mean": 85.2,
      "std": 32.1,
      "min": 12.4,
      "max": 245.8
    },
    {
      "column": "sys_throughput_per_hour",
      "task_type": "regression",
      "is_default": false,
      "null_rate": 0.0,
      "mean": 42.3,
      "std": 8.7,
      "min": 18.1,
      "max": 52.0
    }
  ],
  "entity_targets": [
    {
      "column": "episode_status",
      "task_type": "classification",
      "is_default": true,
      "null_rate": 0.0,
      "unique_count": 3,
      "classes": ["completed", "rejected", "timed_out"],
      "class_balance": {"completed": 0.85, "rejected": 0.05, "timed_out": 0.10}
    },
    {
      "column": "sla_breach",
      "task_type": "classification",
      "is_default": false,
      "null_rate": 0.0,
      "unique_count": 2,
      "classes": ["no", "yes"],
      "class_balance": {"no": 0.78, "yes": 0.22}
    }
  ]
}
```

### 5.3 Experiment Submission

```
POST /api/v1/experiments
Content-Type: application/json

Request body (minimal):
{
  "dataset_id": "ds_20260309_143022_a1b2c3"
}

Request body (with overrides):
{
  "dataset_id": "ds_20260309_143022_a1b2c3",
  "dataset_layer": "snapshots",
  "target": "sys_throughput_per_hour",
  "lookback": "4h",
  "horizon": "1h",
  "algorithms": ["lightgbm"]
}

Response 202 (accepted, training started):
{
  "experiment_id": "exp_20260309_143055_x7y8z9",
  "status": "running",
  "experiment_type": "time_series",
  "resolved_config": { ... full resolved config ... }
}
```

### 5.4 Experiment Validation Only

```
POST /api/v1/experiments/validate
Content-Type: application/json

Request body: (same as experiment submission)

Response 200 (valid):
{
  "valid": true,
  "resolved_config": { ... },
  "errors": [],
  "warnings": ["Column 'sys_cumulative_arrivals' is monotonically increasing — poor forecast target"]
}

Response 200 (invalid):
{
  "valid": false,
  "resolved_config": null,
  "errors": [
    {
      "code": "V-03",
      "field": "features.target",
      "message": "Target 'nonexistent_col' not found in dataset",
      "suggestion": "Available targets: sys_avg_wait_time, sys_throughput_per_hour, sys_wip, ..."
    },
    {
      "code": "VT-02",
      "field": "features.lookback",
      "message": "Lookback (24h) + horizon (1h) exceeds data duration (8h)",
      "suggestion": "Reduce lookback to 7h or less"
    }
  ],
  "warnings": []
}
```

### 5.5 Experiment Result

```
GET /api/v1/experiments/{id}

Response 200 (completed):
{
  "experiment_id": "exp_20260309_143055_x7y8z9",
  "name": "throughput_forecast",
  "status": "completed",
  "experiment_type": "time_series",
  "created_at": "2026-03-09T14:30:55Z",
  "completed_at": "2026-03-09T14:31:12Z",
  "duration_seconds": 17.3,

  "resolved_config": {
    "name": "throughput_forecast",
    "experiment_type": "time_series",
    "seed": 42,
    "dataset_id": "ds_20260309_143022_a1b2c3",
    "dataset_layer": "snapshots",
    "features": {
      "target": "sys_throughput_per_hour",
      "lookback": "4h",
      "horizon": "1h",
      "lag_intervals": ["1h", "2h", "4h"],
      "rolling_windows": ["2h", "4h"],
      "include_trend": true,
      "include_ratios": true,
      "include_cross_node": true,
      "feature_columns": ["sys_wip", "sys_throughput_per_hour", "sys_avg_wait_time", "..."]
    },
    "model": {
      "algorithms": ["lightgbm"],
      "selection": "best_cv",
      "cross_validation": {"strategy": "temporal", "folds": 5},
      "handle_imbalance": false,
      "hyperparameter_tuning": false
    },
    "evaluation": {
      "metrics": ["rmse", "mae", "mape"],
      "generate_report": true,
      "plot_predictions": true,
      "plot_feature_importance": true,
      "plot_confusion_matrix": false,
      "plot_roc_curve": false
    }
  },

  "best_algorithm": "lightgbm",
  "metrics": {
    "rmse": 3.21,
    "mae": 2.14,
    "mape": 0.048
  },
  "all_algorithm_scores": [
    {
      "algorithm": "lightgbm",
      "metrics": {"rmse": 3.21, "mae": 2.14, "mape": 0.048},
      "metrics_std": {"rmse": 0.45, "mae": 0.31, "mape": 0.008}
    }
  ],
  "feature_importance": {
    "sys_throughput_per_hour_lag_1h": 0.234,
    "sys_wip_lag_1h": 0.187,
    "sys_avg_wait_time_lag_1h": 0.142,
    "sys_throughput_per_hour_rolling_mean_2h": 0.098,
    "sys_throughput_per_hour_trend_4h": 0.076
  },
  "model_id": "mdl_20260309_143112_p1q2r3",
  "sample_count": 1200,
  "feature_count": 45
}
```

### 5.6 Model Prediction

```
POST /api/v1/models/{id}/predict
Content-Type: application/json

Request body:
{
  "data": [
    {"sys_wip": 15, "sys_throughput_per_hour": 38.2, "sys_avg_wait_time": 120.5, "...": "..."},
    {"sys_wip": 18, "sys_throughput_per_hour": 35.1, "sys_avg_wait_time": 145.2, "...": "..."}
  ]
}

Response 200 (regression):
{
  "model_id": "mdl_20260309_143112_p1q2r3",
  "predictions": [41.3, 38.7],
  "task_type": "regression"
}

Response 200 (classification):
{
  "model_id": "mdl_...",
  "predictions": ["completed", "timed_out"],
  "probabilities": [
    {"completed": 0.85, "rejected": 0.03, "timed_out": 0.12},
    {"completed": 0.25, "rejected": 0.15, "timed_out": 0.60}
  ],
  "task_type": "classification"
}
```

---

## 6. Algorithm Wrapper Specifications

### 6.1 LightGBM Wrapper

```python
class LightGBMAlgorithm:
    name = "lightgbm"
    supports_regression = True
    supports_classification = True

    # Default hyperparameters (not exposed to user in v0.1)
    REGRESSION_PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_jobs": -1,
    }

    CLASSIFICATION_PARAMS = {
        "objective": "multiclass",       # auto-switches to binary if 2 classes
        "metric": "multi_logloss",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "is_unbalance": True,            # handle class imbalance
        "verbose": -1,
        "n_jobs": -1,
    }
```

### 6.2 XGBoost Wrapper

```python
class XGBoostAlgorithm:
    name = "xgboost"
    supports_regression = True
    supports_classification = True

    REGRESSION_PARAMS = {
        "objective": "reg:squarederror",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbosity": 0,
        "n_jobs": -1,
    }

    CLASSIFICATION_PARAMS = {
        "objective": "multi:softprob",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbosity": 0,
        "n_jobs": -1,
    }
```

### 6.3 Random Forest Wrapper

```python
class RandomForestAlgorithm:
    name = "random_forest"
    supports_regression = True
    supports_classification = True

    REGRESSION_PARAMS = {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "n_jobs": -1,
    }

    CLASSIFICATION_PARAMS = {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "n_jobs": -1,
    }
```

---

## 7. Evaluation Report Schema

```python
class EvaluationReport(BaseModel):
    """Full evaluation report — stored as JSON, renderable as HTML."""
    experiment_id: str
    experiment_name: str
    experiment_type: str                   # "time_series" | "entity_classification"
    created_at: str

    # Dataset info
    dataset_id: str
    dataset_name: str
    sample_count: int
    feature_count: int
    target: str
    task_type: str

    # Best model
    best_algorithm: str
    best_metrics: dict[str, float]

    # All algorithms compared
    algorithm_scores: list[AlgorithmScore]

    # Feature importance (top 20)
    feature_importance: list[FeatureImportanceEntry]

    # Predictions sample (first 100 for visualization)
    predictions_sample: list[PredictionSample]

    # Classification-specific
    confusion_matrix: list[list[int]] | None       # n_classes x n_classes
    class_names: list[str] | None
    roc_curve: dict[str, list[float]] | None       # {"fpr": [...], "tpr": [...], "auc": float}

    # Config used
    resolved_config: dict

class AlgorithmScore(BaseModel):
    algorithm: str
    metrics: dict[str, float]
    metrics_std: dict[str, float]
    rank: int

class FeatureImportanceEntry(BaseModel):
    feature: str
    importance: float
    rank: int

class PredictionSample(BaseModel):
    index: int
    actual: float | str
    predicted: float | str
    # Classification: add probabilities
    probabilities: dict[str, float] | None
```

---

## 8. File Storage Layout

```
$MLRL_DATA_DIR/                          # default: ./data/
├── datasets/
│   ├── ds_20260309_143022_a1b2c3/
│   │   ├── meta.json                    # DatasetMeta
│   │   ├── snapshots.parquet            # Layer 3 data (if present)
│   │   └── trajectories.parquet         # Layer 2 data (if present)
│   └── ds_20260310_091500_d4e5f6/
│       └── ...

$MLRL_MODELS_DIR/                        # default: ./models/
├── mdl_20260309_143112_p1q2r3/
│   ├── meta.json                        # ModelMeta (provenance)
│   ├── model.joblib                     # serialized sklearn/lgbm/xgb model
│   └── feature_importance.json
└── mdl_.../

$MLRL_EXPERIMENTS_DIR/                   # default: ./experiments/
├── exp_20260309_143055_x7y8z9/
│   ├── config.json                      # ResolvedExperimentConfig
│   ├── result.json                      # ExperimentResult summary
│   ├── report.json                      # Full EvaluationReport
│   └── report.html                      # Rendered HTML report (if generated)
└── exp_.../
```

**ID generation:**

```python
import uuid
from datetime import datetime, timezone

def generate_id(prefix: str) -> str:
    """Generate unique ID: {prefix}_{timestamp}_{random6}.

    Example: ds_20260309_143022_a1b2c3
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:6]
    return f"{prefix}_{ts}_{rand}"
```
