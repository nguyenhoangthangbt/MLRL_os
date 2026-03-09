# CONSTITUTION

> Principles, glossary, constraints, and non-goals for ML/RL OS.
>
> This document is the authority on terminology and design philosophy. Every other design document inherits from it. If a term is used elsewhere and not defined here, that is a bug.
>
> Last updated: 2026-03-09.

---

## 1. Purpose & Vision

**One-line:** Load operational data. Build an experiment. Validate it. Train models. Get intelligence. Prove it.

ML/RL OS is a predictive intelligence instrument for operational systems. It consumes structured operational data — from Simulation OS exports, CSV files, Parquet datasets, or external sources — and produces trained models, forecasts, classifications, and evaluation reports through a validated experiment pipeline.

### What This Is Not

This is not a Jupyter notebook with a web interface. Jupyter gives you a blank canvas and leaves you to write everything: data loading, feature engineering, model selection, evaluation, reproducibility. PyCaret gives you AutoML but hides the decisions. Neither gives you a validated experiment pipeline that produces operational intelligence out of the box.

ML/RL OS inverts the typical ML workflow:

| Traditional ML | ML/RL OS |
|---|---|
| Write Python scripts per experiment | Configure YAML or use visual Builder |
| Manually select features | System auto-discovers, user overrides |
| Pick model, tune hyperparameters | Submit config, system selects best |
| Build custom evaluation logic | Get automatic evaluation reports |
| Track experiments in spreadsheets | Built-in experiment registry with provenance |
| Struggle with reproducibility | Seed + config = identical results, guaranteed |

### What Has Shipped (v0.1 Target)

Target capabilities for v0.1:

- **Two prediction problem types:** time-series forecasting and entity classification
- **Experiment Builder:** visual interface to select features, targets, and model settings
- **Validation gate:** experiments must pass validation before training begins
- **Config-driven experiments:** YAML configuration with sensible defaults
- **Auto-discovery:** system inspects data and exposes available features/targets
- **Three-tier configuration:** zero-config (auto-pilot), YAML overrides, or Builder UI
- **SimOS data ingestion:** native support for SimOS 5-layer ML export format
- **External data support:** CSV and Parquet ingestion with schema validation
- **Model engine:** LightGBM, XGBoost, Random Forest with unified interface
- **Evaluation engine:** metrics, comparison, and report generation
- **Model registry:** versioned model storage with full training provenance
- **Experiment tracker:** history of all experiments with artifacts
- **REST API:** FastAPI endpoints for programmatic access

### What Ships in v0.2

- Reinforcement learning: policy training against SimOS environments
- Sequence models: LSTM/Transformer for time-series (if lag features prove insufficient)
- AgentsOS integration: experiment orchestration and knowledge accumulation
- Streaming inference: real-time predictions during live simulation

---

## 2. Glossary

Every term used in this project. Alphabetical.

| Term | Definition |
|---|---|
| **Algorithm** | A specific ML implementation (e.g., LightGBM regressor). Algorithms conform to the Algorithm protocol and are registered in the Algorithm Registry. |
| **Available Targets** | The set of columns in a dataset that can serve as prediction targets. Auto-discovered by inspecting data schema. |
| **Builder** | The visual web interface for constructing experiments. Users select dataset, features, target, model settings, then submit for validation. |
| **Cross-Validation** | Model evaluation strategy that partitions data into train/test folds. Time-series uses temporal CV (no future leakage). Entity classification uses stratified k-fold. |
| **Dataset** | A versioned, validated collection of operational data ready for feature engineering. Immutable after registration. |
| **Default Config** | Sensible default values for all experiment settings. Applied when user provides no override. See §4 Default Policy. |
| **Entity Classification** | Problem type B. Given an entity's state at a point in its journey, predict its outcome (completed/rejected/timed_out) or SLA breach risk. |
| **Evaluation Report** | Structured output from the Evaluation Engine: metrics, feature importance, predictions vs actuals, model comparison. |
| **Experiment** | The fundamental unit of work. An experiment has: id, config (resolved), dataset reference, trained model(s), evaluation results, and provenance metadata. |
| **Experiment Builder** | See Builder. |
| **Feature** | A single input column fed to a model. Features are either raw (from data) or engineered (derived from raw features via transformations). |
| **Feature Engineering** | Transformations applied to raw data to create model-ready features. Includes: lag values, rolling statistics, trend extraction, ratio computation, progress indicators. |
| **Feature Store** | Registry of reusable feature definitions. A feature definition specifies the transformation logic and can be applied to any compatible dataset. |
| **Horizon** | For time-series forecasting: how far ahead to predict. Example: horizon=1h means predict system state 1 hour into the future. |
| **Lag Feature** | A feature created by shifting a time-series column backwards. `queue_lag_1h` = queue depth 1 hour ago. |
| **Layer 2 (Trajectories)** | SimOS export layer containing per-entity, per-step MDP records with state vectors. Primary data source for entity classification. |
| **Layer 3 (Snapshots)** | SimOS export layer containing periodic system state observations. Primary data source for time-series forecasting. |
| **Lookback** | For time-series forecasting: how much historical data to use as features. Example: lookback=8h means use last 8 hours of snapshots. |
| **Model** | A trained artifact: algorithm + learned parameters + training provenance. Immutable after registration. |
| **Model Registry** | Versioned storage for trained models. Each entry includes: model artifact, training config, dataset reference, evaluation metrics, timestamp. |
| **Observation Point** | For entity classification: at which step(s) of an entity's journey to make predictions. Options: `entry_only`, `midpoint`, `all_steps`. |
| **Problem Type** | The class of prediction task. v0.1 supports two: `time_series` and `entity_classification`. Auto-detected from dataset layer or explicitly specified. |
| **Provenance** | Complete lineage of a trained model: dataset version, experiment config, seed, algorithm version, training timestamp. Enables full reproducibility. |
| **Resolved Config** | The fully-specified experiment configuration after merging user overrides onto defaults. The experiment runner only operates on resolved configs. |
| **Rolling Feature** | A feature computed over a sliding window. `queue_rolling_mean_4h` = mean queue depth over last 4 hours. |
| **Seed** | Integer controlling all random processes. Same seed + same config + same data = identical results. Non-negotiable. |
| **SimOS Export** | The 5-layer JSON/CSV output from Simulation OS containing event streams, entity trajectories, state snapshots, domain enrichment, and stress scenarios. |
| **Target** | The column being predicted. For time-series: a future numeric value. For entity classification: a categorical outcome. |
| **Temporal CV** | Cross-validation strategy for time-series that respects chronological ordering. Training data always precedes test data. No future leakage. |
| **Time-Series Forecasting** | Problem type A. Given historical system snapshots, predict future system state (lead time, throughput, queue depth, etc.). |
| **Trend Feature** | A feature capturing the direction of change. `utilization_slope_4h` = linear regression slope of utilization over last 4 hours. |
| **Validation Gate** | The checkpoint between experiment submission and training execution. Config must pass all validation rules before training begins. Invalid experiments are rejected with specific error messages. |

---

## 3. Design Principles

### P1 — Validate Before You Train

No experiment runs without passing the validation gate. Validation checks:

- Dataset exists, is readable, and has sufficient rows
- Target column exists in dataset
- Feature columns exist and have compatible types
- No data leakage (target not in features)
- Time-series: lookback + horizon does not exceed data duration
- Entity classification: target classes have minimum sample count
- Cross-validation strategy is compatible with problem type
- Algorithm supports the task type (regression/classification)

Invalid experiments return all errors, not just the first one. Users fix and resubmit.

### P2 — Convention Over Configuration

Every experiment setting has a sensible default. Users configure only what they want to change. Zero-config experiments are valid and produce reasonable results.

The configuration resolution order:
1. System defaults (built-in, per problem type)
2. User YAML overrides (partial config)
3. Builder UI selections (equivalent to YAML)
4. Validation gate (catches conflicts)
5. Resolved config (fully specified, logged with experiment)

### P3 — Reproducibility Is Non-Negotiable

Every experiment is fully reproducible. Requirements:

- Global seed controls all randomness
- Seeded RNG per component (same pattern as SimOS: `seed_hash(component_name, global_seed)`)
- Dataset version is recorded (content hash)
- Resolved config is stored with results
- Algorithm version is recorded
- Same inputs = identical outputs, always

### P4 — Auto-Discovery Over Hard-Coding

Features and targets are discovered from data, not hard-coded in the platform:

- System inspects dataset schema and exposes available columns
- Numeric columns → candidate features and regression targets
- Categorical columns → candidate classification targets
- Temporal columns → enable time-series problem type
- User selects from discovered options (or accepts defaults)

New SimOS export formats, external datasets with different schemas, and domain-specific attributes all work without code changes.

### P5 — Analysis Is the Product

Training a model is infrastructure. The product is:

- Evaluation reports with metrics and confidence intervals
- Feature importance analysis (which inputs drive predictions)
- Prediction accuracy visualization (predicted vs actual)
- Model comparison (which algorithm/config performed best)
- Forecast visualization with uncertainty bands

### P6 — Builder Is the Interface

The Experiment Builder is how most users interact with ML/RL OS. The workflow:

```
Select Dataset → Explore Data → Pick Features → Pick Target
    → Configure Model (or accept defaults) → Submit
    → Validation Gate (pass/fail with errors)
    → Train → Evaluate → Report
```

The Builder produces YAML. Users who prefer YAML can skip the Builder. Both paths converge at the same validation gate.

### P7 — Instrument Architecture

ML/RL OS is an instrument, not a notebook. It runs structured experiment pipelines:

```
dataset → validation → feature engineering → training → evaluation → reporting
```

No ad-hoc code execution. No unstructured exploration. Experiments are the unit of work.

### P8 — Two Problem Types, One Pipeline Pattern

Time-series forecasting and entity classification share the same pipeline structure. The difference is in feature engineering and evaluation:

| Pipeline Stage | Time-Series | Entity Classification |
|---|---|---|
| Data source | Layer 3 snapshots | Layer 2 trajectories |
| Feature engineering | Windowing, lag, rolling, trend | Entity state, node state, system state |
| Target | Future numeric value | Categorical outcome |
| CV strategy | Temporal (no leakage) | Stratified k-fold |
| Evaluation metrics | RMSE, MAE, MAPE | F1, AUC-ROC, precision, recall |

The Model Engine, Experiment Runner, Evaluation Engine, and Model Registry are shared.

### P9 — Loose Coupling with SimOS

ML/RL OS consumes SimOS output data. It does not import SimOS code, call SimOS APIs during training, or depend on SimOS being available. Integration is through data files:

- SimOS exports JSON/CSV/Parquet → ML/RL OS ingests them
- ML/RL OS has a SimOS-aware data loader that understands the 5-layer export format
- External data (non-SimOS) is equally supported via CSV/Parquet loaders
- No runtime dependency on SimOS

### P10 — Progressive Complexity

The platform serves three user levels:

| Level | Interface | Config Effort | User Profile |
|---|---|---|---|
| Beginner | Builder UI | Zero (accept defaults) | Operations analyst, first-time ML |
| Intermediate | Builder UI + overrides | Select features/target, tweak settings | Data-savvy operations manager |
| Advanced | YAML + API | Full control over pipeline | Data scientist, researcher |

All three levels produce the same quality of results. The difference is control, not capability.

---

## 4. Default Policy

When a user provides no configuration for a setting, the system applies these defaults.

### Time-Series Forecasting Defaults

```yaml
features:
  target: "sys_avg_lead_time"
  lookback: "8h"
  horizon: "1h"
  lag_intervals: ["1h", "2h", "4h", "8h"]
  rolling_windows: ["2h", "4h"]
  include_trend: true
  include_ratios: true
  include_cross_node: true

model:
  algorithms: ["lightgbm", "xgboost"]
  selection: "best_cv"
  cross_validation:
    strategy: "temporal"
    folds: 5
  hyperparameter_tuning: false

evaluation:
  metrics: ["rmse", "mae", "mape"]
  generate_report: true
  plot_predictions: true
  plot_feature_importance: true
```

### Entity Classification Defaults

```yaml
features:
  target: "episode_status"
  observation_point: "all_steps"
  include_entity_state: true
  include_node_state: true
  include_system_state: true
  add_progress_ratio: true
  add_wait_trend: true

model:
  algorithms: ["lightgbm"]
  selection: "best_cv"
  cross_validation:
    strategy: "stratified_kfold"
    folds: 5
  handle_imbalance: true
  hyperparameter_tuning: false

evaluation:
  metrics: ["f1_weighted", "auc_roc", "precision", "recall"]
  generate_report: true
  plot_confusion_matrix: true
  plot_feature_importance: true
  plot_roc_curve: true
```

### Override Rules

- User values always win over defaults
- Partial override is valid (override `target` only, keep all other defaults)
- Invalid overrides are caught at validation gate (e.g., `target: "nonexistent_column"`)
- Resolved config (defaults + overrides) is stored with every experiment for reproducibility

---

## 5. Validation Gate Rules

Experiments must pass ALL rules before training begins. All violations are returned, not just the first.

### Universal Rules (both problem types)

| Rule | Check | Error Message |
|---|---|---|
| V-01 | Dataset path exists and is readable | `Dataset not found: {path}` |
| V-02 | Dataset has minimum row count (≥50 for TS, ≥100 for entity) | `Insufficient data: {n} rows, minimum {min} required` |
| V-03 | Target column exists in dataset | `Target '{target}' not found. Available: {columns}` |
| V-04 | Target not included in feature columns | `Data leakage: target '{target}' is in feature set` |
| V-05 | All specified feature columns exist | `Feature '{col}' not found. Available: {columns}` |
| V-06 | Feature columns have compatible types (numeric) | `Feature '{col}' is {type}, expected numeric` |
| V-07 | Seed is a non-negative integer | `Seed must be a non-negative integer, got {seed}` |
| V-08 | Algorithm names are registered | `Unknown algorithm '{name}'. Available: {algorithms}` |
| V-09 | CV folds ≥ 2 | `Cross-validation folds must be ≥ 2, got {folds}` |
| V-10 | Metric names are registered | `Unknown metric '{name}'. Available: {metrics}` |
| V-11 | No NaN/null in target column exceeds 10% | `Target '{target}' has {pct}% missing values, max 10%` |

### Time-Series Rules

| Rule | Check | Error Message |
|---|---|---|
| VT-01 | Dataset has temporal ordering column | `No timestamp column found for time-series` |
| VT-02 | Lookback + horizon ≤ data duration | `Lookback ({lb}) + horizon ({hz}) exceeds data duration ({dur})` |
| VT-03 | Lookback produces ≥ 50 usable windows | `Lookback {lb} produces only {n} windows, minimum 50` |
| VT-04 | Target is numeric type | `Time-series target must be numeric, '{target}' is {type}` |
| VT-05 | Lag intervals ≤ lookback | `Lag interval {lag} exceeds lookback {lb}` |
| VT-06 | CV strategy is temporal (not random k-fold) | `Time-series requires temporal CV, got '{strategy}'` |

### Entity Classification Rules

| Rule | Check | Error Message |
|---|---|---|
| VE-01 | Target is categorical or can be derived | `Entity target must be categorical, '{target}' is {type}` |
| VE-02 | All target classes have ≥ 10 samples | `Class '{cls}' has only {n} samples, minimum 10` |
| VE-03 | Number of classes ≤ 20 | `Too many classes ({n}), maximum 20` |
| VE-04 | Observation point is valid | `Invalid observation_point '{val}'. Options: entry_only, midpoint, all_steps` |

---

## 6. Constraints

### C-01 No Training Without Validation

The experiment runner refuses to train if validation fails. No bypass flag. No override. Fix the config and resubmit.

### C-02 No Future Leakage in Time-Series

Temporal cross-validation is mandatory for time-series. Random k-fold is rejected at validation. Rolling window features are computed using only past data. The system guarantees that no future information leaks into training.

### C-03 Seeded Everything

Every random process uses a deterministic seed derived from the experiment's global seed:

```python
component_seed = seed_hash(component_name, global_seed)
```

Components include: train/test split, cross-validation fold assignment, algorithm initialization, feature sampling (if applicable).

### C-04 Immutable Artifacts

Once registered, datasets and models are immutable. New versions create new entries. This ensures provenance chains are never broken.

### C-05 No Heavy Dependencies in Core

Core modules must not import LightGBM, XGBoost, or other algorithm-specific libraries at module level. Algorithms are lazy-loaded through the registry. This keeps startup fast and allows users to install only the algorithms they need.

### C-06 SimOS Is a Data Source, Not a Dependency

ML/RL OS never imports from `simulation_os`. SimOS exports are treated as data files with a known schema. If SimOS changes its export format, only the SimOS data loader in ML/RL OS needs updating.

### C-07 Builder Produces YAML

The web Builder does not have its own experiment format. It produces the same YAML that a user would write by hand. This ensures a single code path: YAML → validation → training.

---

## 7. Non-Goals

Things ML/RL OS intentionally does NOT do:

| Non-Goal | Reason |
|---|---|
| Run simulations | That's SimOS's job. Simulation is cheap — just run it. |
| Predict config → outcome | SimOS L1 already does this. A simulation run is faster and more accurate than a surrogate model. |
| Deep learning research | Not a research framework. Use PyTorch/TensorFlow directly for novel architectures. |
| Unstructured data (images, text, audio) | Focus is structured operational data and decision optimization. |
| Notebook-style exploration | This is an instrument with structured pipelines, not an IDE. |
| Real-time streaming inference (v0.1) | Deferred to v0.2. v0.1 operates on batch data. |
| AutoML hyperparameter search (v0.1) | Defaults are good enough. Tuning adds complexity without proportional value for v0.1. |
| Replace PyCaret / scikit-learn | ML/RL OS orchestrates these libraries. It is not a replacement. |

---

## 8. Data Sources

### SimOS Export (Primary)

ML/RL OS natively understands SimOS's 5-layer ML export format (export_version 3.0):

| Layer | Name | ML/RL OS Usage |
|---|---|---|
| Layer 1 | Event Stream | Not used in v0.1 (raw events, too granular) |
| Layer 2 | Entity Trajectories | Entity classification: features + targets |
| Layer 3 | State Snapshots | Time-series forecasting: features + targets |
| Layer 4 | Domain Enrichment | Additional entity features (domain-specific) |
| Layer 5 | Stress Scenarios | Not used in v0.1 (config change descriptors for curriculum RL) |

### External Data (Secondary)

CSV and Parquet files with a schema header. The system auto-discovers columns and types. Users must specify whether the data represents time-series observations or entity records.

### Data Quality Requirements

| Check | Threshold | Behavior |
|---|---|---|
| Minimum rows | 50 (TS) / 100 (entity) | Reject at validation |
| Missing values in target | ≤ 10% | Reject if exceeded |
| Missing values in features | ≤ 30% per column | Warn; impute with median (numeric) |
| Constant columns | 0 variance | Auto-exclude from features, warn |
| Duplicate rows | Any | Warn; deduplicate |

---

## 9. Experiment Builder Workflow

The Builder is a web interface (React, matching SimOS's web stack) that guides users through experiment construction.

### Builder Flow

```
Step 1: SELECT DATASET
  - Upload new data (CSV/Parquet/SimOS JSON)
  - Or select from dataset registry
  - System shows: row count, column count, data quality summary

Step 2: EXPLORE DATA
  - Column list with types and basic statistics
  - Distribution previews for numeric columns
  - Class balance preview for categorical columns
  - Data quality indicators (missing %, constant columns)

Step 3: SELECT PROBLEM TYPE
  - Auto-suggested based on data structure
  - User confirms: "Time-Series Forecasting" or "Entity Classification"

Step 4: SELECT TARGET
  - System shows available targets (auto-discovered from data)
  - Default target is pre-selected
  - User picks or accepts default
  - For classification: shows class distribution
  - For time-series: shows target over time

Step 5: SELECT FEATURES
  - All available features shown with relevance indicators
  - Defaults pre-selected
  - User adds/removes features
  - Feature engineering options:
    - Time-series: lag intervals, rolling windows, trend, ratios
    - Entity: include node state, system state, progress ratio
  - Preview: shows engineered feature matrix sample

Step 6: CONFIGURE MODEL (optional)
  - Algorithm selection (defaults pre-selected)
  - CV strategy (default shown, override available)
  - Metrics (default shown, add/remove)
  - All optional — defaults work out of the box

Step 7: SUBMIT
  - Builder generates YAML from selections
  - YAML shown to user (editable)
  - User clicks "Validate & Run"

Step 8: VALIDATION GATE
  - Pass → experiment starts training
  - Fail → error list shown, user returns to fix
  - Each error links to the relevant Builder step

Step 9: TRAINING & RESULTS
  - Progress indicator
  - Results page: metrics, feature importance, predictions vs actuals
  - Model registered in Model Registry
  - Experiment stored in Experiment Tracker
```

### Builder Produces YAML

The Builder's output is a standard experiment YAML. Example output from a Builder session:

```yaml
# Generated by ML/RL OS Builder
# Experiment: throughput_forecast_2026-03-09_14-30
experiment:
  name: "throughput_forecast"
  type: "time_series"
  seed: 42

dataset:
  id: "ds_sc01_20260309"
  layer: "snapshots"

features:
  target: "sys_throughput_per_hour"
  lookback: "8h"
  horizon: "1h"
  lag_intervals: ["1h", "2h", "4h", "8h"]
  rolling_windows: ["2h", "4h"]
  include_trend: true
  include_ratios: true

model:
  algorithms: ["lightgbm", "xgboost"]
  selection: "best_cv"
  cross_validation:
    strategy: "temporal"
    folds: 5

evaluation:
  metrics: ["rmse", "mae", "mape"]
  generate_report: true
```

Users can export this YAML, modify it, and resubmit via API — enabling both UI and CLI workflows.

---

## 10. Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| Language | Python 3.13+ | Matches SimOS, richest ML ecosystem |
| API | FastAPI + Pydantic v2 | Consistency with SimOS, async-ready, type-safe |
| ML Core | scikit-learn | Foundation: preprocessing, CV, metrics |
| Gradient Boosting | LightGBM + XGBoost | Production-proven, fast CPU training, top tabular performance |
| Ensemble | scikit-learn | Random Forest, Extra Trees |
| Data | Polars (primary) + pandas (compatibility) | Polars: 10-50x faster, lazy eval, lower memory |
| Dataset Format | Parquet (storage) + JSON (SimOS interchange) | Columnar, compressed, fast I/O |
| Web UI | React 19 + TypeScript + Vite | Matches SimOS web stack |
| Charts | Plotly.js (lazy loaded) | Matches SimOS, rich interactive charts |
| Testing | pytest (strict) | 90%+ coverage target from day one |
| Type Checking | mypy strict | Matches SimOS discipline |
| Linting | ruff | Matches SimOS |
| Config | Pydantic BaseSettings | Environment variable management |

### What Is NOT in v0.1

| Technology | When | Reason to Defer |
|---|---|---|
| PyTorch / TensorFlow | v0.2 (RL) | Not needed for tabular ML |
| Ray / Dask | v0.3+ | Overkill for single-machine training |
| MLflow / W&B | Never (built-in) | Own experiment tracking, no external dependency |
| Docker | Deployment phase | Dev first, containerize later |
| PostgreSQL | v0.2 | In-memory + file storage sufficient for v0.1 |

---

## 11. Position in the Ecosystem

```
AgentsOS (future)
   ↓ orchestrates experiments
ML/RL OS
   ↑ consumes data from
SimOS
```

### Data Flow

```
SimOS
  → runs simulation
  → exports 5-layer JSON (via API or file)
  → ML/RL OS ingests export
  → user builds experiment via Builder or YAML
  → validation gate
  → training pipeline
  → evaluation report
  → model registered
  → experiment tracked

AgentsOS (v0.3+)
  → proposes experiments
  → submits to ML/RL OS API
  → analyzes results
  → stores insights in knowledge base
  → proposes next experiment
```

### Independence Guarantee

Each instrument operates independently:

- SimOS runs without ML/RL OS
- ML/RL OS runs without SimOS (using external data)
- AgentsOS coordinates both but neither depends on it

---

## 12. Success Criteria for v0.1

| Criterion | Measurable Target |
|---|---|
| Time-series forecasting works end-to-end | Load SimOS snapshots → train → evaluate → report |
| Entity classification works end-to-end | Load SimOS trajectories → train → evaluate → report |
| Zero-config experiments produce valid results | Submit dataset-only YAML → get trained model + report |
| User overrides work correctly | Override target/features/algorithm → results reflect changes |
| Builder generates valid YAML | Every Builder submission produces parseable, valid YAML |
| Validation gate catches all errors | Invalid configs rejected with specific error messages |
| Reproducibility holds | Same seed + config + data = identical metrics across runs |
| Test coverage ≥ 90% | pytest --cov reports ≥ 90% on all core modules |
| SimOS export ingestion works | All 5-layer JSON exports load without error |
| External CSV/Parquet ingestion works | Standard tabular files load with auto-discovery |
