# ML/RL OS — Development Guide for Claude Code

> **Context:** ML/RL OS is a predictive intelligence instrument for operational systems. It consumes structured operational data (primarily from Simulation OS exports) and produces trained models, forecasts, entity classifications, and evaluation reports through a validated experiment pipeline. The project is in early development (v0.1).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Web UI (web/)                                   Port 5175  │
│  React 19 + Vite + TypeScript + Tailwind                    │
│  Experiment Builder (7-step flow → YAML → validate → run)   │
│  Results Dashboard (metrics, charts, feature importance)    │
├─────────────────────────────────────────────────────────────┤
│  Backend API (src/mlrl_os/)                      Port 8001  │
│  Python 3.13 + FastAPI + Pydantic v2                        │
│  16 REST endpoints                                          │
├─────────────────────────────────────────────────────────────┤
│  Storage: File System (v0.1)                                │
│  ./data/ (datasets) │ ./models/ │ ./experiments/            │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.13+, FastAPI, Pydantic v2 |
| **ML Core** | scikit-learn, LightGBM, XGBoost |
| **Data** | Polars (primary), pandas (compatibility) |
| **Dataset Format** | Parquet (storage), JSON (SimOS interchange), CSV (external) |
| **Web UI** | React 19, TypeScript, Vite, Tailwind CSS |
| **Charts** | Plotly.js (lazy loaded) |
| **State** | TanStack React Query (server), Zustand (client) |
| **Testing** | pytest (backend), Vitest (frontend) |
| **Linting** | ruff (backend), ESLint (frontend) |
| **Type Checking** | mypy strict (backend), TypeScript strict (frontend) |

## Design Documents

All documentation lives in `docs_v1/`. Read these before making significant changes.

| Document | What It Tells You |
|---|---|
| [CONSTITUTION](docs_v1/design/CONSTITUTION.md) | Glossary, design principles P1-P10, constraints C-01..C-07, non-goals, default policy, validation gate rules, Builder workflow, tech stack |
| [BLUEPRINT](docs_v1/design/BLUEPRINT.md) | Full-stack architecture, module interfaces with signatures, API endpoints, source layout, implementation phases |
| [CONTRACTS](docs_v1/design/CONTRACTS.md) | Data decoupling architecture, exact schemas (SimOS export, internal canonical, API request/response, YAML config), concrete examples at every pipeline stage, algorithm specs, file storage layout |
| [SKELETON](docs_v1/design/SKELETON.md) | File-by-file implementation guide with build order, dependencies, function signatures, test specs |
| [IMPLEMENTATION_PLAN](docs_v1/design/IMPLEMENTATION_PLAN.md) | Phased build order, testing strategy, quality gates |

## What This Project Does

ML/RL OS supports exactly **two prediction problem types**:

### 1. Time-Series Forecasting
- **Input:** SimOS Layer 3 state snapshots (periodic system observations)
- **Question:** "System ran 8 hours. What happens in the next hour?"
- **Features:** Lag values, rolling statistics, trend slopes, ratio features
- **Target:** Future numeric value (lead_time, throughput, queue_depth, etc.)
- **Algorithms:** LightGBM, XGBoost with lag features (tabular approach)

### 2. Entity Classification
- **Input:** SimOS Layer 2 entity trajectories (per-entity, per-step MDP records)
- **Question:** "This entity is at step 3. Will it complete? Will it breach SLA?"
- **Features:** Entity state (10), node state (9), system state (5), derived features
- **Target:** Categorical outcome (episode_status, sla_breach, delay_severity)
- **Algorithms:** LightGBM classifier with class imbalance handling

### What This Project Does NOT Do
- **No config → outcome prediction.** SimOS simulation is cheap — just run it.
- **No deep learning** in v0.1. Lag features + gradient boosting is competitive.
- **No RL training** in v0.1. Deferred to v0.2.
- **No SimOS code imports.** SimOS is a data source, not a dependency.

## Source Layout

### Backend (`src/mlrl_os/`)

```
src/mlrl_os/
├── core/                        # Core types and protocols
│   ├── types.py                 # ProblemType, FeatureMatrix, TrainedModel, enums
│   ├── experiment.py            # ExperimentResult, ExperimentRecord
│   └── dataset.py               # RawDataset, DatasetMeta, AvailableTargets
│
├── data/                        # Data ingestion & management
│   ├── simos_loader.py          # SimOS 5-layer export loader
│   ├── external_loader.py       # CSV / Parquet loader
│   ├── registry.py              # Dataset versioning & registry
│   ├── discovery.py             # Target & feature auto-discovery
│   └── validation.py            # Data quality checks
│
├── features/                    # Feature engineering
│   ├── time_series.py           # Windowing, lag, rolling, trend, ratio
│   ├── entity.py                # Entity state, node state, system state, derived
│   ├── target_derivation.py     # Derived targets (sla_breach, delay_severity, wait_ratio_class)
│   ├── detection.py             # Problem type auto-detection
│   └── store.py                 # Feature definition registry
│
├── config/                      # Configuration management
│   ├── defaults.py              # BaseSettings (env vars), default configs
│   ├── resolver.py              # Merge user config + defaults → resolved
│   └── schemas.py               # Pydantic models for all config types
│
├── validation/                  # Validation gate
│   └── gate.py                  # All validation rules (V-01..VE-04)
│
├── models/                      # Model training & management
│   ├── engine.py                # Train, evaluate, select best
│   ├── registry.py              # Model versioning & storage
│   └── algorithms/
│       ├── protocol.py          # Algorithm protocol
│       ├── registry.py          # Algorithm registry
│       ├── lightgbm.py          # LightGBM wrapper
│       ├── xgboost.py           # XGBoost wrapper
│       ├── random_forest.py     # sklearn RandomForest wrapper
│       ├── extra_trees.py       # sklearn ExtraTrees wrapper
│       └── linear.py            # Ridge / LogisticRegression
│
├── evaluation/                  # Evaluation & reporting
│   ├── metrics.py               # Metric registry & computation
│   ├── reports.py               # Report generation (JSON + HTML)
│   └── comparison.py            # Multi-experiment comparison
│
├── experiment/                  # Experiment orchestration
│   ├── runner.py                # Full pipeline: config → result
│   ├── tracker.py               # Experiment history & artifacts
│   └── seed.py                  # Seeded RNG utilities (seed_hash)
│
├── api/                         # FastAPI application
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
web/src/
├── api/                         # Axios client + endpoint modules
├── components/
│   ├── builder/                 # 7-step Experiment Builder
│   │   ├── StepDataset.tsx      # Step 1: select/upload dataset
│   │   ├── StepExplore.tsx      # Step 2: explore data
│   │   ├── StepProblemType.tsx  # Step 3: select problem type
│   │   ├── StepTarget.tsx       # Step 4: select target
│   │   ├── StepFeatures.tsx     # Step 5: select features
│   │   ├── StepModel.tsx        # Step 6: configure model
│   │   └── StepReview.tsx       # Step 7: review YAML & submit
│   ├── results/                 # Metrics, charts, feature importance
│   ├── data/                    # Dataset list, schema viewer, preview
│   ├── layout/                  # App shell, sidebar
│   └── shared/                  # Reusable components
├── pages/                       # Dashboard, Builder, Datasets, Experiments, Results, Models
├── store/                       # Zustand stores (builder, app)
└── lib/                         # YAML generator, formatting
```

### Tests (`tests/`)

```
tests/
├── unit/                        # Per-module unit tests
│   ├── data/
│   ├── features/
│   ├── config/
│   ├── validation/
│   ├── models/
│   ├── evaluation/
│   └── experiment/
├── integration/                 # End-to-end pipeline tests
└── fixtures/                    # Sample SimOS exports, synthetic data
```

## API Routes

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/datasets` | Upload and register dataset |
| GET | `/api/v1/datasets` | List registered datasets |
| GET | `/api/v1/datasets/{id}` | Get dataset metadata |
| GET | `/api/v1/datasets/{id}/schema` | Get column schema with types and stats |
| GET | `/api/v1/datasets/{id}/available-targets` | Discover available targets |
| GET | `/api/v1/datasets/{id}/preview` | Preview first N rows |
| POST | `/api/v1/experiments` | Submit experiment (YAML or JSON) |
| GET | `/api/v1/experiments` | List experiments |
| GET | `/api/v1/experiments/{id}` | Get experiment result |
| GET | `/api/v1/experiments/{id}/report` | Get evaluation report |
| POST | `/api/v1/experiments/validate` | Validate config without running |
| GET | `/api/v1/experiments/defaults/{problem_type}` | Get default config |
| GET | `/api/v1/models` | List registered models |
| GET | `/api/v1/models/{id}` | Get model metadata |
| POST | `/api/v1/models/{id}/predict` | Run prediction |
| GET | `/api/v1/models/{id}/feature-importance` | Get feature importance |
| POST | `/api/v1/config/resolve` | Preview resolved config |
| GET | `/api/v1/health` | Health check |

## Configuration System

### Three Tiers (user chooses complexity level)

1. **Zero config** — provide dataset path only, system auto-detects everything
2. **YAML overrides** — override specific settings (target, features, algorithms)
3. **Builder UI** — visual 7-step flow that generates YAML

### Config Resolution Flow

```
User config (partial or empty)
    ↓
System loads defaults for detected problem type
    ↓
User overrides merged on top (user wins)
    ↓
Resolved config (fully specified, no optional fields)
    ↓
Validation gate (pass or reject with all errors)
    ↓
Pipeline runs with resolved config
```

### Default Targets

- **Time-series:** `sys_avg_lead_time` (overridable to any numeric snapshot column)
- **Entity classification:** `episode_status` (overridable to `sla_breach`, `delay_severity`, etc.)

## Key Patterns

### Schema Adapter Pattern — Data Source Decoupling
ML/RL OS uses **canonical internal column names** everywhere. The ONLY place that knows SimOS field names is `SimosSchemaAdapter` in `data/simos_loader.py`. All downstream code (features, models, evaluation) uses canonical names only.

```
SimOS export → SimosSchemaAdapter → canonical DataFrame → pipeline
External CSV  → ExternalLoader     → canonical DataFrame → pipeline
Future source → NewAdapter          → canonical DataFrame → pipeline
```

If SimOS changes its export schema, update ONE class. No pipeline code changes.

See CONTRACTS.md §1 for the full canonical name mapping.

### Seeded RNG — Non-Negotiable (same as SimOS)
```python
import hashlib, random

def seed_hash(name: str, global_seed: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{name}:{global_seed}".encode()).digest()[:8],
        byteorder="big"
    )
```
Every component gets its own deterministic RNG. Same seed + config + data = identical results.

### Algorithm Registry
All ML algorithms implement the `Algorithm` protocol and are registered by name. Lazy-loaded to avoid import errors when optional deps are missing.

```python
class Algorithm(Protocol):
    @property
    def name(self) -> str: ...
    def train(self, X, y, task, seed, **kwargs) -> TrainedModel: ...
    def predict(self, model, X) -> np.ndarray: ...
    def feature_importance(self, model) -> dict[str, float] | None: ...
```

### Validation Gate
Experiments MUST pass validation before training. All errors are returned at once, not just the first. No bypass flag exists. Rules are defined in CONSTITUTION.md §5.

### Builder Produces YAML
The web Builder generates standard experiment YAML — same format a user would write by hand. This ensures a single code path: YAML → validation → training. No special Builder format.

### Config Resolver
`ConfigResolver.resolve()` merges user config onto problem-type defaults. The output is a `ResolvedExperimentConfig` with NO optional fields — every setting has a concrete value.

## SimOS Data Integration

ML/RL OS consumes SimOS's 5-layer ML export (export_version 3.0):

| Layer | Used By | Content |
|---|---|---|
| Layer 1 (Event Stream) | Not used in v0.1 | Raw chronological events |
| Layer 2 (Trajectories) | Entity classification | Per-entity, per-step MDP records with state vectors |
| Layer 3 (Snapshots) | Time-series forecasting | Periodic system state observations |
| Layer 4 (Domain Enrichment) | Entity classification | Domain-specific features (healthcare/supply chain/service) |
| Layer 5 (Stress Scenarios) | Not used in v0.1 | Config change descriptors for curriculum RL |

**SimOS export is loaded via `SimosLoader` — no SimOS code is imported.**

SimOS exports via: `POST /api/v1/simulations/{job_id}/export-ml` or file export from Web UI.

### SimOS API Access (Local Dev)

SimOS requires two headers for programmatic API access:

```bash
# Required headers for all SimOS API calls
-H "X-API-Key: sk-premium-test-003"    # Premium tier test account
-H "X-SimOS-Client: web"               # Bypasses api_access tier gate
```

The `X-SimOS-Client: web` header is checked by `require_web_client()` in `billing/dependencies.py`. Without it, requests are rejected with "Direct API access is not available on your current plan" even with a valid API key (all tiers have `api_access=False`).

**Smoke test workflow:**
```bash
# 1. Get template config
curl -s http://localhost:8000/api/v1/templates/healthcare_er \
  -H "X-API-Key: sk-premium-test-003" -H "X-SimOS-Client: web"

# 2. Submit simulation (POST the config object from step 1)
curl -s -X POST http://localhost:8000/api/v1/simulations \
  -H "X-API-Key: sk-premium-test-003" -H "X-SimOS-Client: web" \
  -H "Content-Type: application/json" -d '<config json>'

# 3. Poll status until "completed"
curl -s http://localhost:8000/api/v1/simulations/{job_id} \
  -H "X-API-Key: sk-premium-test-003" -H "X-SimOS-Client: web"

# 4. Export ML data (5-layer export v3.0)
curl -s -X POST http://localhost:8000/api/v1/simulations/{job_id}/export-ml?bucket_seconds=60 \
  -H "X-API-Key: sk-premium-test-003" -H "X-SimOS-Client: web"
```

**Test accounts:** See `simulation_os/docs_v2/TEST_ACCOUNTS.md` for all keys (free/pro/premium/admin).

## Environment Variables

All prefixed with `MLRL_`.

| Variable | Default | Description |
|---|---|---|
| `MLRL_ENV` | `development` | Environment |
| `MLRL_DATA_DIR` | `./data` | Dataset storage |
| `MLRL_MODELS_DIR` | `./models` | Model artifacts |
| `MLRL_EXPERIMENTS_DIR` | `./experiments` | Experiment history |
| `MLRL_LOG_LEVEL` | `INFO` | Logging level |
| `MLRL_API_PORT` | `8001` | Backend API port |
| `MLRL_CORS_ALLOW_ORIGINS` | `http://localhost:5175` | CORS origins |
| `MLRL_MAX_TRAINING_ROWS` | `1000000` | Max training dataset rows |
| `MLRL_CV_FOLDS_DEFAULT` | `5` | Default CV folds |
| `MLRL_SEED_DEFAULT` | `42` | Default experiment seed |

## Running the Platform

```bash
# Backend only
pip install -e ".[dev]"
mlrl-os serve                              # API on port 8001

# Web UI only
cd web && npm install && npm run dev       # Vite on port 5175

# CLI
mlrl-os run experiment.yaml               # Run experiment from YAML
mlrl-os validate experiment.yaml           # Validate without running
mlrl-os datasets list                      # List registered datasets

# Tests
pytest                                     # All tests
pytest tests/unit/                         # Unit tests only
pytest --cov=mlrl_os --cov-report=term-missing  # With coverage (90% threshold)
ruff check src/ tests/                     # Lint
mypy                                       # Type check
```

## Rules

1. **No training without validation.** The experiment runner refuses to train if the validation gate fails. No bypass flag. No override. Fix and resubmit.
2. **Convention over configuration.** Every setting has a sensible default. Zero-config experiments are valid.
3. **Builder produces YAML.** The Builder UI does not have its own format. It generates the same YAML a user would write.
4. **Temporal CV for time-series.** Random k-fold is rejected at validation for time-series problems. No future leakage, ever.
5. **Seeded everything.** Every random process uses `seed_hash(component_name, global_seed)`. Same inputs = identical outputs.
6. **Immutable artifacts.** Registered datasets and models cannot be modified. New versions create new entries.
7. **No heavy imports at module level.** LightGBM, XGBoost are lazy-loaded through the algorithm registry.
8. **SimOS is data, not code.** Never import from `simulation_os`. SimOS exports are data files with a known schema.
9. **Auto-discover, don't hard-code.** Features and targets are discovered from data schema, not hard-coded in platform code.
10. **All errors at once.** Validation returns ALL errors, not just the first. Users fix everything in one pass.
11. **Tests before or alongside code.** No module ships without tests. Target 90%+ coverage.
12. **Resolved config has no optionals.** After `ConfigResolver.resolve()`, every field has a concrete value. Pipeline code never checks for None.

## Common Pitfalls

### Backend
- **Don't hard-code feature columns.** Use auto-discovery from dataset schema. Different SimOS exports have different node/resource names.
- **Don't use random k-fold for time-series.** Temporal CV only. The validation gate rejects random k-fold for time-series problems.
- **Don't skip the validation gate.** Even in tests, run validation to ensure config is valid. No shortcut.
- **Don't import SimOS.** `from simulation_os import ...` is forbidden. SimOS is a data source.
- **Don't use SimOS field names outside `SimosSchemaAdapter`.** All pipeline code uses canonical column names only. See CONTRACTS.md §1.
- **Don't import LightGBM/XGBoost at module level.** Use the algorithm registry's lazy loading.
- **Don't share RNG between components.** Each gets its own seeded `random.Random` via `seed_hash`.
- **Don't add optional fields to ResolvedExperimentConfig.** After resolution, everything is concrete.
- **Don't return just the first validation error.** Collect and return ALL errors.

### Frontend
- **Don't bypass the YAML step.** Builder must show generated YAML before submission.
- **Don't call training endpoints without validation.** Always call `/experiments/validate` first (or let `/experiments` do it).
- **Don't hard-code target options.** Fetch from `/datasets/{id}/available-targets`.
- **Don't skip loading states.** Use skeleton components while data loads.
- **Don't add non-lazy Plotly imports.** Plotly is heavy — always lazy load.

## Ecosystem Context

ML/RL OS is one of three instruments in the Operational Intelligence Platform:

```
AgentsOS (future)
   ↓ orchestrates experiments
ML/RL OS (this project)
   ↑ consumes data from
SimOS (production-ready)
```

- **SimOS** runs on ports 8000 (API), 5173 (Web), 5174 (Live-Viz)
- **ML/RL OS** runs on ports 8001 (API), 5175 (Web)
- **AgentsOS** — architecture drafted, not yet implemented

Each instrument operates independently. SimOS runs without ML/RL OS. ML/RL OS runs without SimOS (using external CSV/Parquet data).

## Implementation Status

| Phase | Status | Content |
|---|---|---|
| Phase 1: Core Engine | **DONE** | Types, data loaders, features, config, validation, models (5 algorithms), evaluation, runner — 416 tests |
| Phase 2: API Layer | **DONE** | FastAPI app factory, 18 REST endpoints, 28 API tests |
| Phase 3: Web UI | **DONE** | React 19 + Vite + Tailwind, 7-step Builder, Dashboard, Dataset/Experiment/Model pages (22 files) |
| Phase 4: CLI & Polish | **DONE** | CLI (serve, run, validate, datasets list/import), TypeScript type-checks clean, Vite builds clean |
| Smoke Test | **DONE** | healthcare_er template → SimOS export → SimosLoader → DatasetRegistry → ExperimentRunner → LightGBM entity classification (560 samples, 88 features, 0.6s) |

Current phase: **v0.1 complete (2026-03-10).** Next: v0.2 (RL training against SimOS environments, AgentsOS integration).

### v0.1 Smoke Test Results (2026-03-10)

| Template | Target | Observation | Samples | Features | Best Algo | F1 | AUC |
|---|---|---|---|---|---|---|---|
| healthcare_er | delay_severity | all_steps | 560 | 40 | lightgbm | 1.000 | 1.000 |
| logistics_otd | delay_severity | all_steps | 3688 | 84 | lightgbm | 0.999 | 1.000 |
| call_center | wait_ratio_class | all_steps | 2119 | 91 | lightgbm | 1.000 | 1.000 |
| logistics_otd | delay_severity | entry_only | 996 | 84 | random_forest | 0.911 | 0.990 |

`entry_only` (predict from first step only) confirms genuine predictive power without leakage. `all_steps` near-perfect scores are expected — later steps carry increasingly informative state.
