# IMPLEMENTATION PLAN

> Phased build order for ML/RL OS v0.1.
>
> Inherits from CONSTITUTION.md and BLUEPRINT.md.
>
> Last updated: 2026-03-10.

---

## Scope Recap

ML/RL OS v0.1 ships two prediction capabilities:

1. **Time-Series Forecasting** — given historical system snapshots, predict future system state
2. **Entity Classification** — given entity state at any point in its journey, predict outcome

Delivered via:
- **Experiment Builder** (web UI, 7-step flow → YAML → validate → train → report)
- **YAML config** (direct submission for power users)
- **REST API** (programmatic access)
- **CLI** (command-line execution)

---

## Build Order

### Phase 1: Core Engine

**Goal:** End-to-end pipeline works from Python code. No API, no UI.

```
Week 1-2:
  1.1  Core types (ProblemType, FeatureMatrix, TrainedModel, enums)
  1.2  Seed utilities (seed_hash, deterministic RNG)
  1.3  SimOS loader (Layer 2 trajectories, Layer 3 snapshots)
  1.4  External loader (CSV, Parquet)
  1.5  Data validation (quality checks)
  1.6  Dataset registry (register, version, retrieve)
  1.7  Target discovery (auto-discover available targets from data)

Week 2-3:
  1.8  Problem type detection (auto-detect from data + config)
  1.9  Time-series feature engineering (lag, rolling, trend, ratio)
  1.10 Entity feature engineering (state extraction, progress ratio)
  1.11 Config schemas (Pydantic models)
  1.12 Default configs (per problem type)
  1.13 Config resolver (merge user + defaults → resolved)

Week 3-4:
  1.14 Validation gate (all rules V-01 through VE-04)
  1.15 Algorithm protocol + registry
  1.16 LightGBM algorithm wrapper
  1.17 XGBoost algorithm wrapper
  1.18 Random Forest algorithm wrapper
  1.19 Model engine (train, CV, select best)
  1.20 Model registry (store, version, retrieve)

Week 4-5:
  1.21 Metric registry + computation
  1.22 Evaluation reports (JSON structure)
  1.23 Experiment runner (full pipeline orchestration)
  1.24 Experiment tracker (history, artifacts)
  1.25 Integration tests (end-to-end with SimOS fixtures)
```

**Phase 1 deliverable:** Python library that takes a SimOS export file + optional YAML config and produces trained model + evaluation report.

```python
# Phase 1 usage:
from mlrl_os.experiment.runner import ExperimentRunner

runner = ExperimentRunner()

# Zero config — system handles everything
result = runner.run({"dataset": {"path": "./data/simos_export.json"}})
print(result.best_algorithm)      # "lightgbm"
print(result.metrics)             # {"rmse": 12.3, "mae": 8.7}
print(result.feature_importance)  # {"queue_lag_1h": 0.23, ...}

# With overrides
result = runner.run({
    "dataset": {"path": "./data/simos_export.json", "layer": "trajectories"},
    "features": {"target": "sla_breach"},
    "model": {"algorithms": ["lightgbm", "xgboost"]},
})
```

---

### Phase 2: API Layer

**Goal:** All Phase 1 capabilities accessible via REST API.

```
Week 5-6:
  2.1  FastAPI app factory (create_app)
  2.2  Request/response schemas
  2.3  Dataset routes (upload, list, schema, targets, preview)
  2.4  Experiment routes (submit, validate, list, get result, get report)
  2.5  Model routes (list, get, predict, feature importance)
  2.6  Config routes (resolve preview, defaults, health)
  2.7  API integration tests
```

**Phase 2 deliverable:** Running API server that accepts experiment submissions and returns results.

```bash
# Phase 2 usage:
mlrl-os serve                     # start API on port 8001

# Upload dataset
curl -X POST http://localhost:8001/api/v1/datasets \
  -F "file=@simos_export.json"

# Discover targets
curl http://localhost:8001/api/v1/datasets/ds_001/available-targets

# Submit experiment (zero config)
curl -X POST http://localhost:8001/api/v1/experiments \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "ds_001"}'

# Get results
curl http://localhost:8001/api/v1/experiments/exp_001
```

---

### Phase 3: Web UI (Experiment Builder)

**Goal:** Visual experiment builder with 7-step flow.

```
Week 6-8:
  3.1  Project scaffold (React 19, Vite, TypeScript, Tailwind)
  3.2  API client + TypeScript types
  3.3  App shell (sidebar, routing, layout)
  3.4  Dashboard page (recent experiments, model count)
  3.5  Dataset page (upload, list, explore)
  3.6  Builder — Step 1: Dataset selection
  3.7  Builder — Step 2: Data exploration
  3.8  Builder — Step 3: Problem type selection
  3.9  Builder — Step 4: Target selection (with distribution preview)
  3.10 Builder — Step 5: Feature selection (with engineering options)
  3.11 Builder — Step 6: Model configuration
  3.12 Builder — Step 7: Review YAML + submit
  3.13 Validation error display
  3.14 Results page (metrics, feature importance charts)
  3.15 Experiment history page
  3.16 Model registry page
```

**Phase 3 deliverable:** Full web application. User uploads data, builds experiment visually, submits, sees results.

---

### Phase 4: Polish

```
Week 8-9:
  4.1  CLI (mlrl-os run, mlrl-os validate, mlrl-os datasets)
  4.2  HTML report export
  4.3  Feature Store persistence
  4.4  Model comparison view
  4.5  Dark mode
  4.6  User documentation
```

---

## Testing Strategy

### Test-First Development

Every module gets tests BEFORE or alongside implementation. No module ships without tests.

### Test Categories

| Category | Count (est.) | Focus |
|---|---|---|
| Unit — data | 80 | Loaders, registry, discovery, validation |
| Unit — features | 65 | Time-series engineering, entity engineering, detection |
| Unit — config | 50 | Resolver, defaults, schemas |
| Unit — validation | 35 | Every gate rule individually |
| Unit — models | 70 | Algorithm wrappers, engine, registry |
| Unit — evaluation | 35 | Metrics, reports |
| Unit — experiment | 35 | Runner, tracker, seed |
| Integration | 30 | End-to-end pipelines, API endpoints |
| **Total** | **~400** | |

### Test Fixtures

Pre-built SimOS export fixtures for deterministic testing:

- `simos_export_healthcare.json` — healthcare ER simulation (small, ~100 entities)
- `simos_export_supply_chain.json` — supply chain DC network (medium, ~500 entities)
- `sample_timeseries.csv` — generic time-series (synthetic, ~1000 rows)
- `sample_entities.csv` — generic entity records (synthetic, ~500 rows)

All fixtures have known expected outputs for verification.

### Quality Gates

| Gate | Threshold |
|---|---|
| Test coverage | ≥ 90% |
| All tests pass | 100% |
| Type check (mypy strict) | 0 errors |
| Lint (ruff) | 0 errors |
| Reproducibility | Same seed + config = identical metrics |

---

## Completion Status

### v0.1 (Complete — 2026-03-10, 416 tests)

| Phase | Status | Tests | Notes |
|---|---|---|---|
| Phase 1: Core Engine | DONE | 368 | 43 source files, 23 test files |
| Phase 2: API Layer | DONE | 28 | 6 API files, 18 endpoints |
| Phase 3: Web UI | DONE | — | 22 React+TS files, 7-step Builder |
| Phase 4: CLI & Polish | DONE | — | CLI, TypeScript clean, Vite builds |
| Derived Targets | DONE | 20 | `sla_breach`, `delay_severity`, `wait_ratio_class` |
| v0.1 Smoke Tests | DONE | — | 3 templates, 4 experiments, realistic metrics |

### v0.2 (Complete — 2026-03-10, 710 total tests)

See [v0.2 Design](../plans/2026-03-10-v02-design.md) and [v0.2 Implementation Plan](../plans/2026-03-10-v02-implementation-plan.md) for full details.

| Phase | Status | Tests | Notes |
|---|---|---|---|
| Storage Backend | DONE | 61 | StorageBackend protocol, file + PostgreSQL backends |
| Hyperparameter Tuning | DONE | — | n_trials wired through config → resolver → engine → API |
| RL Foundations | DONE | 67 | Spaces, rewards, replay buffers, networks, curriculum |
| RL Algorithms | DONE | 59 | Custom DQN, PPO, protocol, registry |
| RL Environment/Runner | DONE | 69 | SimOS WebSocket client, environment, runner, API routes |
| LSTM Algorithm | DONE | 14 | PyTorch LSTM via algorithm registry |
| Streaming Inference | DONE | 4 | WebSocket prediction endpoints |
| HTML Reports | DONE | 17 | Standalone HTML export with inline SVG charts |
| v0.2 Smoke Test | DONE | — | 8/8 experiments, 2 templates — [report](../reports/2026-03-10-v02-smoke-test-report.md) |

### v0.2 Smoke Test Results

| Template | Target | Observation | Samples | Features | Best Algo | F1 | AUC |
|---|---|---|---|---|---|---|---|
| healthcare_er | delay_severity | all_steps | 560 | 91 | lightgbm | 1.000 | 1.000 |
| healthcare_er | delay_severity | entry_only | 187 | 91 | random_forest | 0.924 | 0.993 |
| healthcare_er | sla_breach | all_steps | 560 | 91 | random_forest | 0.988 | 1.000 |
| healthcare_er | sla_breach | entry_only | 187 | 91 | lightgbm | 1.000 | 1.000 |
| logistics_otd | delay_severity | all_steps | 3688 | 84 | lightgbm | 0.999 | 1.000 |
| logistics_otd | delay_severity | entry_only | 996 | 84 | random_forest | 0.911 | 0.990 |
| logistics_otd | sla_breach | all_steps | 3688 | 84 | random_forest | 1.000 | 0.000* |
| logistics_otd | sla_breach | entry_only | 996 | 84 | random_forest | 1.000 | 0.000* |

*AUC=0.0 for near-single-class target — see report Finding F-03.

### Next: v0.3

- RL integration testing with live SimOS WebSocket
- PostgreSQL integration testing
- AgentsOS integration for experiment orchestration
- Address enhancement proposals from v0.2 smoke test report
