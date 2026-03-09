# ML/RL OS v0.1 — Full Test Report

**Date:** 2026-03-10
**Phase:** 1 (Core Engine) + 2 (API Layer) + 3 (Web UI) + 4 (CLI) + Smoke Tests
**Status:** ALL TESTS PASSING

## Summary

| Metric | Value |
|---|---|
| Backend tests | 416 |
| Passed | 416 |
| Failed | 0 |
| Skipped | 0 |
| Execution time | ~48s |
| Backend source files | 51 |
| Backend test files | 29 |
| Frontend files | 22 |
| Backend source lines | ~6,800 |
| Test lines | ~5,300 |
| Frontend TypeScript lines | ~1,600 |
| Python version | 3.13.12 |
| Platform | Windows 11 Pro |

## Test Coverage by Module

### Core (`tests/unit/core/`) — 3 test files

| File | Tests | What's Covered |
|---|---|---|
| `test_types.py` | 24 | All enums (ProblemType, TaskType, ExperimentStatus, ObservationPoint, CVStrategy), ColumnInfo, TargetInfo, AvailableTargets, FeatureMatrix (validation, properties) |
| `test_dataset.py` | 10 | DatasetMeta serialization, RawDataset construction/validation, compute_column_info (numeric stats, categorical stats, null handling, edge cases) |
| `test_experiment.py` | 5 | AlgorithmScore, FeatureImportanceEntry, ExperimentResult (full fields, defaults, serialization round-trip) |

### Data Layer (`tests/unit/data/`) — 4 test files

| File | Tests | What's Covered |
|---|---|---|
| `test_simos_loader.py` | 48 | SimosSchemaAdapter (snapshot/trajectory/state/reward/attr field mapping, dynamic node/resource columns, full record mapping with nested state flattening), SimosLoader (load from fixture file, canonical column names, sort order, error handling) |
| `test_external_loader.py` | 8 | CSV/Parquet loading, column info computation, file-not-found handling |
| `test_validation.py` | 12 | Data quality checks from validation module |
| `test_discovery.py` | 20 | Time-series target discovery (numeric columns, structural column exclusion, default target selection, fallback), Entity target discovery (categorical columns, unique value thresholds, class balance computation) |

### Features (`tests/unit/features/`) — 5 test files

| File | Tests | What's Covered |
|---|---|---|
| `test_time_series.py` | 14 | TimeSeriesFeatureEngine.build_features (feature matrix types, lag/rolling/trend feature names, window trimming, error handling for too-small data, minimal parameter mode), parse_duration_to_seconds |
| `test_entity.py` | 14 | EntityFeatureEngine.build_features (feature matrix types, target encoding, derived features, observation point filtering, auto-selection, exclude_columns, error handling) |
| `test_target_derivation.py` | 20 | Derived target computation: sla_breach (7 tests: SLA column, correct labels, all steps, explicit threshold, p75 fallback, no completed, requires eid), delay_severity (4 tests: 3 classes, balanced, all steps, no completed), wait_ratio_class (4 tests: valid classes, fixed thresholds, all steps, no completed), utility functions (5 tests: is_derived_target, constants, leakage columns, unknown target) |
| `test_detection.py` | 10 | Problem type auto-detection from dataset metadata |
| `test_store.py` | 7 | FeatureStore CRUD operations (register, get, list, has, count, KeyError on missing) |

### Config (`tests/unit/config/`) — 3 test files

| File | Tests | What's Covered |
|---|---|---|
| `test_schemas.py` | 14 | All Pydantic config models (ResolvedTimeSeriesFeatures, ResolvedEntityFeatures, ResolvedCVConfig, ResolvedModelConfig, ResolvedEvaluationConfig, ResolvedExperimentConfig), serialization round-trips |
| `test_defaults.py` | 13 | MLRLSettings defaults/env/custom, get_defaults for both problem types, copy isolation, expected keys/values |
| `test_resolver.py` | 13 | parse_duration, ConfigResolver.resolve (zero-config, explicit type, auto-detection from dataset, user overrides, no-None guarantee, name generation, dataset_id/layer propagation, feature auto-selection) |

### Validation (`tests/unit/validation/`) — 1 test file

| File | Tests | What's Covered |
|---|---|---|
| `test_gate.py` | 21 | ValidationGate — all rules: V-02 (min rows), V-03 (target exists), V-04 (target not in features), V-05 (feature columns exist), V-06 (numeric features), V-07 (seed non-negative), V-08 (registered algorithms), V-09 (CV folds >= 2), V-10 (registered metrics), V-11 (target null rate), VT-01 (ts column), VT-04 (numeric TS target), VT-06 (temporal CV for TS), VE-01 (categorical entity target), VE-03 (max classes), multiple errors collected, cumulative target warnings |

### Models (`tests/unit/models/`) — 5 test files

| File | Tests | What's Covered |
|---|---|---|
| `test_protocol.py` | 3 | TrainedModel construction and repr |
| `test_registry.py` | 9 | AlgorithmRegistry (register, get, has, list_available, duplicate rejection, unknown-name error), default_registry with all 5 algorithms |
| `test_linear.py` | 8 | LinearAlgorithm (properties, train regression/classification, predict shape, predict_proba for both tasks, feature importance extraction) |
| `test_tuning.py` | 11 | has_search_space for all algorithms, suggest_params (LightGBM, XGBoost, RandomForest, ExtraTrees, Linear regression/classification), unknown algorithm returns empty dict |
| `test_engine.py` | 11 | ModelEngine.train_and_evaluate (classification, regression, multi-algorithm ranking, all-fail RuntimeError, feature importance, temporal CV, handle_imbalance), Optuna hyperparameter tuning (classification, regression, random_forest, deterministic reproducibility) |

### Evaluation (`tests/unit/evaluation/`) — 2 test files

| File | Tests | What's Covered |
|---|---|---|
| `test_metrics.py` | 15 | compute_metric for all 10 registered metrics, unknown metric error, compute_metrics batch, RMSE non-negativity, R2 perfect predictions, accuracy perfect predictions, AUC-ROC with/without proba, MAPE zero-value safety |
| `test_reports.py` | 8 | ReportGenerator.generate (full report, prediction samples, confusion matrix for classification, no confusion matrix for regression, max_samples limit) |

### Experiment (`tests/unit/experiment/`) — 1 test file

| File | Tests | What's Covered |
|---|---|---|
| `test_seed.py` | 5 | seed_hash determinism, different-name divergence, different-seed divergence, make_rng determinism, sequence reproducibility |

### API (`tests/unit/api/`) — 5 test files

| File | Tests | What's Covered |
|---|---|---|
| `test_app.py` | 3 | App factory creates valid FastAPI app, state dependencies attached, health endpoint returns ok |
| `test_data_routes.py` | 10 | Upload CSV, upload with/without name, list datasets (empty + populated), get dataset detail, get missing → 404, schema endpoint, preview with row limit, available targets discovery |
| `test_experiment_routes.py` | 8 | Get TS/entity defaults, invalid problem type → 422, validate valid config, validate missing dataset, submit experiment (sync), submit missing dataset → 404, list experiments, get missing → 404 |
| `test_model_routes.py` | 4 | List models (empty), get missing → 404, predict missing → 404, feature importance missing → 404 |
| `test_config_routes.py` | 3 | Resolve full config, resolve missing dataset → 404 |

## Source Modules Implemented (Phase 1)

### Core Types & Models (3 files)
- `core/types.py` — All shared enums, ColumnInfo, TargetInfo, AvailableTargets, FeatureMatrix
- `core/dataset.py` — DatasetMeta, RawDataset, compute_column_info
- `core/experiment.py` — AlgorithmScore, FeatureImportanceEntry, ExperimentResult

### Data Ingestion (4 files)
- `data/simos_loader.py` — SimosSchemaAdapter (single coupling point for SimOS field names) + SimosLoader
- `data/external_loader.py` — CSV/Parquet loader for non-SimOS data
- `data/validation.py` — Data quality validation
- `data/discovery.py` — Target and feature auto-discovery from data schema

### Feature Engineering (5 files)
- `features/time_series.py` — Lag, rolling stats, trend slopes, ratio features, cross-node imbalance
- `features/entity.py` — Entity/node/system state extraction, derived features, observation point filtering
- `features/target_derivation.py` — Derived target computation (sla_breach, delay_severity, wait_ratio_class) with leakage prevention
- `features/detection.py` — Problem type auto-detection
- `features/store.py` — Feature definition registry

### Configuration (3 files)
- `config/schemas.py` — All resolved config Pydantic models (no optional fields after resolution)
- `config/defaults.py` — MLRLSettings, TS_DEFAULTS, ENTITY_DEFAULTS
- `config/resolver.py` — ConfigResolver: merge user config + defaults → fully specified config

### Validation (1 file)
- `validation/gate.py` — All 21 validation rules (V-01..V-11, VT-01..VT-06, VE-01..VE-04), warnings

### Models & Algorithms (9 files)
- `models/algorithms/protocol.py` — Algorithm protocol, TrainedModel container
- `models/algorithms/registry.py` — AlgorithmRegistry with lazy loading
- `models/algorithms/lightgbm.py` — LightGBM wrapper (with `handle_imbalance` → `is_unbalance`)
- `models/algorithms/xgboost.py` — XGBoost wrapper (with `handle_imbalance` → `scale_pos_weight`)
- `models/algorithms/random_forest.py` — sklearn RandomForest wrapper (with `handle_imbalance` → `class_weight="balanced"`)
- `models/algorithms/extra_trees.py` — sklearn ExtraTrees wrapper (with `handle_imbalance` → `class_weight="balanced"`)
- `models/algorithms/linear.py` — Ridge/LogisticRegression wrapper (with `handle_imbalance` → `class_weight="balanced"`)
- `models/engine.py` — ModelEngine: train, cross-validate, select best algorithm, Optuna hyperparameter tuning
- `models/tuning.py` — Optuna-based hyperparameter search spaces for all 5 algorithms
- `models/registry.py` — ModelRegistry: file-system model artifact storage

### Evaluation (3 files)
- `evaluation/metrics.py` — 10 metrics (rmse, mae, mape, r2, f1_weighted, f1_macro, precision, recall, accuracy, auc_roc)
- `evaluation/reports.py` — ReportGenerator: structured evaluation reports
- `evaluation/comparison.py` — ComparisonEngine: multi-experiment comparison

### Experiment Orchestration (3 files)
- `experiment/seed.py` — Deterministic seeded RNG (seed_hash, make_rng)
- `experiment/tracker.py` — ExperimentTracker: file-system experiment history
- `experiment/runner.py` — ExperimentRunner: full pipeline orchestrator (config → validation → features → training → evaluation → storage)

### Data Storage (1 file)
- `data/registry.py` — DatasetRegistry: file-system dataset storage with Parquet

### API Layer (6 files)
- `api/app.py` — FastAPI app factory with CORS, exception handlers, dependency injection
- `api/schemas.py` — 20+ Pydantic request/response models for all endpoints
- `api/data_routes.py` — 6 dataset endpoints (upload, list, get, schema, targets, preview)
- `api/experiment_routes.py` — 6 experiment endpoints (submit, list, get, report, validate, defaults)
- `api/model_routes.py` — 4 model endpoints (list, get, predict, feature importance)
- `api/config_routes.py` — 1 config endpoint (resolve)

### CLI (1 file)
- `cli.py` — CLI entry point: `serve`, `run <config.yaml>`, `validate <config.yaml>`, `datasets list`, `datasets import <file>`

### Web UI (22 files — React 19 + TypeScript + Vite + Tailwind)
- `web/src/api/` — Axios client + typed API modules (datasets, experiments, models)
- `web/src/store/builder.ts` — Zustand store for 7-step Builder state
- `web/src/components/layout/AppShell.tsx` — Sidebar + main content layout
- `web/src/components/shared/` — Card, StatusBadge, Skeleton reusable components
- `web/src/components/builder/` — 7 Builder step components (StepDataset through StepReview)
- `web/src/pages/` — Dashboard, Datasets, DatasetDetail, Builder, Experiments, ExperimentDetail, Models

## Bugs Found and Fixed During Testing

| # | Module | Issue | Fix |
|---|---|---|---|
| 1 | `features/time_series.py` | Target column excluded from feature_df but later referenced via `pl.col(target).shift()` causing ColumnNotFoundError | Ensured target column is included in the `select()` call when building feature_df |
| 2 | `models/algorithms/linear.py` | `multi_class="auto"` parameter removed in scikit-learn >= 1.7 | Removed `multi_class` from CLASSIFICATION_PARAMS |
| 3 | `tests/unit/validation/test_gate.py` | Test data (100 rows × 300s) too small for 1h lookback (VT-03 requires 50 windows) | Reduced lookback/horizon to "5m" in test helper |
| 4 | `experiment/seed.py` | `seed_hash` returned 8-byte integers (up to 2^64-1), exceeding scikit-learn and Optuna's 2^32-1 limit for random seeds | Reduced to 4-byte digest (`digest()[:4]`) so seeds fit in 32-bit range |
| 5 | All algorithm wrappers | `handle_imbalance=True` kwarg passed through to sklearn constructors which don't accept it, causing `TypeError` | Added `kwargs.pop("handle_imbalance", False)` in all 5 algorithm wrappers, each applying the flag using the algorithm's native mechanism |

## Test Fixtures

| Fixture | Location | Description |
|---|---|---|
| `simos_export_small.json` | `tests/fixtures/` | Complete SimOS v3.0 export with 3 snapshots, 6 trajectory records (2 entities × 3 steps), metadata, and summary |
| `healthcare_er_export.json` | `tests/fixtures/` | Real SimOS healthcare ER export (187 entities, 560 trajectory rows, 94 columns) |
| `call_center_export.json` | `tests/fixtures/` | Real SimOS call center export (~700 entities, 2119 trajectory rows, 91 columns) |
| `logistics_otd_export.json` | `tests/fixtures/` | Real SimOS logistics OTD export (996 entities, 3688 trajectory rows, 91 columns) |
| `conftest.py` fixtures | `tests/conftest.py` | Synthetic snapshot_df (100 rows), trajectory_df (200+ rows), regression/classification FeatureMatrix, DatasetMeta, RawDataset |

## Remaining Work (Phase 2-4)

| Phase | Status | Content |
|---|---|---|
| Phase 2: API Layer | DONE | FastAPI app factory, 18 REST endpoints, 28 API tests |
| Phase 3: Web UI | DONE | React 19 + Vite + Tailwind, 7-step Builder, Dashboard, Dataset/Experiment/Model pages (22 files) |
| Phase 4: CLI & Polish | DONE | CLI (serve, run, validate, datasets list/import), TypeScript type-checks clean, Vite builds clean |
| Smoke Test | DONE | 3 templates (healthcare_er, call_center, logistics_otd), derived targets, multi-algorithm |
| Derived Targets | DONE | `sla_breach`, `delay_severity`, `wait_ratio_class` — 20 unit tests + 4 smoke tests |

## How to Run

```bash
# Install
cd MLRL_os
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific module
pytest tests/unit/features/ -v

# Run with coverage
pytest --cov=mlrl_os --cov-report=term-missing
```
