# ML/RL OS

Predictive Intelligence Instrument for Operational Systems.

## Overview

ML/RL OS consumes structured operational data (SimOS exports, CSV, Parquet) and produces trained models, forecasts, entity classifications, RL policies, and evaluation reports through a validated experiment pipeline.

**Version:** 0.2.0 (2026-03-10) | **Tests:** 710 passing

## Capabilities

| Problem Type | Description | Algorithms |
|---|---|---|
| **Time-Series Forecasting** | Predict future system state from historical snapshots | LightGBM, XGBoost, LSTM, Random Forest, ExtraTrees, Linear |
| **Entity Classification** | Predict entity outcome from trajectory state | LightGBM, Random Forest, ExtraTrees, LSTM, Linear |
| **Reinforcement Learning** | Train routing/scheduling policies against live SimOS | Custom DQN, PPO |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest

# Start backend API (port 8001)
mlrl-os serve

# Start web UI (port 5175)
cd web && npm install && npm run dev

# Run experiment from YAML
mlrl-os run experiment.yaml

# Validate without running
mlrl-os validate experiment.yaml
```

## Architecture

```
Web UI (React 19 + Vite + Tailwind)         Port 5175
    ↕
Backend API (FastAPI + Pydantic v2)          Port 8001
    ↕                    ↕
ML Engine               RL Engine
(6 algorithms)          (DQN/PPO)
    ↕                    ↕
Storage Backend (File or PostgreSQL)
    ↕
SimOS Data (REST export or WebSocket live)   Port 8000
```

## Documentation

| Document | Purpose |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Full development guide, source layout, API routes, rules |
| [CONSTITUTION](docs_v1/design/CONSTITUTION.md) | Design principles, glossary, constraints |
| [BLUEPRINT](docs_v1/design/BLUEPRINT.md) | Architecture, module interfaces |
| [CONTRACTS](docs_v1/design/CONTRACTS.md) | Data schemas, API contracts |
| [v0.2 Design](docs/plans/2026-03-10-v02-design.md) | RL engine, PostgreSQL, streaming, tuning |
| [v0.2 Smoke Test](docs/reports/2026-03-10-v02-smoke-test-report.md) | 8/8 experiments, findings, enhancements |

## Ecosystem

Part of the Operational Intelligence Platform:

- **SimOS** — simulation engine (production-ready, ports 8000/5173/5174)
- **ML/RL OS** — predictive intelligence (this project, ports 8001/5175)
- **AgentsOS** — autonomous orchestration (implemented)
