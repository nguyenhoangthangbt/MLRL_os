# ML/RL OS Design Documentation Index

> v0.1 — Predictive Intelligence Instrument for Operational Systems

---

## Design Specifications

| # | Document | Purpose | Read When |
|---|---|---|---|
| 1 | [CONSTITUTION](CONSTITUTION.md) | Glossary, principles, constraints, non-goals, defaults, validation rules, Builder workflow | First — defines vocabulary and rules |
| 2 | [BLUEPRINT](BLUEPRINT.md) | Architecture, module interfaces, API endpoints, source layout, phases | Before implementing any module |
| 3 | [CONTRACTS](CONTRACTS.md) | Data decoupling (Schema Adapter), exact schemas, API contracts, YAML config, concrete examples | Before writing data loaders, API routes, or feature engineering |
| 4 | [SKELETON](SKELETON.md) | File-by-file build order (30 files) with dependencies, function signatures, test specs | Before starting implementation — tells you exactly what to build next |
| 5 | [IMPLEMENTATION_PLAN](IMPLEMENTATION_PLAN.md) | Phased schedule, testing strategy, quality gates | For project planning and status tracking |

## Quick Start for New Session

1. Read `MLRL_os/CLAUDE.md` — development rules, pitfalls, ecosystem context
2. Read `SKELETON.md` — find the next file to implement (numbered build order)
3. Read `CONTRACTS.md` §1 — understand the Schema Adapter decoupling pattern
4. Run `pytest` after every module

## Architecture Summary

```
SimOS export → SchemaAdapter → Canonical DataFrame
                                      ↓
External CSV/Parquet → Loader → Canonical DataFrame
                                      ↓
                              Dataset Registry
                                      ↓
              Builder/YAML → Config Resolver → Resolved Config
                                      ↓
                              Validation Gate (pass/fail)
                                      ↓
                              Feature Engine (TS or Entity)
                                      ↓
                              Model Engine (train + CV + select best)
                                      ↓
                              Evaluation Engine (metrics + report)
                                      ↓
                              Model Registry + Experiment Tracker
```

## Key Design Decisions

- **Schema Adapter pattern** — SimOS field names mapped to canonical names in ONE class. All pipeline code uses canonical names only. SimOS schema changes → update one adapter.
- **Two problem types** — Time-series forecasting + Entity classification
- **Three config tiers** — Zero config (auto-pilot), YAML overrides, Builder UI
- **Builder produces YAML** — single code path for all interfaces
- **Validation gate** — mandatory, no bypass, all errors at once
- **Reproducibility** — seeded RNG per component (same pattern as SimOS)
- **No config→outcome prediction** — SimOS simulation is cheap, just run it
