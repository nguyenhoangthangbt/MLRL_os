# ML/RL OS — Architecture Review

> From: Architecture Lead (platforms_os integration validation)
> Date: 2026-03-10
> Context: Cross-platform integration readiness review for the 4-platform Unified Operational Intelligence Platform.
> Action: Dedicated session should validate findings against current codebase and implement fixes.
>
> **For implementers:** After completing each fix, update the status in the tracking table below and add a brief note. The architecture lead will re-validate once all items are marked DONE.

---

## Ecosystem Context

This review is part of a **4-platform Unified Operational Intelligence Platform** validation. Each platform is an independent instrument wired together via REST APIs:

```
LitReview OS (8003) ──grounds──→ AgentsOS (8002) ──orchestrates──→ SimOS (8000) ──data──→ ML/RL OS (8001)
     ↑                              ↑                                  │                      │
     │                              └──────── knowledge ───────────────┘                      │
     └──────────────────────────── learns from ───────────────────────────────────────────────┘
```

**Full research loop:** LitReview OS grounds research in literature → AgentsOS designs experiments → SimOS runs simulations → ML/RL OS learns from data → AgentsOS stores knowledge → cycle repeats.

**Master architecture:** `platforms_os/docs_v1/design/MASTER_ARCHITECTURE.md`
**Integration validation:** `platforms_os/docs_v1/design/INTEGRATION_VALIDATION_REPORT.md`

| Platform | Port | Tests | Status |
|---|---|---|---|
| SimOS | 8000 | 3250+ | Production-ready |
| ML/RL OS | 8001 | 710 | v0.2 complete |
| AgentsOS | 8002 | 449 | v0.1 complete |
| LitReview OS | 8003 | 475 | v0.1 complete |

**Key principle:** No code imports between platforms. All integration via REST API + file contracts (JSON).

---

## Platform Role

ML/RL OS is the **predictive intelligence instrument**. It consumes SimOS 5-layer exports for supervised learning (time-series forecasting, entity classification) and connects to SimOS WebSocket for RL training. AgentsOS orchestrates experiments via REST.

## Findings

### M-02 — MEDIUM: RL endpoint schemas not in CONTRACTS.md

**Problem:** v0.2 added 5 RL endpoints and 2 WebSocket endpoints, but CONTRACTS.md only covers v0.1 ML endpoints. The RL schemas exist in code (`api/rl_routes.py`, `config/rl_schemas.py`) but are not formalized in design docs.

**Missing from CONTRACTS.md:**
- `POST /api/v1/rl/experiments` — request and response schemas
- `GET /api/v1/rl/experiments/{id}` — response schema (training curves format?)
- `GET /api/v1/rl/policies[/{id}]` — response schema
- `WS /ws/v1/predict/{model_id}` — message format
- `WS /ws/v1/policy/{policy_id}` — message format

**Fix:** Add a new §5b to CONTRACTS.md covering RL API contracts. Extract Pydantic models from `rl_routes.py` and `rl_schemas.py` into the doc.

### M-05 — MEDIUM: No authentication mechanism

**Problem:** ML/RL OS API has zero authentication. No API key, JWT, or auth middleware documented or implemented. All endpoints are open.

**Why it matters:** When AgentsOS calls ML/RL OS from a different host (or in production), there's no access control.

**Fix:** This is a platform-wide decision. Options:
1. API key (simple, matches SimOS pattern)
2. Gateway-only auth (all services behind reverse proxy)
3. mTLS for service-to-service

Recommend aligning with SimOS's `X-API-Key` pattern for consistency. Add `MLRL_API_KEYS` env var.

### M-01 — LOW: Stale doc text for snapshot key

**Problem:** CONTRACTS.md §2.1 refers to the SimOS export top-level key as `snapshots`, but SimOS actually exports it as `state_snapshots`. Code in `simos_loader.py` already handles this correctly.

**Fix:** Update CONTRACTS.md §2.1 text: `"snapshots": [...]` → `"state_snapshots": [...]`.

### M-04 — LOW: HTTP status codes not specified

**Problem:** No endpoint in CONTRACTS.md specifies HTTP status codes. For example, `POST /api/v1/experiments` returns 202 (async training started), but the doc just says "Response" without status code.

**Fix:** Add a status code column to the endpoint tables in CONTRACTS.md §5.

### M-06 — MEDIUM: No cross-instrument provenance

**Problem:** When AgentsOS submits a SimOS export to ML/RL OS, there's no metadata linking the resulting model back to the original SimOS `job_id` and template name. AgentsOS can't query "all experiments from template healthcare_er."

**Fix:** Add optional `source_metadata` field to dataset upload:
```python
class DatasetUploadMeta(BaseModel):
    source_instrument: str | None = None  # "simos"
    source_job_id: str | None = None
    source_template: str | None = None
```
This is stored in `DatasetMeta` and propagated to `ExperimentResult`. Low effort, high value for AgentsOS workflows.

---

## Summary

| ID | Severity | Action | Type |
|---|---|---|---|
| M-02 | MEDIUM | Add RL endpoint schemas to CONTRACTS.md | Docs |
| M-05 | MEDIUM | Define authentication mechanism | Design + Code |
| M-06 | MEDIUM | Add source_metadata for cross-instrument provenance | Code + Docs |
| M-01 | LOW | Fix `snapshots` → `state_snapshots` in CONTRACTS.md | Docs |
| M-04 | LOW | Add HTTP status codes to CONTRACTS.md | Docs |

---

## Implementation Tracking

> **Implementers:** Update this table as you complete each item. Change status to `DONE` and add the date + brief note (e.g., commit hash, files changed). The architecture lead will re-validate once all items show DONE.

| ID | Severity | Status | Completed | Notes |
|---|---|---|---|---|
| M-02 | MEDIUM | DONE | 2026-03-10 | Added §5.7 (RL experiments), §5.8 (RL listing/policies), §5.9 (WS inference) to CONTRACTS.md |
| M-05 | MEDIUM | DEFERRED | | Valid finding. Requires platform-wide design decision (API key vs gateway vs mTLS). Recommend aligning with SimOS `X-API-Key` pattern. Awaiting owner approval. |
| M-06 | MEDIUM | DONE | 2026-03-10 | Added `source_instrument`, `source_job_id`, `source_template` to DatasetMeta, DatasetRegistry.register(), upload endpoint, DatasetDetailResponse, and CONTRACTS.md §5.1 |
| M-01 | LOW | DONE | 2026-03-10 | Fixed `"snapshots"` → `"state_snapshots"` in CONTRACTS.md §2.1 |
| M-04 | LOW | DONE | 2026-03-10 | Added §5.10 HTTP Status Code Summary table covering all 21 endpoints to CONTRACTS.md |
| M-07 | LOW | TODO | | Health endpoint missing `storage_backend` field — found in E2E test CC-05 |
| M-08 | LOW | TODO | | POST endpoints return 307 redirect without trailing slash — found in E2E test DP-01 |

### E2E Integration Test Results (2026-03-10)

Phase 1 E2E tests validated ML/RL OS in the cross-platform pipeline. All critical tests **PASSED** with 2 minor findings.

| Test | Result | Notes |
|---|---|---|
| DP-01 healthcare_er pipeline | PASS | Dataset upload (multipart), provenance (`source_type=simos`), target discovery (`sla_breach`, `status`, `delay_severity`), entity_classification training (F1=1.0, AUC=1.0, LightGBM) |
| DP-01 logistics_otd pipeline | PASS | 44.3 MB multipart upload handled correctly, 3688 trajectories, F1=1.0 |
| SR-05 Seed Reproducibility | PASS | Two identical runs with seed=42 produced identical metrics (F1=1.0, AUC=1.0, best_algorithm=lightgbm) |
| CC-05 Health Contract | CONDITIONAL PASS | `status` present, `storage_backend` missing from response (see M-07) |

**New findings:**

#### M-07 — LOW: Health endpoint missing `storage_backend` field

**Problem:** `GET /api/v1/health` returns only `{"status":"ok","version":"0.1.0"}`. The cross-platform contract (CC-05) expects a `storage_backend` field (e.g., `"file"` or `"postgres"`) matching SimOS health response pattern.

**Impact:** Informational only. Cross-platform health monitoring cannot determine MLRL storage backend type. Does not affect functionality.

**Fix:** Add `storage_backend` field to health endpoint response in `api/app.py`.

#### M-08 — LOW: POST endpoints require trailing slash

**Problem:** `POST /api/v1/datasets` (without trailing slash) returns `307 Temporary Redirect` to `/api/v1/datasets/`. Clients must use trailing slash or follow redirects (`curl -L`). This differs from SimOS which accepts both forms.

**Impact:** Minor DX issue. E2E test pipeline needed adjustment. AgentsOS `MLRLOSClient` should use trailing slashes or enable redirect-following.

**Fix:** Either add `redirect_slashes=False` to FastAPI app config, or document the trailing-slash requirement in CONTRACTS.md.
