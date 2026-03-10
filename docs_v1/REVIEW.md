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
