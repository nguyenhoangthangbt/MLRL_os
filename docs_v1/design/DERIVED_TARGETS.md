# Derived Target Specification — Entity Classification

> **Purpose:** MLRL_os computes predictive classification targets from raw SimOS trajectory data.
> SimOS produces operational data; MLRL_os decides what to predict.

## Design Principle

SimOS exports raw entity trajectory records with per-step MDP state vectors.
The `episode_status` field reflects **current** state (`in_progress`/`completed`),
not a prediction-worthy outcome. MLRL_os derives **eventual outcome labels** and
propagates them back to all intermediate steps, enabling mid-journey prediction.

```
SimOS YAML config
  └─ constraints.sla[] → per-entity SLA limits in export
  └─ entities.attributes → priority_class, domain flags
  └─ simulation.duration → determines which entities complete

SimOS ML Export (trajectories)
  └─ episode_total_time  → actual completion time (null if not done)
  └─ sla_*               → per-entity SLA thresholds (from config)
  └─ domain_*            → domain enrichment flags

MLRL_os Target Derivation
  └─ Groups by entity_id
  └─ Finds final record (episode_done=True)
  └─ Computes derived outcome from final record
  └─ Labels ALL steps of that entity with eventual outcome
  └─ Excludes raw outcome columns from features (prevent leakage)
```

## Derived Targets

### 1. `sla_breach` — Binary Classification

**Question:** "Will this entity breach its SLA?"

**Derivation:**
```
For each completed entity:
  breach = episode_total_time > primary_sla_limit
  label = "breach" if breach else "no_breach"
```

**Primary SLA limit selection** (per domain):
| Domain | Primary SLA Field | Fallback |
|---|---|---|
| healthcare | `sla_standard_visit_limit` | first `sla_*` field found |
| service | `sla_resolution_limit` | first `sla_*` field found |
| supply_chain | `sla_order_completion_limit` | first `sla_*` field found |
| *(any)* | user-specified or first `sla_*` field | p75 of total_time |

When no SLA field exists, falls back to p75 of `episode_total_time` as threshold.

**Observed breach rates (test fixtures):**
| Template | SLA Field | Breach Rate |
|---|---|---|
| healthcare_er | sla_standard_visit_limit | 21.4% |
| call_center | sla_resolution_limit | 0.0% |
| logistics_otd | sla_order_completion_limit | 0.0% |

Note: Some templates have 0% breach under default configs. The `delay_severity`
target provides class variation for these cases.

### 2. `delay_severity` — Multi-class Classification

**Question:** "How delayed will this entity be?"

**Derivation:**
```
For each completed entity:
  compute wait_ratio = cumulative_wait / episode_total_time
  bin into severity classes based on distribution percentiles:
    "on_time"      : wait_ratio <= p33
    "minor_delay"  : p33 < wait_ratio <= p66
    "major_delay"  : wait_ratio > p66
```

This target always produces 3 balanced classes (~33% each) regardless of template.

### 3. `wait_ratio_class` — Multi-class Classification

**Question:** "Will this entity experience high waiting relative to processing?"

**Derivation:**
```
For each completed entity:
  wait_ratio = cumulative_wait / (cumulative_wait + cumulative_processing)
  bin by fixed thresholds:
    "efficient"   : wait_ratio <= 0.3
    "moderate"    : 0.3 < wait_ratio <= 0.6
    "congested"   : wait_ratio > 0.6
```

Fixed thresholds enable cross-template comparison.

## Leakage Prevention

When a derived target is selected, the following columns are **automatically excluded**
from features (they reveal the outcome):

| Excluded Column | Reason |
|---|---|
| `total_time` | Direct input to SLA breach / delay severity |
| `done` | Binary flag equivalent of status |
| `status` | Current state, not predictive |
| `rw_episode_completion` | Reward signal only available at completion |
| `rw_step_completion` | Correlated with completion |
| `rw_sla_budget_remaining` | Directly derived from SLA limit - elapsed |
| `rw_sla_penalty` | Zero until breach occurs |
| `t_complete` | Completion timestamp |

## Full Pipeline Trace per Template

### healthcare_er (Emergency Room)

```
SimOS YAML Config:
  entities.type: patient
  entities.attributes.acuity: categorical [1,2,3,4,5]
  constraints.sla:
    - esi_1_immediate: node=triage, max_wait=600
    - esi_2_urgent: node=diagnosis, max_wait=1800
    - total_visit_under_4h: max_total=14400
  simulation.duration: 86400 (24h)

SimOS Export (trajectories, 560 rows):
  SLA fields: sla_critical_wait_threshold=600, sla_standard_visit_limit=21600
  Domain fields: domain_acuity_level, domain_acuity_is_critical
  Outcome: 187 entities completed, 0 in-progress at end

MLRL_os Derivation:
  sla_breach:
    primary_sla = sla_standard_visit_limit (21600s = 6h)
    breach_rate = 21.4% (40/187) → good class balance
  delay_severity:
    total_time distribution: p33=5636s p66=16088s
    on_time/minor_delay/major_delay → ~33% each
  wait_ratio_class:
    efficient/moderate/congested based on wait/(wait+processing)

Predictive Features (mid-journey, step < final):
  Entity state: s_priority (acuity), s_elapsed, s_cum_wait, s_cum_processing
  Node state: s_node_util, s_node_avg_queue (diagnosis/treatment congestion)
  System state: s_sys_util, s_sys_bottleneck_util
  Domain: domain_acuity_level, domain_acuity_is_critical
  Derived: d_progress_ratio, d_wait_trend, d_util_relative
```

### call_center (Telecom Call Center)

```
SimOS YAML Config:
  entities.type: call
  entities.attributes.priority_class: categorical [standard, vip]
  constraints.sla:
    - first_response: max_wait=300
    - resolution: max_total=3600
  simulation.duration: 28800 (8h)

SimOS Export (trajectories, 2119 rows):
  SLA fields: sla_first_response_limit=300, sla_resolution_limit=3600
  Domain fields: domain_is_vip, domain_is_escalated
  Outcome: 432 completed, 0 breach on resolution_limit

MLRL_os Derivation:
  sla_breach:
    primary_sla = sla_resolution_limit (3600s)
    breach_rate = 0.0% → NOT useful (single class)
    fallback: use delay_severity instead
  delay_severity:
    total_time p33=529s, p66=691s
    3 balanced classes → useful
  wait_ratio_class:
    fixed thresholds → depends on call routing efficiency

Predictive Features:
  Entity state: s_priority, s_elapsed, s_cum_wait
  Node state: s_node_util (ivr/routing/agent nodes)
  System state: s_sys_util, s_sys_throughput
  Domain: domain_is_vip, domain_is_escalated
  Resource: s_r_general_agents_util, s_r_technical_agents_util
```

### logistics_otd (Warehouse Fulfillment)

```
SimOS YAML Config:
  entities.type: order
  entities.attributes.priority_class: categorical [standard, express]
  constraints.sla:
    - order_completion: max_total=14400
    - express_delivery: max_total=14400
    - critical_supply: max_total=10800
    - standard_delivery: max_total=86400
  simulation.duration: 28800 (8h)

SimOS Export (trajectories, 3688 rows):
  SLA fields: sla_order_completion_limit=14400, sla_express_delivery_limit=14400,
              sla_critical_supply_threshold=10800, sla_standard_delivery_limit=86400
  Domain fields: domain_is_express, domain_is_critical_item,
                 domain_is_premium_product, domain_is_backup_sourced
  Outcome: 996 completed, 2.7% breach on critical_supply

MLRL_os Derivation:
  sla_breach:
    primary_sla = sla_order_completion_limit (14400s)
    breach_rate = 0.0% → NOT useful
    fallback: use sla_critical_supply_threshold → 2.7% still low
    best option: delay_severity
  delay_severity:
    total_time distribution highly bimodal (fast picks vs slow orders)
    3 balanced classes → useful
  wait_ratio_class:
    Shows pick station vs pack station congestion patterns

Predictive Features:
  Entity state: s_priority, s_elapsed, s_cum_wait, s_cum_cost
  Node state: s_node_util (intake/pick/pack/dispatch)
  Domain: domain_is_express, domain_is_critical_item
  Resource: s_r_pickers_util
```

## Implementation Location

| Component | File | Responsibility |
|---|---|---|
| Target derivation | `features/target_derivation.py` | Compute derived columns, label all steps |
| Feature engine | `features/entity.py` | Call derivation when target is derived |
| Target discovery | `data/discovery.py` | Advertise derived targets as available |
| Validation gate | `validation/gate.py` | Accept derived target names |
| Tests | `tests/unit/features/test_target_derivation.py` | Full test coverage |
