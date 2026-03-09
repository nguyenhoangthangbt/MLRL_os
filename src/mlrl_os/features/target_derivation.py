"""Derive predictive classification targets from raw entity trajectory data.

SimOS exports raw trajectory records where ``episode_status`` reflects current
state (in_progress/completed), not a prediction-worthy outcome. This module
computes **eventual outcome labels** and propagates them back to all
intermediate steps, enabling mid-journey prediction.

Three derived targets are supported:

- ``sla_breach``: Binary — will this entity breach its SLA?
- ``delay_severity``: 3-class — how delayed will this entity be?
- ``wait_ratio_class``: 3-class — congested, moderate, or efficient?

See ``docs_v1/design/DERIVED_TARGETS.md`` for full specification.
"""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)

# All recognised derived target names
DERIVED_TARGETS = frozenset({"sla_breach", "delay_severity", "wait_ratio_class"})

# Columns that must be excluded from features when using derived targets
# (they leak the eventual outcome)
LEAKAGE_COLUMNS = frozenset({
    "total_time",
    "done",
    "status",
    "rw_episode_completion",
    "rw_step_completion",
    "rw_sla_budget_remaining",
    "rw_sla_penalty",
    "t_complete",
})

# Domain → preferred SLA field for sla_breach derivation
_DOMAIN_SLA_FIELD: dict[str, str] = {
    "healthcare": "sla_standard_visit_limit",
    "service": "sla_resolution_limit",
    "supply_chain": "sla_order_completion_limit",
}


def is_derived_target(target: str) -> bool:
    """Return True if ``target`` is a derived target name."""
    return target in DERIVED_TARGETS


def get_leakage_columns() -> frozenset[str]:
    """Return the set of columns that must be excluded when using derived targets."""
    return LEAKAGE_COLUMNS


def derive_target(
    df: pl.DataFrame,
    target: str,
    *,
    sla_column: str | None = None,
    sla_threshold: float | None = None,
) -> pl.DataFrame:
    """Add a derived target column to the trajectory DataFrame.

    For each entity, computes the eventual outcome from its final
    (``done=True``) record and labels **all** intermediate steps with
    that outcome. Entities that never complete are labeled ``"unknown"``
    and should be filtered out before training.

    Args:
        df: Trajectory DataFrame with canonical column names.
        target: One of ``"sla_breach"``, ``"delay_severity"``,
            ``"wait_ratio_class"``.
        sla_column: Explicit SLA column for ``sla_breach``.
            Auto-detected from domain if not provided.
        sla_threshold: Explicit SLA threshold in seconds.
            Overrides the per-entity SLA column value.

    Returns:
        DataFrame with the derived target column added.

    Raises:
        ValueError: If ``target`` is not a recognised derived target.
    """
    if target not in DERIVED_TARGETS:
        msg = (
            f"Unknown derived target '{target}'. "
            f"Valid options: {sorted(DERIVED_TARGETS)}"
        )
        raise ValueError(msg)

    if target == "sla_breach":
        return _derive_sla_breach(df, sla_column=sla_column, sla_threshold=sla_threshold)
    if target == "delay_severity":
        return _derive_delay_severity(df)
    if target == "wait_ratio_class":
        return _derive_wait_ratio_class(df)

    msg = f"Unhandled derived target: {target}"  # pragma: no cover
    raise ValueError(msg)


# ── SLA Breach ───────────────────────────────────────────────────────


def _detect_sla_column(df: pl.DataFrame) -> str | None:
    """Auto-detect the best SLA column from available columns.

    Strategy:
    1. If a domain-preferred SLA field exists, use it.
    2. Otherwise, use the first ``sla_*`` column found.
    """
    sla_cols = sorted(c for c in df.columns if c.startswith("sla_"))
    if not sla_cols:
        return None

    # Check for domain-preferred column (requires domain info in data)
    # The SLA columns themselves hint at the domain
    for _domain, preferred in _DOMAIN_SLA_FIELD.items():
        if preferred in sla_cols:
            return preferred

    return sla_cols[0]


def _derive_sla_breach(
    df: pl.DataFrame,
    *,
    sla_column: str | None = None,
    sla_threshold: float | None = None,
) -> pl.DataFrame:
    """Derive binary ``sla_breach`` target.

    Each completed entity is labeled ``"breach"`` if its total time
    exceeds its SLA limit, ``"no_breach"`` otherwise. The label is
    propagated to all steps of that entity.

    When no SLA column is found and no threshold is given, falls back
    to p75 of ``total_time`` as the breach threshold.
    """
    if "eid" not in df.columns:
        msg = "Column 'eid' required for target derivation"
        raise ValueError(msg)

    # Get completed entities (final record)
    completed = df.filter(pl.col("done") == True)  # noqa: E712
    if len(completed) == 0:
        # No completed entities — label everything "unknown"
        return df.with_columns(pl.lit("unknown").alias("sla_breach"))

    # Determine threshold per entity
    if sla_threshold is not None:
        # Fixed threshold for all entities
        breached_eids = (
            completed.filter(pl.col("total_time") > sla_threshold)["eid"].to_list()
        )
        all_eids = completed.select("eid").unique()
        entity_labels = all_eids.with_columns(
            pl.when(pl.col("eid").is_in(breached_eids))
            .then(pl.lit("breach"))
            .otherwise(pl.lit("no_breach"))
            .alias("sla_breach")
        )
    elif sla_column and sla_column in df.columns:
        # Per-entity SLA from column
        entity_outcomes = (
            completed
            .group_by("eid")
            .agg([
                pl.col("total_time").last(),
                pl.col(sla_column).last(),
            ])
            .with_columns(
                pl.when(pl.col("total_time") > pl.col(sla_column))
                .then(pl.lit("breach"))
                .otherwise(pl.lit("no_breach"))
                .alias("sla_breach")
            )
            .select(["eid", "sla_breach"])
        )
        entity_labels = entity_outcomes
    else:
        # Auto-detect SLA column
        detected = sla_column or _detect_sla_column(df)
        if detected and detected in df.columns:
            return _derive_sla_breach(df, sla_column=detected)

        # Fallback: p75 of total_time
        total_times = completed["total_time"].drop_nulls()
        p75 = float(total_times.quantile(0.75))
        logger.info(
            "No SLA column found; using p75 of total_time (%.0fs) as breach threshold",
            p75,
        )
        return _derive_sla_breach(df, sla_threshold=p75)

    # Join labels back to all rows
    df = df.join(entity_labels, on="eid", how="left")

    # Entities that never completed get "unknown"
    df = df.with_columns(
        pl.col("sla_breach").fill_null(pl.lit("unknown"))
    )

    _log_class_distribution(df, "sla_breach")
    return df


# ── Delay Severity ───────────────────────────────────────────────────


def _derive_delay_severity(df: pl.DataFrame) -> pl.DataFrame:
    """Derive 3-class ``delay_severity`` target based on wait ratio.

    Uses cumulative wait / total_time to determine severity.
    Bins by percentiles of the completed entity distribution
    to ensure ~33% per class.
    """
    if "eid" not in df.columns:
        msg = "Column 'eid' required for target derivation"
        raise ValueError(msg)

    completed = df.filter(pl.col("done") == True)  # noqa: E712
    if len(completed) == 0:
        return df.with_columns(pl.lit("unknown").alias("delay_severity"))

    # Compute wait ratio per completed entity
    entity_outcomes = (
        completed
        .group_by("eid")
        .agg([
            pl.col("total_time").last(),
            pl.col("s_cum_wait").last().alias("final_cum_wait"),
        ])
        .with_columns(
            pl.when(pl.col("total_time") > 0)
            .then(pl.col("final_cum_wait") / pl.col("total_time"))
            .otherwise(0.0)
            .alias("_wait_ratio")
        )
    )

    # Compute percentile thresholds for balanced classes
    ratios = entity_outcomes["_wait_ratio"].drop_nulls()
    p33 = float(ratios.quantile(0.33))
    p66 = float(ratios.quantile(0.66))

    logger.info(
        "delay_severity thresholds: on_time<=%.3f, minor_delay<=%.3f, major_delay>%.3f",
        p33, p66, p66,
    )

    entity_labels = entity_outcomes.with_columns(
        pl.when(pl.col("_wait_ratio") <= p33)
        .then(pl.lit("on_time"))
        .when(pl.col("_wait_ratio") <= p66)
        .then(pl.lit("minor_delay"))
        .otherwise(pl.lit("major_delay"))
        .alias("delay_severity")
    ).select(["eid", "delay_severity"])

    df = df.join(entity_labels, on="eid", how="left")
    df = df.with_columns(
        pl.col("delay_severity").fill_null(pl.lit("unknown"))
    )

    _log_class_distribution(df, "delay_severity")
    return df


# ── Wait Ratio Class ─────────────────────────────────────────────────


def _derive_wait_ratio_class(df: pl.DataFrame) -> pl.DataFrame:
    """Derive 3-class ``wait_ratio_class`` target with fixed thresholds.

    Uses wait / (wait + processing) to classify entities as
    efficient (<=0.3), moderate (0.3-0.6), or congested (>0.6).
    Fixed thresholds enable cross-template comparison.
    """
    if "eid" not in df.columns:
        msg = "Column 'eid' required for target derivation"
        raise ValueError(msg)

    completed = df.filter(pl.col("done") == True)  # noqa: E712
    if len(completed) == 0:
        return df.with_columns(pl.lit("unknown").alias("wait_ratio_class"))

    entity_outcomes = (
        completed
        .group_by("eid")
        .agg([
            pl.col("s_cum_wait").last().alias("final_cum_wait"),
            pl.col("s_cum_processing").last().alias("final_cum_proc"),
        ])
        .with_columns(
            pl.when((pl.col("final_cum_wait") + pl.col("final_cum_proc")) > 0)
            .then(
                pl.col("final_cum_wait")
                / (pl.col("final_cum_wait") + pl.col("final_cum_proc"))
            )
            .otherwise(0.0)
            .alias("_wr")
        )
    )

    entity_labels = entity_outcomes.with_columns(
        pl.when(pl.col("_wr") <= 0.3)
        .then(pl.lit("efficient"))
        .when(pl.col("_wr") <= 0.6)
        .then(pl.lit("moderate"))
        .otherwise(pl.lit("congested"))
        .alias("wait_ratio_class")
    ).select(["eid", "wait_ratio_class"])

    df = df.join(entity_labels, on="eid", how="left")
    df = df.with_columns(
        pl.col("wait_ratio_class").fill_null(pl.lit("unknown"))
    )

    _log_class_distribution(df, "wait_ratio_class")
    return df


# ── Helpers ──────────────────────────────────────────────────────────


def _log_class_distribution(df: pl.DataFrame, target: str) -> None:
    """Log the class distribution of a derived target."""
    counts = df[target].value_counts().sort("count", descending=True)
    total = len(df)
    parts: list[str] = []
    for row in counts.iter_rows(named=True):
        label = row[target]
        count = row["count"]
        pct = count / total * 100 if total > 0 else 0
        parts.append(f"{label}={count} ({pct:.1f}%)")
    logger.info("Derived target '%s' distribution: %s", target, ", ".join(parts))
