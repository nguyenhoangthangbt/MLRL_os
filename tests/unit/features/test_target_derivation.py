"""Tests for derived target computation."""

from __future__ import annotations

import polars as pl
import pytest

from mlrl_os.features.target_derivation import (
    DERIVED_TARGETS,
    LEAKAGE_COLUMNS,
    derive_target,
    get_leakage_columns,
    is_derived_target,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_trajectory_df(
    n_entities: int = 20,
    steps_per_entity: int = 4,
    *,
    include_sla: bool = True,
) -> pl.DataFrame:
    """Build a synthetic trajectory DataFrame for testing.

    Each entity goes through ``steps_per_entity`` steps. The last step
    has ``done=True``. Total time varies linearly by entity ID.
    """
    rows: list[dict] = []
    for eid in range(n_entities):
        total_time = 100.0 + eid * 50.0  # 100..1050
        # Vary wait ratio from ~10% to ~80% across entities
        wait_fraction = 0.1 + 0.7 * (eid / max(n_entities - 1, 1))
        cum_wait = total_time * wait_fraction
        cum_proc = total_time - cum_wait
        sla_limit = 600.0  # Fixed SLA for all entities

        for step in range(steps_per_entity):
            is_final = step == steps_per_entity - 1
            row = {
                "eid": f"e{eid:03d}",
                "etype": "order",
                "step": step,
                "node": f"node_{step}",
                "done": is_final,
                "status": "completed" if is_final else "in_progress",
                "total_time": total_time if is_final else None,
                "t_enter": float(step * 100),
                "t_complete": float(step * 100 + 50),
                "wait": cum_wait / steps_per_entity,
                "processing": cum_proc / steps_per_entity,
                "s_elapsed": float(step * 100 + 50),
                "s_cum_wait": cum_wait * (step + 1) / steps_per_entity,
                "s_cum_processing": cum_proc * (step + 1) / steps_per_entity,
                "s_wait_ratio": cum_wait / total_time,
                "s_steps_done": step + 1,
                "s_node_util": 0.7 + eid * 0.01,
                "s_sys_util": 0.65,
                "rw_sla_budget_remaining": sla_limit - (step + 1) * 50,
                "rw_episode_completion": 15.0 if is_final else None,
            }
            if include_sla:
                row["sla_order_completion_limit"] = sla_limit
            rows.append(row)

    return pl.DataFrame(rows)


# ── Unit tests: utility functions ────────────────────────────────────


class TestUtilityFunctions:
    def test_is_derived_target_recognized(self):
        assert is_derived_target("sla_breach")
        assert is_derived_target("delay_severity")
        assert is_derived_target("wait_ratio_class")

    def test_is_derived_target_unrecognized(self):
        assert not is_derived_target("status")
        assert not is_derived_target("total_time")
        assert not is_derived_target("unknown_target")

    def test_derived_targets_constant(self):
        assert DERIVED_TARGETS == {"sla_breach", "delay_severity", "wait_ratio_class"}

    def test_leakage_columns_includes_key_columns(self):
        lc = get_leakage_columns()
        assert "total_time" in lc
        assert "done" in lc
        assert "status" in lc
        assert "rw_episode_completion" in lc

    def test_unknown_target_raises(self):
        df = _make_trajectory_df()
        with pytest.raises(ValueError, match="Unknown derived target"):
            derive_target(df, "not_a_target")


# ── SLA Breach ───────────────────────────────────────────────────────


class TestSLABreach:
    def test_with_sla_column(self):
        df = _make_trajectory_df(n_entities=20, include_sla=True)
        result = derive_target(df, "sla_breach")

        assert "sla_breach" in result.columns
        values = set(result["sla_breach"].unique().to_list())
        assert values <= {"breach", "no_breach"}

    def test_breach_labels_correct(self):
        """Entities with total_time > sla_limit should be labeled 'breach'."""
        df = _make_trajectory_df(n_entities=20, include_sla=True)
        # SLA limit is 600. Entities with total_time > 600 should breach.
        # total_time = 100 + eid*50, so eid >= 11 has total_time >= 650
        result = derive_target(df, "sla_breach")

        # Check a specific entity
        e00 = result.filter(pl.col("eid") == "e000")
        assert e00["sla_breach"][0] == "no_breach"  # total_time=100 < 600

        e15 = result.filter(pl.col("eid") == "e015")
        assert e15["sla_breach"][0] == "breach"  # total_time=850 > 600

    def test_all_steps_labeled(self):
        """All steps of an entity should have the same label."""
        df = _make_trajectory_df(n_entities=10)
        result = derive_target(df, "sla_breach")

        for eid in result["eid"].unique().to_list():
            entity_rows = result.filter(pl.col("eid") == eid)
            labels = entity_rows["sla_breach"].unique().to_list()
            assert len(labels) == 1, f"Entity {eid} has mixed labels: {labels}"

    def test_with_explicit_threshold(self):
        df = _make_trajectory_df(n_entities=10, include_sla=False)
        result = derive_target(df, "sla_breach", sla_threshold=300.0)

        assert "sla_breach" in result.columns
        breach_count = result.filter(
            (pl.col("sla_breach") == "breach") & (pl.col("done") == True)
        ).height
        no_breach_count = result.filter(
            (pl.col("sla_breach") == "no_breach") & (pl.col("done") == True)
        ).height
        assert breach_count + no_breach_count == 10

    def test_fallback_to_p75_when_no_sla(self):
        """When no SLA column and no threshold, falls back to p75."""
        df = _make_trajectory_df(n_entities=20, include_sla=False)
        result = derive_target(df, "sla_breach")

        assert "sla_breach" in result.columns
        # Should have both classes (p75 splits the data)
        unique = set(result["sla_breach"].unique().to_list())
        assert "no_breach" in unique

    def test_no_completed_entities(self):
        """When no entities are done, all get 'unknown'."""
        df = _make_trajectory_df(n_entities=5)
        # Remove all done=True rows
        df = df.with_columns(pl.lit(False).alias("done"))
        result = derive_target(df, "sla_breach")
        assert set(result["sla_breach"].unique().to_list()) == {"unknown"}

    def test_requires_eid_column(self):
        df = pl.DataFrame({"a": [1, 2], "done": [True, False]})
        with pytest.raises(ValueError, match="'eid' required"):
            derive_target(df, "sla_breach")


# ── Delay Severity ───────────────────────────────────────────────────


class TestDelaySeverity:
    def test_produces_three_classes(self):
        df = _make_trajectory_df(n_entities=30)
        result = derive_target(df, "delay_severity")

        assert "delay_severity" in result.columns
        classes = set(result["delay_severity"].unique().to_list())
        assert classes == {"on_time", "minor_delay", "major_delay"}

    def test_roughly_balanced(self):
        """Each class should have roughly 33% of completed entities."""
        df = _make_trajectory_df(n_entities=60)
        result = derive_target(df, "delay_severity")

        # Check balance on completed entities only
        completed = result.filter(pl.col("done") == True)
        counts = completed["delay_severity"].value_counts()
        for row in counts.iter_rows(named=True):
            pct = row["count"] / len(completed)
            assert 0.15 <= pct <= 0.50, f"Class {row['delay_severity']} is {pct:.0%}"

    def test_all_steps_labeled(self):
        df = _make_trajectory_df(n_entities=10)
        result = derive_target(df, "delay_severity")

        for eid in result["eid"].unique().to_list():
            entity_rows = result.filter(pl.col("eid") == eid)
            labels = entity_rows["delay_severity"].unique().to_list()
            assert len(labels) == 1

    def test_no_completed_entities(self):
        df = _make_trajectory_df(n_entities=5)
        df = df.with_columns(pl.lit(False).alias("done"))
        result = derive_target(df, "delay_severity")
        assert set(result["delay_severity"].unique().to_list()) == {"unknown"}


# ── Wait Ratio Class ─────────────────────────────────────────────────


class TestWaitRatioClass:
    def test_produces_valid_classes(self):
        df = _make_trajectory_df(n_entities=20)
        result = derive_target(df, "wait_ratio_class")

        assert "wait_ratio_class" in result.columns
        classes = set(result["wait_ratio_class"].unique().to_list())
        assert classes <= {"efficient", "moderate", "congested"}

    def test_fixed_thresholds(self):
        """Verify fixed threshold logic: <=0.3 efficient, <=0.6 moderate, >0.6 congested."""
        # Create entities with known wait ratios
        rows = []
        for eid, wait, proc in [
            ("low", 10.0, 90.0),     # 10% → efficient
            ("mid", 50.0, 50.0),     # 50% → moderate
            ("high", 80.0, 20.0),    # 80% → congested
        ]:
            for step in range(3):
                is_final = step == 2
                rows.append({
                    "eid": eid,
                    "step": step,
                    "done": is_final,
                    "total_time": 100.0 if is_final else None,
                    "s_cum_wait": wait,
                    "s_cum_processing": proc,
                })

        df = pl.DataFrame(rows)
        result = derive_target(df, "wait_ratio_class")

        low = result.filter(pl.col("eid") == "low")["wait_ratio_class"][0]
        mid = result.filter(pl.col("eid") == "mid")["wait_ratio_class"][0]
        high = result.filter(pl.col("eid") == "high")["wait_ratio_class"][0]

        assert low == "efficient"
        assert mid == "moderate"
        assert high == "congested"

    def test_all_steps_labeled(self):
        df = _make_trajectory_df(n_entities=10)
        result = derive_target(df, "wait_ratio_class")

        for eid in result["eid"].unique().to_list():
            entity_rows = result.filter(pl.col("eid") == eid)
            labels = entity_rows["wait_ratio_class"].unique().to_list()
            assert len(labels) == 1

    def test_no_completed_entities(self):
        df = _make_trajectory_df(n_entities=5)
        df = df.with_columns(pl.lit(False).alias("done"))
        result = derive_target(df, "wait_ratio_class")
        assert set(result["wait_ratio_class"].unique().to_list()) == {"unknown"}
