"""Tests for mlrl_os.rl.rewards — Reward functions and composite shaping."""

from __future__ import annotations

import pytest

from mlrl_os.rl.rewards import (
    CompositeReward,
    CostMinimizationReward,
    SLAComplianceReward,
    ThroughputReward,
)

# ---------------------------------------------------------------------------
# ThroughputReward
# ---------------------------------------------------------------------------


class TestThroughputReward:
    """Tests for ThroughputReward."""

    def test_positive_reward_on_completion(self) -> None:
        reward_fn = ThroughputReward()
        prev = {"completed_count": 5}
        curr = {"completed_count": 7}
        assert reward_fn.compute(prev, action=0, curr_state=curr, done=False) == 2.0

    def test_zero_reward_no_completions(self) -> None:
        reward_fn = ThroughputReward()
        prev = {"completed_count": 5}
        curr = {"completed_count": 5}
        assert reward_fn.compute(prev, action=0, curr_state=curr, done=False) == 0.0

    def test_missing_keys_default_to_zero(self) -> None:
        reward_fn = ThroughputReward()
        assert reward_fn.compute({}, action=0, curr_state={}, done=False) == 0.0

    def test_returns_float(self) -> None:
        reward_fn = ThroughputReward()
        result = reward_fn.compute(
            {"completed_count": 0}, action=0, curr_state={"completed_count": 1}, done=True,
        )
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# SLAComplianceReward
# ---------------------------------------------------------------------------


class TestSLAComplianceReward:
    """Tests for SLAComplianceReward."""

    def test_penalty_on_breach(self) -> None:
        reward_fn = SLAComplianceReward(breach_penalty=-10.0)
        prev = {"sla_breaches": 0}
        curr = {"sla_breaches": 2}
        assert reward_fn.compute(prev, action=0, curr_state=curr, done=False) == -20.0

    def test_bonus_on_completion_without_breach(self) -> None:
        reward_fn = SLAComplianceReward(compliance_bonus=1.0)
        prev = {"sla_breaches": 0, "completed_count": 5}
        curr = {"sla_breaches": 0, "completed_count": 6}
        assert reward_fn.compute(prev, action=0, curr_state=curr, done=True) == 1.0

    def test_zero_reward_no_event(self) -> None:
        reward_fn = SLAComplianceReward()
        prev = {"sla_breaches": 0, "completed_count": 5}
        curr = {"sla_breaches": 0, "completed_count": 5}
        assert reward_fn.compute(prev, action=0, curr_state=curr, done=False) == 0.0

    def test_custom_penalty_and_bonus(self) -> None:
        reward_fn = SLAComplianceReward(breach_penalty=-5.0, compliance_bonus=2.0)
        prev = {"sla_breaches": 1}
        curr = {"sla_breaches": 2}
        assert reward_fn.compute(prev, action=0, curr_state=curr, done=False) == -5.0

    def test_no_bonus_when_not_done(self) -> None:
        reward_fn = SLAComplianceReward(compliance_bonus=1.0)
        prev = {"sla_breaches": 0, "completed_count": 5}
        curr = {"sla_breaches": 0, "completed_count": 6}
        assert reward_fn.compute(prev, action=0, curr_state=curr, done=False) == 0.0


# ---------------------------------------------------------------------------
# CostMinimizationReward
# ---------------------------------------------------------------------------


class TestCostMinimizationReward:
    """Tests for CostMinimizationReward."""

    def test_negative_cost_increment(self) -> None:
        reward_fn = CostMinimizationReward()
        prev = {"total_cost": 100.0}
        curr = {"total_cost": 115.0}
        assert reward_fn.compute(prev, action=0, curr_state=curr, done=False) == -15.0

    def test_zero_cost_no_change(self) -> None:
        reward_fn = CostMinimizationReward()
        prev = {"total_cost": 100.0}
        curr = {"total_cost": 100.0}
        assert reward_fn.compute(prev, action=0, curr_state=curr, done=False) == 0.0

    def test_missing_keys_default_to_zero(self) -> None:
        reward_fn = CostMinimizationReward()
        assert reward_fn.compute({}, action=0, curr_state={"total_cost": 5.0}, done=False) == -5.0


# ---------------------------------------------------------------------------
# CompositeReward
# ---------------------------------------------------------------------------


class TestCompositeReward:
    """Tests for CompositeReward weighted combination."""

    def test_weighted_sum(self) -> None:
        throughput = ThroughputReward()
        cost = CostMinimizationReward()
        composite = CompositeReward(components=[(throughput, 1.0), (cost, 0.5)])

        prev = {"completed_count": 5, "total_cost": 100.0}
        curr = {"completed_count": 7, "total_cost": 120.0}

        # throughput: 2.0 * 1.0 = 2.0; cost: -20.0 * 0.5 = -10.0 => total -8.0
        result = composite.compute(prev, action=0, curr_state=curr, done=False)
        assert result == pytest.approx(-8.0)

    def test_single_component(self) -> None:
        throughput = ThroughputReward()
        composite = CompositeReward(components=[(throughput, 2.0)])

        prev = {"completed_count": 0}
        curr = {"completed_count": 3}
        assert composite.compute(prev, action=0, curr_state=curr, done=False) == 6.0

    def test_empty_components(self) -> None:
        composite = CompositeReward(components=[])
        assert composite.compute({}, action=0, curr_state={}, done=False) == 0.0

    def test_all_reward_types_combined(self) -> None:
        composite = CompositeReward(
            components=[
                (ThroughputReward(), 1.0),
                (SLAComplianceReward(breach_penalty=-10.0), 1.0),
                (CostMinimizationReward(), 0.1),
            ]
        )
        prev = {"completed_count": 0, "sla_breaches": 0, "total_cost": 0.0}
        curr = {"completed_count": 1, "sla_breaches": 1, "total_cost": 50.0}

        # throughput: 1.0; sla: -10.0; cost: -50.0 * 0.1 = -5.0 => total -14.0
        result = composite.compute(prev, action=0, curr_state=curr, done=False)
        assert result == pytest.approx(-14.0)
