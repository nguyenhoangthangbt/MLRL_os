"""Tests for mlrl_os.rl.curriculum — Curriculum learning manager."""

from __future__ import annotations

import pytest

from mlrl_os.rl.curriculum import CurriculumManager, StressScenario

# ---------------------------------------------------------------------------
# StressScenario
# ---------------------------------------------------------------------------


class TestStressScenario:
    """Tests for the StressScenario dataclass."""

    def test_creation(self) -> None:
        s = StressScenario(id=1, description="baseline", difficulty=1)
        assert s.id == 1
        assert s.description == "baseline"
        assert s.difficulty == 1
        assert s.config_change == {}

    def test_with_config_change(self) -> None:
        s = StressScenario(
            id=2,
            description="high load",
            difficulty=3,
            config_change={"arrival_rate": 2.0},
        )
        assert s.config_change == {"arrival_rate": 2.0}


# ---------------------------------------------------------------------------
# CurriculumManager
# ---------------------------------------------------------------------------


def _sample_scenarios() -> list[StressScenario]:
    """Three scenarios of increasing difficulty."""
    return [
        StressScenario(id=1, description="easy", difficulty=1),
        StressScenario(id=2, description="medium", difficulty=5),
        StressScenario(id=3, description="hard", difficulty=10),
    ]


class TestCurriculumManager:
    """Tests for CurriculumManager progression logic."""

    def test_starts_at_easiest_scenario(self) -> None:
        mgr = CurriculumManager(scenarios=_sample_scenarios())
        assert mgr.current_scenario().difficulty == 1

    def test_sorts_by_difficulty(self) -> None:
        # Pass scenarios in reverse order
        reversed_scenarios = list(reversed(_sample_scenarios()))
        mgr = CurriculumManager(scenarios=reversed_scenarios)
        assert mgr.current_scenario().difficulty == 1

    def test_should_advance_above_threshold(self) -> None:
        mgr = CurriculumManager(scenarios=_sample_scenarios(), threshold=0.8)
        assert mgr.should_advance(success_rate=0.85)

    def test_should_not_advance_below_threshold(self) -> None:
        mgr = CurriculumManager(scenarios=_sample_scenarios(), threshold=0.8)
        assert not mgr.should_advance(success_rate=0.7)

    def test_should_not_advance_at_last_scenario(self) -> None:
        mgr = CurriculumManager(scenarios=_sample_scenarios(), threshold=0.8)
        mgr.advance()  # -> medium
        mgr.advance()  # -> hard (last)
        assert not mgr.should_advance(success_rate=1.0)

    def test_advance_returns_next_scenario(self) -> None:
        mgr = CurriculumManager(scenarios=_sample_scenarios())
        next_scenario = mgr.advance()
        assert next_scenario is not None
        assert next_scenario.difficulty == 5
        assert mgr.current_scenario().difficulty == 5

    def test_advance_at_last_returns_none(self) -> None:
        mgr = CurriculumManager(scenarios=_sample_scenarios())
        mgr.advance()
        mgr.advance()
        result = mgr.advance()
        assert result is None

    def test_is_complete_at_last_scenario(self) -> None:
        mgr = CurriculumManager(scenarios=_sample_scenarios())
        assert not mgr.is_complete
        mgr.advance()
        assert not mgr.is_complete
        mgr.advance()
        assert mgr.is_complete

    def test_progress_tracks_advancement(self) -> None:
        mgr = CurriculumManager(scenarios=_sample_scenarios())
        assert mgr.progress == pytest.approx(1 / 3)
        mgr.advance()
        assert mgr.progress == pytest.approx(2 / 3)
        mgr.advance()
        assert mgr.progress == pytest.approx(1.0)

    def test_single_scenario(self) -> None:
        scenarios = [StressScenario(id=1, description="only", difficulty=1)]
        mgr = CurriculumManager(scenarios=scenarios)
        assert mgr.is_complete
        assert mgr.progress == 1.0
        assert mgr.advance() is None

    def test_threshold_exact_boundary(self) -> None:
        mgr = CurriculumManager(scenarios=_sample_scenarios(), threshold=0.8)
        assert mgr.should_advance(success_rate=0.8)

    def test_custom_threshold(self) -> None:
        mgr = CurriculumManager(scenarios=_sample_scenarios(), threshold=0.95)
        assert not mgr.should_advance(success_rate=0.9)
        assert mgr.should_advance(success_rate=0.95)
