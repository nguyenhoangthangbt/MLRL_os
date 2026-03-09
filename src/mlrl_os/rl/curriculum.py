"""Curriculum learning manager for progressive difficulty training.

Uses SimOS Layer 5 stress scenarios to train agents on increasingly
difficult configurations.  The manager tracks progress and decides
when the agent is ready to advance to the next difficulty level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StressScenario:
    """A single training scenario with associated difficulty."""

    id: int
    description: str
    difficulty: int
    config_change: dict[str, Any] = field(default_factory=dict)


class CurriculumManager:
    """Manages progression through ordered difficulty levels.

    Scenarios are sorted by difficulty on construction.  The agent
    advances when its success rate meets or exceeds the threshold.

    Args:
        scenarios: List of stress scenarios (sorted internally by difficulty).
        threshold: Minimum success rate required to advance.
    """

    def __init__(
        self,
        scenarios: list[StressScenario],
        threshold: float = 0.8,
    ) -> None:
        self._scenarios = sorted(scenarios, key=lambda s: s.difficulty)
        self._threshold = threshold
        self._current_idx = 0

    def current_scenario(self) -> StressScenario:
        """Return the active training scenario."""
        return self._scenarios[self._current_idx]

    def should_advance(self, success_rate: float) -> bool:
        """Return whether the agent should move to the next difficulty level."""
        return success_rate >= self._threshold and self._current_idx < len(self._scenarios) - 1

    def advance(self) -> StressScenario | None:
        """Move to the next scenario if available, returning it (or ``None``)."""
        if self._current_idx < len(self._scenarios) - 1:
            self._current_idx += 1
            return self._scenarios[self._current_idx]
        return None

    @property
    def is_complete(self) -> bool:
        """Whether the curriculum has reached its final scenario."""
        return self._current_idx >= len(self._scenarios) - 1

    @property
    def progress(self) -> float:
        """Fraction of the curriculum completed (0.0 to 1.0)."""
        return (self._current_idx + 1) / len(self._scenarios)
