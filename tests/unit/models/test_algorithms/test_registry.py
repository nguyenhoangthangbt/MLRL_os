"""Tests for mlrl_os.models.algorithms.registry."""

from __future__ import annotations

import pytest

from mlrl_os.models.algorithms.registry import (
    KNOWN_ALGORITHMS,
    AlgorithmRegistry,
    default_registry,
)


class _DummyAlgorithm:
    """Minimal algorithm stub for registry tests."""

    @property
    def name(self) -> str:
        return "dummy"


class _DummyAlgorithm2:
    """Second stub to test duplicate registration."""

    @property
    def name(self) -> str:
        return "dummy2"


class TestAlgorithmRegistry:
    """Tests for AlgorithmRegistry."""

    def test_register_adds_algorithm(self) -> None:
        """register() makes the algorithm available in the registry."""
        reg = AlgorithmRegistry()
        reg.register("dummy", _DummyAlgorithm)

        assert reg.has("dummy")

    def test_get_returns_instance(self) -> None:
        """get() returns a fresh instance of the registered class."""
        reg = AlgorithmRegistry()
        reg.register("dummy", _DummyAlgorithm)

        algo = reg.get("dummy")
        assert isinstance(algo, _DummyAlgorithm)

    def test_get_unknown_raises_key_error(self) -> None:
        """get() raises KeyError for an unregistered algorithm name."""
        reg = AlgorithmRegistry()

        with pytest.raises(KeyError, match="Unknown algorithm"):
            reg.get("nonexistent")

    def test_has_returns_true_for_registered(self) -> None:
        """has() returns True for a registered algorithm."""
        reg = AlgorithmRegistry()
        reg.register("dummy", _DummyAlgorithm)

        assert reg.has("dummy") is True

    def test_has_returns_false_for_unregistered(self) -> None:
        """has() returns False for an unregistered name."""
        reg = AlgorithmRegistry()

        assert reg.has("nonexistent") is False

    def test_list_available_returns_sorted_names(self) -> None:
        """list_available() returns names in sorted order."""
        reg = AlgorithmRegistry()
        reg.register("zeta", _DummyAlgorithm)
        reg.register("alpha", _DummyAlgorithm2)

        result = reg.list_available()
        assert result == ["alpha", "zeta"]

    def test_register_duplicate_raises_value_error(self) -> None:
        """register() raises ValueError when name is already taken."""
        reg = AlgorithmRegistry()
        reg.register("dummy", _DummyAlgorithm)

        with pytest.raises(ValueError, match="already registered"):
            reg.register("dummy", _DummyAlgorithm2)


class TestDefaultRegistry:
    """Tests for default_registry() factory function."""

    def test_default_registry_has_all_known_algorithms(self) -> None:
        """default_registry() returns a registry with all 5 built-in algorithms."""
        reg = default_registry()

        available = reg.list_available()
        assert len(available) == 5

        for algo_name in KNOWN_ALGORITHMS:
            assert reg.has(algo_name), f"Missing algorithm: {algo_name}"
