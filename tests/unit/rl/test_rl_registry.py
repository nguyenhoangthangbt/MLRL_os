"""Tests for mlrl_os.rl.algorithms.registry -- RL algorithm registry."""

from __future__ import annotations

import pytest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


# ---------------------------------------------------------------------------
# RLAlgorithmRegistry
# ---------------------------------------------------------------------------


class TestRLAlgorithmRegistry:
    """Tests for RLAlgorithmRegistry class."""

    def test_register_and_list(self) -> None:
        from mlrl_os.rl.algorithms.registry import RLAlgorithmRegistry

        registry = RLAlgorithmRegistry()
        assert registry.list_available() == []

        class FakeAlgo:
            def __init__(self, **kwargs: object) -> None:
                pass

        registry.register("fake", FakeAlgo)
        assert registry.list_available() == ["fake"]
        assert registry.has("fake")

    def test_register_duplicate_raises_value_error(self) -> None:
        from mlrl_os.rl.algorithms.registry import RLAlgorithmRegistry

        registry = RLAlgorithmRegistry()

        class FakeAlgo:
            def __init__(self, **kwargs: object) -> None:
                pass

        registry.register("fake", FakeAlgo)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("fake", FakeAlgo)

    def test_get_unknown_raises_key_error(self) -> None:
        from mlrl_os.rl.algorithms.registry import RLAlgorithmRegistry

        registry = RLAlgorithmRegistry()
        with pytest.raises(KeyError, match="Unknown RL algorithm"):
            registry.get("nonexistent")

    def test_get_passes_kwargs_to_constructor(self) -> None:
        from mlrl_os.rl.algorithms.registry import RLAlgorithmRegistry

        registry = RLAlgorithmRegistry()

        class FakeAlgo:
            def __init__(self, obs_dim: int = 0, act_dim: int = 0, **kwargs: object) -> None:
                self.obs_dim = obs_dim
                self.act_dim = act_dim

        registry.register("fake", FakeAlgo)
        algo = registry.get("fake", obs_dim=10, act_dim=4)
        assert algo.obs_dim == 10  # type: ignore[attr-defined]
        assert algo.act_dim == 4  # type: ignore[attr-defined]

    def test_has_returns_false_for_missing(self) -> None:
        from mlrl_os.rl.algorithms.registry import RLAlgorithmRegistry

        registry = RLAlgorithmRegistry()
        assert not registry.has("missing")

    def test_list_available_is_sorted(self) -> None:
        from mlrl_os.rl.algorithms.registry import RLAlgorithmRegistry

        registry = RLAlgorithmRegistry()

        class A:
            def __init__(self, **kwargs: object) -> None:
                pass

        class B:
            def __init__(self, **kwargs: object) -> None:
                pass

        registry.register("zebra", A)
        registry.register("alpha", B)
        assert registry.list_available() == ["alpha", "zebra"]


# ---------------------------------------------------------------------------
# default_rl_registry
# ---------------------------------------------------------------------------


class TestDefaultRLRegistry:
    """Tests for the default_rl_registry factory function."""

    def test_default_registry_has_dqn_and_ppo(self) -> None:
        from mlrl_os.rl.algorithms.registry import default_rl_registry

        registry = default_rl_registry()
        assert registry.has("dqn")
        assert registry.has("ppo")
        assert sorted(registry.list_available()) == ["dqn", "ppo"]

    def test_default_registry_instantiates_dqn(self) -> None:
        from mlrl_os.rl.algorithms.registry import default_rl_registry

        registry = default_rl_registry()
        dqn = registry.get("dqn", obs_dim=4, act_dim=2)
        assert dqn.name == "dqn"  # type: ignore[attr-defined]

    def test_default_registry_instantiates_ppo(self) -> None:
        from mlrl_os.rl.algorithms.registry import default_rl_registry

        registry = default_rl_registry()
        ppo = registry.get("ppo", obs_dim=4, act_dim=2)
        assert ppo.name == "ppo"  # type: ignore[attr-defined]
