"""Tests for mlrl_os.rl.networks — MLP and ActorCritic neural networks.

All tests are skipped if PyTorch is not installed, since these components
lazy-import torch to avoid hard dependencies.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed", exc_type=ImportError)

from mlrl_os.rl.networks import MLP, ActorCritic  # noqa: E402

# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class TestMLP:
    """Tests for the configurable MLP wrapper."""

    def test_forward_produces_correct_shape(self) -> None:
        net = MLP(input_dim=24, output_dim=4, hidden_dims=[32, 32])
        x = torch.randn(8, 24)
        out = net(x)
        assert out.shape == (8, 4)

    def test_default_hidden_dims(self) -> None:
        net = MLP(input_dim=10, output_dim=3)
        x = torch.randn(1, 10)
        out = net(x)
        assert out.shape == (1, 3)

    def test_tanh_activation(self) -> None:
        net = MLP(input_dim=5, output_dim=2, hidden_dims=[16], activation="tanh")
        x = torch.randn(4, 5)
        out = net(x)
        assert out.shape == (4, 2)

    def test_state_dict_roundtrip(self) -> None:
        net1 = MLP(input_dim=5, output_dim=2, hidden_dims=[8])
        sd = net1.state_dict()
        net2 = MLP(input_dim=5, output_dim=2, hidden_dims=[8])
        net2.load_state_dict(sd)

        x = torch.randn(1, 5)
        torch.testing.assert_close(net1(x), net2(x))

    def test_parameters_are_iterable(self) -> None:
        net = MLP(input_dim=5, output_dim=2, hidden_dims=[8])
        params = list(net.parameters)
        assert len(params) > 0
        assert all(isinstance(p, torch.nn.Parameter) for p in params)


# ---------------------------------------------------------------------------
# ActorCritic
# ---------------------------------------------------------------------------


class TestActorCritic:
    """Tests for the shared-backbone actor-critic network."""

    def test_forward_produces_distribution_and_value(self) -> None:
        ac = ActorCritic(obs_dim=24, act_dim=4)
        obs = torch.randn(8, 24)
        dist, value = ac(obs)

        assert isinstance(dist, torch.distributions.Categorical)
        assert dist.logits.shape == (8, 4)
        assert value.shape == (8, 1)

    def test_custom_hidden_dims(self) -> None:
        ac = ActorCritic(obs_dim=10, act_dim=3, hidden_dims=[16, 16])
        obs = torch.randn(2, 10)
        dist, value = ac(obs)
        assert dist.logits.shape == (2, 3)
        assert value.shape == (2, 1)

    def test_sample_action_from_distribution(self) -> None:
        ac = ActorCritic(obs_dim=5, act_dim=3)
        obs = torch.randn(1, 5)
        dist, _ = ac(obs)
        action = dist.sample()
        assert action.shape == (1,)
        assert 0 <= action.item() < 3

    def test_state_dict_roundtrip(self) -> None:
        ac1 = ActorCritic(obs_dim=5, act_dim=3, hidden_dims=[8])
        sd = ac1.state_dict()
        ac2 = ActorCritic(obs_dim=5, act_dim=3, hidden_dims=[8])
        ac2.load_state_dict(sd)

        obs = torch.randn(1, 5)
        dist1, val1 = ac1(obs)
        dist2, val2 = ac2(obs)
        torch.testing.assert_close(dist1.logits, dist2.logits)
        torch.testing.assert_close(val1, val2)

    def test_parameters_include_all_components(self) -> None:
        ac = ActorCritic(obs_dim=5, act_dim=3, hidden_dims=[8])
        params = list(ac.parameters)
        # backbone(2) + actor_head(2) + critic_head(2) = 6 param tensors
        assert len(params) == 6

    def test_state_dict_has_expected_keys(self) -> None:
        ac = ActorCritic(obs_dim=5, act_dim=3)
        sd = ac.state_dict()
        assert "backbone" in sd
        assert "actor" in sd
        assert "critic" in sd
