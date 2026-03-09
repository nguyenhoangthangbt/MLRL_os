"""Tests for mlrl_os.rl.environment — SimOS RL environment.

All SimOS interactions are mocked. No real SimOS or WebSocket is required.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mlrl_os.rl.environment import SimOSEnvironment
from mlrl_os.rl.rewards import ThroughputReward


def _make_mock_client(
    topology: dict | None = None,
    step_responses: list[dict] | None = None,
) -> AsyncMock:
    """Create a mock SimOSWebSocketClient with configurable responses."""
    mock = AsyncMock()
    mock._topology = topology or {"metadata": {"node_names": ["triage", "treatment"]}}
    mock.create_simulation.return_value = "job-test-123"
    mock.connect.return_value = None
    mock.pause.return_value = None
    mock.close.return_value = None

    if step_responses:
        mock.step.side_effect = step_responses
    else:
        mock.step.return_value = {
            "type": "snapshot",
            "snapshot": {"completed_count": 1, "throughput": 5.0},
        }

    return mock


class TestEnvironmentInit:
    """Tests for environment initialization."""

    def test_default_properties(self) -> None:
        env = SimOSEnvironment()
        assert env.obs_dim == 24  # default before specs are built
        assert env.act_dim == 2  # default before specs are built
        assert env._max_steps == 10_000
        assert env._template == "healthcare_er"

    def test_custom_properties(self) -> None:
        env = SimOSEnvironment(
            template="logistics_otd",
            seed=99,
            max_steps_per_episode=500,
        )
        assert env._template == "logistics_otd"
        assert env._max_steps == 500


class TestEnvironmentReset:
    """Tests for reset — starts new simulation and returns initial obs."""

    @pytest.mark.asyncio()
    async def test_reset_returns_numpy_array(self) -> None:
        mock_client = _make_mock_client()

        with patch("mlrl_os.rl.environment.SimOSWebSocketClient", return_value=mock_client):
            env = SimOSEnvironment()
            env._client = mock_client
            obs = await env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

    @pytest.mark.asyncio()
    async def test_reset_calls_client_lifecycle(self) -> None:
        mock_client = _make_mock_client()

        with patch("mlrl_os.rl.environment.SimOSWebSocketClient", return_value=mock_client):
            env = SimOSEnvironment()
            env._client = mock_client
            await env.reset()

        mock_client.close.assert_called_once()
        mock_client.create_simulation.assert_called_once_with("healthcare_er")
        mock_client.connect.assert_called_once_with("job-test-123")
        mock_client.pause.assert_called_once()
        mock_client.step.assert_called_once_with(count=1)

    @pytest.mark.asyncio()
    async def test_reset_resets_step_count(self) -> None:
        mock_client = _make_mock_client()

        with patch("mlrl_os.rl.environment.SimOSWebSocketClient", return_value=mock_client):
            env = SimOSEnvironment()
            env._client = mock_client
            env._step_count = 100
            await env.reset()

        assert env._step_count == 0

    @pytest.mark.asyncio()
    async def test_reset_builds_specs_from_topology(self) -> None:
        topology = {"metadata": {"node_names": ["a", "b", "c"]}}
        mock_client = _make_mock_client(topology=topology)

        with patch("mlrl_os.rl.environment.SimOSWebSocketClient", return_value=mock_client):
            env = SimOSEnvironment()
            env._client = mock_client
            await env.reset()

        assert env._obs_spec is not None
        assert env._act_spec is not None
        assert env.obs_dim == 24  # entity(10) + node(9) + system(5)
        assert env.act_dim == 3  # 3 nodes


class TestEnvironmentStep:
    """Tests for step — execute action and return (obs, reward, done, info)."""

    @pytest.mark.asyncio()
    async def test_step_returns_tuple(self) -> None:
        snapshot_1 = {"type": "snapshot", "snapshot": {"completed_count": 0}}
        snapshot_2 = {"type": "snapshot", "snapshot": {"completed_count": 1, "throughput": 5.0}}

        mock_client = _make_mock_client(step_responses=[snapshot_1, snapshot_2])

        with patch("mlrl_os.rl.environment.SimOSWebSocketClient", return_value=mock_client):
            env = SimOSEnvironment(reward_fn=ThroughputReward())
            env._client = mock_client
            await env.reset()

            obs, reward, done, info = await env.step(action=0)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    @pytest.mark.asyncio()
    async def test_step_increments_count(self) -> None:
        snapshot_1 = {"type": "snapshot", "snapshot": {"completed_count": 0}}
        snapshot_2 = {"type": "snapshot", "snapshot": {"completed_count": 1}}
        snapshot_3 = {"type": "snapshot", "snapshot": {"completed_count": 2}}

        mock_client = _make_mock_client(
            step_responses=[snapshot_1, snapshot_2, snapshot_3]
        )

        with patch("mlrl_os.rl.environment.SimOSWebSocketClient", return_value=mock_client):
            env = SimOSEnvironment(reward_fn=ThroughputReward())
            env._client = mock_client
            await env.reset()
            await env.step(action=0)
            await env.step(action=1)

        assert env._step_count == 2

    @pytest.mark.asyncio()
    async def test_step_computes_reward(self) -> None:
        snapshot_init = {"type": "snapshot", "snapshot": {"completed_count": 5}}
        snapshot_next = {"type": "snapshot", "snapshot": {"completed_count": 8}}

        mock_client = _make_mock_client(
            step_responses=[snapshot_init, snapshot_next]
        )

        with patch("mlrl_os.rl.environment.SimOSWebSocketClient", return_value=mock_client):
            env = SimOSEnvironment(reward_fn=ThroughputReward())
            env._client = mock_client
            await env.reset()
            _, reward, _, _ = await env.step(action=0)

        assert reward == 3.0  # 8 - 5

    @pytest.mark.asyncio()
    async def test_step_done_on_simulation_end(self) -> None:
        snapshot_init = {"type": "snapshot", "snapshot": {"completed_count": 0}}
        end_msg = {"type": "simulation_end"}

        mock_client = _make_mock_client(
            step_responses=[snapshot_init, end_msg]
        )

        with patch("mlrl_os.rl.environment.SimOSWebSocketClient", return_value=mock_client):
            env = SimOSEnvironment(reward_fn=ThroughputReward())
            env._client = mock_client
            await env.reset()
            _, _, done, _ = await env.step(action=0)

        assert done is True

    @pytest.mark.asyncio()
    async def test_step_done_on_max_steps(self) -> None:
        snapshot_init = {"type": "snapshot", "snapshot": {"completed_count": 0}}
        snapshot_next = {"type": "snapshot", "snapshot": {"completed_count": 1}}

        mock_client = _make_mock_client(
            step_responses=[snapshot_init, snapshot_next]
        )

        with patch("mlrl_os.rl.environment.SimOSWebSocketClient", return_value=mock_client):
            env = SimOSEnvironment(
                reward_fn=ThroughputReward(),
                max_steps_per_episode=1,
            )
            env._client = mock_client
            await env.reset()
            _, _, done, _ = await env.step(action=0)

        assert done is True

    @pytest.mark.asyncio()
    async def test_step_info_contains_step_count(self) -> None:
        snapshot_init = {"type": "snapshot", "snapshot": {"completed_count": 0}}
        snapshot_next = {"type": "snapshot", "snapshot": {"completed_count": 1}}

        mock_client = _make_mock_client(
            step_responses=[snapshot_init, snapshot_next]
        )

        with patch("mlrl_os.rl.environment.SimOSWebSocketClient", return_value=mock_client):
            env = SimOSEnvironment(reward_fn=ThroughputReward())
            env._client = mock_client
            await env.reset()
            _, _, _, info = await env.step(action=0)

        assert info["step"] == 1


class TestEnvironmentClose:
    """Tests for closing the environment."""

    @pytest.mark.asyncio()
    async def test_close_delegates_to_client(self) -> None:
        mock_client = _make_mock_client()

        with patch("mlrl_os.rl.environment.SimOSWebSocketClient", return_value=mock_client):
            env = SimOSEnvironment()
            env._client = mock_client
            await env.close()

        mock_client.close.assert_called()


class TestStateExtraction:
    """Tests for _extract_state and _state_to_obs."""

    def test_extract_state_from_snapshot(self) -> None:
        env = SimOSEnvironment()
        msg = {"snapshot": {"throughput": 5, "utilization": 0.8}}
        state = env._extract_state(msg)
        assert state["throughput"] == 5

    def test_extract_state_from_state_key(self) -> None:
        env = SimOSEnvironment()
        msg = {"state": {"throughput": 10}}
        state = env._extract_state(msg)
        assert state["throughput"] == 10

    def test_extract_state_fallback_to_data(self) -> None:
        env = SimOSEnvironment()
        msg = {"data": {"value": 42}}
        state = env._extract_state(msg)
        assert state["value"] == 42

    def test_state_to_obs_without_spec(self) -> None:
        env = SimOSEnvironment()
        state = {"a": 1.0, "b": 2.0, "c": "text"}
        obs = env._state_to_obs(state)
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert len(obs) == 24  # padded to 24

    def test_state_to_obs_with_spec(self) -> None:
        env = SimOSEnvironment()
        # Build a minimal spec
        from mlrl_os.rl.spaces import ObservationSpec
        env._obs_spec = ObservationSpec(
            dim=3,
            names=["a", "b", "c"],
            low=np.zeros(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
        )
        state = {"a": 0.1, "b": 0.5, "c": 0.9}
        obs = env._state_to_obs(state)
        assert len(obs) == 3
        assert obs[0] == pytest.approx(0.1)
        assert obs[2] == pytest.approx(0.9)

    def test_state_to_obs_missing_keys_default_to_zero(self) -> None:
        env = SimOSEnvironment()
        from mlrl_os.rl.spaces import ObservationSpec
        env._obs_spec = ObservationSpec(
            dim=3,
            names=["a", "b", "c"],
            low=np.zeros(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
        )
        state = {"a": 0.5}  # b and c missing
        obs = env._state_to_obs(state)
        assert obs[0] == pytest.approx(0.5)
        assert obs[1] == 0.0
        assert obs[2] == 0.0
