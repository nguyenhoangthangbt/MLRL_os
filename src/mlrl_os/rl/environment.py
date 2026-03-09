"""Pure Python RL environment backed by SimOS WebSocket stepping.

No Gymnasium dependency. The environment communicates with SimOS via
:class:`SimOSWebSocketClient` to create simulations, step through
events, and extract observation vectors for RL training.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mlrl_os.experiment.seed import seed_hash
from mlrl_os.rl.rewards import RewardFunction, ThroughputReward
from mlrl_os.rl.simos_client import SimOSWebSocketClient
from mlrl_os.rl.spaces import ActionSpec, ObservationSpec, build_specs_from_template

logger = logging.getLogger(__name__)


class SimOSEnvironment:
    """Pure Python RL environment backed by SimOS WebSocket stepping.

    Each ``reset()`` creates a fresh SimOS simulation from the given
    template. Each ``step()`` advances the simulation by one event and
    returns ``(observation, reward, done, info)``.

    Args:
        simos_url: SimOS WebSocket base URL.
        template: SimOS template name for simulation creation.
        reward_fn: Reward function to compute scalar rewards.
        seed: Global seed for deterministic behaviour.
        api_key: SimOS API key.
        max_steps_per_episode: Maximum steps before forced termination.
    """

    def __init__(
        self,
        simos_url: str = "ws://localhost:8000",
        template: str = "healthcare_er",
        reward_fn: RewardFunction | None = None,
        seed: int = 42,
        api_key: str = "sk-premium-test-003",
        max_steps_per_episode: int = 10_000,
    ) -> None:
        self._simos_url = simos_url
        self._template = template
        self._reward_fn: RewardFunction = reward_fn or ThroughputReward()
        self._seed = seed_hash("environment", seed)
        self._rng = np.random.default_rng(self._seed)
        self._api_key = api_key
        self._max_steps = max_steps_per_episode
        self._client = SimOSWebSocketClient(simos_url, api_key)
        self._step_count: int = 0
        self._prev_state: dict[str, Any] = {}
        self._obs_spec: ObservationSpec | None = None
        self._act_spec: ActionSpec | None = None
        self._job_id: str | None = None

    # -- Default fallback dimensions (before specs are built) -----

    _DEFAULT_OBS_DIM: int = 24
    _DEFAULT_ACT_DIM: int = 2

    @property
    def obs_dim(self) -> int:
        """Observation vector dimensionality."""
        if self._obs_spec is not None:
            return self._obs_spec.dim
        return self._DEFAULT_OBS_DIM

    @property
    def act_dim(self) -> int:
        """Number of discrete actions."""
        if self._act_spec is not None and self._act_spec.n is not None:
            return self._act_spec.n
        return self._DEFAULT_ACT_DIM

    # -- Core RL interface ----------------------------------------

    async def reset(self) -> np.ndarray:
        """Start a new simulation episode and return the initial observation.

        Closes any previous connection, creates a new simulation via
        the REST API, connects the WebSocket, pauses, and takes one
        initial step to obtain the starting state.

        Returns:
            Observation vector as a float32 numpy array.
        """
        await self._client.close()

        self._job_id = await self._client.create_simulation(self._template)
        await self._client.connect(self._job_id)
        await self._client.pause()

        state_msg = await self._client.step(count=1)
        self._prev_state = self._extract_state(state_msg)
        self._step_count = 0

        if hasattr(self._client, "_topology"):
            metadata = self._client._topology.get("metadata", {})
            self._obs_spec, self._act_spec = build_specs_from_template(metadata)

        return self._state_to_obs(self._prev_state)

    async def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Execute an action, step the simulation, and return the transition.

        Args:
            action: Integer action index.

        Returns:
            Tuple of ``(observation, reward, done, info)``.
        """
        self._step_count += 1

        state_msg = await self._client.step(count=1)

        done = (
            state_msg.get("type") == "simulation_end"
            or self._step_count >= self._max_steps
        )

        curr_state = self._extract_state(state_msg)

        reward = self._reward_fn.compute(
            self._prev_state, action, curr_state, done
        )

        obs = self._state_to_obs(curr_state)

        info: dict[str, Any] = {
            "step": self._step_count,
            "raw_state": curr_state,
        }

        self._prev_state = curr_state
        return obs, reward, done, info

    async def close(self) -> None:
        """Close the WebSocket connection to SimOS."""
        await self._client.close()

    # -- Internal helpers -----------------------------------------

    def _extract_state(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Extract a flat state dictionary from a SimOS message.

        Checks ``snapshot``, ``state``, and ``data`` keys in priority order.
        """
        if "snapshot" in msg:
            return msg["snapshot"]
        if "state" in msg:
            return msg["state"]
        return msg.get("data", msg)

    def _state_to_obs(self, state: dict[str, Any]) -> np.ndarray:
        """Convert a state dictionary to a fixed-size numpy observation vector.

        When an ``ObservationSpec`` is available, uses the spec's feature
        names to extract values in order. Otherwise falls back to taking
        the first N numeric values and zero-padding to 24 dimensions.
        """
        if self._obs_spec is not None:
            return np.array(
                [state.get(name, 0.0) for name in self._obs_spec.names],
                dtype=np.float32,
            )

        values = [
            float(v) for v in state.values() if isinstance(v, (int, float))
        ]
        target_dim = self._DEFAULT_OBS_DIM
        if len(values) > target_dim:
            values = values[:target_dim]
        else:
            values = values + [0.0] * (target_dim - len(values))
        return np.array(values, dtype=np.float32)
