"""RL experiment runner — orchestrates the full RL training pipeline.

Pipeline: config -> validate -> environment -> train -> evaluate -> store.

Mirrors the supervised ``ExperimentRunner`` pattern but adapted for
the RL setting: no dataset, no cross-validation. Instead it creates
a SimOS environment, trains an agent, and evaluates its policy.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mlrl_os.config.rl_schemas import ResolvedRLConfig
from mlrl_os.core.experiment import ExperimentResult
from mlrl_os.core.types import ExperimentStatus, ProblemType

logger = logging.getLogger(__name__)


class RLExperimentResult:
    """Accumulated metrics from an RL training run.

    Not a Pydantic model because it is internal bookkeeping, not
    a serialised artifact.
    """

    def __init__(self) -> None:
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.eval_rewards: list[float] = []
        self.training_losses: list[dict[str, float]] = []
        self.curriculum_progress: list[int] = []


class RLExperimentRunner:
    """Orchestrates RL training: config -> environment -> train -> evaluate -> store.

    Args:
        artifacts_dir: Directory for saving trained policy checkpoints.
    """

    def __init__(self, artifacts_dir: Path | str = "./models") -> None:
        self._artifacts_dir = Path(artifacts_dir)
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, config: dict[str, Any]) -> ExperimentResult:
        """Run a full RL experiment from a configuration dictionary.

        Args:
            config: Raw configuration dict (validated into ``ResolvedRLConfig``).

        Returns:
            An ``ExperimentResult`` with COMPLETED or FAILED status.
        """
        start_time = time.time()
        resolved = ResolvedRLConfig.model_validate(config)

        exp_id = hashlib.sha256(
            f"{resolved.name}:{datetime.now(UTC).isoformat()}".encode()
        ).hexdigest()[:12]

        try:
            env = await self._build_environment(resolved)
            algo = self._build_algorithm(resolved, env.obs_dim, env.act_dim)

            await self._train(resolved, env, algo)
            eval_metrics = await self._evaluate(resolved, env, algo)

            policy_path = self._artifacts_dir / f"rl_{exp_id}"
            policy_path.mkdir(parents=True, exist_ok=True)
            algo.save(policy_path / "policy.pt")

            await env.close()

            duration = time.time() - start_time

            return ExperimentResult(
                experiment_id=exp_id,
                name=resolved.name,
                status=ExperimentStatus.COMPLETED,
                experiment_type=ProblemType.REINFORCEMENT_LEARNING,
                created_at=datetime.now(UTC).isoformat(),
                completed_at=datetime.now(UTC).isoformat(),
                duration_seconds=duration,
                best_algorithm=resolved.algorithm,
                metrics=eval_metrics,
                resolved_config=resolved.model_dump(),
            )
        except Exception as exc:
            logger.error("RL experiment failed: %s", exc)
            return ExperimentResult(
                experiment_id=exp_id,
                name=resolved.name,
                status=ExperimentStatus.FAILED,
                experiment_type=ProblemType.REINFORCEMENT_LEARNING,
                created_at=datetime.now(UTC).isoformat(),
                error_message=str(exc),
            )

    async def _build_environment(self, config: ResolvedRLConfig) -> Any:
        """Construct a SimOSEnvironment with the appropriate reward function.

        Returns:
            A configured ``SimOSEnvironment`` instance.
        """
        from mlrl_os.rl.environment import SimOSEnvironment
        from mlrl_os.rl.rewards import (
            CompositeReward,
            CostMinimizationReward,
            SLAComplianceReward,
            ThroughputReward,
        )

        reward_map: dict[str, type] = {
            "throughput": ThroughputReward,
            "sla_compliance": SLAComplianceReward,
            "cost_minimization": CostMinimizationReward,
        }

        if config.reward.function == "composite":
            components = []
            for comp in config.reward.components:
                cls = reward_map.get(comp["name"], ThroughputReward)
                components.append((cls(), comp.get("weight", 1.0)))
            reward_fn = CompositeReward(components)
        else:
            reward_cls = reward_map.get(config.reward.function, ThroughputReward)
            reward_fn = reward_cls()

        return SimOSEnvironment(
            simos_url=config.simos_url,
            template=config.template,
            reward_fn=reward_fn,
            seed=config.seed,
            api_key=config.api_key,
            max_steps_per_episode=config.max_steps_per_episode,
        )

    def _build_algorithm(
        self, config: ResolvedRLConfig, obs_dim: int, act_dim: int
    ) -> Any:
        """Instantiate an RL algorithm from the registry.

        Args:
            config: Resolved RL configuration.
            obs_dim: Observation vector dimensionality.
            act_dim: Number of discrete actions.

        Returns:
            An instantiated RL algorithm.

        Raises:
            KeyError: If the algorithm name is not registered.
        """
        from mlrl_os.rl.algorithms.registry import default_rl_registry

        registry = default_rl_registry()
        return registry.get(
            config.algorithm,
            obs_dim=obs_dim,
            act_dim=act_dim,
            seed=config.seed,
            lr=config.learning_rate,
            gamma=config.gamma,
            eps_start=config.eps_start,
            eps_end=config.eps_end,
            eps_decay=config.eps_decay,
            clip_eps=config.clip_eps,
            entropy_coef=config.entropy_coef,
            value_coef=config.value_coef,
        )

    async def _train(
        self, config: ResolvedRLConfig, env: Any, algo: Any
    ) -> RLExperimentResult:
        """Run the training loop for the configured number of episodes.

        Args:
            config: Resolved RL configuration.
            env: The RL environment.
            algo: The RL algorithm.

        Returns:
            Accumulated training metrics.
        """
        from mlrl_os.rl.replay_buffer import ReplayBuffer

        result = RLExperimentResult()
        buffer = ReplayBuffer(capacity=config.replay_buffer_size, seed=config.seed)

        for episode in range(config.max_episodes):
            obs = await env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                action = algo.select_action(obs, explore=True)
                next_obs, reward, done, _info = await env.step(action)

                buffer.push(obs, action, reward, next_obs, done)

                if len(buffer) >= config.batch_size:
                    batch = buffer.sample(config.batch_size)
                    losses = algo.train_step(batch)
                    result.training_losses.append(losses)

                obs = next_obs
                episode_reward += reward
                episode_length += 1

            result.episode_rewards.append(episode_reward)
            result.episode_lengths.append(episode_length)

            if (episode + 1) % 10 == 0:
                logger.info(
                    "Episode %d/%d reward=%.2f",
                    episode + 1,
                    config.max_episodes,
                    episode_reward,
                )

        return result

    async def _evaluate(
        self, config: ResolvedRLConfig, env: Any, algo: Any
    ) -> dict[str, float]:
        """Evaluate the trained policy over multiple episodes.

        Args:
            config: Resolved RL configuration.
            env: The RL environment.
            algo: The RL algorithm (used in greedy mode).

        Returns:
            Dictionary of evaluation metrics.
        """
        rewards: list[float] = []
        successes = 0

        for _ in range(config.eval_episodes):
            obs = await env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                action = algo.select_action(obs, explore=False)
                obs, reward, done, _info = await env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)
            if episode_reward > 0:
                successes += 1

        rewards_arr = np.array(rewards)
        return {
            "mean_reward": float(np.mean(rewards_arr)),
            "std_reward": float(np.std(rewards_arr)),
            "min_reward": float(np.min(rewards_arr)),
            "max_reward": float(np.max(rewards_arr)),
            "success_rate": successes / max(len(rewards), 1),
        }
