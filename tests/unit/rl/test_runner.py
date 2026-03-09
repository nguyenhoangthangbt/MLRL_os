"""Tests for mlrl_os.rl.runner — RL experiment orchestration.

All environment interactions and algorithm training are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mlrl_os.config.rl_schemas import ResolvedRLConfig
from mlrl_os.core.types import ExperimentStatus, ProblemType
from mlrl_os.rl.runner import RLExperimentResult, RLExperimentRunner


def _make_mock_env(
    obs_dim: int = 4,
    act_dim: int = 2,
    episode_length: int = 3,
) -> AsyncMock:
    """Create a mock SimOSEnvironment that runs short episodes."""
    mock_env = AsyncMock()
    mock_env.obs_dim = obs_dim
    mock_env.act_dim = act_dim

    obs = np.zeros(obs_dim, dtype=np.float32)
    mock_env.reset.return_value = obs

    step_count = 0

    async def mock_step(action: int) -> tuple:
        nonlocal step_count
        step_count += 1
        next_obs = np.ones(obs_dim, dtype=np.float32) * step_count
        reward = 1.0
        done = step_count >= episode_length
        if done:
            step_count = 0
        info = {"step": step_count}
        return next_obs, reward, done, info

    mock_env.step.side_effect = mock_step
    mock_env.close.return_value = None
    return mock_env


def _make_mock_algo() -> MagicMock:
    """Create a mock RL algorithm."""
    mock_algo = MagicMock()
    mock_algo.name = "dqn"
    mock_algo.select_action.return_value = 0
    mock_algo.train_step.return_value = {"loss": 0.1, "epsilon": 0.5}
    mock_algo.save.return_value = None
    return mock_algo


class TestRLExperimentResult:
    """Tests for the RLExperimentResult data container."""

    def test_empty_result(self) -> None:
        result = RLExperimentResult()
        assert result.episode_rewards == []
        assert result.episode_lengths == []
        assert result.eval_rewards == []
        assert result.training_losses == []
        assert result.curriculum_progress == []

    def test_accumulate_results(self) -> None:
        result = RLExperimentResult()
        result.episode_rewards.append(10.0)
        result.episode_lengths.append(100)
        result.training_losses.append({"loss": 0.5})
        assert len(result.episode_rewards) == 1
        assert result.episode_lengths[0] == 100


class TestRLExperimentRunnerBuildEnvironment:
    """Tests for _build_environment with various reward configs."""

    @pytest.mark.asyncio()
    async def test_build_environment_throughput(self, tmp_path: Path) -> None:
        runner = RLExperimentRunner(artifacts_dir=tmp_path / "models")
        config = ResolvedRLConfig(name="test")

        with patch("mlrl_os.rl.environment.SimOSEnvironment") as MockEnv:
            MockEnv.return_value = _make_mock_env()
            env = await runner._build_environment(config)

        MockEnv.assert_called_once()
        call_kwargs = MockEnv.call_args.kwargs
        assert call_kwargs["template"] == "healthcare_er"

    @pytest.mark.asyncio()
    async def test_build_environment_composite_reward(self, tmp_path: Path) -> None:
        from mlrl_os.config.rl_schemas import ResolvedRewardConfig

        runner = RLExperimentRunner(artifacts_dir=tmp_path / "models")
        config = ResolvedRLConfig(
            name="composite-test",
            reward=ResolvedRewardConfig(
                function="composite",
                components=[
                    {"name": "throughput", "weight": 1.0},
                    {"name": "cost_minimization", "weight": 0.5},
                ],
            ),
        )

        with patch("mlrl_os.rl.environment.SimOSEnvironment") as MockEnv:
            MockEnv.return_value = _make_mock_env()
            env = await runner._build_environment(config)

        call_kwargs = MockEnv.call_args.kwargs
        from mlrl_os.rl.rewards import CompositeReward
        assert isinstance(call_kwargs["reward_fn"], CompositeReward)


class TestRLExperimentRunnerBuildAlgorithm:
    """Tests for _build_algorithm."""

    def test_build_dqn(self, tmp_path: Path) -> None:
        runner = RLExperimentRunner(artifacts_dir=tmp_path / "models")
        config = ResolvedRLConfig(name="test", algorithm="dqn")

        algo = runner._build_algorithm(config, obs_dim=4, act_dim=2)
        assert algo.name == "dqn"

    def test_build_ppo(self, tmp_path: Path) -> None:
        runner = RLExperimentRunner(artifacts_dir=tmp_path / "models")
        config = ResolvedRLConfig(name="test", algorithm="ppo")

        algo = runner._build_algorithm(config, obs_dim=4, act_dim=2)
        assert algo.name == "ppo"

    def test_build_unknown_raises(self, tmp_path: Path) -> None:
        runner = RLExperimentRunner(artifacts_dir=tmp_path / "models")
        config = ResolvedRLConfig(name="test", algorithm="unknown_algo")

        with pytest.raises(KeyError, match="unknown_algo"):
            runner._build_algorithm(config, obs_dim=4, act_dim=2)


class TestRLExperimentRunnerTrain:
    """Tests for the training loop."""

    @pytest.mark.asyncio()
    async def test_train_collects_episode_rewards(self, tmp_path: Path) -> None:
        runner = RLExperimentRunner(artifacts_dir=tmp_path / "models")
        config = ResolvedRLConfig(
            name="test",
            max_episodes=3,
            batch_size=2,
            replay_buffer_size=100,
        )
        env = _make_mock_env(obs_dim=4, episode_length=3)
        algo = _make_mock_algo()

        result = await runner._train(config, env, algo)

        assert len(result.episode_rewards) == 3
        assert all(isinstance(r, float) for r in result.episode_rewards)
        assert len(result.episode_lengths) == 3

    @pytest.mark.asyncio()
    async def test_train_calls_algo_train_step(self, tmp_path: Path) -> None:
        runner = RLExperimentRunner(artifacts_dir=tmp_path / "models")
        config = ResolvedRLConfig(
            name="test",
            max_episodes=2,
            batch_size=2,
            replay_buffer_size=100,
        )
        env = _make_mock_env(obs_dim=4, episode_length=5)
        algo = _make_mock_algo()

        await runner._train(config, env, algo)

        # After enough steps, train_step should have been called
        assert algo.train_step.call_count > 0


class TestRLExperimentRunnerEvaluate:
    """Tests for the evaluation loop."""

    @pytest.mark.asyncio()
    async def test_evaluate_returns_metrics(self, tmp_path: Path) -> None:
        runner = RLExperimentRunner(artifacts_dir=tmp_path / "models")
        config = ResolvedRLConfig(name="test", eval_episodes=5)
        env = _make_mock_env(obs_dim=4, episode_length=3)
        algo = _make_mock_algo()

        metrics = await runner._evaluate(config, env, algo)

        assert "mean_reward" in metrics
        assert "std_reward" in metrics
        assert "min_reward" in metrics
        assert "max_reward" in metrics
        assert "success_rate" in metrics
        assert isinstance(metrics["mean_reward"], float)


class TestRLExperimentRunnerRun:
    """Tests for the full run pipeline."""

    @pytest.mark.asyncio()
    async def test_run_returns_completed_result(self, tmp_path: Path) -> None:
        runner = RLExperimentRunner(artifacts_dir=tmp_path / "models")
        config = {
            "name": "full-test",
            "algorithm": "dqn",
            "max_episodes": 2,
            "batch_size": 2,
            "replay_buffer_size": 100,
            "eval_episodes": 2,
            "max_steps_per_episode": 50,
        }

        mock_env = _make_mock_env(obs_dim=4, episode_length=3)
        mock_algo = _make_mock_algo()

        with (
            patch.object(runner, "_build_environment", return_value=mock_env),
            patch.object(runner, "_build_algorithm", return_value=mock_algo),
        ):
            result = await runner.run(config)

        assert result.status == ExperimentStatus.COMPLETED
        assert result.experiment_type == ProblemType.REINFORCEMENT_LEARNING
        assert result.name == "full-test"
        assert result.best_algorithm == "dqn"
        assert result.metrics is not None
        assert result.duration_seconds is not None
        assert result.duration_seconds > 0

    @pytest.mark.asyncio()
    async def test_run_returns_failed_on_error(self, tmp_path: Path) -> None:
        runner = RLExperimentRunner(artifacts_dir=tmp_path / "models")
        config = {"name": "fail-test", "algorithm": "dqn"}

        with patch.object(
            runner,
            "_build_environment",
            side_effect=RuntimeError("Connection refused"),
        ):
            result = await runner.run(config)

        assert result.status == ExperimentStatus.FAILED
        assert "Connection refused" in (result.error_message or "")

    @pytest.mark.asyncio()
    async def test_run_saves_policy_artifact(self, tmp_path: Path) -> None:
        runner = RLExperimentRunner(artifacts_dir=tmp_path / "models")
        config = {
            "name": "save-test",
            "max_episodes": 1,
            "batch_size": 2,
            "replay_buffer_size": 100,
            "eval_episodes": 1,
        }

        mock_env = _make_mock_env(obs_dim=4, episode_length=3)
        mock_algo = _make_mock_algo()

        with (
            patch.object(runner, "_build_environment", return_value=mock_env),
            patch.object(runner, "_build_algorithm", return_value=mock_algo),
        ):
            result = await runner.run(config)

        assert result.status == ExperimentStatus.COMPLETED
        mock_algo.save.assert_called_once()

    @pytest.mark.asyncio()
    async def test_run_creates_artifacts_dir(self, tmp_path: Path) -> None:
        artifacts = tmp_path / "nested" / "models"
        runner = RLExperimentRunner(artifacts_dir=artifacts)
        assert artifacts.exists()
