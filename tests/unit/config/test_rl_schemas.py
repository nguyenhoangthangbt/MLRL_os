"""Tests for mlrl_os.config.rl_schemas — RL experiment configuration models."""

from __future__ import annotations

import pytest

from mlrl_os.config.rl_schemas import (
    ResolvedCurriculumConfig,
    ResolvedRewardConfig,
    ResolvedRLConfig,
)


class TestResolvedRewardConfig:
    """Tests for ResolvedRewardConfig defaults and validation."""

    def test_defaults(self) -> None:
        cfg = ResolvedRewardConfig()
        assert cfg.function == "throughput"
        assert cfg.components == []

    def test_custom_function(self) -> None:
        cfg = ResolvedRewardConfig(function="sla_compliance")
        assert cfg.function == "sla_compliance"

    def test_composite_components(self) -> None:
        components = [
            {"name": "throughput", "weight": 1.0},
            {"name": "cost_minimization", "weight": 0.5},
        ]
        cfg = ResolvedRewardConfig(function="composite", components=components)
        assert cfg.function == "composite"
        assert len(cfg.components) == 2
        assert cfg.components[0]["name"] == "throughput"


class TestResolvedCurriculumConfig:
    """Tests for ResolvedCurriculumConfig defaults and validation."""

    def test_defaults(self) -> None:
        cfg = ResolvedCurriculumConfig()
        assert cfg.enabled is False
        assert cfg.threshold == 0.8

    def test_enabled_with_custom_threshold(self) -> None:
        cfg = ResolvedCurriculumConfig(enabled=True, threshold=0.9)
        assert cfg.enabled is True
        assert cfg.threshold == 0.9


class TestResolvedRLConfig:
    """Tests for ResolvedRLConfig full configuration."""

    def test_minimal_config(self) -> None:
        cfg = ResolvedRLConfig(name="test-rl-exp")
        assert cfg.name == "test-rl-exp"
        assert cfg.experiment_type == "reinforcement_learning"
        assert cfg.seed == 42
        assert cfg.algorithm == "dqn"

    def test_all_defaults_populated(self) -> None:
        cfg = ResolvedRLConfig(name="full-defaults")
        assert cfg.simos_url == "ws://localhost:8000"
        assert cfg.template == "healthcare_er"
        assert cfg.api_key == "sk-premium-test-003"
        assert cfg.max_steps_per_episode == 10_000
        assert cfg.max_episodes == 1000
        assert cfg.batch_size == 64
        assert cfg.learning_rate == 0.0003
        assert cfg.gamma == 0.99
        # DQN defaults
        assert cfg.eps_start == 1.0
        assert cfg.eps_end == 0.01
        assert cfg.eps_decay == 10_000
        assert cfg.target_update_freq == 100
        assert cfg.replay_buffer_size == 10_000
        # PPO defaults
        assert cfg.clip_eps == 0.2
        assert cfg.entropy_coef == 0.01
        assert cfg.value_coef == 0.5
        assert cfg.gae_lambda == 0.95
        # Reward
        assert cfg.reward.function == "throughput"
        # Curriculum
        assert cfg.curriculum.enabled is False
        # Evaluation
        assert cfg.eval_episodes == 50
        assert cfg.eval_metrics == ["mean_reward", "success_rate"]

    def test_custom_algorithm(self) -> None:
        cfg = ResolvedRLConfig(name="ppo-exp", algorithm="ppo")
        assert cfg.algorithm == "ppo"

    def test_custom_reward(self) -> None:
        reward = ResolvedRewardConfig(function="composite", components=[
            {"name": "throughput", "weight": 1.0},
        ])
        cfg = ResolvedRLConfig(name="composite-reward", reward=reward)
        assert cfg.reward.function == "composite"
        assert len(cfg.reward.components) == 1

    def test_model_dump_roundtrip(self) -> None:
        cfg = ResolvedRLConfig(name="roundtrip-test", algorithm="ppo", seed=123)
        data = cfg.model_dump()
        restored = ResolvedRLConfig.model_validate(data)
        assert restored.name == "roundtrip-test"
        assert restored.algorithm == "ppo"
        assert restored.seed == 123

    def test_model_validate_from_dict(self) -> None:
        data = {
            "name": "from-dict",
            "algorithm": "dqn",
            "max_episodes": 500,
            "learning_rate": 0.001,
        }
        cfg = ResolvedRLConfig.model_validate(data)
        assert cfg.name == "from-dict"
        assert cfg.max_episodes == 500
        assert cfg.learning_rate == 0.001
        # Other fields get defaults
        assert cfg.experiment_type == "reinforcement_learning"

    def test_experiment_type_is_literal(self) -> None:
        cfg = ResolvedRLConfig(name="literal-check")
        assert cfg.experiment_type == "reinforcement_learning"

    def test_curriculum_enabled(self) -> None:
        curriculum = ResolvedCurriculumConfig(enabled=True, threshold=0.7)
        cfg = ResolvedRLConfig(name="curriculum-exp", curriculum=curriculum)
        assert cfg.curriculum.enabled is True
        assert cfg.curriculum.threshold == 0.7
