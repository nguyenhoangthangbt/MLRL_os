"""Pydantic models for RL experiment configuration.

Provides the ``ResolvedRLConfig`` schema that mirrors the supervised
``ResolvedExperimentConfig`` pattern -- after resolution every field has
a concrete value with no optionals.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ResolvedRewardConfig(BaseModel):
    """Configuration for the reward function used during RL training."""

    function: str = "throughput"
    components: list[dict[str, Any]] = Field(default_factory=list)


class ResolvedCurriculumConfig(BaseModel):
    """Configuration for curriculum learning progression."""

    enabled: bool = False
    threshold: float = 0.8


class ResolvedRLConfig(BaseModel):
    """Fully resolved RL experiment configuration.

    After resolution every field has a concrete value -- pipeline code
    never checks for ``None``.
    """

    name: str
    experiment_type: Literal["reinforcement_learning"] = "reinforcement_learning"
    seed: int = 42

    # SimOS connection
    simos_url: str = "ws://localhost:8000"
    template: str = "healthcare_er"
    api_key: str = "sk-premium-test-003"
    max_steps_per_episode: int = 10_000

    # Training
    algorithm: str = "dqn"
    max_episodes: int = 1000
    batch_size: int = 64
    learning_rate: float = 0.0003
    gamma: float = 0.99

    # PPO-specific
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gae_lambda: float = 0.95

    # DQN-specific
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: int = 10_000
    target_update_freq: int = 100
    replay_buffer_size: int = 10_000

    # Reward
    reward: ResolvedRewardConfig = Field(default_factory=ResolvedRewardConfig)

    # Curriculum
    curriculum: ResolvedCurriculumConfig = Field(default_factory=ResolvedCurriculumConfig)

    # Evaluation
    eval_episodes: int = 50
    eval_metrics: list[str] = Field(default_factory=lambda: ["mean_reward", "success_rate"])
