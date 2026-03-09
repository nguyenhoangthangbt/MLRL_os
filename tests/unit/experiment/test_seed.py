"""Tests for mlrl_os.experiment.seed."""

from __future__ import annotations

import random

import pytest

from mlrl_os.experiment.seed import make_rng, seed_hash


class TestSeedHash:
    """Tests for seed_hash()."""

    def test_deterministic_same_inputs(self) -> None:
        """seed_hash returns the same value for identical inputs."""
        a = seed_hash("component_a", 42)
        b = seed_hash("component_a", 42)
        assert a == b

    def test_differs_for_different_names(self) -> None:
        """seed_hash returns different values for different component names."""
        a = seed_hash("component_a", 42)
        b = seed_hash("component_b", 42)
        assert a != b

    def test_differs_for_different_seeds(self) -> None:
        """seed_hash returns different values for different global seeds."""
        a = seed_hash("component_a", 42)
        b = seed_hash("component_a", 99)
        assert a != b


class TestMakeRng:
    """Tests for make_rng()."""

    def test_returns_deterministic_random(self) -> None:
        """make_rng returns a random.Random instance."""
        rng = make_rng("test", 42)
        assert isinstance(rng, random.Random)

    def test_same_args_produce_same_sequence(self) -> None:
        """Two RNGs created with the same args produce identical sequences."""
        rng1 = make_rng("cv_split", 42)
        rng2 = make_rng("cv_split", 42)

        seq1 = [rng1.random() for _ in range(10)]
        seq2 = [rng2.random() for _ in range(10)]
        assert seq1 == seq2
