"""Tests for mlrl_os.models.algorithms.protocol."""

from __future__ import annotations

import numpy as np
import pytest

from mlrl_os.models.algorithms.protocol import TrainedModel


class TestTrainedModel:
    """Tests for TrainedModel dataclass."""

    def test_construction_with_all_fields(self) -> None:
        """TrainedModel stores model, algorithm_name, task, and feature_names."""
        dummy_model = object()
        tm = TrainedModel(
            model=dummy_model,
            algorithm_name="lightgbm",
            task="regression",
            feature_names=["f0", "f1", "f2"],
        )

        assert tm.model is dummy_model
        assert tm.algorithm_name == "lightgbm"
        assert tm.task == "regression"
        assert tm.feature_names == ["f0", "f1", "f2"]

    def test_repr(self) -> None:
        """__repr__ includes algorithm name, task, and feature count."""
        tm = TrainedModel(
            model=None,
            algorithm_name="xgboost",
            task="classification",
            feature_names=["a", "b"],
        )

        r = repr(tm)
        assert "xgboost" in r
        assert "classification" in r
        assert "features=2" in r
