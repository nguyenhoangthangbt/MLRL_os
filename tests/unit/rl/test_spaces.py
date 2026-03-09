"""Tests for mlrl_os.rl.spaces — Observation and Action specifications."""

from __future__ import annotations

import numpy as np

from mlrl_os.rl.spaces import ActionSpec, ObservationSpec, build_specs_from_template

# ---------------------------------------------------------------------------
# ObservationSpec
# ---------------------------------------------------------------------------


class TestObservationSpec:
    """Tests for ObservationSpec dataclass."""

    def test_creation_with_valid_dimensions(self) -> None:
        spec = ObservationSpec(
            dim=3,
            names=["a", "b", "c"],
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        assert spec.dim == 3
        assert spec.names == ["a", "b", "c"]
        assert spec.low.shape == (3,)
        assert spec.high.shape == (3,)

    def test_contains_within_bounds(self) -> None:
        spec = ObservationSpec(
            dim=2,
            names=["x", "y"],
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )
        assert spec.contains(np.array([0.5, 0.0], dtype=np.float32))

    def test_contains_at_boundaries(self) -> None:
        spec = ObservationSpec(
            dim=2,
            names=["x", "y"],
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )
        assert spec.contains(np.array([0.0, 0.0], dtype=np.float32))
        assert spec.contains(np.array([1.0, 1.0], dtype=np.float32))

    def test_contains_outside_bounds(self) -> None:
        spec = ObservationSpec(
            dim=2,
            names=["x", "y"],
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )
        assert not spec.contains(np.array([1.5, 0.5], dtype=np.float32))
        assert not spec.contains(np.array([-0.1, 0.5], dtype=np.float32))

    def test_contains_with_infinite_bounds(self) -> None:
        spec = ObservationSpec(
            dim=2,
            names=["x", "y"],
            low=np.full(2, -np.inf, dtype=np.float32),
            high=np.full(2, np.inf, dtype=np.float32),
        )
        assert spec.contains(np.array([1e10, -1e10], dtype=np.float32))


# ---------------------------------------------------------------------------
# ActionSpec
# ---------------------------------------------------------------------------


class TestActionSpec:
    """Tests for ActionSpec dataclass."""

    def test_discrete_action_spec(self) -> None:
        spec = ActionSpec(type="discrete", n=4, labels=["a", "b", "c", "d"])
        assert spec.type == "discrete"
        assert spec.n == 4
        assert spec.labels == ["a", "b", "c", "d"]
        assert spec.shape is None
        assert spec.low is None
        assert spec.high is None

    def test_continuous_action_spec(self) -> None:
        spec = ActionSpec(
            type="continuous",
            shape=(2,),
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
        )
        assert spec.type == "continuous"
        assert spec.shape == (2,)
        assert spec.n is None

    def test_discrete_without_labels(self) -> None:
        spec = ActionSpec(type="discrete", n=3)
        assert spec.labels is None
        assert spec.n == 3


# ---------------------------------------------------------------------------
# build_specs_from_template
# ---------------------------------------------------------------------------


class TestBuildSpecsFromTemplate:
    """Tests for build_specs_from_template factory function."""

    def test_with_node_and_resource_names(self) -> None:
        metadata = {
            "node_names": ["triage", "treatment", "discharge"],
            "resource_names": ["doctor", "nurse"],
        }
        obs_spec, act_spec = build_specs_from_template(metadata)

        # 10 entity + 9 node + 5 system = 24
        assert obs_spec.dim == 24
        assert len(obs_spec.names) == 24
        assert obs_spec.low.shape == (24,)
        assert obs_spec.high.shape == (24,)

        assert act_spec.type == "discrete"
        assert act_spec.n == 3
        assert act_spec.labels == ["triage", "treatment", "discharge"]

    def test_with_empty_metadata(self) -> None:
        metadata: dict = {}
        obs_spec, act_spec = build_specs_from_template(metadata)

        assert obs_spec.dim == 24
        assert act_spec.type == "discrete"
        assert act_spec.n == 2  # minimum 2
        assert act_spec.labels is None

    def test_observation_names_include_all_feature_groups(self) -> None:
        metadata = {"node_names": ["a"], "resource_names": []}
        obs_spec, _ = build_specs_from_template(metadata)

        entity_prefix = "entity_"
        node_prefix = "node_"
        sys_prefix = "sys_"

        entity_count = sum(1 for n in obs_spec.names if n.startswith(entity_prefix))
        node_count = sum(1 for n in obs_spec.names if n.startswith(node_prefix))
        sys_count = sum(1 for n in obs_spec.names if n.startswith(sys_prefix))

        assert entity_count == 10
        assert node_count == 9
        assert sys_count == 5

    def test_infinite_bounds_by_default(self) -> None:
        metadata = {"node_names": ["x"]}
        obs_spec, _ = build_specs_from_template(metadata)

        assert np.all(obs_spec.low == -np.inf)
        assert np.all(obs_spec.high == np.inf)

    def test_single_node_gets_minimum_two_actions(self) -> None:
        metadata = {"node_names": ["only_one"]}
        _, act_spec = build_specs_from_template(metadata)

        assert act_spec.n == 2
