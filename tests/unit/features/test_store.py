"""Tests for mlrl_os.features.store module."""

from __future__ import annotations

import pytest

from mlrl_os.features.store import FeatureDefinition, FeatureStore


@pytest.fixture()
def sample_definition() -> FeatureDefinition:
    return FeatureDefinition(
        name="wip_lag_1h",
        source_column="wip",
        transform_type="lag",
        parameters={"interval": "1h", "steps": 12},
        description="WIP lagged by 1 hour",
    )


@pytest.fixture()
def store() -> FeatureStore:
    return FeatureStore()


class TestFeatureDefinition:
    """Tests for FeatureDefinition creation."""

    def test_creation_with_all_fields(self) -> None:
        fd = FeatureDefinition(
            name="throughput_rmean_2h",
            source_column="throughput",
            transform_type="rolling_mean",
            parameters={"window": "2h"},
            description="Rolling mean of throughput over 2h",
        )
        assert fd.name == "throughput_rmean_2h"
        assert fd.source_column == "throughput"
        assert fd.transform_type == "rolling_mean"
        assert fd.parameters == {"window": "2h"}

    def test_creation_with_defaults(self) -> None:
        fd = FeatureDefinition(
            name="test_feat",
            source_column="col",
            transform_type="derived",
        )
        assert fd.parameters == {}
        assert fd.description == ""


class TestFeatureStore:
    """Tests for FeatureStore registry."""

    def test_register_adds_definition(
        self, store: FeatureStore, sample_definition: FeatureDefinition
    ) -> None:
        name = store.register(sample_definition)
        assert name == sample_definition.name
        assert store.has(name)

    def test_get_retrieves_definition(
        self, store: FeatureStore, sample_definition: FeatureDefinition
    ) -> None:
        store.register(sample_definition)
        retrieved = store.get(sample_definition.name)
        assert retrieved.name == sample_definition.name
        assert retrieved.source_column == sample_definition.source_column

    def test_get_unknown_raises_key_error(self, store: FeatureStore) -> None:
        with pytest.raises(KeyError, match="not found"):
            store.get("nonexistent_feature")

    def test_list_definitions_returns_all(
        self, store: FeatureStore
    ) -> None:
        fd1 = FeatureDefinition(
            name="feat_a", source_column="a", transform_type="lag"
        )
        fd2 = FeatureDefinition(
            name="feat_b", source_column="b", transform_type="trend"
        )
        store.register(fd1)
        store.register(fd2)
        defs = store.list_definitions()
        assert len(defs) == 2
        names = {d.name for d in defs}
        assert names == {"feat_a", "feat_b"}

    def test_has_returns_true_for_registered(
        self, store: FeatureStore, sample_definition: FeatureDefinition
    ) -> None:
        store.register(sample_definition)
        assert store.has(sample_definition.name) is True

    def test_has_returns_false_for_unregistered(
        self, store: FeatureStore
    ) -> None:
        assert store.has("no_such_feature") is False

    def test_count_property(self, store: FeatureStore) -> None:
        assert store.count == 0
        store.register(
            FeatureDefinition(name="f1", source_column="c1", transform_type="lag")
        )
        assert store.count == 1
        store.register(
            FeatureDefinition(name="f2", source_column="c2", transform_type="trend")
        )
        assert store.count == 2
