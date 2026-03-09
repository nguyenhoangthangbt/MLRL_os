"""SimOS v3.0 export loader with Schema Adapter pattern.

This module is the ONLY place that knows SimOS field names.
All downstream code uses canonical column names exclusively.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import polars as pl

from mlrl_os.core.dataset import RawDataset, compute_column_info

logger = logging.getLogger(__name__)


class SimosSchemaAdapter:
    """Maps SimOS v3.0 export field names to ML/RL OS canonical names.

    This is the single point of coupling between SimOS field naming
    and the internal canonical schema. If SimOS changes its export
    format, only this class needs updating.
    """

    # ── Snapshot system-level mapping ────────────────────────────────
    SNAPSHOT_FIELD_MAP: dict[str, str] = {
        "timestamp": "ts",
        "bucket_index": "bucket_idx",
        "sys_wip": "wip",
        "sys_total_in_queue": "in_queue",
        "sys_total_busy": "busy",
        "sys_cumulative_arrivals": "cum_arrivals",
        "sys_cumulative_completions": "cum_completions",
        "sys_arrival_rate_per_hour": "arrival_rate",
        "sys_throughput_per_hour": "throughput",
        "sys_wip_ratio": "wip_ratio",
        "sys_avg_wait_time": "avg_wait",
        "sys_avg_processing_time": "avg_processing",
        "sys_wait_cost_in_bucket": "wait_cost_bucket",
        "sys_idle_cost_in_bucket": "idle_cost_bucket",
        "sys_revenue_in_bucket": "revenue_bucket",
        "sys_cumulative_wait_cost": "cum_wait_cost",
        "sys_cumulative_idle_cost": "cum_idle_cost",
        "sys_cumulative_revenue": "cum_revenue",
        "sys_cumulative_net_cost": "cum_net_cost",
        "sys_sla_breaches_in_bucket": "sla_breaches_bucket",
        "sys_cumulative_sla_breaches": "cum_sla_breaches",
        "sys_sla_compliance": "sla_compliance",
    }

    # ── Node metric mapping (suffix only) ───────────────────────────
    NODE_METRIC_MAP: dict[str, str] = {
        "queue": "queue",
        "busy": "busy",
        "concurrency": "concurrency",
        "utilization": "util",
        "cumulative_processed": "cum_processed",
        "arrivals_in_bucket": "arrivals_bucket",
        "completions_in_bucket": "completions_bucket",
        "throughput_per_hour": "throughput",
        "avg_wait_time": "avg_wait",
        "avg_processing_time": "avg_processing",
    }

    # ── Resource metric mapping (suffix only) ───────────────────────
    RESOURCE_METRIC_MAP: dict[str, str] = {
        "capacity": "capacity",
        "utilization": "util",
    }

    # ── Trajectory top-level mapping ────────────────────────────────
    TRAJECTORY_FIELD_MAP: dict[str, str] = {
        "entity_id": "eid",
        "entity_type": "etype",
        "entity_priority": "epriority",
        "source_index": "source_idx",
        "step_index": "step",
        "node_name": "node",
        "sim_time_enter": "t_enter",
        "sim_time_complete": "t_complete",
        "wait_time": "wait",
        "processing_time": "processing",
        "setup_time": "setup",
        "transit_time": "transit",
        "transport_cost": "transit_cost",
        "episode_done": "done",
        "episode_status": "status",
        "episode_total_time": "total_time",
        "attr_priority_class": "priority_class",
    }

    # Top-level trajectory fields that are passed through with their
    # original name (SLA thresholds, domain enrichment flags).
    # These are template-specific and cannot be enumerated statically.
    _TRAJECTORY_PASSTHROUGH_PREFIXES = ("sla_", "domain_")

    # ── State dict mapping ──────────────────────────────────────────
    STATE_FIELD_MAP: dict[str, str] = {
        "entity_priority": "s_priority",
        "entity_elapsed_time": "s_elapsed",
        "entity_cumulative_wait": "s_cum_wait",
        "entity_cumulative_processing": "s_cum_processing",
        "entity_wait_ratio": "s_wait_ratio",
        "entity_steps_completed": "s_steps_done",
        "entity_cumulative_cost": "s_cum_cost",
        "entity_cumulative_transport_cost": "s_cum_transit_cost",
        "entity_cumulative_setup_time": "s_cum_setup",
        "entity_source_index": "s_source_idx",
        "node_utilization": "s_node_util",
        "node_avg_queue_depth": "s_node_avg_queue",
        "node_max_queue_depth": "s_node_max_queue",
        "node_avg_wait_time": "s_node_avg_wait",
        "node_avg_processing_time": "s_node_avg_processing",
        "node_throughput_per_hour": "s_node_throughput",
        "node_concurrency": "s_node_concurrency",
        "node_setup_count": "s_node_setup_count",
        "node_mutation_count": "s_node_mutation_count",
        "sys_utilization": "s_sys_util",
        "sys_throughput_per_hour": "s_sys_throughput",
        "sys_bottleneck_utilization": "s_sys_bottleneck_util",
        "sys_node_count": "s_sys_node_count",
        "sys_total_capacity": "s_sys_total_capacity",
    }

    # ── Reward dict mapping ─────────────────────────────────────────
    REWARD_FIELD_MAP: dict[str, str] = {
        "time_penalty": "rw_time_penalty",
        "cost_penalty": "rw_cost_penalty",
        "transport_cost_penalty": "rw_transport_cost_penalty",
        "step_completion": "rw_step_completion",
        "sla_penalty": "rw_sla_penalty",
        "processing_efficiency": "rw_processing_efficiency",
        "sla_budget_remaining": "rw_sla_budget_remaining",
        "episode_completion": "rw_episode_completion",
    }

    # ── Compiled patterns for dynamic fields ────────────────────────
    _NODE_PATTERN = re.compile(r"^node_([a-zA-Z0-9_]+?)_(.+)$")
    _RESOURCE_PATTERN = re.compile(r"^resource_([a-zA-Z0-9_]+?)_(.+)$")
    _STATE_RESOURCE_PATTERN = re.compile(
        r"^resource_([a-zA-Z0-9_]+?)_(utilization|capacity)$"
    )
    _STATE_ATTR_PATTERN = re.compile(r"^attr_(.+)$")

    def map_snapshot_field(self, simos_name: str) -> str | None:
        """Map a single SimOS snapshot field name to its canonical name.

        Returns None if the field is unrecognised.
        """
        # System-level field
        if simos_name in self.SNAPSHOT_FIELD_MAP:
            return self.SNAPSHOT_FIELD_MAP[simos_name]

        # Node-level field: node_{name}_{metric}
        m = self._NODE_PATTERN.match(simos_name)
        if m:
            node_name, metric = m.group(1), m.group(2)
            canonical_metric = self.NODE_METRIC_MAP.get(metric)
            if canonical_metric is not None:
                return f"n_{node_name}_{canonical_metric}"

        # Resource-level field: resource_{name}_{metric}
        m = self._RESOURCE_PATTERN.match(simos_name)
        if m:
            res_name, metric = m.group(1), m.group(2)
            canonical_metric = self.RESOURCE_METRIC_MAP.get(metric)
            if canonical_metric is not None:
                return f"r_{res_name}_{canonical_metric}"

        logger.debug("Unrecognised SimOS snapshot field: %s", simos_name)
        return None

    def map_trajectory_top_level(self, simos_name: str) -> str | None:
        """Map a top-level trajectory field to its canonical name.

        Fields with known prefixes (``sla_``, ``domain_``) are passed
        through with their original name since they are template-specific
        and cannot be enumerated statically.
        """
        canonical = self.TRAJECTORY_FIELD_MAP.get(simos_name)
        if canonical is not None:
            return canonical

        # Passthrough: SLA thresholds and domain enrichment flags
        for prefix in self._TRAJECTORY_PASSTHROUGH_PREFIXES:
            if simos_name.startswith(prefix):
                return simos_name

        return None

    def map_state_field(self, simos_name: str) -> str | None:
        """Map a state-dict field to its canonical name (s_ prefix).

        Handles static fields, dynamic resource fields, and dynamic attr fields.
        """
        # Static mapping
        if simos_name in self.STATE_FIELD_MAP:
            return self.STATE_FIELD_MAP[simos_name]

        # Dynamic resource state: resource_{name}_{utilization|capacity}
        m = self._STATE_RESOURCE_PATTERN.match(simos_name)
        if m:
            res_name, metric = m.group(1), m.group(2)
            canonical_metric = self.RESOURCE_METRIC_MAP.get(metric, metric)
            return f"s_r_{res_name}_{canonical_metric}"

        # Dynamic attr: attr_{name}
        m = self._STATE_ATTR_PATTERN.match(simos_name)
        if m:
            attr_name = m.group(1)
            return f"s_attr_{attr_name}"

        logger.debug("Unrecognised SimOS state field: %s", simos_name)
        return None

    def map_reward_field(self, simos_name: str) -> str | None:
        """Map a reward-dict field to its canonical name (rw_ prefix)."""
        return self.REWARD_FIELD_MAP.get(simos_name)

    def map_snapshot_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Map all fields in a single snapshot record to canonical names.

        Unknown fields are silently dropped.
        """
        mapped: dict[str, Any] = {}
        for key, value in record.items():
            canonical = self.map_snapshot_field(key)
            if canonical is not None:
                mapped[canonical] = value
        return mapped

    def map_trajectory_record(
        self, record: dict[str, Any]
    ) -> dict[str, Any]:
        """Map and flatten a single trajectory record to canonical names.

        Flattens nested ``state``, ``reward``, and ``next_state`` dicts
        into top-level columns with appropriate prefixes.
        """
        mapped: dict[str, Any] = {}

        # Top-level fields
        for key, value in record.items():
            if key in ("state", "reward", "next_state"):
                continue
            canonical = self.map_trajectory_top_level(key)
            if canonical is not None:
                mapped[canonical] = value

        # State dict → s_ prefixed columns
        state = record.get("state")
        if isinstance(state, dict):
            for key, value in state.items():
                canonical = self.map_state_field(key)
                if canonical is not None:
                    mapped[canonical] = value

        # Reward dict → rw_ prefixed columns
        reward = record.get("reward")
        if isinstance(reward, dict):
            for key, value in reward.items():
                canonical = self.map_reward_field(key)
                if canonical is not None:
                    mapped[canonical] = value

        # Next-state dict → ns_ prefixed columns (mirror of state mapping)
        next_state = record.get("next_state")
        if isinstance(next_state, dict):
            for key, value in next_state.items():
                canonical = self.map_state_field(key)
                if canonical is not None:
                    # Replace s_ prefix with ns_ for next-state
                    ns_canonical = "ns_" + canonical[2:]
                    mapped[ns_canonical] = value

        return mapped


class SimosLoader:
    """Load SimOS v3.0 JSON exports into canonical RawDataset.

    Usage::

        loader = SimosLoader()
        dataset = loader.load(Path("export.json"))
    """

    def __init__(self) -> None:
        self._adapter = SimosSchemaAdapter()

    def load(self, path: Path) -> RawDataset:
        """Load a SimOS JSON export file and return a RawDataset.

        Args:
            path: Path to the SimOS JSON export file.

        Returns:
            RawDataset with canonical column names.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the file cannot be parsed or contains no data.
        """
        if not path.exists():
            msg = f"SimOS export file not found: {path}"
            raise FileNotFoundError(msg)

        logger.info("Loading SimOS export from %s", path)

        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            msg = f"Expected JSON object at top level, got {type(raw).__name__}"
            raise ValueError(msg)

        # Extract optional metadata/summary
        metadata_dict = raw.get("metadata")
        summary_dict = raw.get("summary")

        # Extract snapshots (SimOS export uses "state_snapshots" key)
        raw_snapshots = raw.get("state_snapshots") or raw.get("snapshots")
        snapshots_df: pl.DataFrame | None = None
        col_info_snapshots = None
        if raw_snapshots and isinstance(raw_snapshots, list):
            snapshots_df = self.extract_snapshots(raw_snapshots)
            col_info_snapshots = compute_column_info(snapshots_df)
            logger.info(
                "Extracted %d snapshot rows, %d columns",
                len(snapshots_df),
                len(snapshots_df.columns),
            )

        # Extract trajectories
        raw_trajectories = raw.get("trajectories")
        trajectories_df: pl.DataFrame | None = None
        col_info_trajectories = None
        if raw_trajectories and isinstance(raw_trajectories, list):
            trajectories_df = self.extract_trajectories(raw_trajectories)
            col_info_trajectories = compute_column_info(trajectories_df)
            logger.info(
                "Extracted %d trajectory rows, %d columns",
                len(trajectories_df),
                len(trajectories_df.columns),
            )

        if snapshots_df is None and trajectories_df is None:
            msg = "SimOS export contains neither snapshots nor trajectories"
            raise ValueError(msg)

        return RawDataset(
            source_type="simos",
            source_path=str(path),
            snapshots=snapshots_df,
            trajectories=trajectories_df,
            metadata_dict=metadata_dict,
            summary_dict=summary_dict,
            column_info_snapshots=col_info_snapshots,
            column_info_trajectories=col_info_trajectories,
        )

    def extract_snapshots(self, raw_snapshots: list[dict[str, Any]]) -> pl.DataFrame:
        """Convert raw SimOS snapshot dicts to a canonical Polars DataFrame.

        Args:
            raw_snapshots: List of snapshot records from the SimOS export.

        Returns:
            DataFrame with canonical column names, sorted by ``ts``.
        """
        mapped_records = [
            self._adapter.map_snapshot_record(rec) for rec in raw_snapshots
        ]

        # Filter out empty records
        mapped_records = [rec for rec in mapped_records if rec]

        if not mapped_records:
            msg = "No valid snapshot records after mapping"
            raise ValueError(msg)

        df = pl.DataFrame(mapped_records)

        # Sort by timestamp if present
        if "ts" in df.columns:
            df = df.sort("ts")

        return df

    def extract_trajectories(
        self, raw_trajectories: list[dict[str, Any]]
    ) -> pl.DataFrame:
        """Convert raw SimOS trajectory dicts to a canonical Polars DataFrame.

        Flattens nested state, reward, and next_state dicts into top-level
        columns with appropriate prefixes (s\\_, rw\\_, ns\\_).

        Args:
            raw_trajectories: List of trajectory records from the SimOS export.

        Returns:
            DataFrame with canonical column names, sorted by ``eid`` and ``step``.
        """
        mapped_records = [
            self._adapter.map_trajectory_record(rec) for rec in raw_trajectories
        ]

        # Filter out empty records
        mapped_records = [rec for rec in mapped_records if rec]

        if not mapped_records:
            msg = "No valid trajectory records after mapping"
            raise ValueError(msg)

        df = pl.DataFrame(mapped_records)

        # Sort by entity id and step if present
        sort_cols = [c for c in ("eid", "step") if c in df.columns]
        if sort_cols:
            df = df.sort(sort_cols)

        return df
