"""Multi-experiment comparison.

Compares results across multiple experiments and produces a structured
comparison report with rankings.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mlrl_os.core.experiment import AlgorithmScore, ExperimentResult
from mlrl_os.core.types import ExperimentStatus

# Metrics where lower is better.
_MINIMIZE_METRICS: set[str] = {"rmse", "mae", "mape"}


class ExperimentComparison(BaseModel):
    """Comparison of multiple experiment results."""

    experiment_ids: list[str]
    metric_names: list[str]
    rankings: list[dict[str, Any]]
    best_experiment_id: str
    best_experiment_name: str
    best_metrics: dict[str, float]


class ComparisonEngine:
    """Compare multiple experiment results."""

    def compare(
        self,
        results: list[ExperimentResult],
        metric_names: list[str] | None = None,
        rank_by: str | None = None,
    ) -> ExperimentComparison:
        """Compare experiments and rank them.

        Args:
            results: List of completed experiment results.
            metric_names: Metrics to include in comparison. If ``None``, use
                all common metrics across completed experiments.
            rank_by: Metric to rank by. If ``None``, use first metric.

        Returns:
            ExperimentComparison with rankings.

        Raises:
            ValueError: If no completed experiments remain or the ranking
                metric is not found in any result.
        """
        completed = [
            r for r in results if r.status == ExperimentStatus.COMPLETED
        ]
        if not completed:
            raise ValueError("No completed experiments to compare.")

        # Determine common metric names across all completed experiments.
        if metric_names is None:
            common: set[str] | None = None
            for r in completed:
                if r.metrics:
                    keys = set(r.metrics.keys())
                    common = keys if common is None else common & keys
            metric_names = sorted(common) if common else []

        if not metric_names:
            raise ValueError("No common metrics found across experiments.")

        rank_metric = rank_by if rank_by is not None else metric_names[0]

        # Verify at least one result has the ranking metric.
        has_rank_metric = any(
            r.metrics and rank_metric in r.metrics for r in completed
        )
        if not has_rank_metric:
            raise ValueError(
                f"Ranking metric '{rank_metric}' not found in any completed "
                "experiment results."
            )

        minimize = rank_metric in _MINIMIZE_METRICS

        # Build unsorted rows with metric values.
        rows: list[dict[str, Any]] = []
        for r in completed:
            row: dict[str, Any] = {
                "experiment_id": r.experiment_id,
                "name": r.name,
            }
            for m in metric_names:
                row[m] = r.metrics.get(m) if r.metrics else None
            rows.append(row)

        # Sort by the ranking metric.  Entries missing the metric sort last.
        def _sort_key(row: dict[str, Any]) -> tuple[int, float]:
            val = row.get(rank_metric)
            if val is None:
                return (1, 0.0)
            return (0, val if minimize else -val)

        rows.sort(key=_sort_key)

        # Assign ranks (1-based).
        for idx, row in enumerate(rows, start=1):
            row["rank"] = idx

        best_row = rows[0]
        best_metrics = {
            m: best_row[m] for m in metric_names if best_row.get(m) is not None
        }

        return ExperimentComparison(
            experiment_ids=[r.experiment_id for r in completed],
            metric_names=metric_names,
            rankings=rows,
            best_experiment_id=best_row["experiment_id"],
            best_experiment_name=best_row["name"],
            best_metrics=best_metrics,
        )
