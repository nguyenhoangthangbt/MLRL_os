"""Experiment history tracking and artifact storage.

Manages experiment records on the file system. Each experiment's config,
result, and report are stored as JSON under::

    {experiments_dir}/experiments/{experiment_id}/
        config.json
        result.json
        report.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlrl_os.config.defaults import MLRLSettings
from mlrl_os.core.experiment import ExperimentResult
from mlrl_os.core.types import ExperimentStatus
from mlrl_os.evaluation.reports import EvaluationReport

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """File-system based experiment history tracker."""

    def __init__(self, settings: MLRLSettings | None = None) -> None:
        self._settings = settings or MLRLSettings()
        self._base_dir = Path(self._settings.experiments_dir) / "experiments"
        self._base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _experiment_dir(self, experiment_id: str) -> Path:
        """Return the directory for a given experiment, without checking existence."""
        return self._base_dir / experiment_id

    def _require_dir(self, experiment_id: str) -> Path:
        """Return the experiment directory, raising *KeyError* if it does not exist."""
        path = self._experiment_dir(experiment_id)
        if not path.is_dir():
            msg = f"Experiment not found: {experiment_id}"
            raise KeyError(msg)
        return path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(self, experiment_id: str, name: str, config_dict: dict) -> None:  # type: ignore[type-arg]
        """Create a new experiment record with PENDING status.

        Persists the supplied configuration and an initial result stub so that
        the experiment appears in :meth:`list_experiments` immediately.

        Args:
            experiment_id: Unique identifier for the experiment.
            name: Human-readable experiment name.
            config_dict: Resolved experiment configuration dictionary.

        Raises:
            FileExistsError: If the experiment directory already exists.
        """
        exp_dir = self._experiment_dir(experiment_id)
        if exp_dir.exists():
            msg = f"Experiment already exists: {experiment_id}"
            raise FileExistsError(msg)

        exp_dir.mkdir(parents=True, exist_ok=True)

        # Persist config
        config_path = exp_dir / "config.json"
        config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

        # Persist initial result stub
        now = datetime.now(timezone.utc).isoformat()
        result = ExperimentResult(
            experiment_id=experiment_id,
            name=name,
            status=ExperimentStatus.PENDING,
            experiment_type=config_dict.get("problem_type", "time_series"),
            created_at=now,
            resolved_config=config_dict,
        )
        result_path = exp_dir / "result.json"
        result_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )

        logger.info("Created experiment record: %s (%s)", experiment_id, name)

    def update_status(self, experiment_id: str, status: ExperimentStatus) -> None:
        """Update experiment status.

        Args:
            experiment_id: The experiment to update.
            status: New status value.

        Raises:
            KeyError: If the experiment does not exist.
        """
        exp_dir = self._require_dir(experiment_id)
        result_path = exp_dir / "result.json"
        data = json.loads(result_path.read_text(encoding="utf-8"))
        data["status"] = status.value
        if status == ExperimentStatus.COMPLETED or status == ExperimentStatus.FAILED:
            data["completed_at"] = datetime.now(timezone.utc).isoformat()
        result_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Updated experiment %s status to %s", experiment_id, status.value)

    def save_result(self, result: ExperimentResult) -> None:
        """Save the experiment result.

        Args:
            result: The full experiment result to persist.

        Raises:
            KeyError: If the experiment does not exist.
        """
        exp_dir = self._require_dir(result.experiment_id)
        result_path = exp_dir / "result.json"
        result_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        logger.info("Saved result for experiment %s", result.experiment_id)

    def save_report(self, experiment_id: str, report: EvaluationReport) -> None:
        """Save an evaluation report for the experiment.

        Args:
            experiment_id: The experiment the report belongs to.
            report: The evaluation report to persist.

        Raises:
            KeyError: If the experiment does not exist.
        """
        exp_dir = self._require_dir(experiment_id)
        report_path = exp_dir / "report.json"
        report_path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        logger.info("Saved report for experiment %s", experiment_id)

    def get_result(self, experiment_id: str) -> ExperimentResult:
        """Load experiment result.

        Args:
            experiment_id: The experiment to load.

        Returns:
            The persisted :class:`ExperimentResult`.

        Raises:
            KeyError: If the experiment does not exist.
        """
        exp_dir = self._require_dir(experiment_id)
        result_path = exp_dir / "result.json"
        data = json.loads(result_path.read_text(encoding="utf-8"))
        return ExperimentResult.model_validate(data)

    def get_report(self, experiment_id: str) -> EvaluationReport:
        """Load evaluation report.

        Args:
            experiment_id: The experiment to load the report for.

        Returns:
            The persisted :class:`EvaluationReport`.

        Raises:
            KeyError: If the experiment or its report does not exist.
        """
        exp_dir = self._require_dir(experiment_id)
        report_path = exp_dir / "report.json"
        if not report_path.exists():
            msg = f"Report not found for experiment: {experiment_id}"
            raise KeyError(msg)
        data = json.loads(report_path.read_text(encoding="utf-8"))
        return EvaluationReport.model_validate(data)

    def list_experiments(self) -> list[ExperimentResult]:
        """List all experiments, sorted by created_at descending.

        Returns:
            A list of :class:`ExperimentResult` objects ordered from newest
            to oldest.
        """
        results: list[ExperimentResult] = []
        if not self._base_dir.exists():
            return results

        for exp_dir in self._base_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            result_path = exp_dir / "result.json"
            if not result_path.exists():
                continue
            try:
                data = json.loads(result_path.read_text(encoding="utf-8"))
                results.append(ExperimentResult.model_validate(data))
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to load experiment result from %s", result_path
                )

        results.sort(key=lambda r: r.created_at, reverse=True)
        return results

    def record(
        self,
        experiment_id: str,
        config: Any,
        result: ExperimentResult,
        report: EvaluationReport | None = None,
    ) -> None:
        """Convenience method: create/update an experiment with config, result, and optional report.

        This is the high-level method used by :class:`ExperimentRunner`.
        """
        config_dict = config.model_dump(mode="json") if hasattr(config, "model_dump") else config
        exp_dir = self._experiment_dir(experiment_id)

        if not exp_dir.exists():
            self.create(experiment_id, result.name, config_dict)

        self.save_result(result)

        if report is not None:
            self.save_report(experiment_id, report)

    def has(self, experiment_id: str) -> bool:
        """Check if an experiment exists.

        Args:
            experiment_id: The experiment identifier to check.

        Returns:
            ``True`` if the experiment directory exists, ``False`` otherwise.
        """
        return self._experiment_dir(experiment_id).is_dir()
