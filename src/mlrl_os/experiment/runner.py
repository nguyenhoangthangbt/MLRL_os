"""Full experiment pipeline orchestrator.

This is the main entry point that ties together:
config resolution -> validation -> feature engineering -> training -> evaluation -> storage.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import polars as pl

from mlrl_os.config.defaults import MLRLSettings
from mlrl_os.config.resolver import ConfigResolver
from mlrl_os.config.schemas import (
    ResolvedEntityFeatures,
    ResolvedExperimentConfig,
    ResolvedTimeSeriesFeatures,
)
from mlrl_os.core.dataset import DatasetMeta
from mlrl_os.core.experiment import ExperimentResult
from mlrl_os.core.types import ExperimentStatus, FeatureMatrix, ProblemType
from mlrl_os.data.registry import DatasetRegistry
from mlrl_os.evaluation.reports import EvaluationReport, ReportGenerator
from mlrl_os.experiment.tracker import ExperimentTracker
from mlrl_os.features.entity import EntityFeatureEngine
from mlrl_os.features.time_series import TimeSeriesFeatureEngine
from mlrl_os.models.engine import ModelEngine, TrainingResult
from mlrl_os.models.registry import ModelRegistry
from mlrl_os.validation.gate import ValidationGate, ValidationResult

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates the full experiment pipeline."""

    def __init__(
        self,
        settings: MLRLSettings | None = None,
        dataset_registry: DatasetRegistry | None = None,
        model_registry: ModelRegistry | None = None,
        experiment_tracker: ExperimentTracker | None = None,
        model_engine: ModelEngine | None = None,
    ) -> None:
        self._settings = settings or MLRLSettings()
        self._dataset_registry = dataset_registry or DatasetRegistry(self._settings)
        self._model_registry = model_registry or ModelRegistry(self._settings)
        self._experiment_tracker = experiment_tracker or ExperimentTracker(self._settings)
        self._model_engine = model_engine or ModelEngine()

        self._config_resolver = ConfigResolver(self._settings)
        self._validation_gate = ValidationGate()
        self._report_generator = ReportGenerator()
        self._ts_engine = TimeSeriesFeatureEngine()
        self._entity_engine = EntityFeatureEngine()

    def run(
        self,
        user_config: dict[str, Any],
        dataset_meta: DatasetMeta,
    ) -> ExperimentResult:
        """Run a complete experiment pipeline.

        Steps:
        1. Resolve config (merge user + defaults)
        2. Validate config against data
        3. Load data from registry
        4. Build features (time-series or entity)
        5. Train & evaluate all algorithms with CV
        6. Save best model to model registry
        7. Generate evaluation report
        8. Save result and report to experiment tracker
        9. Return ExperimentResult

        If validation fails, returns a FAILED result with error details.
        If training fails, returns a FAILED result with error message.
        """
        experiment_id = uuid.uuid4().hex[:12]
        created_at = datetime.now(tz=timezone.utc).isoformat()

        # Step 1: Resolve config
        try:
            config = self._config_resolver.resolve(user_config, dataset_meta)
        except Exception as exc:
            logger.error("Config resolution failed: %s", exc)
            return ExperimentResult(
                experiment_id=experiment_id,
                name=user_config.get("name", "failed_experiment"),
                status=ExperimentStatus.FAILED,
                experiment_type=ProblemType.TIME_SERIES,
                created_at=created_at,
                completed_at=datetime.now(tz=timezone.utc).isoformat(),
                error_message=f"Config resolution failed: {exc}",
            )

        start_time = datetime.now(tz=timezone.utc)

        try:
            # Step 2: Load data from registry
            logger.info(
                "Loading data for dataset %s, layer %s",
                dataset_meta.id,
                config.dataset_layer,
            )
            df = self._dataset_registry.get_data(dataset_meta.id, config.dataset_layer)

            # Step 3: Validate config against data
            logger.info("Validating experiment config")
            validation = self._validation_gate.validate(config, df)
            if not validation.valid:
                error_messages = "; ".join(
                    f"[{e.code}] {e.message}" for e in validation.errors
                )
                logger.warning("Validation failed: %s", error_messages)
                return ExperimentResult(
                    experiment_id=experiment_id,
                    name=config.name,
                    status=ExperimentStatus.FAILED,
                    experiment_type=config.experiment_type,
                    created_at=created_at,
                    completed_at=datetime.now(tz=timezone.utc).isoformat(),
                    resolved_config=config.model_dump(mode="json"),
                    error_message=f"Validation failed: {error_messages}",
                )

            # Step 4: Build features
            logger.info("Building features for %s", config.experiment_type.value)
            feature_matrix = self._build_features(config, df)

            # Step 5: Train & evaluate all algorithms with CV
            logger.info(
                "Training algorithms: %s", config.model.algorithms,
            )
            training_result: TrainingResult = self._model_engine.train_and_evaluate(
                feature_matrix=feature_matrix,
                algorithm_names=config.model.algorithms,
                metric_names=config.evaluation.metrics,
                cv_strategy=config.model.cross_validation.strategy,
                cv_folds=config.model.cross_validation.folds,
                seed=config.seed,
                selection=config.model.selection,
                handle_imbalance=config.model.handle_imbalance,
                hyperparameter_tuning=config.model.hyperparameter_tuning,
                n_trials=config.model.n_trials,
            )

            # Step 6: Save best model to model registry
            logger.info("Saving best model: %s", training_result.best_algorithm)
            model_meta = self._model_registry.register(
                trained_model=training_result.best_model,
                experiment_id=experiment_id,
                metrics=training_result.best_metrics,
            )

            end_time = datetime.now(tz=timezone.utc)
            duration_seconds = (end_time - start_time).total_seconds()

            # Build experiment result
            result = ExperimentResult(
                experiment_id=experiment_id,
                name=config.name,
                status=ExperimentStatus.COMPLETED,
                experiment_type=config.experiment_type,
                created_at=created_at,
                completed_at=end_time.isoformat(),
                duration_seconds=duration_seconds,
                best_algorithm=training_result.best_algorithm,
                metrics=training_result.best_metrics,
                all_algorithm_scores=training_result.all_scores,
                feature_importance=training_result.feature_importance,
                model_id=model_meta.id,
                sample_count=feature_matrix.sample_count,
                feature_count=feature_matrix.feature_count,
                resolved_config=config.model_dump(mode="json"),
            )

            # Step 7: Generate evaluation report
            if config.evaluation.generate_report:
                logger.info("Generating evaluation report")
                report: EvaluationReport = self._report_generator.generate(
                    result=result,
                    y_true=feature_matrix.y,
                    y_pred=training_result.y_pred,
                    y_proba=training_result.y_proba,
                    class_names=feature_matrix.class_names,
                    dataset_name=dataset_meta.name,
                )

                # Step 8: Save result and report to experiment tracker
                self._experiment_tracker.record(
                    experiment_id=experiment_id,
                    config=config,
                    result=result,
                    report=report,
                )
            else:
                self._experiment_tracker.record(
                    experiment_id=experiment_id,
                    config=config,
                    result=result,
                )

            logger.info(
                "Experiment %s completed in %.1fs — best: %s (%s)",
                experiment_id,
                duration_seconds,
                training_result.best_algorithm,
                {k: round(v, 4) for k, v in training_result.best_metrics.items()},
            )

            return result

        except Exception as exc:
            end_time = datetime.now(tz=timezone.utc)
            duration_seconds = (end_time - start_time).total_seconds()
            logger.error(
                "Experiment %s failed after %.1fs: %s",
                experiment_id,
                duration_seconds,
                exc,
                exc_info=True,
            )
            return ExperimentResult(
                experiment_id=experiment_id,
                name=config.name,
                status=ExperimentStatus.FAILED,
                experiment_type=config.experiment_type,
                created_at=created_at,
                completed_at=end_time.isoformat(),
                duration_seconds=duration_seconds,
                resolved_config=config.model_dump(mode="json"),
                error_message=str(exc),
            )

    def validate_only(
        self,
        user_config: dict[str, Any],
        dataset_meta: DatasetMeta,
    ) -> ValidationResult:
        """Validate a config without running the experiment."""
        config = self._config_resolver.resolve(user_config, dataset_meta)
        df = self._dataset_registry.get_data(dataset_meta.id, config.dataset_layer)
        return self._validation_gate.validate(config, df)

    def _build_features(
        self,
        config: ResolvedExperimentConfig,
        df: pl.DataFrame,
    ) -> FeatureMatrix:
        """Build features based on problem type."""
        features = config.features

        if config.experiment_type == ProblemType.TIME_SERIES:
            if not isinstance(features, ResolvedTimeSeriesFeatures):
                msg = "Expected ResolvedTimeSeriesFeatures for TIME_SERIES problem type"
                raise TypeError(msg)
            return self._ts_engine.build_features(
                df=df,
                target=features.target,
                lookback=features.lookback,
                horizon=features.horizon,
                lag_intervals=features.lag_intervals,
                rolling_windows=features.rolling_windows,
                include_trend=features.include_trend,
                include_ratios=features.include_ratios,
                include_cross_node=features.include_cross_node,
                feature_columns=features.feature_columns or None,
                exclude_columns=features.exclude_columns or None,
            )

        # ENTITY_CLASSIFICATION
        if not isinstance(features, ResolvedEntityFeatures):
            msg = "Expected ResolvedEntityFeatures for ENTITY_CLASSIFICATION problem type"
            raise TypeError(msg)
        return self._entity_engine.build_features(
            df=df,
            target=features.target,
            observation_point=features.observation_point,
            include_entity_state=features.include_entity_state,
            include_node_state=features.include_node_state,
            include_system_state=features.include_system_state,
            add_progress_ratio=features.add_progress_ratio,
            add_wait_trend=features.add_wait_trend,
            feature_columns=features.feature_columns or None,
            exclude_columns=features.exclude_columns or None,
            sla_column=features.sla_column,
            sla_threshold=features.sla_threshold,
        )
