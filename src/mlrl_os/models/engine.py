"""Model training engine — train, cross-validate, and select the best algorithm."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from mlrl_os.core.experiment import AlgorithmScore, FeatureImportanceEntry
from mlrl_os.core.types import CVStrategy, FeatureMatrix, TaskType
from mlrl_os.evaluation.metrics import compute_metrics
from mlrl_os.experiment.seed import seed_hash
from mlrl_os.models.algorithms.protocol import Algorithm, TrainedModel
from mlrl_os.models.algorithms.registry import AlgorithmRegistry, default_registry
from mlrl_os.models.tuning import DEFAULT_N_TRIALS, has_search_space, suggest_params

logger = logging.getLogger(__name__)

# Metrics where lower is better (regression error metrics).
_MINIMIZE_METRICS: frozenset[str] = frozenset({"rmse", "mae", "mape"})


@dataclass
class TrainingResult:
    """Result of training/evaluating multiple algorithms."""

    best_algorithm: str
    best_model: TrainedModel
    best_metrics: dict[str, float]
    all_scores: list[AlgorithmScore]
    feature_importance: list[FeatureImportanceEntry]
    y_pred: np.ndarray  # predictions from best model on full data
    y_proba: np.ndarray | None  # class probabilities if classification


class ModelEngine:
    """Orchestrates training and evaluation of multiple algorithms.

    For each requested algorithm the engine runs cross-validation, ranks
    algorithms by the primary metric (first in *metric_names*), retrains
    the best algorithm on the full dataset, and returns consolidated
    results including feature importance.
    """

    def __init__(self, registry: AlgorithmRegistry | None = None) -> None:
        self._registry = registry or default_registry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_and_evaluate(
        self,
        feature_matrix: FeatureMatrix,
        algorithm_names: list[str],
        metric_names: list[str],
        cv_strategy: CVStrategy,
        cv_folds: int,
        seed: int,
        selection: str = "best_cv",
        handle_imbalance: bool = False,
        hyperparameter_tuning: bool = False,
    ) -> TrainingResult:
        """Train all algorithms with CV, select best, retrain on full data.

        Args:
            feature_matrix: Model-ready data (X, y, metadata).
            algorithm_names: Which algorithms to evaluate.
            metric_names: Metrics to compute; the first is used for selection.
            cv_strategy: Cross-validation splitting strategy.
            cv_folds: Number of CV folds.
            seed: Global random seed.
            selection: Selection criterion (currently only ``"best_cv"``).
            handle_imbalance: If *True*, pass ``handle_imbalance=True`` as a
                keyword argument to each algorithm's ``train`` method.
            hyperparameter_tuning: If *True*, run a randomized search over
                per-algorithm hyperparameter spaces before final training.

        Returns:
            A :class:`TrainingResult` with the best model, predictions, and
            per-algorithm scores.

        Raises:
            RuntimeError: If every algorithm fails during cross-validation.
        """
        X = feature_matrix.X
        y = feature_matrix.y
        task = feature_matrix.task_type.value  # "regression" | "classification"
        primary_metric = metric_names[0]

        # ---- cross-validate each algorithm (with optional tuning) ----
        algo_results: list[
            tuple[str, Algorithm, dict[str, float], dict[str, float], dict[str, Any]]
        ] = []

        for name in algorithm_names:
            try:
                algorithm = self._registry.get(name)
            except KeyError:
                logger.warning("Algorithm %r not found in registry — skipping.", name)
                continue

            # Guard: does the algorithm support this task?
            if task == TaskType.REGRESSION.value and not algorithm.supports_regression:
                logger.warning(
                    "Algorithm %r does not support regression — skipping.", name
                )
                continue
            if task == TaskType.CLASSIFICATION.value and not algorithm.supports_classification:
                logger.warning(
                    "Algorithm %r does not support classification — skipping.", name
                )
                continue

            try:
                if hyperparameter_tuning:
                    mean_metrics, std_metrics, best_params = self._tune_and_validate(
                        algorithm=algorithm,
                        X=X,
                        y=y,
                        task=task,
                        metric_names=metric_names,
                        cv_strategy=cv_strategy,
                        cv_folds=cv_folds,
                        seed=seed,
                        handle_imbalance=handle_imbalance,
                    )
                else:
                    mean_metrics, std_metrics = self._cross_validate(
                        algorithm=algorithm,
                        X=X,
                        y=y,
                        task=task,
                        metric_names=metric_names,
                        cv_strategy=cv_strategy,
                        cv_folds=cv_folds,
                        seed=seed,
                        handle_imbalance=handle_imbalance,
                    )
                    best_params = {}
                algo_results.append(
                    (name, algorithm, mean_metrics, std_metrics, best_params)
                )
            except Exception:
                logger.warning(
                    "Algorithm %r failed during cross-validation — skipping.",
                    name,
                    exc_info=True,
                )

        if not algo_results:
            msg = "All algorithms failed during cross-validation."
            raise RuntimeError(msg)

        # ---- rank and select best ----
        minimize = primary_metric in _MINIMIZE_METRICS
        algo_results.sort(
            key=lambda t: t[2][primary_metric],
            reverse=not minimize,
        )

        all_scores: list[AlgorithmScore] = []
        for rank_idx, (name, _algo, mean_m, std_m, _params) in enumerate(
            algo_results, start=1
        ):
            all_scores.append(
                AlgorithmScore(
                    algorithm=name,
                    metrics=mean_m,
                    metrics_std=std_m,
                    rank=rank_idx,
                )
            )

        best_name, best_algo, best_cv_metrics, _, tuned_params = algo_results[0]

        # ---- retrain best on full data ----
        algo_seed = seed_hash(f"{best_name}_train", seed)
        train_kwargs: dict[str, object] = {**tuned_params}
        if handle_imbalance:
            train_kwargs["handle_imbalance"] = True

        best_model = best_algo.train(
            X,
            y,
            task=task,
            seed=algo_seed,
            feature_names=feature_matrix.feature_names,
            **train_kwargs,
        )

        # ---- predictions on full data ----
        y_pred = best_algo.predict(best_model, X)
        y_proba: np.ndarray | None = None
        if task == TaskType.CLASSIFICATION.value:
            y_proba = best_algo.predict_proba(best_model, X)

        # ---- full-data metrics ----
        best_metrics = compute_metrics(metric_names, y, y_pred, y_proba)

        # ---- feature importance ----
        importance_entries = self._extract_feature_importance(best_algo, best_model)

        return TrainingResult(
            best_algorithm=best_name,
            best_model=best_model,
            best_metrics=best_metrics,
            all_scores=all_scores,
            feature_importance=importance_entries,
            y_pred=y_pred,
            y_proba=y_proba,
        )

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def _cross_validate(
        self,
        algorithm: Algorithm,
        X: np.ndarray,
        y: np.ndarray,
        task: str,
        metric_names: list[str],
        cv_strategy: CVStrategy,
        cv_folds: int,
        seed: int,
        handle_imbalance: bool = False,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Run CV for one algorithm, return (mean_metrics, std_metrics)."""
        splitter = self._make_splitter(cv_strategy, cv_folds, seed)

        fold_metrics: dict[str, list[float]] = {m: [] for m in metric_names}
        algo_seed = seed_hash(f"{algorithm.name}_train", seed)

        train_kwargs: dict[str, object] = {}
        if handle_imbalance:
            train_kwargs["handle_imbalance"] = True

        split_iter = (
            splitter.split(X, y)
            if cv_strategy == CVStrategy.STRATIFIED_KFOLD
            else splitter.split(X)
        )

        for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_seed = seed_hash(f"{algorithm.name}_fold{fold_idx}", algo_seed)

            model = algorithm.train(
                X_train,
                y_train,
                task=task,
                seed=fold_seed,
                **train_kwargs,
            )

            y_pred = algorithm.predict(model, X_val)
            y_proba: np.ndarray | None = None
            if task == TaskType.CLASSIFICATION.value:
                y_proba = algorithm.predict_proba(model, X_val)

            scores = compute_metrics(metric_names, y_val, y_pred, y_proba)
            for m_name, value in scores.items():
                fold_metrics[m_name].append(value)

        mean_metrics = {m: float(np.mean(vals)) for m, vals in fold_metrics.items()}
        std_metrics = {m: float(np.std(vals)) for m, vals in fold_metrics.items()}
        return mean_metrics, std_metrics

    # ------------------------------------------------------------------
    # Hyperparameter tuning (Optuna)
    # ------------------------------------------------------------------

    def _tune_and_validate(
        self,
        algorithm: Algorithm,
        X: np.ndarray,
        y: np.ndarray,
        task: str,
        metric_names: list[str],
        cv_strategy: CVStrategy,
        cv_folds: int,
        seed: int,
        handle_imbalance: bool = False,
        n_trials: int = DEFAULT_N_TRIALS,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
        """Run Optuna hyperparameter search with CV, return best metrics and params.

        Args:
            algorithm: Algorithm instance.
            X, y: Training data.
            task: "regression" or "classification".
            metric_names: Metrics to compute; first is used for optimization.
            cv_strategy: CV splitting strategy.
            cv_folds: Number of CV folds.
            seed: Global seed.
            handle_imbalance: Pass through to algorithm.
            n_trials: Number of Optuna trials to run.

        Returns:
            Tuple of (mean_metrics, std_metrics, best_params).
        """
        import optuna

        # If no search space for this algorithm, fall back to default CV
        if not has_search_space(algorithm.name):
            mean_m, std_m = self._cross_validate(
                algorithm, X, y, task, metric_names,
                cv_strategy, cv_folds, seed, handle_imbalance,
            )
            return mean_m, std_m, {}

        primary_metric = metric_names[0]
        minimize = primary_metric in _MINIMIZE_METRICS
        direction = "minimize" if minimize else "maximize"

        # Silence Optuna's verbose logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=seed_hash(f"{algorithm.name}_optuna", seed))
        study = optuna.create_study(direction=direction, sampler=sampler)

        def objective(trial: optuna.trial.Trial) -> float:
            params = suggest_params(trial, algorithm.name, task)

            # Run CV with these params
            splitter = self._make_splitter(cv_strategy, cv_folds, seed)
            algo_seed = seed_hash(f"{algorithm.name}_train", seed)

            train_kwargs: dict[str, object] = {**params}
            if handle_imbalance:
                train_kwargs["handle_imbalance"] = True

            split_iter = (
                splitter.split(X, y)
                if cv_strategy == CVStrategy.STRATIFIED_KFOLD
                else splitter.split(X)
            )

            fold_scores: list[float] = []
            for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                fold_seed = seed_hash(f"{algorithm.name}_fold{fold_idx}", algo_seed)

                model = algorithm.train(
                    X_train, y_train, task=task, seed=fold_seed, **train_kwargs,
                )
                y_pred = algorithm.predict(model, X_val)
                y_proba = None
                if task == TaskType.CLASSIFICATION.value:
                    y_proba = algorithm.predict_proba(model, X_val)

                scores = compute_metrics([primary_metric], y_val, y_pred, y_proba)
                fold_scores.append(scores[primary_metric])

                # Optuna pruning: report intermediate value
                trial.report(float(np.mean(fold_scores)), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(fold_scores))

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_trial.params
        logger.info(
            "Optuna tuning for %s: best %s=%.4f in %d trials",
            algorithm.name, primary_metric, study.best_value, len(study.trials),
        )

        # Re-run full CV with best params to get all metrics + std
        mean_metrics, std_metrics = self._cross_validate_with_params(
            algorithm, X, y, task, metric_names,
            cv_strategy, cv_folds, seed, handle_imbalance, best_params,
        )

        return mean_metrics, std_metrics, best_params

    def _cross_validate_with_params(
        self,
        algorithm: Algorithm,
        X: np.ndarray,
        y: np.ndarray,
        task: str,
        metric_names: list[str],
        cv_strategy: CVStrategy,
        cv_folds: int,
        seed: int,
        handle_imbalance: bool,
        extra_params: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Run CV with specific hyperparameters, return (mean_metrics, std_metrics)."""
        splitter = self._make_splitter(cv_strategy, cv_folds, seed)

        fold_metrics: dict[str, list[float]] = {m: [] for m in metric_names}
        algo_seed = seed_hash(f"{algorithm.name}_train", seed)

        train_kwargs: dict[str, object] = {**extra_params}
        if handle_imbalance:
            train_kwargs["handle_imbalance"] = True

        split_iter = (
            splitter.split(X, y)
            if cv_strategy == CVStrategy.STRATIFIED_KFOLD
            else splitter.split(X)
        )

        for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_seed = seed_hash(f"{algorithm.name}_fold{fold_idx}", algo_seed)
            model = algorithm.train(
                X_train, y_train, task=task, seed=fold_seed, **train_kwargs,
            )

            y_pred = algorithm.predict(model, X_val)
            y_proba = None
            if task == TaskType.CLASSIFICATION.value:
                y_proba = algorithm.predict_proba(model, X_val)

            scores = compute_metrics(metric_names, y_val, y_pred, y_proba)
            for m_name, value in scores.items():
                fold_metrics[m_name].append(value)

        mean_metrics = {m: float(np.mean(vals)) for m, vals in fold_metrics.items()}
        std_metrics = {m: float(np.std(vals)) for m, vals in fold_metrics.items()}
        return mean_metrics, std_metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_splitter(
        cv_strategy: CVStrategy,
        cv_folds: int,
        seed: int,
    ) -> TimeSeriesSplit | StratifiedKFold | KFold:
        """Create the appropriate scikit-learn CV splitter."""
        if cv_strategy == CVStrategy.TEMPORAL:
            return TimeSeriesSplit(n_splits=cv_folds)
        if cv_strategy == CVStrategy.STRATIFIED_KFOLD:
            return StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=seed
            )
        # kfold (default)
        return KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    @staticmethod
    def _extract_feature_importance(
        algorithm: Algorithm,
        model: TrainedModel,
    ) -> list[FeatureImportanceEntry]:
        """Extract, normalise, and rank feature importance from the best model."""
        raw = algorithm.feature_importance(model)
        if not raw:
            return []

        total = sum(abs(v) for v in raw.values())
        if total == 0:
            return []

        normalized = {k: abs(v) / total for k, v in raw.items()}

        # Sort descending by importance
        sorted_features = sorted(normalized.items(), key=lambda kv: kv[1], reverse=True)

        return [
            FeatureImportanceEntry(feature=name, importance=imp, rank=rank)
            for rank, (name, imp) in enumerate(sorted_features, start=1)
        ]
