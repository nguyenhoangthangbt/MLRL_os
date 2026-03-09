"""Tests for mlrl_os.api.rl_routes — RL experiment API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from mlrl_os.api.app import create_app
from mlrl_os.config.defaults import MLRLSettings
from mlrl_os.core.experiment import ExperimentResult
from mlrl_os.core.types import ExperimentStatus, ProblemType


@pytest.fixture()
def client(tmp_path) -> TestClient:
    settings = MLRLSettings(
        data_dir=str(tmp_path / "data"),
        models_dir=str(tmp_path / "models"),
        experiments_dir=str(tmp_path / "experiments"),
    )
    app = create_app(settings)
    return TestClient(app)


class TestListRLExperiments:
    """Tests for GET /api/v1/rl/experiments."""

    def test_returns_empty_list(self, client: TestClient) -> None:
        resp = client.get("/api/v1/rl/experiments")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_filters_rl_experiments_only(self, client: TestClient) -> None:
        # Store a supervised experiment and an RL experiment
        tracker = client.app.state.experiment_tracker

        # Supervised experiment
        supervised_result = ExperimentResult(
            experiment_id="sup-001",
            name="supervised-test",
            status=ExperimentStatus.COMPLETED,
            experiment_type=ProblemType.ENTITY_CLASSIFICATION,
            created_at="2026-03-10T00:00:00Z",
        )
        tracker.create("sup-001", "supervised-test", {"problem_type": "entity_classification"})
        tracker.save_result(supervised_result)

        # RL experiment
        rl_result = ExperimentResult(
            experiment_id="rl-001",
            name="rl-test",
            status=ExperimentStatus.COMPLETED,
            experiment_type=ProblemType.REINFORCEMENT_LEARNING,
            created_at="2026-03-10T01:00:00Z",
        )
        tracker.create("rl-001", "rl-test", {"problem_type": "reinforcement_learning"})
        tracker.save_result(rl_result)

        resp = client.get("/api/v1/rl/experiments")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["experiment_id"] == "rl-001"
        assert data[0]["experiment_type"] == "reinforcement_learning"


class TestListRLPolicies:
    """Tests for GET /api/v1/rl/policies."""

    def test_returns_empty_list(self, client: TestClient) -> None:
        resp = client.get("/api/v1/rl/policies")
        assert resp.status_code == 200
        assert resp.json() == []


class TestSubmitRLExperiment:
    """Tests for POST /api/v1/rl/experiments."""

    def test_submit_returns_202(self, client: TestClient) -> None:
        mock_result = ExperimentResult(
            experiment_id="rl-submit-001",
            name="test-submit",
            status=ExperimentStatus.COMPLETED,
            experiment_type=ProblemType.REINFORCEMENT_LEARNING,
            created_at="2026-03-10T00:00:00Z",
            best_algorithm="dqn",
            metrics={"mean_reward": 10.0},
        )

        with patch("mlrl_os.rl.runner.RLExperimentRunner") as MockRunner:
            mock_runner_instance = AsyncMock()
            mock_runner_instance.run.return_value = mock_result
            MockRunner.return_value = mock_runner_instance

            resp = client.post(
                "/api/v1/rl/experiments",
                json={
                    "name": "test-submit",
                    "template": "healthcare_er",
                    "algorithm": "dqn",
                    "max_episodes": 10,
                },
            )

        assert resp.status_code == 202
        data = resp.json()
        assert data["experiment_id"] == "rl-submit-001"
        assert data["status"] == "completed"
        assert data["name"] == "test-submit"

    def test_submit_with_defaults(self, client: TestClient) -> None:
        mock_result = ExperimentResult(
            experiment_id="rl-submit-002",
            name="minimal",
            status=ExperimentStatus.COMPLETED,
            experiment_type=ProblemType.REINFORCEMENT_LEARNING,
            created_at="2026-03-10T00:00:00Z",
        )

        with patch("mlrl_os.rl.runner.RLExperimentRunner") as MockRunner:
            mock_runner_instance = AsyncMock()
            mock_runner_instance.run.return_value = mock_result
            MockRunner.return_value = mock_runner_instance

            resp = client.post(
                "/api/v1/rl/experiments",
                json={"name": "minimal"},
            )

        assert resp.status_code == 202
        assert resp.json()["name"] == "minimal"

    def test_submit_passes_config_to_runner(self, client: TestClient) -> None:
        mock_result = ExperimentResult(
            experiment_id="rl-config-check",
            name="config-check",
            status=ExperimentStatus.COMPLETED,
            experiment_type=ProblemType.REINFORCEMENT_LEARNING,
            created_at="2026-03-10T00:00:00Z",
        )

        with patch("mlrl_os.rl.runner.RLExperimentRunner") as MockRunner:
            mock_runner_instance = AsyncMock()
            mock_runner_instance.run.return_value = mock_result
            MockRunner.return_value = mock_runner_instance

            client.post(
                "/api/v1/rl/experiments",
                json={
                    "name": "config-check",
                    "algorithm": "ppo",
                    "reward_function": "sla_compliance",
                },
            )

            call_args = mock_runner_instance.run.call_args[0][0]
            assert call_args["algorithm"] == "ppo"
            assert call_args["reward"]["function"] == "sla_compliance"
            assert call_args["experiment_type"] == "reinforcement_learning"

    def test_submit_missing_name_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/rl/experiments",
            json={"algorithm": "dqn"},
        )
        assert resp.status_code == 422


class TestRLRoutesRegistered:
    """Tests that RL routes are properly registered in the app."""

    def test_rl_routes_exist(self, client: TestClient) -> None:
        routes = [route.path for route in client.app.routes]
        assert "/api/v1/rl/experiments" in routes
        assert "/api/v1/rl/policies" in routes
