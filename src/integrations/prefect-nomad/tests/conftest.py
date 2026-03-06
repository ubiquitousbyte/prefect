import asyncio
import uuid
from unittest.mock import MagicMock, patch

import pytest
from anyio import to_thread

from prefect.client.schemas import FlowRun
from prefect.server.database.alembic_commands import alembic_upgrade
from prefect.testing.fixtures import *  # noqa
from prefect.testing.utilities import prefect_test_harness


@pytest.fixture(scope="session")
def test_database_connection_url() -> str | None:
    """Required by hosted_api_server from prefect.testing.fixtures."""
    return None


@pytest.fixture(scope="session", autouse=True)
def prefect_db():
    """Sets up test harness for temporary DB during test runs."""
    with prefect_test_harness():
        asyncio.run(to_thread.run_sync(alembic_upgrade))
        yield


@pytest.fixture
def flow_run():
    """Creates a minimal FlowRun for testing."""
    return FlowRun(flow_id=uuid.uuid4())


def _make_mock_nomad_client(
    *,
    job_allocations: list[dict] | None = None,
    log_output: str = "",
    evaluation_status: str = "complete",
    evaluation_failed_tg_allocs: dict | None = None,
    evaluation_next_eval: str = "",
) -> MagicMock:
    """Helper to construct a fully-mocked ``nomad.Nomad`` client.

    Args:
        job_allocations: List of allocation dicts returned by
            ``job.get_allocations()``. Defaults to a single completed
            allocation with exit code 0.
        log_output: String returned by ``client.stream_logs.stream()``.
        evaluation_status: The evaluation status to return ("complete", "failed", etc.).
        evaluation_failed_tg_allocs: Optional FailedTGAllocs dict for the evaluation.
        evaluation_next_eval: Optional NextEval UUID for evaluation chaining.
    """
    if job_allocations is None:
        job_allocations = [
            {
                "ID": "alloc-abc123",
                "ClientStatus": "complete",
                "TaskStates": {
                    "prefect-job": {
                        "State": "dead",
                        "Events": [
                            {
                                "Type": "Terminated",
                                "ExitCode": 0,
                            }
                        ],
                    }
                },
            }
        ]

    mock = MagicMock()

    # nomad.Nomad attributes used for PID construction
    mock.host = "127.0.0.1"
    mock.port = 4646
    mock.secure = False

    # Job API
    mock.job.register_job.return_value = {"EvalID": "eval-123"}
    mock.job.get_allocations.return_value = job_allocations
    mock.job.deregister_job.return_value = None

    # Evaluation API
    mock.evaluation.get_evaluation.return_value = {
        "ID": "eval-123",
        "Status": evaluation_status,
        "FailedTGAllocs": evaluation_failed_tg_allocs,
        "NextEval": evaluation_next_eval,
        "StatusDescription": "",
    }

    # Log streaming
    mock.client.stream_logs.stream.return_value = log_output

    return mock


@pytest.fixture
def mock_nomad_client():
    """A mock ``nomad.Nomad`` client that returns a completed allocation."""
    return _make_mock_nomad_client()


@pytest.fixture
def mock_nomad_client_running_then_complete():
    """A mock client that first returns running, then complete allocations.

    Useful for testing the polling loop.
    """
    running_alloc = {
        "ID": "alloc-run123",
        "ClientStatus": "running",
        "TaskStates": {
            "prefect-job": {
                "State": "running",
                "Events": [{"Type": "Started"}],
            }
        },
    }
    complete_alloc = {
        "ID": "alloc-run123",
        "ClientStatus": "complete",
        "TaskStates": {
            "prefect-job": {
                "State": "dead",
                "Events": [
                    {"Type": "Terminated", "ExitCode": 0},
                ],
            }
        },
    }

    mock = _make_mock_nomad_client()
    mock.job.get_allocations.side_effect = [
        [running_alloc],
        [complete_alloc],
    ]
    return mock


@pytest.fixture
def mock_nomad_client_failed():
    """A mock client that returns a failed allocation with exit code 1."""
    return _make_mock_nomad_client(
        job_allocations=[
            {
                "ID": "alloc-fail456",
                "ClientStatus": "failed",
                "TaskStates": {
                    "prefect-job": {
                        "State": "dead",
                        "Events": [
                            {"Type": "Terminated", "ExitCode": 1},
                        ],
                    }
                },
            }
        ]
    )


@pytest.fixture
def mock_nomad_client_eval_scheduling_failure():
    """A mock client that returns an evaluation with FailedTGAllocs."""
    return _make_mock_nomad_client(
        evaluation_status="complete",
        evaluation_failed_tg_allocs={
            "web": {
                "NodesAvailable": {"dc1": 0},
                "NodesEvaluated": 0,
                "NodesExhausted": 0,
                "NodesFiltered": 0,
            }
        },
    )


@pytest.fixture
def mock_nomad_client_eval_failed():
    """A mock client that returns a failed evaluation."""
    mock = _make_mock_nomad_client(evaluation_status="failed")
    mock.evaluation.get_evaluation.return_value["StatusDescription"] = (
        "evaluation rejected by scheduler"
    )
    return mock


@pytest.fixture
def mock_nomad_client_eval_canceled():
    """A mock client that returns a canceled evaluation."""
    return _make_mock_nomad_client(evaluation_status="canceled")


@pytest.fixture
def mock_nomad_client_eval_chained():
    """A mock client that chains evaluations via NextEval."""
    mock = _make_mock_nomad_client()
    # First call returns pending eval with NextEval, second call returns complete
    mock.evaluation.get_evaluation.side_effect = [
        {
            "ID": "eval-123",
            "Status": "pending",
            "FailedTGAllocs": None,
            "NextEval": "eval-456",
            "StatusDescription": "",
        },
        {
            "ID": "eval-456",
            "Status": "complete",
            "FailedTGAllocs": None,
            "NextEval": "",
            "StatusDescription": "",
        },
    ]
    return mock


@pytest.fixture
def patched_nomad_module():
    """Context-manager fixture that patches `nomad.Nomad` constructor.

    Usage::

        def test_something(patched_nomad_module, mock_nomad_client):
            with patched_nomad_module(mock_nomad_client):
                # nomad.Nomad() now returns mock_nomad_client
                ...
    """

    def _patch(mock_client: MagicMock):
        return patch("nomad.Nomad", return_value=mock_client)

    return _patch
