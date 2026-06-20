from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agentic_qa_lab.api import RunExecutionManager, RunStore, create_app
from agentic_qa_lab.api.app import create_app as create_default_app
from agentic_qa_lab.api.execution import ExecutionAgent, RunExecutionRequest
from agentic_qa_lab.domain import (
    ActionResult,
    AgentAction,
    FailureCategory,
    Observation,
    RunResult,
    RunStatus,
    TraceStep,
)


def _run(status: RunStatus = RunStatus.SUCCESS) -> RunResult:
    step = TraceStep(
        index=0,
        observation=Observation(step=0, url="https://e.com/", timestamp=1.0),
        action=AgentAction.click("#go"),
        result=ActionResult.ok(),
    )
    category = FailureCategory.NONE if status is RunStatus.SUCCESS else FailureCategory.TIMEOUT
    return RunResult(
        task_id="t1",
        status=status,
        failure_category=category,
        steps=[step],
        started_at=1.0,
        ended_at=2.0,
        duration_seconds=1.0,
    )


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    # Deterministic, monotonically increasing run ids for stable assertions.
    counter = {"n": 0}

    def next_id() -> str:
        counter["n"] += 1
        return f"run-{counter['n']:03d}"

    store = RunStore(tmp_path / "runs", id_factory=next_id)
    with TestClient(create_app(store)) as test_client:
        yield test_client


def _await_execution(client: TestClient, execution_id: str) -> dict[str, object]:
    deadline = time.time() + 2.0
    while time.time() < deadline:
        payload = client.get(f"/executions/{execution_id}").json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"execution {execution_id} did not finish in time")


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_create_and_get_run(client: TestClient) -> None:
    resp = client.post("/runs", json=_run().model_dump(mode="json"))
    assert resp.status_code == 201
    run_id = resp.json()["run_id"]
    assert run_id == "run-001"

    got = client.get(f"/runs/{run_id}")
    assert got.status_code == 200
    assert got.json()["task_id"] == "t1"


def test_list_runs(client: TestClient) -> None:
    client.post("/runs", json=_run(RunStatus.SUCCESS).model_dump(mode="json"))
    client.post("/runs", json=_run(RunStatus.TIMEOUT).model_dump(mode="json"))

    listing = client.get("/runs").json()
    assert len(listing) == 2
    statuses = {row["status"] for row in listing}
    assert statuses == {"success", "timeout"}
    assert all("run_id" in row and "steps" in row for row in listing)


def test_get_trace(client: TestClient) -> None:
    run_id = client.post("/runs", json=_run().model_dump(mode="json")).json()["run_id"]
    trace = client.get(f"/runs/{run_id}/trace").json()

    assert len(trace) == 1
    assert trace[0]["action"]["type"] == "click"


def test_missing_run_returns_404(client: TestClient) -> None:
    assert client.get("/runs/nope").status_code == 404
    assert client.get("/runs/nope/trace").status_code == 404


def test_create_rejects_invalid_run(client: TestClient) -> None:
    # Successful status with a non-none failure category violates the model.
    bad = _run().model_dump(mode="json")
    bad["failure_category"] = "timeout"
    assert client.post("/runs", json=bad).status_code == 422


def test_create_app_uses_store_dir_from_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AGENTIC_QA_STORE_DIR", str(tmp_path / "custom-runs"))

    app = create_default_app()

    root = app.state.store._root  # noqa: SLF001 - validates settings wiring
    assert root == tmp_path / "custom-runs"
    app.state.execution_manager.close()


def test_execute_run_queues_and_stores_completed_run(tmp_path: Path) -> None:
    counter = {"n": 0}

    def next_id() -> str:
        counter["n"] += 1
        return f"run-{counter['n']:03d}"

    def fake_run_factory(_request: object) -> RunResult:
        return _run()

    store = RunStore(tmp_path / "runs", id_factory=next_id)
    execution_manager = RunExecutionManager(
        store,
        artifact_dir=tmp_path / "artifacts",
        run_factory=fake_run_factory,
        id_factory=lambda: "exec-001",
    )
    with TestClient(create_app(store, execution_manager)) as client:
        queued = client.post(
            "/runs/execute",
            json={"task_path": "tasks/example_login.yaml"},
        )
        assert queued.status_code == 202
        assert queued.json()["execution_id"] == "exec-001"

        record = _await_execution(client, "exec-001")
        assert record["status"] == "completed"
        assert record["run_id"] == "run-001"

        runs = client.get("/runs").json()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run-001"


def test_missing_execution_returns_404(client: TestClient) -> None:
    response = client.get("/executions/nope")
    assert response.status_code == 404


def test_list_executions_returns_records(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs", id_factory=lambda: "run-001")
    execution_manager = RunExecutionManager(
        store,
        artifact_dir=tmp_path / "artifacts",
        run_factory=lambda _request: _run(),
        id_factory=lambda: "exec-001",
    )
    with TestClient(create_app(store, execution_manager)) as client:
        client.post("/runs/execute", json={"task_path": "tasks/example_login.yaml"})
        _await_execution(client, "exec-001")

        payload = client.get("/executions")
        assert payload.status_code == 200
        assert payload.json()[0]["execution_id"] == "exec-001"


def test_run_store_get_trace_and_missing_run(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs", id_factory=lambda: "run-001")
    record = store.create(_run())

    trace = store.get_trace(record.run_id)

    assert len(trace) == 1
    assert trace[0].action.selector == "#go"
    with pytest.raises(KeyError):
        store.get("missing")


def test_run_store_list_skips_corrupt_and_unrelated_json(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs", id_factory=lambda: "run-001")
    store.create(_run())
    (tmp_path / "runs" / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "runs" / "other.json").write_text('{"hello":"world"}', encoding="utf-8")

    listing = store.list()

    assert len(listing) == 1
    assert listing[0].run_id == "run-001"


def test_run_store_list_is_sorted_by_run_id(tmp_path: Path) -> None:
    ids = iter(["run-010", "run-002"])
    store = RunStore(tmp_path / "runs", id_factory=lambda: next(ids))
    store.create(_run(RunStatus.SUCCESS))
    store.create(_run(RunStatus.TIMEOUT))

    listing = store.list()

    assert [row.run_id for row in listing] == ["run-002", "run-010"]


def test_execute_run_request_accepts_environment_options(tmp_path: Path) -> None:
    caps = tmp_path / "caps.json"
    request = RunExecutionRequest(
        task_path=Path("tasks/example_login.yaml"),
        agent=ExecutionAgent.LLM,
        environment="appium",
        judge_success=True,
        appium_server="http://127.0.0.1:4725",
        appium_capabilities_file=caps,
    )

    assert request.environment == "appium"
    assert request.judge_success is True
    assert request.appium_server == "http://127.0.0.1:4725"
    assert request.appium_capabilities_file == caps


def test_default_run_factory_uses_cli_environment_builder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        "task_id: api-demo\n"
        "goal: finish quickly\n"
        "start_url: https://api.example.com/\n"
        "plan:\n"
        "  - {type: type_text, selector: '#method', text: GET}\n"
        "  - {type: click, selector: '#send'}\n"
        "  - {type: finish, reason: done}\n",
        encoding="utf-8",
    )

    class FakeEnv:
        def __enter__(self) -> FakeEnv:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    class FakeRunner:
        def __init__(self, *, stop_on_action_failure: bool, success_judge: object = None) -> None:
            self.stop_on_action_failure = stop_on_action_failure
            self.success_judge = success_judge

        def run(self, task: object, agent: object, env: object) -> RunResult:
            return _run()

    seen: dict[str, object] = {}

    def fake_build_environment(
        case: object,
        kind: object,
        *,
        headless: bool = True,
        screenshot_dir: Path | None = None,
        appium_server: str = "http://127.0.0.1:4723",
        appium_capabilities_file: Path | None = None,
    ) -> FakeEnv:
        seen["kind"] = kind
        seen["headless"] = headless
        seen["screenshot_dir"] = screenshot_dir
        seen["appium_server"] = appium_server
        seen["appium_capabilities_file"] = appium_capabilities_file
        return FakeEnv()

    monkeypatch.setattr("agentic_qa_lab.cli.build_environment", fake_build_environment)
    monkeypatch.setattr(
        "agentic_qa_lab.cli.build_success_judge", lambda *, enabled, meter=None: None
    )
    monkeypatch.setattr("agentic_qa_lab.api.execution.Runner", FakeRunner)

    store = RunStore(tmp_path / "runs", id_factory=lambda: "run-001")
    manager = RunExecutionManager(store, artifact_dir=tmp_path / "artifacts")
    try:
        request = RunExecutionRequest(
            task_path=task_path,
            environment="api",
            headless=False,
            appium_server="http://127.0.0.1:4725",
            appium_capabilities_file=tmp_path / "caps.yaml",
        )
        manager._default_run_factory(request)  # noqa: SLF001 - direct unit coverage
    finally:
        manager.close()

    assert str(seen["kind"]) == "api"
    assert seen["headless"] is False
    assert seen["screenshot_dir"] is None
    assert seen["appium_server"] == "http://127.0.0.1:4725"
    assert seen["appium_capabilities_file"] == tmp_path / "caps.yaml"
