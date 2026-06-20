from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from agentic_qa_lab.agents import RuleBasedAgent
from agentic_qa_lab.domain import (
    ActionResult,
    AgentAction,
    FailureCategory,
    Observation,
    RunResult,
    RunStatus,
    TaskSpec,
)
from agentic_qa_lab.environments import BrowserEnvironment
from agentic_qa_lab.evaluation import (
    BenchmarkCase,
    BenchmarkRunner,
    compute_summary,
    export_results,
    load_case,
    load_cases,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeEnv(BrowserEnvironment):
    def __init__(self, *, dom: str = "<html>Welcome</html>") -> None:
        self._dom = dom
        self._step = 0
        self.closed = False

    def _obs(self) -> Observation:
        return Observation(
            step=self._step,
            url="https://e.com/",
            dom_snapshot=self._dom,
            timestamp=1.0 + self._step,
        )

    def open(self, url: str) -> Observation:
        return self._obs()

    def observe(self) -> Observation:
        self._step += 1
        return self._obs()

    def execute(self, action: AgentAction) -> ActionResult:
        return ActionResult.ok()

    def close(self) -> None:
        self.closed = True


def _run(
    status: RunStatus, *, category: FailureCategory, steps: int, retries: int = 0
) -> RunResult:
    return RunResult(
        task_id="t",
        status=status,
        failure_category=category,
        steps=[],
        total_retries=retries,
        duration_seconds=float(steps),
        started_at=1.0,
        ended_at=1.0 + steps,
    )


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_compute_summary_empty() -> None:
    summary = compute_summary([])
    assert summary.total == 0
    assert summary.success_rate == 0.0


def test_compute_summary_mixed() -> None:
    results = [
        _run(RunStatus.SUCCESS, category=FailureCategory.NONE, steps=2),
        _run(RunStatus.SUCCESS, category=FailureCategory.NONE, steps=4),
        _run(RunStatus.TIMEOUT, category=FailureCategory.TIMEOUT, steps=6, retries=3),
        _run(RunStatus.FAILURE, category=FailureCategory.ELEMENT_NOT_FOUND, steps=0),
    ]
    summary = compute_summary(results)

    assert summary.total == 4
    assert summary.successes == 2
    assert summary.success_rate == 0.5
    assert summary.timeout_rate == 0.25
    assert summary.total_retries == 3
    assert summary.failure_categories["timeout"] == 1
    assert summary.failure_categories["none"] == 2


# --------------------------------------------------------------------------- #
# task loading
# --------------------------------------------------------------------------- #
def test_load_yaml_case(tmp_path: Path) -> None:
    path = tmp_path / "t.yaml"
    path.write_text(
        "task_id: a\n"
        "goal: do\n"
        "start_url: https://e.com/\n"
        "plan:\n"
        "  - {type: click, selector: '#go'}\n"
        "  - {type: finish}\n",
        encoding="utf-8",
    )
    case = load_case(path)
    assert case.task.task_id == "a"
    assert len(case.plan) == 2
    assert case.plan[0].selector == "#go"


def test_load_json_case(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps(
            {
                "task_id": "b",
                "goal": "do",
                "start_url": "https://e.com/",
                "plan": [{"type": "finish"}],
            }
        ),
        encoding="utf-8",
    )
    case = load_case(path)
    assert case.task.task_id == "b"
    assert case.plan[0].type.value == "finish"


def test_load_case_resolves_env_text_refs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTIC_QA_TEST_PASSWORD", "s3cret")
    path = tmp_path / "secret.yaml"
    path.write_text(
        "task_id: s\n"
        "goal: log in\n"
        "start_url: https://e.com/\n"
        "plan:\n"
        "  - type: type_text\n"
        "    selector: '#password'\n"
        "    text: {env: AGENTIC_QA_TEST_PASSWORD}\n",
        encoding="utf-8",
    )

    case = load_case(path)

    assert case.plan[0].text == "s3cret"


def test_load_case_raises_on_missing_env_text_ref(tmp_path: Path) -> None:
    path = tmp_path / "secret.yaml"
    path.write_text(
        "task_id: s\n"
        "goal: log in\n"
        "start_url: https://e.com/\n"
        "plan:\n"
        "  - type: type_text\n"
        "    selector: '#password'\n"
        "    text: {env: AGENTIC_QA_MISSING_SECRET}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="AGENTIC_QA_MISSING_SECRET"):
        load_case(path)


def test_load_case_resolves_nested_env_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTIC_QA_TEST_TOKEN", "token-123")
    path = tmp_path / "api.yaml"
    path.write_text(
        "task_id: api\n"
        "goal: call api\n"
        "start_url: https://api.example.com/\n"
        "plan:\n"
        "  - type: type_text\n"
        "    selector: '#header:authorization'\n"
        "    text: {env: AGENTIC_QA_TEST_TOKEN}\n"
        "  - type: click\n"
        "    selector: '#send'\n",
        encoding="utf-8",
    )

    case = load_case(path)

    assert case.plan[0].text == "token-123"
    assert case.plan[1].selector == "#send"


def test_load_case_rejects_invalid_env_ref_name(tmp_path: Path) -> None:
    path = tmp_path / "secret.yaml"
    path.write_text(
        "task_id: s\n"
        "goal: log in\n"
        "start_url: https://e.com/\n"
        "plan:\n"
        "  - type: type_text\n"
        "    selector: '#password'\n"
        "    text: {env: ''}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-empty environment variable name"):
        load_case(path)


def test_load_case_rejects_non_mapping_top_level(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a mapping"):
        load_case(path)


def test_load_case_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("task_id: nope\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported task file extension"):
        load_case(path)


def test_load_cases_sorted_and_deduped(tmp_path: Path) -> None:
    (tmp_path / "z.yaml").write_text(
        "task_id: z\ngoal: g\nstart_url: https://e.com/\n", encoding="utf-8"
    )
    (tmp_path / "a.yaml").write_text(
        "task_id: a\ngoal: g\nstart_url: https://e.com/\n", encoding="utf-8"
    )
    cases = load_cases([str(tmp_path / "*.yaml")])
    assert [c.task.task_id for c in cases] == ["a", "z"]


def test_load_cases_accepts_explicit_file_paths(tmp_path: Path) -> None:
    path = tmp_path / "single.yaml"
    path.write_text("task_id: one\ngoal: g\nstart_url: https://e.com/\n", encoding="utf-8")

    cases = load_cases([str(path)])

    assert [c.task.task_id for c in cases] == ["one"]


def test_load_cases_skips_missing_patterns(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text(
        "task_id: a\ngoal: g\nstart_url: https://e.com/\n", encoding="utf-8"
    )

    cases = load_cases([str(tmp_path / "*.yaml"), str(tmp_path / "missing/*.yaml")])

    assert [c.task.task_id for c in cases] == ["a"]


def test_bundled_example_tasks_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTIC_QA_EXAMPLE_LOGIN_PASSWORD", "s3cret")
    cases = load_cases(["tasks/*.yaml", "tasks/*.json"])
    ids = {c.task.task_id for c in cases}
    assert {"example-login", "example-search"} <= ids


# --------------------------------------------------------------------------- #
# benchmark runner + export
# --------------------------------------------------------------------------- #
def test_benchmark_runner_runs_cases_and_closes_env() -> None:
    case = BenchmarkCase(
        task=TaskSpec(task_id="t", goal="g", start_url="https://e.com/"),
        plan=[AgentAction.finish("done")],
    )
    envs: list[FakeEnv] = []

    def make_agent(c: BenchmarkCase) -> RuleBasedAgent:
        return RuleBasedAgent(c.plan)

    def make_env(c: BenchmarkCase) -> FakeEnv:
        env = FakeEnv()
        envs.append(env)
        return env

    results = BenchmarkRunner().run([case], make_agent, make_env)
    assert len(results) == 1
    assert results[0].status is RunStatus.SUCCESS
    assert envs[0].closed is True  # env context-managed/closed


def test_benchmark_runner_rejects_non_positive_workers() -> None:
    case = BenchmarkCase(
        task=TaskSpec(task_id="t", goal="g", start_url="https://e.com/"),
        plan=[AgentAction.finish("done")],
    )

    def make_agent(c: BenchmarkCase) -> RuleBasedAgent:
        return RuleBasedAgent(c.plan)

    def make_env(c: BenchmarkCase) -> FakeEnv:
        return FakeEnv()

    try:
        BenchmarkRunner().run([case], make_agent, make_env, workers=0)
    except ValueError as exc:
        assert "workers" in str(exc)
    else:
        raise AssertionError("Expected workers=0 to raise ValueError")


def test_benchmark_runner_can_run_cases_in_parallel() -> None:
    cases = [
        BenchmarkCase(
            task=TaskSpec(task_id=f"t{i}", goal="g", start_url="https://e.com/"),
            plan=[AgentAction.finish("done")],
        )
        for i in range(2)
    ]
    started: list[str] = []
    started_lock = threading.Lock()
    release = threading.Event()

    class BlockingEnv(FakeEnv):
        def open(self, url: str) -> Observation:
            with started_lock:
                started.append(url)
                if len(started) == 2:
                    release.set()
            release.wait(timeout=1.0)
            return super().open(url)

    def make_agent(c: BenchmarkCase) -> RuleBasedAgent:
        return RuleBasedAgent(c.plan)

    def make_env(c: BenchmarkCase) -> BlockingEnv:
        return BlockingEnv()

    started_at = time.perf_counter()
    results = BenchmarkRunner().run(cases, make_agent, make_env, workers=2)
    elapsed = time.perf_counter() - started_at

    assert [result.task_id for result in results] == ["t0", "t1"]
    assert all(result.status is RunStatus.SUCCESS for result in results)
    assert len(started) == 2
    assert elapsed < 1.0


def test_export_results_writes_files(tmp_path: Path) -> None:
    results = [
        _run(RunStatus.SUCCESS, category=FailureCategory.NONE, steps=2),
        _run(RunStatus.FAILURE, category=FailureCategory.TIMEOUT, steps=1, retries=1),
    ]
    csv_path, json_path = export_results(results, tmp_path)
    junit_path = tmp_path / "junit.xml"
    allure_dir = tmp_path / "allure-results"

    assert csv_path.exists() and json_path.exists()
    assert junit_path.exists()
    assert allure_dir.is_dir()
    csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert csv_lines[0].startswith("task_id,status")
    assert len(csv_lines) == 3  # header + 2 rows

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total"] == 2
    assert len(payload["runs"]) == 2
    assert "testsuite" in junit_path.read_text(encoding="utf-8")
    allure_files = sorted(allure_dir.glob("*-result.json"))
    assert len(allure_files) == 2
