from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agentic_qa_lab import cli
from agentic_qa_lab.agents import LLMPlannerAgent, ObservationMode, RuleBasedAgent
from agentic_qa_lab.cli import AgentKind, app, build_agent
from agentic_qa_lab.domain import ActionResult, AgentAction, Observation, TaskSpec
from agentic_qa_lab.environments import BrowserEnvironment, PlaywrightEnvironment
from agentic_qa_lab.evaluation import BenchmarkCase

runner = CliRunner()


class FakeEnv(BrowserEnvironment):
    """Headless-free env: every action succeeds; DOM shows the success marker."""

    def __init__(self) -> None:
        self._step = 0

    def _obs(self) -> Observation:
        return Observation(
            step=self._step,
            url="https://e.com/",
            dom_snapshot="<html>Welcome</html>",
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
        pass


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        task=TaskSpec(task_id="t", goal="g", start_url="https://e.com/"),
        plan=[AgentAction.finish("done")],
    )


# --------------------------------------------------------------------------- #
# build_agent
# --------------------------------------------------------------------------- #
def test_build_rule_agent() -> None:
    agent = build_agent(_case(), AgentKind.RULE, ObservationMode.DOM_ONLY)
    assert isinstance(agent, RuleBasedAgent)


def test_build_llm_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    agent = build_agent(_case(), AgentKind.LLM, ObservationMode.COMBINED)
    assert isinstance(agent, LLMPlannerAgent)


# --------------------------------------------------------------------------- #
# `run` command (browser launch monkeypatched)
# --------------------------------------------------------------------------- #
@pytest.fixture
def task_file(tmp_path: Path) -> Path:
    path = tmp_path / "task.yaml"
    path.write_text(
        "task_id: cli-demo\n"
        "goal: finish quickly\n"
        "start_url: https://e.com/\n"
        "plan:\n"
        "  - {type: finish, reason: done}\n",
        encoding="utf-8",
    )
    return path


def _patch_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_launch(**_kwargs: Any) -> FakeEnv:
        return FakeEnv()

    monkeypatch.setattr(PlaywrightEnvironment, "launch", staticmethod(fake_launch))


def test_run_command_writes_trace(
    monkeypatch: pytest.MonkeyPatch, task_file: Path, tmp_path: Path
) -> None:
    _patch_launch(monkeypatch)
    out_dir = tmp_path / "out"

    result = runner.invoke(app, ["run", "--task", str(task_file), "--out-dir", str(out_dir)])

    assert result.exit_code == 0, result.output
    assert "success" in result.output
    trace = out_dir / "cli-demo.jsonl"
    assert trace.exists()
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["record"] == "summary"
    assert records[-1]["status"] == "success"


def test_run_command_exits_nonzero_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A task whose success_selector is never satisfied -> finish resolves to failure.
    path = tmp_path / "task.yaml"
    path.write_text(
        "task_id: doomed\n"
        "goal: never succeeds\n"
        "start_url: https://e.com/\n"
        "success_selector: NEVER_PRESENT\n"
        "plan:\n"
        "  - {type: finish, reason: premature}\n",
        encoding="utf-8",
    )
    _patch_launch(monkeypatch)

    result = runner.invoke(app, ["run", "--task", str(path), "--out-dir", str(tmp_path / "o")])
    assert result.exit_code == 1


def test_run_command_can_enable_success_judge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "task.yaml"
    path.write_text(
        "task_id: judged\n"
        "goal: semantic success\n"
        "start_url: https://e.com/\n"
        "success_judge: page shows success semantically\n"
        "plan:\n"
        "  - {type: finish, reason: done}\n",
        encoding="utf-8",
    )
    _patch_launch(monkeypatch)

    class AlwaysPassJudge:
        def evaluate(self, task: TaskSpec, observation: Observation, trace: list[object]) -> object:
            from agentic_qa_lab.agents import JudgeVerdict

            return JudgeVerdict(success=True, reason="semantic success")

    monkeypatch.setattr(
        cli,
        "build_success_judge",
        lambda *, enabled, meter=None: AlwaysPassJudge(),
    )
    result = runner.invoke(
        app,
        ["run", "--task", str(path), "--judge-success", "--out-dir", str(tmp_path / "o")],
    )

    assert result.exit_code == 0, result.output
    assert "success" in result.output


def test_run_require_approval_denied_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Plan performs a risky click; declining approval must stop the run (exit 1).
    path = tmp_path / "task.yaml"
    path.write_text(
        "task_id: risky\n"
        "goal: delete something\n"
        "start_url: https://e.com/\n"
        "plan:\n"
        "  - {type: click, selector: '#delete-account'}\n"
        "  - {type: finish, reason: done}\n",
        encoding="utf-8",
    )
    _patch_launch(monkeypatch)

    result = runner.invoke(
        app,
        ["run", "--task", str(path), "--require-approval", "--out-dir", str(tmp_path / "o")],
        input="n\n",
    )
    assert result.exit_code == 1


def test_run_require_approval_confirmed_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "task.yaml"
    path.write_text(
        "task_id: risky-ok\n"
        "goal: delete something\n"
        "start_url: https://e.com/\n"
        "plan:\n"
        "  - {type: click, selector: '#delete-account'}\n"
        "  - {type: finish, reason: done}\n",
        encoding="utf-8",
    )
    _patch_launch(monkeypatch)

    result = runner.invoke(
        app,
        ["run", "--task", str(path), "--require-approval", "--out-dir", str(tmp_path / "o")],
        input="y\n",
    )
    assert result.exit_code == 0, result.output
    assert "success" in result.output


def test_run_require_approval_approve_all_reuses_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "task.yaml"
    path.write_text(
        "task_id: risky-twice\n"
        "goal: delete twice\n"
        "start_url: https://e.com/\n"
        "plan:\n"
        "  - {type: click, selector: '#delete-account'}\n"
        "  - {type: click, selector: '#submit-payment'}\n"
        "  - {type: finish, reason: done}\n",
        encoding="utf-8",
    )
    _patch_launch(monkeypatch)

    result = runner.invoke(
        app,
        ["run", "--task", str(path), "--require-approval", "--out-dir", str(tmp_path / "o")],
        input="a\n",
    )
    assert result.exit_code == 0, result.output
    assert result.output.count("Approve risky action") == 1


def test_run_help_lists_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "benchmark" in result.output
    assert "record" in result.output


def test_record_command_writes_task_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out_file = tmp_path / "recorded.yaml"

    def fake_record_case(
        task: TaskSpec,
        *,
        finish_reason: str,
        headless: bool,
        wait_for_finish: Any,
    ) -> BenchmarkCase:
        assert task.task_id == "rec-demo"
        assert finish_reason == "captured"
        assert headless is False
        assert wait_for_finish is not None
        return BenchmarkCase(
            task=task,
            plan=[AgentAction.click("#submit"), AgentAction.finish(finish_reason)],
        )

    monkeypatch.setattr("agentic_qa_lab.evaluation.record_case", fake_record_case)
    result = runner.invoke(
        app,
        [
            "record",
            "--task-id",
            "rec-demo",
            "--goal",
            "Submit the form",
            "--start-url",
            "https://e.com/",
            "--out-file",
            str(out_file),
            "--finish-reason",
            "captured",
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_file.exists()
    text = out_file.read_text(encoding="utf-8")
    assert "task_id: rec-demo" in text
    assert "selector: '#submit'" in text or 'selector: "#submit"' in text


def test_cli_module_has_app() -> None:
    assert hasattr(cli, "app")
