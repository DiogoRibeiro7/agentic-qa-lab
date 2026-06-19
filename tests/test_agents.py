from __future__ import annotations

import json
from pathlib import Path

from agentic_qa_lab.agents import RuleBasedAgent, Runner, write_trace_jsonl
from agentic_qa_lab.domain import (
    ActionResult,
    AgentAction,
    FailureCategory,
    Observation,
    RunStatus,
    TaskSpec,
    TraceStep,
)
from agentic_qa_lab.environments import BrowserEnvironment


class FakeEnv(BrowserEnvironment):
    """Scripted environment for runner tests.

    ``fail_actions`` names selectors/keys whose execution should fail; ``dom``
    is the DOM snapshot returned by every observation.
    """

    def __init__(
        self, *, dom: str = "<html></html>", fail_selectors: set[str] | None = None
    ) -> None:
        self._dom = dom
        self._fail = fail_selectors or set()
        self._step = 0
        self.executed: list[AgentAction] = []
        self.closed = False

    def _obs(self) -> Observation:
        return Observation(
            step=self._step,
            url="https://example.com/",
            dom_snapshot=self._dom,
            timestamp=1.0 + self._step,
        )

    def open(self, url: str) -> Observation:
        return self._obs()

    def observe(self) -> Observation:
        self._step += 1
        return self._obs()

    def execute(self, action: AgentAction) -> ActionResult:
        self.executed.append(action)
        if action.selector is not None and action.selector in self._fail:
            return ActionResult.failed("boom", category=FailureCategory.ELEMENT_NOT_FOUND)
        return ActionResult.ok()

    def close(self) -> None:
        self.closed = True


def _task(**kwargs: object) -> TaskSpec:
    base: dict[str, object] = {
        "task_id": "t1",
        "goal": "do it",
        "start_url": "https://example.com/",
    }
    base.update(kwargs)
    return TaskSpec.model_validate(base)


# --------------------------------------------------------------------------- #
# RuleBasedAgent
# --------------------------------------------------------------------------- #
def test_rule_agent_replays_plan_in_order() -> None:
    plan = [AgentAction.click("#a"), AgentAction.click("#b")]
    agent = RuleBasedAgent(plan)
    task = _task()
    obs = Observation(step=0, url="https://e.com", timestamp=1.0)

    assert agent.next_action(task, obs, []).selector == "#a"
    step = TraceStep(index=0, observation=obs, action=plan[0], result=ActionResult.ok())
    assert agent.next_action(task, obs, [step]).selector == "#b"


def test_rule_agent_finishes_when_plan_exhausted() -> None:
    agent = RuleBasedAgent([])
    action = agent.next_action(_task(), Observation(step=0, url="https://e.com", timestamp=1.0), [])
    assert action.is_terminal


def test_rule_agent_finishes_early_on_success_selector() -> None:
    agent = RuleBasedAgent([AgentAction.click("#a")])
    task = _task(success_selector="Welcome")
    obs = Observation(step=0, url="https://e.com", dom_snapshot="<p>Welcome</p>", timestamp=1.0)
    assert agent.next_action(task, obs, []).is_terminal


# --------------------------------------------------------------------------- #
# Runner — success
# --------------------------------------------------------------------------- #
def test_runner_completes_successful_task() -> None:
    plan = [AgentAction.click("#a"), AgentAction.finish("done")]
    env = FakeEnv()
    run = Runner().run(_task(), RuleBasedAgent(plan), env)

    assert run.status is RunStatus.SUCCESS
    assert run.succeeded
    assert run.step_count == 2
    assert env.closed is False  # Runner does not own env lifecycle


def test_runner_success_requires_selector_when_set() -> None:
    plan = [AgentAction.finish("done")]
    env = FakeEnv(dom="<html>no marker</html>")
    run = Runner().run(_task(success_selector="Welcome"), RuleBasedAgent(plan), env)

    assert run.status is RunStatus.FAILURE


def test_runner_success_with_selector_present() -> None:
    plan = [AgentAction.click("#a")]
    env = FakeEnv(dom="<p>Welcome</p>")
    # Agent will finish early because the selector is visible.
    run = Runner().run(_task(success_selector="Welcome"), RuleBasedAgent(plan), env)
    assert run.status is RunStatus.SUCCESS


# --------------------------------------------------------------------------- #
# Runner — failure / safeguards
# --------------------------------------------------------------------------- #
def test_runner_retries_then_fails() -> None:
    plan = [AgentAction.click("#bad")]
    env = FakeEnv(fail_selectors={"#bad"})
    run = Runner().run(_task(max_retries=2), RuleBasedAgent(plan), env)

    assert run.status is RunStatus.FAILURE
    assert run.failure_category is FailureCategory.ELEMENT_NOT_FOUND
    # 1 initial attempt + 2 retries = 3 executions of the failing action.
    assert len(env.executed) == 3
    assert run.total_retries == 2


def test_runner_hits_max_steps() -> None:
    # A plan that never finishes within the step budget.
    plan = [AgentAction.click("#a")] * 10
    env = FakeEnv()
    run = Runner().run(_task(max_steps=3), RuleBasedAgent(plan), env)

    assert run.status is RunStatus.MAX_STEPS
    assert run.step_count == 3


def test_runner_times_out() -> None:
    clock = iter([0.0, 0.0, 100.0, 100.0])  # start, loop-check (over budget), end

    def fake_mono() -> float:
        return next(clock)

    env = FakeEnv()
    runner = Runner(monotonic=fake_mono, wall_clock=lambda: 5.0)
    run = runner.run(_task(timeout_seconds=10.0), RuleBasedAgent([AgentAction.click("#a")]), env)

    assert run.status is RunStatus.TIMEOUT
    assert run.failure_category is FailureCategory.TIMEOUT


def test_runner_handles_agent_exception() -> None:
    class Boom:
        def next_action(
            self, task: TaskSpec, observation: Observation, trace: list[TraceStep]
        ) -> AgentAction:
            raise RuntimeError("kaboom")

    run = Runner().run(_task(), Boom(), FakeEnv())
    assert run.status is RunStatus.ERROR
    assert run.failure_category is FailureCategory.AGENT_ERROR


# --------------------------------------------------------------------------- #
# JSONL trace output
# --------------------------------------------------------------------------- #
def test_write_trace_jsonl(tmp_path: Path) -> None:
    plan = [AgentAction.click("#a"), AgentAction.finish("done")]
    run = Runner().run(_task(), RuleBasedAgent(plan), FakeEnv())

    out = write_trace_jsonl(run, tmp_path / "traces" / "t1.jsonl")
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]

    assert [r["record"] for r in records] == ["step", "step", "summary"]
    assert records[-1]["status"] == "success"
    assert records[-1]["step_count"] == 2
