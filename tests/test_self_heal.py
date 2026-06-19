from __future__ import annotations

from agentic_qa_lab.agents import Runner, SelfHealingAgent
from agentic_qa_lab.domain import (
    ActionResult,
    ActionType,
    AgentAction,
    FailureCategory,
    Observation,
    RunStatus,
    TaskSpec,
    TraceStep,
)
from agentic_qa_lab.environments import BrowserEnvironment


class StubAgent:
    def __init__(self, action: AgentAction) -> None:
        self._action = action

    def next_action(
        self, task: TaskSpec, observation: Observation, trace: list[TraceStep]
    ) -> AgentAction:
        return self._action


class HealingEnv(BrowserEnvironment):
    """Fails one selector and succeeds on the healed replacement."""

    def __init__(self) -> None:
        self._step = 0
        self.executed: list[AgentAction] = []

    def _obs(self) -> Observation:
        return Observation(
            step=self._step,
            url="https://e.com/",
            dom_snapshot="<button id='submit'>Submit order</button>",
            timestamp=1.0 + self._step,
        )

    def open(self, url: str) -> Observation:
        return self._obs()

    def observe(self) -> Observation:
        self._step += 1
        return self._obs()

    def execute(self, action: AgentAction) -> ActionResult:
        self.executed.append(action)
        if action.selector == "#missing":
            return ActionResult.failed(
                "no node found for selector #missing",
                category=FailureCategory.ELEMENT_NOT_FOUND,
            )
        if action.selector in {"#submit", "text=Submit order", 'role=button[name="Submit order"]'}:
            return ActionResult.ok()
        return ActionResult.failed("unexpected selector", category=FailureCategory.UNKNOWN)

    def close(self) -> None:
        pass


def _task() -> TaskSpec:
    return TaskSpec(task_id="t", goal="g", start_url="https://e.com/", max_steps=6)


def _obs() -> Observation:
    return Observation(
        step=0,
        url="https://e.com/",
        dom_snapshot="<button id='submit'>Submit order</button>",
        timestamp=1.0,
    )


def _failed_step(action: AgentAction) -> TraceStep:
    return TraceStep(
        index=0,
        observation=_obs(),
        action=action,
        result=ActionResult.failed(
            "no node found for selector",
            category=FailureCategory.ELEMENT_NOT_FOUND,
        ),
    )


def test_heals_repeated_missing_selector_from_dom() -> None:
    original = AgentAction.click("#missing")
    healed = SelfHealingAgent(StubAgent(original)).next_action(_task(), _obs(), [_failed_step(original)])

    assert healed.type is ActionType.CLICK
    assert healed.selector == "#submit"
    assert "self-healed" in (healed.reason or "")


def test_skips_candidates_that_already_failed() -> None:
    original = AgentAction.click("#missing")
    trace = [
        _failed_step(original),
        TraceStep(
            index=1,
            observation=_obs(),
            action=AgentAction.click("#submit"),
            result=ActionResult.failed(
                "still missing",
                category=FailureCategory.ELEMENT_NOT_FOUND,
            ),
        ),
    ]

    healed = SelfHealingAgent(StubAgent(original)).next_action(_task(), _obs(), trace)

    assert healed.selector != "#submit"


def test_recovers_end_to_end_when_runner_allows_repair() -> None:
    env = HealingEnv()
    agent = SelfHealingAgent(StubAgent(AgentAction.click("#missing")))
    run = Runner(stop_on_action_failure=False).run(_task(), agent, env)

    assert any(action.selector == "#missing" for action in env.executed)
    assert any(action.selector == "#submit" for action in env.executed)
    assert run.status is RunStatus.MAX_STEPS
