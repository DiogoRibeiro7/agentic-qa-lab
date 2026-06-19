from __future__ import annotations

from agentic_qa_lab.agents import ReflectiveAgent, Runner
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
    """Inner agent that always proposes the same action."""

    def __init__(self, action: AgentAction) -> None:
        self._action = action

    def next_action(
        self, task: TaskSpec, observation: Observation, trace: list[TraceStep]
    ) -> AgentAction:
        return self._action


class FlakyEnv(BrowserEnvironment):
    """Fails the click target a fixed number of times, then succeeds."""

    def __init__(self, *, fail_selector: str, fail_times: int) -> None:
        self._fail_selector = fail_selector
        self._remaining = fail_times
        self._step = 0
        self.executed: list[AgentAction] = []

    def _obs(self) -> Observation:
        return Observation(
            step=self._step,
            url="https://e.com/",
            dom_snapshot="<html/>",
            timestamp=1.0 + self._step,
        )

    def open(self, url: str) -> Observation:
        return self._obs()

    def observe(self) -> Observation:
        self._step += 1
        return self._obs()

    def execute(self, action: AgentAction) -> ActionResult:
        self.executed.append(action)
        if action.selector == self._fail_selector and self._remaining > 0:
            self._remaining -= 1
            return ActionResult.failed("not ready", category=FailureCategory.ELEMENT_NOT_FOUND)
        return ActionResult.ok()

    def close(self) -> None:
        pass


def _task(**kwargs: object) -> TaskSpec:
    base: dict[str, object] = {"task_id": "t", "goal": "g", "start_url": "https://e.com/"}
    base.update(kwargs)
    return TaskSpec.model_validate(base)


def _obs() -> Observation:
    return Observation(step=0, url="https://e.com/", timestamp=1.0)


def _failed_step(action: AgentAction) -> TraceStep:
    return TraceStep(
        index=0,
        observation=_obs(),
        action=action,
        result=ActionResult.failed("x", category=FailureCategory.ELEMENT_NOT_FOUND),
    )


# --------------------------------------------------------------------------- #
# Unit behaviour
# --------------------------------------------------------------------------- #
def test_passes_through_when_no_prior_failure() -> None:
    click = AgentAction.click("#go")
    agent = ReflectiveAgent(StubAgent(click))
    assert agent.next_action(_task(), _obs(), []) == click


def test_inserts_settle_wait_after_a_failure() -> None:
    click = AgentAction.click("#go")
    agent = ReflectiveAgent(StubAgent(click), settle_ms=250)

    action = agent.next_action(_task(), _obs(), [_failed_step(click)])

    assert action.type is ActionType.WAIT
    assert action.duration_ms == 250


def test_retries_after_settling() -> None:
    click = AgentAction.click("#go")
    agent = ReflectiveAgent(StubAgent(click))
    wait_step = TraceStep(
        index=1, observation=_obs(), action=AgentAction.wait(500), result=ActionResult.ok()
    )

    # After a failure followed by a successful settle wait, retry the action.
    action = agent.next_action(_task(), _obs(), [_failed_step(click), wait_step])
    assert action.type is ActionType.CLICK


def test_gives_up_after_max_attempts() -> None:
    click = AgentAction.click("#go")
    agent = ReflectiveAgent(StubAgent(click), max_attempts=2)
    trace = [_failed_step(click), _failed_step(click)]

    action = agent.next_action(_task(), _obs(), trace)
    assert action.type is ActionType.FAIL
    assert "gave up" in (action.reason or "")


def test_terminal_inner_action_passes_through() -> None:
    finish = AgentAction.finish("done")
    agent = ReflectiveAgent(StubAgent(finish))
    assert agent.next_action(_task(), _obs(), [_failed_step(AgentAction.click("#go"))]) == finish


# --------------------------------------------------------------------------- #
# Regression: full loop through the Runner
# --------------------------------------------------------------------------- #
def test_recovers_a_flaky_action_end_to_end() -> None:
    # The click fails once, then succeeds; the wrapper should settle and recover.
    env = FlakyEnv(fail_selector="#go", fail_times=1)
    agent = ReflectiveAgent(StubAgent(AgentAction.click("#go")), max_attempts=3)
    # Let the agent own recovery: the Runner must not stop on the first failure.
    runner = Runner(stop_on_action_failure=False)
    # Inner never finishes; cap steps so a recovered click still terminates the run.
    run = runner.run(_task(max_steps=6, max_retries=0), agent, env)

    # A settle wait must have been issued between the failed and successful click.
    assert any(a.type is ActionType.WAIT for a in env.executed)
    assert any(a.type is ActionType.CLICK for a in env.executed)
    # With retries disabled at the Runner level, recovery comes from the wrapper.
    assert run.status is RunStatus.MAX_STEPS  # never finishes, but no early failure
    assert run.failure_category is FailureCategory.MAX_STEPS_EXCEEDED


def test_gives_up_end_to_end_when_always_failing() -> None:
    env = FlakyEnv(fail_selector="#go", fail_times=99)
    agent = ReflectiveAgent(StubAgent(AgentAction.click("#go")), max_attempts=2)
    run = Runner().run(_task(max_steps=10, max_retries=0), agent, env)

    assert run.status is RunStatus.FAILURE
