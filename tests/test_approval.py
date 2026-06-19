from __future__ import annotations

from agentic_qa_lab.agents import (
    ApprovalDecision,
    ApprovalAgent,
    ReflectiveAgent,
    RiskPolicy,
    Runner,
    allow_all,
    deny_all,
)
from agentic_qa_lab.domain import (
    ActionResult,
    ActionType,
    AgentAction,
    Observation,
    RunStatus,
    TaskSpec,
    TraceStep,
)
from agentic_qa_lab.environments import BrowserEnvironment


class StubAgent:
    """Inner agent that always proposes one fixed action."""

    def __init__(self, action: AgentAction) -> None:
        self._action = action

    def next_action(
        self, task: TaskSpec, observation: Observation, trace: list[TraceStep]
    ) -> AgentAction:
        return self._action


class RecordingEnv(BrowserEnvironment):
    """Env that records executed actions; everything succeeds."""

    def __init__(self) -> None:
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
        return ActionResult.ok()

    def close(self) -> None:
        pass


def _task() -> TaskSpec:
    return TaskSpec(task_id="t", goal="g", start_url="https://e.com/")


def _obs() -> Observation:
    return Observation(step=0, url="https://e.com/", timestamp=1.0)


# --------------------------------------------------------------------------- #
# RiskPolicy
# --------------------------------------------------------------------------- #
def test_policy_flags_keyword_targets() -> None:
    policy = RiskPolicy()
    assert policy.is_risky(AgentAction.click("#delete-account"))
    assert policy.is_risky(AgentAction.click("#submit-payment"))
    assert policy.is_risky(AgentAction.press_key("Enter", selector="#confirm"))


def test_policy_ignores_safe_actions() -> None:
    policy = RiskPolicy()
    assert not policy.is_risky(AgentAction.click("#open-menu"))
    assert not policy.is_risky(
        AgentAction.type_text("delete", selector="#search")
    )  # type_text safe
    assert not policy.is_risky(AgentAction.wait(100))


def test_policy_custom_keywords() -> None:
    policy = RiskPolicy(keywords={"nuke"})
    assert policy.is_risky(AgentAction.click("#nuke"))
    assert not policy.is_risky(AgentAction.click("#delete"))  # default keyword no longer applies


# --------------------------------------------------------------------------- #
# ApprovalAgent
# --------------------------------------------------------------------------- #
def test_safe_action_passes_without_approval() -> None:
    inner = StubAgent(AgentAction.click("#next"))
    agent = ApprovalAgent(inner, approver=deny_all)  # would block, but action is safe
    assert agent.next_action(_task(), _obs(), []).type is ActionType.CLICK


def test_risky_action_denied_becomes_fail() -> None:
    inner = StubAgent(AgentAction.click("#delete"))
    agent = ApprovalAgent(inner, approver=deny_all)
    action = agent.next_action(_task(), _obs(), [])
    assert action.type is ActionType.FAIL
    assert "not approved" in (action.reason or "")


def test_risky_action_approved_passes_through() -> None:
    risky = AgentAction.click("#delete")
    agent = ApprovalAgent(StubAgent(risky), approver=allow_all)
    assert agent.next_action(_task(), _obs(), []) == risky


def test_approver_receives_the_action() -> None:
    seen: list[AgentAction] = []

    def spy(action: AgentAction) -> bool:
        seen.append(action)
        return True

    risky = AgentAction.click("#submit")
    ApprovalAgent(StubAgent(risky), approver=spy).next_action(_task(), _obs(), [])
    assert seen == [risky]


def test_allow_session_skips_later_prompts() -> None:
    seen: list[AgentAction] = []
    risky = AgentAction.click("#delete")

    def approve_session(action: AgentAction) -> ApprovalDecision:
        seen.append(action)
        return ApprovalDecision.ALLOW_SESSION

    agent = ApprovalAgent(StubAgent(risky), approver=approve_session)
    assert agent.next_action(_task(), _obs(), []).type is ActionType.CLICK
    assert agent.next_action(_task(), _obs(), []).type is ActionType.CLICK
    assert seen == [risky]


def test_terminal_action_is_never_gated() -> None:
    agent = ApprovalAgent(StubAgent(AgentAction.finish("done")), approver=deny_all)
    assert agent.next_action(_task(), _obs(), []).type is ActionType.FINISH


def test_composes_over_reflective() -> None:
    # Approval should gate whatever the reflective wrapper ultimately proposes.
    inner = ReflectiveAgent(StubAgent(AgentAction.click("#delete")))
    agent = ApprovalAgent(inner, approver=deny_all)
    assert agent.next_action(_task(), _obs(), []).type is ActionType.FAIL


# --------------------------------------------------------------------------- #
# End-to-end through the Runner
# --------------------------------------------------------------------------- #
def test_denied_risky_action_stops_the_run() -> None:
    env = RecordingEnv()
    agent = ApprovalAgent(StubAgent(AgentAction.click("#delete")), approver=deny_all)
    run = Runner().run(_task(), agent, env)

    assert run.status is RunStatus.FAILURE
    # The risky click must never have been executed against the environment.
    assert all(a.type is not ActionType.CLICK for a in env.executed)


def test_approved_risky_action_executes() -> None:
    env = RecordingEnv()
    inner = StubAgent(AgentAction.click("#delete"))
    agent = ApprovalAgent(inner, approver=allow_all)
    Runner().run(_task(), agent, env)

    assert any(a.selector == "#delete" for a in env.executed)
