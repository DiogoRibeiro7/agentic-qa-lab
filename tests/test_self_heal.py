from __future__ import annotations

from agentic_qa_lab.agents import Runner, SelfHealingAgent
from agentic_qa_lab.agents.self_heal import (
    _attr,
    _candidate_selectors,
    _role_selector,
    _same_shape,
    _selector_terms,
    _text_selector,
)
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
    healed = SelfHealingAgent(StubAgent(original)).next_action(
        _task(), _obs(), [_failed_step(original)]
    )

    assert healed.type is ActionType.CLICK
    assert healed.selector == "#submit"
    assert "self-healed" in (healed.reason or "")


def test_same_shape_ignores_selector_only() -> None:
    assert _same_shape(
        AgentAction.click("#a"),
        AgentAction.click("#b"),
    )
    assert not _same_shape(
        AgentAction.type_text("alice", selector="#a"),
        AgentAction.type_text("bob", selector="#b"),
    )


def test_selector_terms_filters_noise() -> None:
    terms = _selector_terms('css=#submit-button text="Save" role=button')

    assert "submit-button" in terms
    assert "save" in terms
    assert "css" not in terms
    assert "role" not in terms


def test_attr_text_and_role_helpers() -> None:
    attrs = 'id="submit" aria-label="Save now"'

    assert _attr(attrs, "id") == "submit"
    assert _attr(attrs, "missing") is None
    assert _text_selector(" Save\n now ") == "text=Save now"
    assert _text_selector("   ") is None
    assert _role_selector("button", " Save\n now ") == 'role=button[name="Save now"]'
    assert _role_selector("div", "Save") is None


def test_candidate_selectors_rank_interactive_matches() -> None:
    dom = """
    <div>Ignore me</div>
    <button id="submit" data-testid="submit-btn">Submit order</button>
    <input name="username" placeholder="User name" />
    """

    candidates = _candidate_selectors(dom, AgentAction.click("#missing-submit"))

    assert "#submit" in candidates
    assert '[data-testid="submit-btn"]' in candidates
    assert "text=Submit order" in candidates


def test_candidate_selectors_return_empty_without_dom_or_selector() -> None:
    assert _candidate_selectors("", AgentAction.click("#missing")) == []
    assert (
        _candidate_selectors(
            "<button id='submit'>Submit</button>",
            AgentAction.finish("done"),
        )
        == []
    )


def test_candidate_selectors_support_type_text_and_press_key_priorities() -> None:
    dom = """
    <textarea id="message">Draft</textarea>
    <select id="country">Portugal</select>
    """

    type_candidates = _candidate_selectors(dom, AgentAction.type_text("hi", selector="#message"))
    press_candidates = _candidate_selectors(
        dom, AgentAction.press_key("Enter", selector="#country")
    )

    assert "text=Draft" in type_candidates
    assert "text=Portugal" in press_candidates


def test_candidate_selectors_skip_zero_score_non_healable_actions_and_empty_candidates() -> None:
    finish = AgentAction.finish("done").model_copy(update={"selector": "#missing"})
    dom = "<option></option>"

    assert _candidate_selectors(dom, finish) == []


def test_candidate_selectors_deduplicate_and_skip_original_selector() -> None:
    dom = """
    <button id="submit">Submit</button>
    <button id="submit">Submit</button>
    """

    candidates = _candidate_selectors(dom, AgentAction.click("#submit"))

    assert "#submit" not in candidates
    assert candidates.count("text=Submit") == 1


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


def test_returns_proposed_when_no_recent_element_not_found() -> None:
    original = AgentAction.click("#missing")
    trace = [
        TraceStep(
            index=0,
            observation=_obs(),
            action=original,
            result=ActionResult.failed("boom", category=FailureCategory.UNKNOWN),
        )
    ]

    healed = SelfHealingAgent(StubAgent(original)).next_action(_task(), _obs(), trace)

    assert healed.selector == "#missing"


def test_returns_proposed_when_shape_changed_since_failure() -> None:
    original = AgentAction.click("#missing")
    proposed = AgentAction.type_text("alice", selector="#missing")

    healed = SelfHealingAgent(StubAgent(proposed)).next_action(
        _task(), _obs(), [_failed_step(original)]
    )

    assert healed == proposed


def test_returns_proposed_when_all_candidates_already_failed_or_limited() -> None:
    original = AgentAction.click("#missing")
    trace = [
        _failed_step(original),
        TraceStep(
            index=1,
            observation=_obs(),
            action=AgentAction.click("#submit"),
            result=ActionResult.failed("boom", category=FailureCategory.ELEMENT_NOT_FOUND),
        ),
    ]
    agent = SelfHealingAgent(StubAgent(original), max_candidates_per_action=1)

    healed = agent.next_action(_task(), _obs(), trace)

    assert healed == original


def test_returns_proposed_for_terminal_or_unhealable_actions() -> None:
    finish = AgentAction.finish("done")
    wait = AgentAction.wait(50)

    assert SelfHealingAgent(StubAgent(finish)).next_action(_task(), _obs(), []) == finish
    assert SelfHealingAgent(StubAgent(wait)).next_action(_task(), _obs(), []) == wait


def test_last_element_not_found_ignores_waits_and_stops_on_other_failures() -> None:
    missing = _failed_step(AgentAction.click("#missing"))
    wait_step = TraceStep(
        index=1,
        observation=_obs(),
        action=AgentAction.wait(10),
        result=ActionResult.ok(),
    )
    other_failure = TraceStep(
        index=2,
        observation=_obs(),
        action=AgentAction.click("#other"),
        result=ActionResult.failed("boom", category=FailureCategory.UNKNOWN),
    )

    assert SelfHealingAgent._last_element_not_found([missing, wait_step]) == missing  # noqa: SLF001
    assert SelfHealingAgent._last_element_not_found([missing, other_failure]) is None  # noqa: SLF001


def test_failed_selectors_collects_only_same_shape_missing_failures() -> None:
    proposed = AgentAction.click("#missing")
    trace = [
        _failed_step(AgentAction.click("#missing")),
        TraceStep(
            index=1,
            observation=_obs(),
            action=AgentAction.click("#submit"),
            result=ActionResult.failed("boom", category=FailureCategory.ELEMENT_NOT_FOUND),
        ),
        TraceStep(
            index=2,
            observation=_obs(),
            action=AgentAction.type_text("alice", selector="#user"),
            result=ActionResult.failed("boom", category=FailureCategory.ELEMENT_NOT_FOUND),
        ),
    ]

    assert SelfHealingAgent._failed_selectors(trace, proposed) == {"#missing", "#submit"}  # noqa: SLF001


def test_recovers_end_to_end_when_runner_allows_repair() -> None:
    env = HealingEnv()
    agent = SelfHealingAgent(StubAgent(AgentAction.click("#missing")))
    run = Runner(stop_on_action_failure=False).run(_task(), agent, env)

    assert any(action.selector == "#missing" for action in env.executed)
    assert any(action.selector == "#submit" for action in env.executed)
    assert run.status is RunStatus.MAX_STEPS
