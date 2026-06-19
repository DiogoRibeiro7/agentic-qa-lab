"""Success detection keys off rendered visible text, not the DOM snapshot.

Regression coverage for the substring trap: a marker present only in
``<script>`` source (captured by ``dom_snapshot`` but not rendered) must not
count as success, while a marker in the page's visible text must.
"""

from __future__ import annotations

from agentic_qa_lab.agents import RuleBasedAgent, Runner
from agentic_qa_lab.domain import (
    ActionResult,
    AgentAction,
    Observation,
    RunStatus,
    TaskSpec,
)
from agentic_qa_lab.environments import BrowserEnvironment

SCRIPT_DOM = "<html><body><script>var t = 'Welcome, alice!';</script></body></html>"


# --------------------------------------------------------------------------- #
# Observation.search_text / contains_marker
# --------------------------------------------------------------------------- #
def test_search_text_prefers_visible_text() -> None:
    obs = Observation(
        step=0,
        url="https://e.com/",
        dom_snapshot=SCRIPT_DOM,
        visible_text="",  # nothing visible yet
        timestamp=1.0,
    )
    # Marker is in the DOM (script) but not visible -> not a match.
    assert not obs.contains_marker("Welcome")
    assert obs.search_text == ""


def test_visible_marker_matches() -> None:
    obs = Observation(
        step=1,
        url="https://e.com/",
        dom_snapshot=SCRIPT_DOM,
        visible_text="Welcome, alice!",
        timestamp=1.0,
    )
    assert obs.contains_marker("Welcome")


def test_falls_back_to_dom_when_no_visible_text() -> None:
    # Environments that don't capture visible text keep the old behaviour.
    obs = Observation(step=0, url="https://e.com/", dom_snapshot="<p>Welcome</p>", timestamp=1.0)
    assert obs.contains_marker("Welcome")


# --------------------------------------------------------------------------- #
# End-to-end via the Runner
# --------------------------------------------------------------------------- #
class ScriptedEnv(BrowserEnvironment):
    """Env whose DOM always contains the marker but whose visible text varies."""

    def __init__(self, *, visible_text: str) -> None:
        self._visible = visible_text
        self._step = 0

    def _obs(self) -> Observation:
        return Observation(
            step=self._step,
            url="https://e.com/",
            dom_snapshot=SCRIPT_DOM,  # marker is in the script source
            visible_text=self._visible,
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


def _task() -> TaskSpec:
    return TaskSpec(task_id="t", goal="g", start_url="https://e.com/", success_selector="Welcome")


def test_finish_fails_when_marker_only_in_script() -> None:
    plan = [AgentAction.finish("done")]
    run = Runner().run(_task(), RuleBasedAgent(plan), ScriptedEnv(visible_text="nothing here"))
    assert run.status is RunStatus.FAILURE


def test_finish_succeeds_when_marker_is_visible() -> None:
    plan = [AgentAction.finish("done")]
    run = Runner().run(_task(), RuleBasedAgent(plan), ScriptedEnv(visible_text="Welcome, alice!"))
    assert run.status is RunStatus.SUCCESS


def test_rule_agent_does_not_early_finish_on_script_marker() -> None:
    # Even though the DOM (script) has "Welcome", the agent must not finish early
    # because it is not visible; it should follow its plan instead.
    agent = RuleBasedAgent([AgentAction.click("#a")])
    obs = Observation(
        step=0,
        url="https://e.com/",
        dom_snapshot=SCRIPT_DOM,
        visible_text="",
        timestamp=1.0,
    )
    action = agent.next_action(_task(), obs, [])
    assert action.type is AgentAction.click("#a").type
    assert action.selector == "#a"
