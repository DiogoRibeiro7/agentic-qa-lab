from __future__ import annotations

from agentic_qa_lab.agents import (
    LLMMessage,
    LLMPlannerAgent,
    MemorySummary,
    summarize_trace,
)
from agentic_qa_lab.domain import (
    ActionResult,
    AgentAction,
    FailureCategory,
    Observation,
    TaskSpec,
    TraceStep,
)


def _step(
    index: int,
    action: AgentAction,
    *,
    success: bool,
    url: str = "https://e.com/",
    category: FailureCategory = FailureCategory.ELEMENT_NOT_FOUND,
    error: str | None = "not found",
) -> TraceStep:
    obs = Observation(step=index, url=url, dom_snapshot="<html/>", timestamp=1.0 + index)
    result = ActionResult.ok() if success else ActionResult.failed(error or "x", category=category)
    return TraceStep(index=index, observation=obs, action=action, result=result)


# --------------------------------------------------------------------------- #
# summarize_trace
# --------------------------------------------------------------------------- #
def test_empty_trace_is_empty_summary() -> None:
    summary = summarize_trace([])
    assert summary == MemorySummary()
    assert summary.render() == ""


def test_counts_repeated_failures() -> None:
    click = AgentAction.click("#go")
    trace = [
        _step(0, click, success=False),
        _step(1, click, success=False),
        _step(2, click, success=False),
    ]
    summary = summarize_trace(trace)

    assert summary.failed_targets[0].target == "click:#go"
    assert summary.failed_targets[0].count == 3
    assert summary.failed_targets[0].last_category == "element_not_found"
    assert summary.problem_targets  # >= threshold


def test_tracks_successes_and_visited_urls() -> None:
    trace = [
        _step(0, AgentAction.type_text("alice", selector="#user"), success=True, url="https://a/"),
        _step(1, AgentAction.click("#go"), success=True, url="https://b/"),
    ]
    summary = summarize_trace(trace)

    assert "type_text:#user" in summary.succeeded_targets
    assert "click:#go" in summary.succeeded_targets
    assert summary.visited_urls == ["https://a/", "https://b/"]
    assert summary.failed_targets == []


def test_last_error_is_most_recent() -> None:
    trace = [
        _step(0, AgentAction.click("#a"), success=False, error="first"),
        _step(1, AgentAction.click("#b"), success=False, error="second"),
    ]
    assert summarize_trace(trace).last_error == "second"


def test_render_lists_problem_targets_only() -> None:
    click = AgentAction.click("#go")
    trace = [_step(0, click, success=False), _step(1, click, success=False)]
    text = summarize_trace(trace).render()

    assert "MEMORY" in text
    assert "avoid (kept failing): click:#go x2" in text


def test_single_failure_is_not_a_problem_target() -> None:
    trace = [_step(0, AgentAction.click("#go"), success=False)]
    summary = summarize_trace(trace)
    assert summary.failed_targets  # recorded
    assert not summary.problem_targets  # but below threshold
    # render still surfaces the last error even without problem targets
    assert "last error" in summary.render()


# --------------------------------------------------------------------------- #
# Planner integration
# --------------------------------------------------------------------------- #
class RecordingLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.last: list[LLMMessage] = []

    def complete(self, messages: list[LLMMessage]) -> str:
        self.last = messages
        return self._reply


def _task() -> TaskSpec:
    return TaskSpec(task_id="t", goal="g", start_url="https://e.com/")


def _obs() -> Observation:
    return Observation(step=0, url="https://e.com/", dom_snapshot="<html/>", timestamp=9.0)


def test_planner_injects_memory_when_enabled() -> None:
    click = AgentAction.click("#go")
    trace = [_step(0, click, success=False), _step(1, click, success=False)]
    llm = RecordingLLM('{"type": "finish"}')

    LLMPlannerAgent(llm, include_memory=True).next_action(_task(), _obs(), trace)

    user = llm.last[1].content
    assert "MEMORY" in user
    assert "click:#go" in user


def test_planner_omits_memory_when_disabled() -> None:
    click = AgentAction.click("#go")
    trace = [_step(0, click, success=False), _step(1, click, success=False)]
    llm = RecordingLLM('{"type": "finish"}')

    LLMPlannerAgent(llm, include_memory=False).next_action(_task(), _obs(), trace)

    assert "MEMORY" not in llm.last[1].content


def test_planner_memory_absent_on_empty_trace() -> None:
    llm = RecordingLLM('{"type": "finish"}')
    LLMPlannerAgent(llm).next_action(_task(), _obs(), [])
    assert "MEMORY" not in llm.last[1].content
