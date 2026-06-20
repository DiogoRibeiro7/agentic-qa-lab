from __future__ import annotations

from typing import Any

import pytest

from agentic_qa_lab.agents import LLMMessage, LLMPlannerAgent, ObservationMode
from agentic_qa_lab.agents import planner as planner_module
from agentic_qa_lab.domain import (
    ActionResult,
    ActionType,
    AgentAction,
    FailureCategory,
    Observation,
    TaskSpec,
    TraceStep,
)


class FakeLLM:
    """LLM stub returning queued replies and recording prompts."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[list[LLMMessage]] = []

    def complete(self, messages: list[LLMMessage]) -> str:
        self.prompts.append(messages)
        return self._replies.pop(0)


class StructuredFakeLLM:
    """LLM stub that exposes schema-driven completions."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self._replies = list(replies)
        self.prompts: list[list[LLMMessage]] = []
        self.schemas: list[dict[str, Any]] = []

    def complete_json(
        self,
        messages: list[LLMMessage],
        *,
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        assert schema_name == "agent_action"
        self.prompts.append(messages)
        self.schemas.append(schema)
        return self._replies.pop(0)


def _task(**kwargs: object) -> TaskSpec:
    base: dict[str, object] = {
        "task_id": "t1",
        "goal": "Log in",
        "start_url": "https://example.com/",
    }
    base.update(kwargs)
    return TaskSpec.model_validate(base)


def _obs() -> Observation:
    return Observation(
        step=0,
        url="https://example.com/login",
        title="Login",
        dom_snapshot="<input id='user'><button id='go'>Go</button>",
        timestamp=1.0,
    )


def test_parses_plain_json_action() -> None:
    llm = FakeLLM(['{"type": "click", "selector": "#go", "reason": "submit"}'])
    action = LLMPlannerAgent(llm).next_action(_task(), _obs(), [])

    assert action.type is ActionType.CLICK
    assert action.selector == "#go"


def test_parses_action_inside_code_fence() -> None:
    reply = 'Sure!\n```json\n{"type": "type_text", "selector": "#user", "text": "alice"}\n```'
    llm = FakeLLM([reply])
    action = LLMPlannerAgent(llm).next_action(_task(), _obs(), [])

    assert action.type is ActionType.TYPE_TEXT
    assert action.text == "alice"


def test_prompt_includes_goal_and_dom() -> None:
    llm = FakeLLM(['{"type": "finish", "reason": "done"}'])
    LLMPlannerAgent(llm).next_action(_task(), _obs(), [])

    user_msg = llm.prompts[0][1].content
    assert "GOAL: Log in" in user_msg
    assert "PAGE SUMMARY:" in user_msg
    assert "button id=go" in user_msg


def test_prompt_prefers_visible_text_over_raw_dom_dump() -> None:
    llm = FakeLLM(['{"type": "finish", "reason": "done"}'])
    observation = Observation(
        step=0,
        url="https://example.com/login",
        title="Login",
        visible_text="Login form Sign in",
        dom_snapshot="<div><button id='go'>Go</button></div>",
        timestamp=1.0,
    )

    LLMPlannerAgent(llm).next_action(_task(), observation, [])

    user_msg = llm.prompts[0][1].content
    assert "VISIBLE_TEXT:\nLogin form Sign in" in user_msg
    assert "DOM (truncated" not in user_msg


def test_history_is_capped_by_token_budget() -> None:
    trace = [
        TraceStep(
            index=i,
            observation=_obs(),
            action=AgentAction.click(f"#button-{i}"),
            result=ActionResult.failed(
                "missing",
                category=FailureCategory.ELEMENT_NOT_FOUND,
            ),
        )
        for i in range(6)
    ]

    rendered = LLMPlannerAgent._render_history(trace, token_budget=12)
    assert "#button-5" in rendered
    assert "#button-0" not in rendered


def test_memory_is_capped_by_token_budget() -> None:
    llm = FakeLLM(['{"type": "finish", "reason": "done"}'])
    trace = [
        TraceStep(
            index=i,
            observation=_obs(),
            action=AgentAction.click("#go"),
            result=ActionResult.failed(
                f"very long failure message {i} " * 10,
                category=FailureCategory.ELEMENT_NOT_FOUND,
            ),
        )
        for i in range(3)
    ]

    LLMPlannerAgent(llm, memory_token_budget=10).next_action(_task(), _obs(), trace)

    user_msg = llm.prompts[0][1].content
    assert "MEMORY (learned this run):" in user_msg
    assert "..." in user_msg


def test_retries_on_invalid_then_succeeds() -> None:
    llm = FakeLLM(
        [
            "I think you should click the button.",  # no JSON
            '{"type": "click", "selector": "#go"}',  # valid
        ]
    )
    action = LLMPlannerAgent(llm, max_parse_retries=2).next_action(_task(), _obs(), [])

    assert action.type is ActionType.CLICK
    assert len(llm.prompts) == 2
    # The correction prompt must carry the prior bad reply + a fix request.
    assert any("not a valid action" in m.content for m in llm.prompts[1])


def test_retries_on_schema_violation() -> None:
    llm = FakeLLM(
        [
            '{"type": "type_text", "selector": "#user"}',  # missing text -> invalid
            '{"type": "type_text", "selector": "#user", "text": "alice"}',
        ]
    )
    action = LLMPlannerAgent(llm).next_action(_task(), _obs(), [])
    assert action.text == "alice"


def test_gives_up_with_fail_after_retries() -> None:
    llm = FakeLLM(["nope", "still nope", "nope again"])
    action = LLMPlannerAgent(llm, max_parse_retries=2).next_action(_task(), _obs(), [])

    assert action.type is ActionType.FAIL
    assert "no valid action" in (action.reason or "")


def test_prefers_structured_completion_when_available() -> None:
    llm = StructuredFakeLLM([{"type": "click", "selector": "#go"}])

    action = LLMPlannerAgent(llm).next_action(_task(), _obs(), [])

    assert action.type is ActionType.CLICK
    assert action.selector == "#go"
    assert llm.schemas[0]["required"] == ["type"]


def test_combined_mode_attaches_screenshot_and_visual_hint() -> None:
    llm = FakeLLM(['{"type": "finish", "reason": "done"}'])
    observation = Observation(
        step=0,
        url="https://example.com/login",
        title="Login",
        dom_snapshot="<button id='go'>Go</button>",
        screenshot_path="artifacts/login.png",
        timestamp=1.0,
    )

    LLMPlannerAgent(llm, observation_mode=ObservationMode.COMBINED).next_action(
        _task(), observation, []
    )

    system_msg, user_msg = llm.prompts[0]
    assert "A SCREENSHOT of the current page is attached." in system_msg.content
    assert user_msg.images == ("artifacts/login.png",)
    assert "SCREENSHOT: attached" in user_msg.content


def test_screenshot_mode_reports_unavailable_when_missing() -> None:
    llm = FakeLLM(['{"type": "finish", "reason": "done"}'])

    LLMPlannerAgent(llm, observation_mode=ObservationMode.SCREENSHOT_ONLY).next_action(
        _task(), _obs(), []
    )

    user_msg = llm.prompts[0][1].content
    assert "SCREENSHOT: unavailable" in user_msg
    assert "PAGE SUMMARY:" not in user_msg


def test_render_page_summary_falls_back_to_raw_dom_when_needed() -> None:
    observation = Observation(
        step=0,
        url="https://example.com/login",
        title="Login",
        dom_snapshot="<div>plain content only</div>",
        timestamp=1.0,
    )

    rendered = LLMPlannerAgent(FakeLLM([]), dom_char_limit=12)._render_page_summary(observation)

    assert "DOM_FALLBACK:" in rendered
    assert "<div>plai..." in rendered


def test_render_history_marks_success_and_coordinate_targets() -> None:
    trace = [
        TraceStep(
            index=1,
            observation=_obs(),
            action=AgentAction.model_validate({"type": "click", "x": 12, "y": 34}),
            result=ActionResult.ok(),
        )
    ]

    rendered = LLMPlannerAgent._render_history(trace)

    assert "click (12, 34) -> ok" in rendered


def test_fit_lines_to_token_budget_handles_empty_and_oversized_inputs() -> None:
    assert LLMPlannerAgent._fit_lines_to_token_budget(["alpha"], token_budget=0) == "  (none)"

    rendered = LLMPlannerAgent._fit_lines_to_token_budget(
        ["this line is too large for a tiny budget"],
        token_budget=2,
    )

    assert rendered.endswith("...")
    assert "tiny" not in rendered


def test_truncate_to_token_budget_returns_original_when_text_fits() -> None:
    assert LLMPlannerAgent._truncate_to_token_budget("short text", token_budget=20) == "short text"


def test_parse_action_rejects_non_object_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(planner_module, "_extract_json_object", lambda _reply: "[]")

    with pytest.raises(ValueError, match="must be an object"):
        LLMPlannerAgent._parse_action("ignored")


def test_extract_json_object_rejects_unbalanced_input() -> None:
    with pytest.raises(ValueError, match="Unbalanced JSON object"):
        LLMPlannerAgent._parse_action('{"type": "click"')


def test_summarize_interactive_dom_handles_empty_and_limit() -> None:
    assert LLMPlannerAgent._summarize_interactive_dom("") == []

    dom = (
        "".join(f'<button id="b{i}">Button {i}</button>' for i in range(20))
        + '<button id="named">Visible label text that should appear</button>'
    )

    summaries = LLMPlannerAgent._summarize_interactive_dom(dom)

    assert len(summaries) == 12
    assert summaries[0] == "button id=b0"


def test_format_element_summary_includes_visible_text() -> None:
    summary = LLMPlannerAgent._format_element_summary(
        "button",
        'id="submit" role="button"',
        "Submit order",
    )

    assert summary == 'button id=submit role=button text="Submit order"'
