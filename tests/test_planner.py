from __future__ import annotations

from agentic_qa_lab.agents import LLMMessage, LLMPlannerAgent
from agentic_qa_lab.domain import ActionType, Observation, TaskSpec


class FakeLLM:
    """LLM stub returning queued replies and recording prompts."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[list[LLMMessage]] = []

    def complete(self, messages: list[LLMMessage]) -> str:
        self.prompts.append(messages)
        return self._replies.pop(0)


def _task(**kwargs: object) -> TaskSpec:
    base: dict[str, object] = {
        "task_id": "t1",
        "goal": "Log in",
        "start_url": "https://example.com/",
    }
    base.update(kwargs)
    return TaskSpec(**base)  # type: ignore[arg-type]


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
    assert "id='go'" in user_msg  # DOM snapshot included


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
