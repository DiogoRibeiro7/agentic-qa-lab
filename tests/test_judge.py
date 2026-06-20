from __future__ import annotations

from agentic_qa_lab.agents import JudgeVerdict, LLMMessage, LLMSuccessJudge
from agentic_qa_lab.domain import Observation, TaskSpec


class StructuredJudgeLLM:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload
        self.calls = 0

    def complete(self, messages: list[LLMMessage]) -> str:
        raise AssertionError("plain completion should not be used")

    def complete_json(
        self,
        messages: list[LLMMessage],
        *,
        schema_name: str,
        schema: dict[str, object],
    ) -> dict[str, object]:
        self.calls += 1
        return self._payload


class PlainJudgeLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    def complete(self, messages: list[LLMMessage]) -> str:
        return self._reply


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="t",
        goal="Confirm the item was created.",
        start_url="https://e.com/",
        success_judge="The page should clearly show that the item exists.",
    )


def _obs() -> Observation:
    return Observation(
        step=0,
        url="https://e.com/items/1",
        title="Item details",
        visible_text="Item created successfully",
        dom_snapshot="<h1>Item created successfully</h1>",
        timestamp=1.0,
    )


def test_llm_success_judge_uses_structured_completion() -> None:
    judge = LLMSuccessJudge(
        StructuredJudgeLLM({"success": True, "reason": "Visible confirmation present."})
    )

    verdict = judge.evaluate(_task(), _obs(), [])

    assert verdict == JudgeVerdict(success=True, reason="Visible confirmation present.")


def test_llm_success_judge_parses_plain_json_fence() -> None:
    judge = LLMSuccessJudge(
        PlainJudgeLLM('```json\n{"success": false, "reason": "No confirmation shown."}\n```')
    )

    verdict = judge.evaluate(_task(), _obs(), [])

    assert verdict == JudgeVerdict(success=False, reason="No confirmation shown.")
