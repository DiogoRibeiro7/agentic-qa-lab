"""LLM-backed success judge for semantic task completion checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..domain import Observation, TaskSpec, TraceStep
from .llm import LLMClient, LLMMessage, StructuredLLMClient

_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "success": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["success", "reason"],
}


@dataclass(frozen=True)
class JudgeVerdict:
    """Success/failure verdict returned by a semantic success judge."""

    success: bool
    reason: str


@runtime_checkable
class SuccessJudge(Protocol):
    """Optional semantic success checker consulted by the runner."""

    def evaluate(
        self,
        task: TaskSpec,
        observation: Observation,
        trace: list[TraceStep],
    ) -> JudgeVerdict:
        """Return whether the task appears complete under the judge rubric."""
        ...


class LLMSuccessJudge:
    """Use an LLM to grade whether the current page satisfies a task outcome."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def evaluate(
        self,
        task: TaskSpec,
        observation: Observation,
        trace: list[TraceStep],
    ) -> JudgeVerdict:
        """Return a structured semantic success verdict for ``task``."""
        messages = self._messages(task, observation, trace)
        if isinstance(self._client, StructuredLLMClient):
            data = self._client.complete_json(
                messages,
                schema_name="success_judgement",
                schema=_VERDICT_SCHEMA,
            )
        else:
            raw = self._client.complete(messages)
            data = self._parse_json(raw)
        return JudgeVerdict(success=bool(data["success"]), reason=str(data["reason"]).strip())

    @staticmethod
    def _messages(
        task: TaskSpec,
        observation: Observation,
        trace: list[TraceStep],
    ) -> list[LLMMessage]:
        """Build the prompt shown to the semantic success judge."""
        recent = (
            "\n".join(
                (
                    f"- {step.action.type.value} "
                    f"{step.action.selector or step.action.key or step.action.reason or ''}"
                ).strip()
                for step in trace[-5:]
            )
            or "(no prior steps)"
        )
        visible = observation.visible_text or "(none)"
        dom = (observation.dom_snapshot or "(none)")[:4_000]
        criteria = task.success_judge or "Judge whether the task goal is complete."
        return [
            LLMMessage(
                role="system",
                content=(
                    "You are a strict QA success judge. Return JSON only with keys "
                    "'success' (boolean) and 'reason' (short string)."
                ),
            ),
            LLMMessage(
                role="user",
                content=(
                    f"Task goal: {task.goal}\n"
                    f"Judge rubric: {criteria}\n"
                    f"Current URL: {observation.url}\n"
                    f"Current title: {observation.title or '(none)'}\n"
                    f"Visible text:\n{visible}\n\n"
                    f"Recent actions:\n{recent}\n\n"
                    f"DOM excerpt:\n{dom}\n\n"
                    "Decide whether the task has succeeded."
                ),
                images=(
                    (observation.screenshot_path,)
                    if observation.screenshot_path is not None
                    else ()
                ),
            ),
        ]

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Parse a JSON object from a plain-text completion."""
        text = raw.strip()
        if "```" in text:
            parts = [part.strip() for part in text.split("```") if part.strip()]
            for part in parts:
                candidate = part.removeprefix("json").strip()
                try:
                    data = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    return data
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Success judge reply must be a JSON object.")
        return data
