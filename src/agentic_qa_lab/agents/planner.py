"""LLM-backed planning agent.

``LLMPlannerAgent`` turns the task, the latest observation, and the trace into a
chat prompt, asks an :class:`LLMClient` for the next step, and parses the reply
into a validated :class:`AgentAction`. Malformed replies are retried with a
correction message; if the model never produces a valid action the agent emits
a terminal ``fail`` so the run ends cleanly.
"""

from __future__ import annotations

import json
from enum import StrEnum

from pydantic import ValidationError

from ..domain import AgentAction, Observation, TaskSpec, TraceStep
from .llm import LLMClient, LLMMessage
from .memory import summarize_trace

#: Hard cap on how much DOM text is sent to the model per step.
DEFAULT_DOM_CHAR_LIMIT = 4000

SYSTEM_PROMPT = """\
You are a UI-testing agent that controls a web browser one action at a time.
Respond with a SINGLE JSON object and nothing else. The schema is:

  {"type": "<click|type_text|press_key|wait|finish|fail>",
   "selector": "<css selector, optional>",
   "x": <int, optional>, "y": <int, optional>,
   "text": "<text for type_text>",
   "key": "<key name for press_key, e.g. Enter>",
   "duration_ms": <positive int for wait>,
   "reason": "<short rationale>"}

Rules:
- click/type_text/press_key require a selector (or x and y for click).
- type_text requires a non-empty "text".
- press_key requires a "key".
- wait requires a positive "duration_ms".
- Use "finish" when the goal is achieved, "fail" when it cannot be.
Return only the JSON object, optionally inside a ```json code fence.
"""

#: Appended to the system prompt when a screenshot is attached.
VISUAL_HINT = """\

A SCREENSHOT of the current page is attached. Reason about the visual layout —
the position, labels, and state of buttons, inputs, and text — to choose the
next action. When the DOM is also provided, reconcile both; prefer stable CSS
selectors, falling back to (x, y) coordinates read from the screenshot only
when no selector is available.
"""


class ObservationMode(StrEnum):
    """Which observation channels are sent to the model.

    Used to compare grounding strategies (see the vision notebook).
    """

    DOM_ONLY = "dom_only"
    SCREENSHOT_ONLY = "screenshot_only"
    COMBINED = "combined"

    @property
    def uses_dom(self) -> bool:
        """Whether this mode includes the DOM snapshot."""
        return self in {ObservationMode.DOM_ONLY, ObservationMode.COMBINED}

    @property
    def uses_screenshot(self) -> bool:
        """Whether this mode attaches the screenshot."""
        return self in {ObservationMode.SCREENSHOT_ONLY, ObservationMode.COMBINED}


def _extract_json_object(text: str) -> str:
    """Return the first balanced ``{...}`` block found in ``text``.

    Raises
    ------
    ValueError
        If no balanced JSON object is present.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")
    depth = 0
    for i in range(start, len(text)):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("Unbalanced JSON object in model output.")


class LLMPlannerAgent:
    """Decide the next action by prompting an LLM.

    Parameters
    ----------
    client:
        Any :class:`LLMClient` implementation.
    max_parse_retries:
        How many times to re-prompt the model after an invalid reply before
        giving up with a terminal ``fail``.
    dom_char_limit:
        Maximum number of DOM characters included in the prompt.
    observation_mode:
        Which observation channels to ground on — DOM text, the screenshot, or
        both. Defaults to ``DOM_ONLY``.
    include_memory:
        When ``True`` (default) a distilled :class:`MemorySummary` of the trace
        (repeatedly-failing targets, successes, last error) is added to the
        prompt so the planner avoids repeating failed actions.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        max_parse_retries: int = 2,
        dom_char_limit: int = DEFAULT_DOM_CHAR_LIMIT,
        observation_mode: ObservationMode = ObservationMode.DOM_ONLY,
        include_memory: bool = True,
    ) -> None:
        self._client = client
        self._max_parse_retries = max_parse_retries
        self._dom_char_limit = dom_char_limit
        self._mode = observation_mode
        self._include_memory = include_memory

    def next_action(
        self,
        task: TaskSpec,
        observation: Observation,
        trace: list[TraceStep],
    ) -> AgentAction:
        """Prompt the model and return a validated action.

        See :class:`~agentic_qa_lab.agents.base.Agent` for the contract.
        """
        messages = self._build_messages(task, observation, trace)
        last_error = ""
        for _ in range(self._max_parse_retries + 1):
            reply = self._client.complete(messages)
            try:
                return self._parse_action(reply)
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)
                messages = [
                    *messages,
                    LLMMessage(role="assistant", content=reply),
                    LLMMessage(
                        role="user",
                        content=(
                            f"That was not a valid action ({last_error}). "
                            "Reply with one corrected JSON object only."
                        ),
                    ),
                ]
        return AgentAction.fail(f"LLM produced no valid action: {last_error}")

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _build_messages(
        self,
        task: TaskSpec,
        observation: Observation,
        trace: list[TraceStep],
    ) -> list[LLMMessage]:
        """Assemble the system + user messages for one planning step.

        The screenshot is attached only in screenshot/combined modes and only
        when the observation actually carries one.
        """
        system = SYSTEM_PROMPT
        images: tuple[str, ...] = ()
        if self._mode.uses_screenshot and observation.screenshot_path is not None:
            system = SYSTEM_PROMPT + VISUAL_HINT
            images = (observation.screenshot_path,)
        return [
            LLMMessage(role="system", content=system),
            LLMMessage(
                role="user",
                content=self._render_state(task, observation, trace),
                images=images,
            ),
        ]

    def _render_state(
        self,
        task: TaskSpec,
        observation: Observation,
        trace: list[TraceStep],
    ) -> str:
        """Render task, observation, and recent history as prompt text.

        The DOM block is included only when the observation mode uses it.
        """
        history = self._render_history(trace)
        success = task.success_selector or "(none)"
        lines = [
            f"GOAL: {task.goal}",
            f"SUCCESS_SELECTOR: {success}",
            f"STEP: {len(trace)} / {task.max_steps}",
            f"URL: {observation.url}",
            f"TITLE: {observation.title or '(unknown)'}",
            f"RECENT_ACTIONS:\n{history}",
        ]
        if self._include_memory:
            memory = summarize_trace(trace).render()
            if memory:
                lines.append(memory)
        if self._mode.uses_dom:
            dom = (observation.dom_snapshot or "")[: self._dom_char_limit]
            lines.append(f"DOM (truncated to {self._dom_char_limit} chars):\n{dom}")
        if self._mode.uses_screenshot:
            attached = observation.screenshot_path is not None
            lines.append(f"SCREENSHOT: {'attached' if attached else 'unavailable'}")
        lines.append("\nChoose the next action as a single JSON object.")
        return "\n".join(lines)

    @staticmethod
    def _render_history(trace: list[TraceStep], *, limit: int = 5) -> str:
        """Summarize the last ``limit`` steps for the prompt."""
        if not trace:
            return "  (none)"
        lines = []
        for step in trace[-limit:]:
            if step.result.success:
                outcome = "ok"
            else:
                outcome = f"FAILED:{step.result.failure_category.value}"
            lines.append(f"  {step.index}: {step.action.type.value} -> {outcome}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Output parsing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_action(reply: str) -> AgentAction:
        """Parse a model reply into a validated :class:`AgentAction`.

        Raises
        ------
        ValueError
            If the reply contains no JSON object or is not a JSON object.
        pydantic.ValidationError
            If the JSON does not satisfy the action schema.
        """
        raw = _extract_json_object(reply)
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Model output JSON must be an object.")
        return AgentAction.model_validate(data)
