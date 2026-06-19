"""LLM-backed planning agent.

``LLMPlannerAgent`` turns the task, the latest observation, and the trace into a
chat prompt, asks an :class:`LLMClient` for the next step, and parses the reply
into a validated :class:`AgentAction`. Malformed replies are retried with a
correction message; if the model never produces a valid action the agent emits
a terminal ``fail`` so the run ends cleanly.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum

from pydantic import ValidationError

from ..domain import AgentAction, Observation, TaskSpec, TraceStep
from .llm import LLMClient, LLMMessage
from .memory import summarize_trace
from .usage import estimate_tokens

#: Hard cap on how much DOM text is sent to the model per step.
DEFAULT_DOM_CHAR_LIMIT = 4000
DEFAULT_HISTORY_TOKEN_BUDGET = 160
DEFAULT_MEMORY_TOKEN_BUDGET = 120
DEFAULT_VISIBLE_TEXT_CHAR_LIMIT = 1200
DEFAULT_INTERACTIVE_SNIPPET_LIMIT = 12

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
    history_token_budget:
        Approximate token budget reserved for recent action history.
    memory_token_budget:
        Approximate token budget reserved for rendered run memory.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        max_parse_retries: int = 2,
        dom_char_limit: int = DEFAULT_DOM_CHAR_LIMIT,
        observation_mode: ObservationMode = ObservationMode.DOM_ONLY,
        include_memory: bool = True,
        history_token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET,
        memory_token_budget: int = DEFAULT_MEMORY_TOKEN_BUDGET,
    ) -> None:
        self._client = client
        self._max_parse_retries = max_parse_retries
        self._dom_char_limit = dom_char_limit
        self._mode = observation_mode
        self._include_memory = include_memory
        self._history_token_budget = history_token_budget
        self._memory_token_budget = memory_token_budget

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
        history = self._render_history(trace, token_budget=self._history_token_budget)
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
            memory = self._truncate_to_token_budget(
                summarize_trace(trace).render(),
                token_budget=self._memory_token_budget,
            )
            if memory:
                lines.append(memory)
        if self._mode.uses_dom:
            page_summary = self._render_page_summary(observation)
            lines.append(page_summary)
        if self._mode.uses_screenshot:
            attached = observation.screenshot_path is not None
            lines.append(f"SCREENSHOT: {'attached' if attached else 'unavailable'}")
        lines.append("\nChoose the next action as a single JSON object.")
        return "\n".join(lines)

    @staticmethod
    def _render_history(
        trace: list[TraceStep],
        *,
        limit: int = 8,
        token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET,
    ) -> str:
        """Summarize recent steps for the prompt within a token budget."""
        if not trace:
            return "  (none)"
        lines = []
        for step in trace[-limit:]:
            if step.result.success:
                outcome = "ok"
            else:
                outcome = f"FAILED:{step.result.failure_category.value}"
            target = step.action.selector or (
                f"({step.action.x}, {step.action.y})"
                if step.action.x is not None and step.action.y is not None
                else ""
            )
            detail = f" {target}" if target else ""
            lines.append(f"  {step.index}: {step.action.type.value}{detail} -> {outcome}")
        return LLMPlannerAgent._fit_lines_to_token_budget(lines, token_budget=token_budget)

    def _render_page_summary(self, observation: Observation) -> str:
        """Render a compact page summary instead of dumping raw HTML."""
        parts: list[str] = []
        visible = self._truncate_text(
            observation.visible_text or "",
            char_limit=min(self._dom_char_limit, DEFAULT_VISIBLE_TEXT_CHAR_LIMIT),
        )
        if visible:
            parts.append(f"VISIBLE_TEXT:\n{visible}")
        interactive = self._summarize_interactive_dom(observation.dom_snapshot or "")
        if interactive:
            joined = "\n".join(f"  - {line}" for line in interactive)
            parts.append(f"INTERACTIVE_ELEMENTS:\n{joined}")
        if not parts:
            raw = self._truncate_text(
                observation.dom_snapshot or "", char_limit=self._dom_char_limit
            )
            parts.append(f"DOM_FALLBACK:\n{raw or '(none)'}")
        return "PAGE SUMMARY:\n" + "\n".join(parts)

    @staticmethod
    def _truncate_text(text: str, *, char_limit: int) -> str:
        """Collapse whitespace and trim text to ``char_limit`` characters."""
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= char_limit:
            return compact
        return compact[: max(0, char_limit - 3)].rstrip() + "..."

    @staticmethod
    def _summarize_interactive_dom(dom_snapshot: str) -> list[str]:
        """Extract accessibility-like summaries for interactive HTML elements."""
        if not dom_snapshot:
            return []
        pattern = re.compile(
            r"<(?P<tag>button|input|textarea|select|option|a|label)\b(?P<attrs>[^>]*)>"
            r"(?P<text>.*?)"
            r"(?:</(?P=tag)>)?",
            flags=re.IGNORECASE | re.DOTALL,
        )
        summaries: list[str] = []
        seen: set[str] = set()
        for match in pattern.finditer(dom_snapshot):
            tag = match.group("tag").lower()
            attrs = match.group("attrs") or ""
            text = re.sub(r"<[^>]+>", " ", match.group("text") or "")
            summary = LLMPlannerAgent._format_element_summary(tag, attrs, text)
            if summary and summary not in seen:
                seen.add(summary)
                summaries.append(summary)
            if len(summaries) >= DEFAULT_INTERACTIVE_SNIPPET_LIMIT:
                break
        return summaries

    @staticmethod
    def _format_element_summary(tag: str, attrs: str, text: str) -> str:
        """Render one compact interactive-element line."""
        fields: list[str] = [tag]
        for name in ("id", "name", "type", "placeholder", "aria-label", "role", "href"):
            match = re.search(rf'{name}\s*=\s*["\']([^"\']+)["\']', attrs, flags=re.IGNORECASE)
            if match:
                fields.append(f"{name}={match.group(1)}")
        visible = LLMPlannerAgent._truncate_text(text, char_limit=80)
        if visible:
            fields.append(f'text="{visible}"')
        return " ".join(fields)

    @staticmethod
    def _fit_lines_to_token_budget(lines: list[str], *, token_budget: int) -> str:
        """Keep the newest lines that fit within ``token_budget`` tokens."""
        if token_budget <= 0:
            return "  (none)"
        kept: list[str] = []
        total = 0
        for line in reversed(lines):
            tokens = estimate_tokens(line)
            if kept and total + tokens > token_budget:
                break
            if not kept and tokens > token_budget:
                kept.append(
                    LLMPlannerAgent._truncate_to_token_budget(line, token_budget=token_budget)
                )
                break
            kept.append(line)
            total += tokens
        kept.reverse()
        return "\n".join(kept) if kept else "  (none)"

    @staticmethod
    def _truncate_to_token_budget(text: str, *, token_budget: int) -> str:
        """Trim a text block to approximately ``token_budget`` tokens."""
        if token_budget <= 0 or not text:
            return ""
        if estimate_tokens(text) <= token_budget:
            return text
        char_limit = max(1, token_budget * 4 - 3)
        return text[:char_limit].rstrip() + "..."

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
