"""Selector self-healing wrapper around any agent.

When a browser interaction fails with ``element_not_found``, a fixed selector
can be the problem rather than the whole plan. ``SelfHealingAgent`` watches the
recent trace for these failures and, when the inner agent keeps proposing the
same broken selector, swaps in a DOM-derived alternative before giving up.

Heuristics stay intentionally simple and deterministic:

- extract useful tokens from the failed selector (id/name/text-like fragments)
- scan the current DOM for interactive elements whose id/name/label/text match
- propose a nearby CSS/id/name selector first, then text-/role-based selectors
- avoid candidates that already failed earlier in the run
"""

from __future__ import annotations

import html
import re

from ..domain import ActionType, AgentAction, FailureCategory, Observation, TaskSpec, TraceStep
from .base import Agent

_HEALABLE_ACTIONS = frozenset({ActionType.CLICK, ActionType.TYPE_TEXT, ActionType.PRESS_KEY})
_INTERACTIVE_TAGS = ("button", "input", "textarea", "select", "option", "a", "label")


def _same_shape(a: AgentAction, b: AgentAction) -> bool:
    """Return ``True`` when two actions are the same except for selector."""
    return a.type is b.type and a.text == b.text and a.key == b.key and a.x == b.x and a.y == b.y


def _selector_terms(selector: str) -> list[str]:
    """Extract meaningful lowercase tokens from a selector string."""
    parts = re.findall(r"[A-Za-z0-9_-]+", selector.lower())
    return [part for part in parts if len(part) >= 2 and part not in {"css", "text", "role"}]


def _attr(attrs: str, name: str) -> str | None:
    match = re.search(rf'{name}\s*=\s*["\']([^"\']+)["\']', attrs, flags=re.IGNORECASE)
    return html.unescape(match.group(1)) if match else None


def _text_selector(text: str) -> str | None:
    clean = re.sub(r"\s+", " ", html.unescape(text)).strip()
    if not clean:
        return None
    return f"text={clean}"


def _role_selector(tag: str, text: str) -> str | None:
    clean = re.sub(r"\s+", " ", html.unescape(text)).strip()
    role_map = {"button": "button", "a": "link", "input": "textbox", "textarea": "textbox"}
    role = role_map.get(tag)
    if role is None or not clean:
        return None
    return f'role={role}[name="{clean}"]'


def _tag_priority(action_type: ActionType, tag: str) -> int:
    """Prefer candidates whose tag matches the action kind."""
    if action_type is ActionType.CLICK:
        return 2 if tag in {"button", "a", "label"} else 1
    if action_type is ActionType.TYPE_TEXT:
        return 2 if tag in {"input", "textarea"} else 1
    if action_type is ActionType.PRESS_KEY:
        return 2 if tag in {"input", "textarea", "select"} else 1
    return 0


def _candidate_selectors(dom_snapshot: str, action: AgentAction) -> list[str]:
    """Return ranked alternative selectors for ``selector`` from ``dom_snapshot``."""
    selector = action.selector or ""
    if not dom_snapshot or not selector:
        return []
    terms = _selector_terms(selector)

    pattern = re.compile(
        r"<(?P<tag>button|input|textarea|select|option|a|label)\b(?P<attrs>[^>]*)>(?P<text>.*?)</(?P=tag)>|"
        r"<(?P<self_tag>input)\b(?P<self_attrs>[^>]*)/?>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    ranked: list[tuple[int, int, list[str]]] = []
    for match in pattern.finditer(dom_snapshot):
        tag = (match.group("tag") or match.group("self_tag") or "").lower()
        attrs = match.group("attrs") or match.group("self_attrs") or ""
        text = match.group("text") or ""
        haystack = " ".join(
            part
            for part in (
                _attr(attrs, "id"),
                _attr(attrs, "name"),
                _attr(attrs, "aria-label"),
                _attr(attrs, "placeholder"),
                text,
            )
            if part
        ).lower()
        score = sum(term in haystack for term in terms)
        priority = _tag_priority(action.type, tag)
        if score <= 0 and priority <= 0:
            continue

        candidates: list[str] = []
        id_value = _attr(attrs, "id")
        name_value = _attr(attrs, "name")
        testid_value = _attr(attrs, "data-testid")
        if id_value:
            candidates.append(f"#{id_value}")
        if testid_value:
            candidates.append(f'[data-testid="{testid_value}"]')
        if name_value:
            candidates.append(f'[name="{name_value}"]')
        text_value = _text_selector(text)
        if text_value:
            candidates.append(text_value)
        role_value = _role_selector(tag, text)
        if role_value:
            candidates.append(role_value)
        if not candidates:
            continue
        ranked.append((score, priority, candidates))

    seen: set[str] = set()
    alternatives: list[str] = []
    for _score, _priority, candidates in sorted(
        ranked, key=lambda item: (item[0], item[1]), reverse=True
    ):
        for candidate in candidates:
            if candidate == selector or candidate in seen:
                continue
            seen.add(candidate)
            alternatives.append(candidate)
    return alternatives


class SelfHealingAgent:
    """Wrap an agent and retry element-not-found actions with better selectors."""

    def __init__(self, inner: Agent, *, max_candidates_per_action: int = 3) -> None:
        self._inner = inner
        self._max_candidates_per_action = max_candidates_per_action

    def next_action(
        self,
        task: TaskSpec,
        observation: Observation,
        trace: list[TraceStep],
    ) -> AgentAction:
        """Return the inner action or a healed selector variant when possible."""
        proposed = self._inner.next_action(task, observation, trace)
        if proposed.is_terminal or proposed.type not in _HEALABLE_ACTIONS or not proposed.selector:
            return proposed

        last = self._last_element_not_found(trace)
        if last is None:
            return proposed
        failed = last.action
        if not _same_shape(failed, proposed) or failed.selector != proposed.selector:
            return proposed

        failed_selectors = self._failed_selectors(trace, proposed)
        candidates = _candidate_selectors(observation.dom_snapshot or "", proposed)
        for candidate in candidates[: self._max_candidates_per_action]:
            if candidate in failed_selectors:
                continue
            return proposed.model_copy(
                update={
                    "selector": candidate,
                    "reason": (
                        f"{proposed.reason + '; ' if proposed.reason else ''}"
                        f"self-healed from {proposed.selector} to {candidate}"
                    ),
                }
            )
        return proposed

    @staticmethod
    def _last_element_not_found(trace: list[TraceStep]) -> TraceStep | None:
        """Return the most recent element-not-found step, ignoring waits."""
        for step in reversed(trace):
            if step.action.type is ActionType.WAIT:
                continue
            if step.result.failure_category is FailureCategory.ELEMENT_NOT_FOUND:
                return step
            break
        return None

    @staticmethod
    def _failed_selectors(trace: list[TraceStep], proposed: AgentAction) -> set[str]:
        """Return selectors that already failed for this action shape."""
        selectors: set[str] = set()
        for step in trace:
            if (
                step.action.selector
                and not step.result.success
                and step.result.failure_category is FailureCategory.ELEMENT_NOT_FOUND
                and _same_shape(step.action, proposed)
            ):
                selectors.add(step.action.selector)
        return selectors
