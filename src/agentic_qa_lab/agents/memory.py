"""Distil a run's trace into a compact memory the planner can reason over.

The raw trace is verbose and grows every step. A planner that re-reads it each
turn tends to repeat actions that already failed. :func:`summarize_trace`
collapses the trace into a small :class:`MemorySummary` — which targets keep
failing, which already succeeded, and where the agent has been — that renders to
a few prompt lines. It is computed from the trace on demand (no mutable state),
so it stays deterministic and trivially testable.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from ..domain import ActionType, TraceStep

#: Action types that address a concrete UI target worth remembering.
_INTERACTIVE = frozenset({ActionType.CLICK, ActionType.TYPE_TEXT, ActionType.PRESS_KEY})
#: A target is "problematic" once it has failed at least this many times.
REPEAT_FAILURE_THRESHOLD = 2


def _target(step: TraceStep) -> str:
    """Build a stable string key for the action's UI target.

    The target is either the selector or the explicit click coordinates. This
    string is used for counting repeated failures against the same UI target.
    """
    action = step.action
    where = action.selector if action.selector is not None else f"({action.x}, {action.y})"
    return f"{action.type.value}:{where}"


class FailedTarget(BaseModel):
    """A target that failed, with how often and its last failure category."""

    target: str
    count: int = Field(ge=1)
    last_category: str


class MemorySummary(BaseModel):
    """Compact, derived view of what a run has learned so far.

    Attributes
    ----------
    visited_urls:
        Distinct URLs seen, in first-visit order.
    failed_targets:
        Targets that failed at least once, most-failed first.
    succeeded_targets:
        Targets that have at least one successful interaction.
    last_error:
        The most recent failure message, if any.
    """

    visited_urls: list[str] = Field(default_factory=list)
    failed_targets: list[FailedTarget] = Field(default_factory=list)
    succeeded_targets: list[str] = Field(default_factory=list)
    last_error: str | None = None

    @property
    def problem_targets(self) -> list[FailedTarget]:
        """Failed targets at or above :data:`REPEAT_FAILURE_THRESHOLD`."""
        return [f for f in self.failed_targets if f.count >= REPEAT_FAILURE_THRESHOLD]

    def render(self, *, max_items: int = 5) -> str:
        """Render the summary as prompt lines, or ``""`` when empty.

        Only the actionable parts are included: targets to avoid (repeatedly
        failing), targets already done, and the last error.
        """
        lines: list[str] = []
        problems = self.problem_targets[:max_items]
        if problems:
            joined = ", ".join(f"{p.target} x{p.count} [{p.last_category}]" for p in problems)
            lines.append(f"  avoid (kept failing): {joined}")
        if self.succeeded_targets:
            lines.append(f"  already succeeded: {', '.join(self.succeeded_targets[:max_items])}")
        if self.last_error:
            lines.append(f"  last error: {self.last_error}")
        if not lines:
            return ""
        return "MEMORY (learned this run):\n" + "\n".join(lines)


def summarize_trace(trace: list[TraceStep]) -> MemorySummary:
    """Reduce ``trace`` to a :class:`MemorySummary`."""
    visited: list[str] = []
    fail_counts: Counter[str] = Counter()
    fail_category: dict[str, str] = {}
    succeeded: list[str] = []
    last_error: str | None = None

    for step in trace:
        url = step.observation.url
        if url not in visited:
            visited.append(url)

        if step.action.type not in _INTERACTIVE:
            continue
        target = _target(step)
        if step.result.success:
            if target not in succeeded:
                succeeded.append(target)
        else:
            fail_counts[target] += 1
            fail_category[target] = step.result.failure_category.value
            if step.result.error:
                last_error = step.result.error

    failed = [
        FailedTarget(target=target, count=count, last_category=fail_category[target])
        for target, count in fail_counts.most_common()
    ]
    return MemorySummary(
        visited_urls=visited,
        failed_targets=failed,
        succeeded_targets=succeeded,
        last_error=last_error,
    )
