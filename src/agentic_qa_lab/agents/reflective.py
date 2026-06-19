"""Self-reflection and repair wrapper around any agent.

A raw planner can get stuck proposing the same action that just failed —
clicking an element that has not rendered yet, for example. ``ReflectiveAgent``
wraps an inner :class:`~agentic_qa_lab.agents.base.Agent` and adds a bounded
repair policy on top of its decisions:

#. If the inner agent re-proposes an action that just failed, insert a single
   ``wait`` so the page can settle, then let it retry.
#. If the same action keeps failing past ``max_attempts``, stop with a terminal
   ``fail`` instead of looping forever.

The policy is deterministic and inner-agent agnostic, so it composes with the
rule-based baseline or the LLM planner alike.
"""

from __future__ import annotations

from ..domain import ActionType, AgentAction, Observation, TaskSpec, TraceStep
from .base import Agent

#: Default number of failed attempts at one action before giving up.
DEFAULT_MAX_ATTEMPTS = 3
#: Default settle duration (ms) inserted as a repair between retries.
DEFAULT_SETTLE_MS = 500


def _same_target(a: AgentAction, b: AgentAction) -> bool:
    """Return ``True`` when two actions target the same UI interaction.

    The comparison considers action type, selector, coordinates, text payload,
    and key so that retry logic only treats truly repeated interactions as the
    same target.
    """
    return (
        a.type is b.type
        and a.selector == b.selector
        and a.x == b.x
        and a.y == b.y
        and a.text == b.text
        and a.key == b.key
    )


class ReflectiveAgent:
    """Add a bounded settle-and-retry repair loop to an inner agent.

    Parameters
    ----------
    inner:
        The agent whose decisions are supervised.
    max_attempts:
        Number of failed attempts at the *same* action tolerated before the
        wrapper gives up with a terminal ``fail``.
    settle_ms:
        Duration of the ``wait`` inserted as a repair between retries.
    """

    def __init__(
        self,
        inner: Agent,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        settle_ms: int = DEFAULT_SETTLE_MS,
    ) -> None:
        self._inner = inner
        self._max_attempts = max_attempts
        self._settle_ms = settle_ms

    def next_action(
        self,
        task: TaskSpec,
        observation: Observation,
        trace: list[TraceStep],
    ) -> AgentAction:
        """Return the inner action, repairing repeated failures.

        See :class:`~agentic_qa_lab.agents.base.Agent` for the contract.
        """
        proposed = self._inner.next_action(task, observation, trace)
        if proposed.is_terminal:
            return proposed

        failures = self._trailing_failures(proposed, trace)
        if failures >= self._max_attempts:
            return AgentAction.fail(
                f"Repair gave up after {failures} failed attempts at "
                f"'{proposed.type.value}' on {proposed.selector or (proposed.x, proposed.y)}."
            )
        if failures > 0 and not self._just_settled(trace):
            # Reflect: the same action just failed — let the page settle first.
            return AgentAction.wait(self._settle_ms)
        return proposed

    def _trailing_failures(self, proposed: AgentAction, trace: list[TraceStep]) -> int:
        """Count consecutive recent failures of ``proposed`` (ignoring waits)."""
        count = 0
        for step in reversed(trace):
            if step.action.type is ActionType.WAIT:
                continue  # repair waits don't break the failure streak
            if _same_target(step.action, proposed) and not step.result.success:
                count += 1
                continue
            break
        return count

    @staticmethod
    def _just_settled(trace: list[TraceStep]) -> bool:
        """Return ``True`` when the most recent step was a successful wait."""
        if not trace:
            return False
        last = trace[-1]
        return last.action.type is ActionType.WAIT and last.result.success
