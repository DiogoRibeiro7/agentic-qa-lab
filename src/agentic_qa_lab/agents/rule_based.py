"""Deterministic, scripted baseline agent.

``RuleBasedAgent`` replays a fixed plan of actions. It is the control baseline
against which planner/LLM agents are measured: it has no reasoning, so its
success rate is a property of the task and environment alone.

Two light rules make it robust:

* If the task's ``success_selector`` is present in the latest DOM snapshot, the
  agent finishes immediately rather than continuing the script.
* When the script is exhausted it emits ``finish`` so the run terminates
  cleanly instead of stalling.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..domain import AgentAction, Observation, TaskSpec, TraceStep


class RuleBasedAgent:
    """Replay a predefined action plan, step by step.

    Parameters
    ----------
    plan:
        Ordered actions to emit. The action chosen on each turn is indexed by
        the number of steps already executed (``len(trace)``).
    finish_reason:
        Reason attached to the terminal ``finish`` emitted once the plan is
        exhausted or the success selector is found.
    """

    def __init__(
        self,
        plan: Sequence[AgentAction],
        *,
        finish_reason: str = "Plan complete.",
    ) -> None:
        self._plan: tuple[AgentAction, ...] = tuple(plan)
        self._finish_reason = finish_reason

    def next_action(
        self,
        task: TaskSpec,
        observation: Observation,
        trace: list[TraceStep],
    ) -> AgentAction:
        """Return the planned action for the current step.

        See :class:`~agentic_qa_lab.agents.base.Agent` for the contract.
        """
        if self._success_visible(task, observation):
            return AgentAction.finish("Success selector detected.")

        index = len(trace)
        if index < len(self._plan):
            return self._plan[index]
        return AgentAction.finish(self._finish_reason)

    @staticmethod
    def _success_visible(task: TaskSpec, observation: Observation) -> bool:
        """Return ``True`` when the success selector text is in the DOM."""
        if task.success_selector is None or observation.dom_snapshot is None:
            return False
        return task.success_selector in observation.dom_snapshot
