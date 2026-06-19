"""Agent protocol.

An agent is a pure decision function: given the task, the latest observation,
and the trace so far, it returns the next :class:`AgentAction`. Agents must not
perform any I/O — that is the environment's job — which keeps them trivial to
unit test and swap (rule-based, LLM-based, ...).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..domain import AgentAction, Observation, TaskSpec, TraceStep


@runtime_checkable
class Agent(Protocol):
    """Decision-making interface implemented by every agent."""

    def next_action(
        self,
        task: TaskSpec,
        observation: Observation,
        trace: list[TraceStep],
    ) -> AgentAction:
        """Return the next action to take.

        Parameters
        ----------
        task:
            The task being attempted.
        observation:
            The most recent environment observation.
        trace:
            All steps executed so far, in order.

        Returns
        -------
        AgentAction
            The action the environment should execute next.
        """
        ...
