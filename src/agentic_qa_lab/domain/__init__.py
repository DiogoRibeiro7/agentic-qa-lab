"""Core domain models for agentic-qa-lab.

This package defines the provider- and environment-agnostic types that flow
through the system: tasks, observations, actions, per-action results, and the
aggregated run trace. Everything here is plain Pydantic with no I/O so the
models stay easy to test and serialize.
"""

from __future__ import annotations

from .action import TERMINAL_ACTIONS, ActionType, AgentAction
from .observation import Observation
from .result import (
    TERMINAL_STATUSES,
    ActionResult,
    FailureCategory,
    RunStatus,
)
from .task import TaskSpec
from .trace import RunResult, TraceStep

__all__ = [
    "ActionType",
    "ActionResult",
    "AgentAction",
    "FailureCategory",
    "Observation",
    "RunResult",
    "RunStatus",
    "TERMINAL_ACTIONS",
    "TERMINAL_STATUSES",
    "TaskSpec",
    "TraceStep",
]
