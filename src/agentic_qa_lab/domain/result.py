"""Per-action result and terminal run-status definitions."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class RunStatus(StrEnum):
    """Terminal status of a completed run."""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    MAX_STEPS = "max_steps"
    ERROR = "error"


#: Statuses considered terminal for a run.
TERMINAL_STATUSES: frozenset[RunStatus] = frozenset(RunStatus)


class FailureCategory(StrEnum):
    """Coarse taxonomy used to bucket non-successful outcomes."""

    NONE = "none"
    ELEMENT_NOT_FOUND = "element_not_found"
    TIMEOUT = "timeout"
    NAVIGATION_ERROR = "navigation_error"
    INVALID_ACTION = "invalid_action"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    AGENT_ERROR = "agent_error"
    JUDGE_REJECTED = "judge_rejected"
    UNKNOWN = "unknown"


class ActionResult(BaseModel):
    """Outcome of executing a single :class:`AgentAction` in an environment.

    Attributes
    ----------
    success:
        Whether the action was applied without error.
    error:
        Error message when ``success`` is ``False``.
    failure_category:
        Taxonomy bucket for the error.
    retries:
        Number of retries consumed while applying the action.
    duration_ms:
        Wall-clock time spent applying the action.
    """

    success: bool
    error: str | None = Field(default=None)
    failure_category: FailureCategory = Field(default=FailureCategory.NONE)
    retries: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0)

    @classmethod
    def ok(cls, *, retries: int = 0, duration_ms: float = 0.0) -> ActionResult:
        """Construct a successful result."""
        return cls(success=True, retries=retries, duration_ms=duration_ms)

    @classmethod
    def failed(
        cls,
        error: str,
        *,
        category: FailureCategory = FailureCategory.UNKNOWN,
        retries: int = 0,
        duration_ms: float = 0.0,
    ) -> ActionResult:
        """Construct a failed result with a taxonomy category."""
        return cls(
            success=False,
            error=error,
            failure_category=category,
            retries=retries,
            duration_ms=duration_ms,
        )
