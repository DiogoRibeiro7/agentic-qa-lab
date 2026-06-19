"""Trace step and full run-result aggregation models."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from .action import AgentAction
from .observation import Observation
from .result import ActionResult, FailureCategory, RunStatus


class TraceStep(BaseModel):
    """A single (observation, action, result) tuple in a run.

    Attributes
    ----------
    index:
        Zero-based step index.
    observation:
        Environment state the agent acted on.
    action:
        Action the agent chose.
    result:
        Outcome of executing the action.
    """

    index: int = Field(ge=0)
    observation: Observation
    action: AgentAction
    result: ActionResult


class RunResult(BaseModel):
    """Aggregated outcome of a complete agent run.

    Attributes
    ----------
    task_id:
        Identifier of the task that was run.
    status:
        Terminal :class:`RunStatus`.
    failure_category:
        Taxonomy bucket; ``NONE`` on success.
    steps:
        Ordered trace steps.
    total_retries:
        Sum of retries across all steps.
    duration_seconds:
        End-to-end wall-clock duration.
    started_at, ended_at:
        Epoch-second timestamps bounding the run.
    total_tokens:
        LLM tokens consumed by the run, when measured (0 otherwise).
    cost_usd:
        Estimated LLM cost in USD, when measured (0 otherwise).
    """

    task_id: str = Field(min_length=1)
    status: RunStatus
    failure_category: FailureCategory = Field(default=FailureCategory.NONE)
    steps: list[TraceStep] = Field(default_factory=list)
    total_retries: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0)
    started_at: float = Field(gt=0)
    ended_at: float = Field(gt=0)
    total_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)

    @property
    def step_latency_ms(self) -> list[float]:
        """Per-step action latencies in milliseconds."""
        return [step.result.duration_ms for step in self.steps]

    @model_validator(mode="after")
    def _validate_consistency(self) -> RunResult:
        """Enforce internal consistency of the aggregated result.

        Raises
        ------
        ValueError
            If timestamps are out of order or success/failure categories are
            inconsistent with the terminal status.
        """
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must be greater than or equal to started_at.")
        if self.status is RunStatus.SUCCESS and self.failure_category is not FailureCategory.NONE:
            raise ValueError("Successful runs must have failure_category 'none'.")
        if self.status is not RunStatus.SUCCESS and self.failure_category is FailureCategory.NONE:
            raise ValueError("Non-successful runs require a failure_category other than 'none'.")
        return self

    @property
    def step_count(self) -> int:
        """Return the number of executed steps."""
        return len(self.steps)

    @property
    def succeeded(self) -> bool:
        """Return ``True`` when the run terminated successfully."""
        return self.status is RunStatus.SUCCESS
