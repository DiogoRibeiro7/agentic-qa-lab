"""Task specification for a single agent run."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TaskSpec(BaseModel):
    """Declarative description of a UI/game task for the agent to complete.

    A ``TaskSpec`` is environment-agnostic: it states *what* to achieve and the
    safeguards for the run, not *how* to achieve it.

    Attributes
    ----------
    task_id:
        Stable identifier, unique within a benchmark.
    goal:
        Natural-language goal handed to the planner.
    start_url:
        Initial page/URL the environment should open.
    success_selector:
        Optional selector whose presence signals success.
    success_judge:
        Optional natural-language success rubric evaluated by an LLM judge when
        enabled at runtime.
    max_steps:
        Hard cap on the number of agent steps.
    max_retries:
        Maximum retries allowed per failing action.
    timeout_seconds:
        Wall-clock budget for the whole run.
    metadata:
        Free-form labels (difficulty, category, ...).
    """

    task_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    start_url: str = Field(min_length=1)
    success_selector: str | None = Field(default=None)
    success_judge: str | None = Field(default=None)
    max_steps: int = Field(default=25, ge=1, le=1000)
    max_retries: int = Field(default=2, ge=0, le=20)
    timeout_seconds: float = Field(default=120.0, gt=0)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("start_url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        """Ensure ``start_url`` looks like a navigable target."""
        allowed_prefixes = ("http://", "https://", "file://", "about:")
        if not value.startswith(allowed_prefixes):
            raise ValueError("start_url must begin with one of: " + ", ".join(allowed_prefixes))
        return value
