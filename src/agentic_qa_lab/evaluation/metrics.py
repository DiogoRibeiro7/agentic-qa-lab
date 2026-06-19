"""Aggregate metrics computed over a set of runs."""

from __future__ import annotations

from collections import Counter
from statistics import mean, median

from pydantic import BaseModel, Field

from ..domain import RunResult, RunStatus


class BenchmarkSummary(BaseModel):
    """Headline metrics for a benchmark of one agent over many tasks.

    Attributes
    ----------
    total:
        Number of runs.
    successes:
        Number of runs with a ``success`` status.
    success_rate:
        ``successes / total`` (``0.0`` when there are no runs).
    mean_steps, median_steps:
        Central tendency of steps-to-completion across runs.
    total_retries:
        Sum of retries across all runs.
    timeout_rate:
        Fraction of runs that ended in ``timeout``.
    failure_categories:
        Count of runs per terminal :class:`FailureCategory` value.
    """

    total: int = Field(ge=0)
    successes: int = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)
    mean_steps: float = Field(ge=0.0)
    median_steps: float = Field(ge=0.0)
    total_retries: int = Field(ge=0)
    timeout_rate: float = Field(ge=0.0, le=1.0)
    failure_categories: dict[str, int] = Field(default_factory=dict)


def compute_summary(results: list[RunResult]) -> BenchmarkSummary:
    """Reduce a list of :class:`RunResult` to a :class:`BenchmarkSummary`.

    An empty input yields an all-zero summary rather than raising, so callers
    can summarize partial benchmarks safely.
    """
    total = len(results)
    if total == 0:
        return BenchmarkSummary(
            total=0,
            successes=0,
            success_rate=0.0,
            mean_steps=0.0,
            median_steps=0.0,
            total_retries=0,
            timeout_rate=0.0,
            failure_categories={},
        )

    successes = sum(1 for r in results if r.status is RunStatus.SUCCESS)
    timeouts = sum(1 for r in results if r.status is RunStatus.TIMEOUT)
    step_counts = [r.step_count for r in results]
    categories = Counter(r.failure_category.value for r in results)

    return BenchmarkSummary(
        total=total,
        successes=successes,
        success_rate=successes / total,
        mean_steps=float(mean(step_counts)),
        median_steps=float(median(step_counts)),
        total_retries=sum(r.total_retries for r in results),
        timeout_rate=timeouts / total,
        failure_categories=dict(categories),
    )
