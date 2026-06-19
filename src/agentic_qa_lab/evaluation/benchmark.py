"""Run a set of benchmark cases and export aggregate results."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

from ..agents import Runner
from ..agents.base import Agent
from ..domain import RunResult
from ..environments import BrowserEnvironment
from .metrics import BenchmarkSummary, compute_summary
from .tasks import BenchmarkCase

#: Builds the agent used for a given case.
AgentFactory = Callable[[BenchmarkCase], Agent]
#: Builds the environment used for a given case.
EnvFactory = Callable[[BenchmarkCase], BrowserEnvironment]


class BenchmarkRunner:
    """Execute many :class:`BenchmarkCase` runs and aggregate the outcomes.

    Parameters
    ----------
    runner:
        The :class:`Runner` used for each case. A default is created when not
        supplied.
    """

    def __init__(self, runner: Runner | None = None) -> None:
        self._runner = runner or Runner()

    def run(
        self,
        cases: list[BenchmarkCase],
        agent_factory: AgentFactory,
        env_factory: EnvFactory,
    ) -> list[RunResult]:
        """Run every case and return the per-case :class:`RunResult` list.

        Each case gets a fresh agent and environment from the factories. If the
        environment is a context manager it is closed after the run.
        """
        results: list[RunResult] = []
        for case in cases:
            agent = agent_factory(case)
            env = env_factory(case)
            manager = env if isinstance(env, BrowserEnvironment) else nullcontext(env)
            with manager:
                results.append(self._runner.run(case.task, agent, env))
        return results


def export_results(
    results: list[RunResult],
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Write per-task CSV and a summary JSON to ``out_dir``.

    Produces ``benchmark_summary.csv`` (one row per run plus aggregate fields)
    and ``benchmark_summary.json`` (the :class:`BenchmarkSummary` plus per-run
    detail). Returns the two paths.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    summary = compute_summary(results)

    csv_path = directory / "benchmark_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "task_id",
                "status",
                "failure_category",
                "steps",
                "retries",
                "duration_seconds",
                "tokens",
                "cost_usd",
            ]
        )
        for run in results:
            writer.writerow(
                [
                    run.task_id,
                    run.status.value,
                    run.failure_category.value,
                    run.step_count,
                    run.total_retries,
                    f"{run.duration_seconds:.4f}",
                    run.total_tokens,
                    f"{run.cost_usd:.6f}",
                ]
            )

    json_path = directory / "benchmark_summary.json"
    payload = {
        "summary": summary.model_dump(),
        "runs": [
            {
                "task_id": run.task_id,
                "status": run.status.value,
                "failure_category": run.failure_category.value,
                "steps": run.step_count,
                "retries": run.total_retries,
                "duration_seconds": run.duration_seconds,
                "tokens": run.total_tokens,
                "cost_usd": run.cost_usd,
            }
            for run in results
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


def summarize(results: list[RunResult]) -> BenchmarkSummary:
    """Convenience re-export of :func:`compute_summary`."""
    return compute_summary(results)
