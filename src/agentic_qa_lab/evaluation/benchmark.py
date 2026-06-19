"""Run a set of benchmark cases and export aggregate results."""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

from ..agents import Runner
from ..agents.base import Agent
from ..domain import RunResult, RunStatus
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
        *,
        workers: int = 1,
    ) -> list[RunResult]:
        """Run every case and return the per-case :class:`RunResult` list.

        Each case gets a fresh agent and environment from the factories. If the
        environment is a context manager it is closed after the run. ``workers``
        controls how many cases may execute concurrently.
        """
        if workers <= 0:
            raise ValueError("workers must be >= 1.")
        if workers == 1 or len(cases) <= 1:
            return [self._run_case(case, agent_factory, env_factory) for case in cases]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(
                executor.map(lambda case: self._run_case(case, agent_factory, env_factory), cases)
            )

    def _run_case(
        self,
        case: BenchmarkCase,
        agent_factory: AgentFactory,
        env_factory: EnvFactory,
    ) -> RunResult:
        """Run one case with a fresh agent and environment."""
        agent = agent_factory(case)
        env = env_factory(case)
        # Some environments are themselves context managers. Use a no-op
        # context manager otherwise so the runner code can stay uniform.
        manager = env if isinstance(env, BrowserEnvironment) else nullcontext(env)
        with manager:
            return self._runner.run(case.task, agent, env)


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
    _export_junit(results, directory / "junit.xml")
    _export_allure(results, directory / "allure-results")
    return csv_path, json_path


def summarize(results: list[RunResult]) -> BenchmarkSummary:
    """Convenience re-export of :func:`compute_summary`."""
    return compute_summary(results)


def _export_junit(results: list[RunResult], path: Path) -> Path:
    """Write benchmark results as a JUnit XML report."""
    suite = ET.Element(
        "testsuite",
        name="agentic-qa benchmark",
        tests=str(len(results)),
        failures=str(
            sum(not run.succeeded and run.status is not RunStatus.ERROR for run in results)
        ),
        errors=str(sum(run.status is RunStatus.ERROR for run in results)),
        skipped="0",
        time=f"{sum(run.duration_seconds for run in results):.4f}",
    )

    for run in results:
        case = ET.SubElement(
            suite,
            "testcase",
            classname="agentic_qa_lab.benchmark",
            name=run.task_id,
            time=f"{run.duration_seconds:.4f}",
        )
        if run.status is RunStatus.SUCCESS:
            pass
        elif run.status is RunStatus.ERROR:
            error = ET.SubElement(
                case,
                "error",
                message=run.failure_category.value,
                type=run.status.value,
            )
            error.text = _failure_text(run)
        else:
            failure = ET.SubElement(
                case,
                "failure",
                message=run.failure_category.value,
                type=run.status.value,
            )
            failure.text = _failure_text(run)

        system_out = ET.SubElement(case, "system-out")
        system_out.text = _trace_summary(run)

    tree = ET.ElementTree(suite)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def _export_allure(results: list[RunResult], directory: Path) -> Path:
    """Write one minimal Allure result JSON file per run."""
    directory.mkdir(parents=True, exist_ok=True)
    for index, run in enumerate(results):
        payload = {
            "name": run.task_id,
            "fullName": f"agentic_qa_lab.benchmark::{run.task_id}",
            "status": _allure_status(run),
            "statusDetails": {
                "message": run.failure_category.value,
                "trace": _failure_text(run),
            },
            "stage": "finished",
            "start": int(run.started_at * 1000),
            "stop": int(run.ended_at * 1000),
            "labels": [
                {"name": "suite", "value": "agentic-qa benchmark"},
                {"name": "host", "value": "local"},
                {"name": "failure_category", "value": run.failure_category.value},
            ],
            "parameters": [
                {"name": "status", "value": run.status.value},
                {"name": "retries", "value": str(run.total_retries)},
                {"name": "steps", "value": str(run.step_count)},
            ],
            "attachments": [],
            "description": _trace_summary(run),
        }
        target = directory / f"{index:03d}-{run.task_id}-result.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return directory


def _trace_summary(run: RunResult) -> str:
    """Render a short textual step summary for exported reports."""
    if not run.steps:
        return "No recorded steps."
    lines = []
    for step in run.steps:
        target = step.action.selector or (
            f"({step.action.x}, {step.action.y})"
            if step.action.x is not None and step.action.y is not None
            else "-"
        )
        result = "ok" if step.result.success else f"failed:{step.result.failure_category.value}"
        lines.append(f"{step.index}: {step.action.type.value} {target} -> {result}")
    return "\n".join(lines)


def _failure_text(run: RunResult) -> str:
    """Return a compact failure description for one run."""
    if run.succeeded:
        return "Run succeeded."
    if run.steps and run.steps[-1].result.error:
        return run.steps[-1].result.error or run.failure_category.value
    return run.failure_category.value


def _allure_status(run: RunResult) -> str:
    """Map the run outcome onto a coarse Allure status."""
    if run.status is RunStatus.SUCCESS:
        return "passed"
    if run.status is RunStatus.ERROR:
        return "broken"
    return "failed"
