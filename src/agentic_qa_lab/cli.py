"""Command line interface for the project."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

from .agents import ObservationMode
from .config import RuntimeSettings

if TYPE_CHECKING:
    from .agents import Agent
    from .evaluation import BenchmarkCase

app = typer.Typer(help="Portfolio project command line interface.")
console = Console()


class AgentKind(StrEnum):
    """Selectable agent implementations for the ``run`` command."""

    RULE = "rule"
    LLM = "llm"


def build_agent(case: BenchmarkCase, kind: AgentKind, mode: ObservationMode) -> Agent:
    """Construct the agent named by ``kind`` for ``case``.

    The ``rule`` baseline replays the case's plan; the ``llm`` planner uses an
    OpenAI-compatible client with the given observation ``mode``.
    """
    from .agents import LLMPlannerAgent, OpenAICompatibleClient, RuleBasedAgent

    if kind is AgentKind.RULE:
        return RuleBasedAgent(case.plan)
    return LLMPlannerAgent(OpenAICompatibleClient(), observation_mode=mode)


@app.command()
def info() -> None:
    """Print validated runtime settings."""
    settings = RuntimeSettings()
    console.print(settings.model_dump())


@app.command()
def benchmark(
    tasks: Annotated[
        list[str],
        typer.Option("--tasks", help="Task files or glob patterns, e.g. tasks/*.yaml"),
    ],
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", help="Directory for the summary CSV/JSON."),
    ] = Path("artifacts/benchmark"),
    headless: Annotated[bool, typer.Option(help="Run the browser headless.")] = True,
) -> None:
    """Run the rule-based baseline over tasks and export summary metrics.

    Each task's ``plan`` drives the :class:`RuleBasedAgent`; a fresh
    Playwright browser is launched per task. Requires Chromium binaries
    (``playwright install chromium``).
    """
    from .agents import RuleBasedAgent
    from .environments import PlaywrightEnvironment
    from .evaluation import (
        BenchmarkRunner,
        compute_summary,
        export_results,
        load_cases,
    )

    cases = load_cases(list(tasks))
    if not cases:
        console.print("[red]No task files matched.[/red]")
        raise typer.Exit(code=1)

    console.print(f"Loaded [bold]{len(cases)}[/bold] task(s).")

    def make_agent(case: BenchmarkCase) -> RuleBasedAgent:
        return RuleBasedAgent(case.plan)

    def make_env(case: BenchmarkCase) -> PlaywrightEnvironment:
        return PlaywrightEnvironment.launch(headless=headless)

    results = BenchmarkRunner().run(cases, make_agent, make_env)
    csv_path, json_path = export_results(results, out_dir)
    summary = compute_summary(results)

    table = Table(title="Benchmark summary")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("tasks", str(summary.total))
    table.add_row("success rate", f"{summary.success_rate:.0%}")
    table.add_row("mean steps", f"{summary.mean_steps:.2f}")
    table.add_row("median steps", f"{summary.median_steps:.2f}")
    table.add_row("total retries", str(summary.total_retries))
    table.add_row("timeout rate", f"{summary.timeout_rate:.0%}")
    console.print(table)
    console.print(f"Wrote {csv_path} and {json_path}")


@app.command()
def run(
    task: Annotated[Path, typer.Option("--task", help="A single task YAML/JSON file.")],
    agent: Annotated[
        AgentKind, typer.Option(help="Agent: 'rule' baseline or 'llm' planner.")
    ] = AgentKind.RULE,
    mode: Annotated[
        ObservationMode, typer.Option(help="LLM grounding channel (llm agent only).")
    ] = ObservationMode.DOM_ONLY,
    reflect: Annotated[
        bool, typer.Option(help="Wrap the agent in a settle-and-retry repair loop.")
    ] = False,
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Directory for the trace JSONL + screenshots.")
    ] = Path("artifacts/runs"),
    headless: Annotated[bool, typer.Option(help="Run the browser headless.")] = True,
) -> None:
    """Run a single task with one agent and write its trace.

    Launches a fresh Playwright browser (requires ``playwright install
    chromium``), executes the task, prints the outcome, and stores the trace as
    ``<out_dir>/<task_id>.jsonl``.
    """
    from .agents import ReflectiveAgent, Runner, write_trace_jsonl
    from .environments import PlaywrightEnvironment
    from .evaluation import load_case

    case = load_case(task)
    chosen = build_agent(case, agent, mode)
    if reflect:
        chosen = ReflectiveAgent(chosen)
    runner = Runner(stop_on_action_failure=not reflect)

    screenshots = out_dir / case.task.task_id / "screenshots"
    with PlaywrightEnvironment.launch(headless=headless, screenshot_dir=screenshots) as env:
        result = runner.run(case.task, chosen, env)

    trace_path = write_trace_jsonl(result, out_dir / f"{case.task.task_id}.jsonl")

    table = Table(title=f"Run: {case.task.task_id}")
    table.add_column("field")
    table.add_column("value", justify="right")
    table.add_row("status", result.status.value)
    table.add_row("failure_category", result.failure_category.value)
    table.add_row("steps", str(result.step_count))
    table.add_row("retries", str(result.total_retries))
    table.add_row("duration_s", f"{result.duration_seconds:.2f}")
    console.print(table)
    console.print(f"Wrote {trace_path}")
    if not result.succeeded:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
