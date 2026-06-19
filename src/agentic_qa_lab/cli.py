"""Command line interface for the project."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .config import RuntimeSettings

app = typer.Typer(help="Portfolio project command line interface.")
console = Console()


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
        BenchmarkCase,
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


if __name__ == "__main__":
    app()
