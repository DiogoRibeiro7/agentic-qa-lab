"""Command line interface for the project."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from .agents import ApprovalDecision, ObservationMode
from .config import RuntimeSettings

if TYPE_CHECKING:
    from .agents import Agent, LLMClient, SuccessJudge, TokenMeter
    from .domain import AgentAction
    from .environments import BrowserEnvironment
    from .evaluation import BenchmarkCase

app = typer.Typer(help="Portfolio project command line interface.")
console = Console()


class AgentKind(StrEnum):
    """Selectable agent implementations for the ``run`` command."""

    RULE = "rule"
    LLM = "llm"


class EnvironmentKind(StrEnum):
    """Selectable environment implementations for CLI execution."""

    AUTO = "auto"
    PLAYWRIGHT = "playwright"
    SELENIUM = "selenium"
    APPIUM = "appium"
    API = "api"


def build_agent(
    case: BenchmarkCase,
    kind: AgentKind,
    mode: ObservationMode,
    *,
    meter: TokenMeter | None = None,
) -> Agent:
    """Construct the agent named by ``kind`` for ``case``.

    The ``rule`` baseline replays the case's plan; the ``llm`` planner uses an
    OpenAI-compatible client with the given observation ``mode``. When ``meter``
    is supplied the LLM client is wrapped so token usage/cost is recorded.
    """
    from .agents import (
        LLMPlannerAgent,
        MeteredClient,
        OpenAICompatibleClient,
        RuleBasedAgent,
    )

    if kind is AgentKind.RULE:
        return RuleBasedAgent(case.plan)
    client: LLMClient = OpenAICompatibleClient()
    if meter is not None:
        client = MeteredClient(client, meter)
    return LLMPlannerAgent(client, observation_mode=mode)


def build_success_judge(*, enabled: bool, meter: TokenMeter | None = None) -> SuccessJudge | None:
    """Construct an optional LLM-backed semantic success judge."""
    if not enabled:
        return None
    from .agents import LLMSuccessJudge, MeteredClient, OpenAICompatibleClient

    client: LLMClient = OpenAICompatibleClient()
    if meter is not None:
        client = MeteredClient(client, meter)
    return LLMSuccessJudge(client)


def build_environment(
    case: BenchmarkCase,
    kind: EnvironmentKind,
    *,
    headless: bool = True,
    screenshot_dir: Path | None = None,
    appium_server: str = "http://127.0.0.1:4723",
    appium_capabilities_file: Path | None = None,
) -> BrowserEnvironment:
    """Construct the selected execution environment for ``case``."""
    from .environments import (
        APIEnvironment,
        AppiumEnvironment,
        PlaywrightEnvironment,
        SeleniumEnvironment,
    )

    resolved = _resolve_environment_kind(case, kind)
    if resolved is EnvironmentKind.API:
        return APIEnvironment(case.task.start_url)
    if resolved is EnvironmentKind.PLAYWRIGHT:
        return PlaywrightEnvironment.launch(headless=headless, screenshot_dir=screenshot_dir)
    if resolved is EnvironmentKind.SELENIUM:
        return SeleniumEnvironment.launch(headless=headless, screenshot_dir=screenshot_dir)
    if resolved is EnvironmentKind.APPIUM:
        return AppiumEnvironment.launch(
            command_executor=appium_server,
            capabilities=_load_capabilities(appium_capabilities_file),
            screenshot_dir=screenshot_dir,
        )
    raise ValueError(f"Unsupported environment kind: {resolved}")


def _resolve_environment_kind(case: BenchmarkCase, kind: EnvironmentKind) -> EnvironmentKind:
    """Resolve ``auto`` into a concrete environment choice."""
    if kind is not EnvironmentKind.AUTO:
        return kind
    if _looks_like_api_case(case):
        return EnvironmentKind.API
    start_url = case.task.start_url
    if start_url.startswith(("appium://", "appium:")):
        return EnvironmentKind.APPIUM
    if start_url.startswith(("http://", "https://", "file://", "about:")):
        return EnvironmentKind.PLAYWRIGHT
    return EnvironmentKind.API


def _looks_like_api_case(case: BenchmarkCase) -> bool:
    """Heuristically detect task files authored for ``APIEnvironment``."""
    api_selectors = ("#method", "#path", "#body", "#send", "#header:", "#query:")
    for action in case.plan:
        selector = action.selector or ""
        if selector == "#send":
            return True
        if any(selector.startswith(prefix) for prefix in api_selectors):
            return True
    return case.task.metadata.get("environment") == "api"


def _load_capabilities(path: Path | None) -> dict[str, object]:
    """Load an Appium capabilities mapping from JSON or YAML."""
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported capabilities file extension: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError("Appium capabilities file must contain a mapping/object.")
    return dict(data)


def _console_approver(action: AgentAction) -> ApprovalDecision:
    """Approver that asks the operator to confirm a risky action on the console."""
    target = action.selector or (action.x, action.y)
    prompt = f"Approve risky action '{action.type.value}' on {target}? " "[y]es/[a]ll/[n]o"
    while True:
        choice = typer.prompt(prompt, default="n").strip().lower()
        if choice in {"y", "yes"}:
            return ApprovalDecision.ALLOW_ONCE
        if choice in {"a", "all"}:
            return ApprovalDecision.ALLOW_SESSION
        if choice in {"n", "no"}:
            return ApprovalDecision.DENY
        console.print("[red]Enter 'y', 'a', or 'n'.[/red]")


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
    workers: Annotated[
        int,
        typer.Option("--workers", min=1, help="Number of benchmark cases to run concurrently."),
    ] = 1,
    environment: Annotated[
        EnvironmentKind,
        typer.Option(help="Execution environment: auto/playwright/selenium/appium/api."),
    ] = EnvironmentKind.AUTO,
    judge_success: Annotated[
        bool, typer.Option(help="Enable LLM judging for tasks that define success_judge.")
    ] = False,
    appium_server: Annotated[
        str, typer.Option(help="Appium server URL (appium environment only).")
    ] = "http://127.0.0.1:4723",
    appium_capabilities_file: Annotated[
        Path | None,
        typer.Option(help="JSON/YAML Appium capabilities file (appium environment only)."),
    ] = None,
    headless: Annotated[bool, typer.Option(help="Run the browser headless.")] = True,
) -> None:
    """Run the rule-based baseline over tasks and export summary metrics.

    Each task's ``plan`` drives the :class:`RuleBasedAgent`; a fresh
    Playwright browser is launched per task. Requires Chromium binaries
    (``playwright install chromium``).
    """
    from .agents import RuleBasedAgent, Runner
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
        """Create the rule-based agent for a benchmark case."""
        return RuleBasedAgent(case.plan)

    def make_env(case: BenchmarkCase) -> BrowserEnvironment:
        """Launch a fresh environment for a benchmark case."""
        return build_environment(
            case,
            environment,
            headless=headless,
            appium_server=appium_server,
            appium_capabilities_file=appium_capabilities_file,
        )

    judge = build_success_judge(enabled=judge_success)
    results = BenchmarkRunner(runner=Runner(success_judge=judge)).run(
        cases, make_agent, make_env, workers=workers
    )
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
    table.add_row("mean step latency (ms)", f"{summary.mean_step_latency_ms:.1f}")
    table.add_row("p95 step latency (ms)", f"{summary.p95_step_latency_ms:.1f}")
    table.add_row("mean obs latency (ms)", f"{summary.mean_observation_latency_ms:.1f}")
    console.print(table)
    console.print(f"Wrote {csv_path} and {json_path}")


@app.command()
def record(
    task_id: Annotated[str, typer.Option(help="Stable id for the recorded task file.")],
    goal: Annotated[str, typer.Option(help="Goal text to store in the task file.")],
    start_url: Annotated[str, typer.Option(help="URL to open before recording starts.")],
    out_file: Annotated[
        Path, typer.Option("--out-file", help="Output YAML/JSON task file to write.")
    ],
    success_selector: Annotated[
        str | None,
        typer.Option(help="Optional selector or visible text used as the success marker."),
    ] = None,
    finish_reason: Annotated[
        str, typer.Option(help="Reason attached to the final recorded finish action.")
    ] = "recorded session complete",
    max_steps: Annotated[int, typer.Option(help="Task max_steps safeguard.")] = 25,
    max_retries: Annotated[int, typer.Option(help="Task max_retries safeguard.")] = 2,
    timeout_seconds: Annotated[float, typer.Option(help="Task timeout_seconds safeguard.")] = 120.0,
    headless: Annotated[
        bool,
        typer.Option(help="Run the browser headless; headed is usually better for recording."),
    ] = False,
) -> None:
    """Record a manual browser session into a reusable task file."""
    from .domain import TaskSpec
    from .evaluation import dump_case, record_case

    task = TaskSpec(
        task_id=task_id,
        goal=goal,
        start_url=start_url,
        success_selector=success_selector,
        max_steps=max_steps,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
    )

    console.print(
        "Recording manual session. Interact with the browser, then press Enter here to save."
    )

    case = record_case(
        task,
        finish_reason=finish_reason,
        headless=headless,
        wait_for_finish=lambda: typer.prompt(
            "Press Enter when the manual flow is complete", default="", show_default=False
        ),
    )
    path = dump_case(case, out_file)

    table = Table(title=f"Recorded: {task.task_id}")
    table.add_column("field")
    table.add_column("value", justify="right")
    table.add_row("start_url", task.start_url)
    table.add_row("actions", str(len(case.plan)))
    table.add_row("output", str(path))
    console.print(table)


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
    self_heal: Annotated[
        bool,
        typer.Option(help="Retry element-not-found actions with nearby selector alternatives."),
    ] = False,
    environment: Annotated[
        EnvironmentKind,
        typer.Option(help="Execution environment: auto/playwright/selenium/appium/api."),
    ] = EnvironmentKind.AUTO,
    judge_success: Annotated[
        bool, typer.Option(help="Enable LLM judging for tasks that define success_judge.")
    ] = False,
    require_approval: Annotated[
        bool,
        typer.Option(
            help="Prompt before risky actions; supports yes/no or approve-all-for-this-run."
        ),
    ] = False,
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Directory for the trace JSONL + screenshots.")
    ] = Path("artifacts/runs"),
    appium_server: Annotated[
        str, typer.Option(help="Appium server URL (appium environment only).")
    ] = "http://127.0.0.1:4723",
    appium_capabilities_file: Annotated[
        Path | None,
        typer.Option(help="JSON/YAML Appium capabilities file (appium environment only)."),
    ] = None,
    headless: Annotated[bool, typer.Option(help="Run the browser headless.")] = True,
    price_in: Annotated[float, typer.Option(help="USD per 1k input tokens (llm agent).")] = 0.0,
    price_out: Annotated[float, typer.Option(help="USD per 1k output tokens (llm agent).")] = 0.0,
) -> None:
    """Run a single task with one agent and write its trace.

    Launches a fresh environment, executes the task, prints the outcome, and stores the trace as
    ``<out_dir>/<task_id>.jsonl``. With the ``llm`` agent, token usage is metered
    and (given ``--price-in``/``--price-out``) costed.
    """
    from .agents import (
        ApprovalAgent,
        ReflectiveAgent,
        Runner,
        SelfHealingAgent,
        TokenMeter,
        write_trace_jsonl,
    )
    from .evaluation import load_case

    case = load_case(task)
    meter = (
        TokenMeter(price_per_1k_input=price_in, price_per_1k_output=price_out)
        if agent is AgentKind.LLM or judge_success
        else None
    )
    chosen = build_agent(case, agent, mode, meter=meter)
    if self_heal:
        chosen = SelfHealingAgent(chosen)
    if reflect:
        chosen = ReflectiveAgent(chosen)
    if require_approval:
        # Gate risky actions outermost, so it sees what would actually execute.
        chosen = ApprovalAgent(chosen, approver=_console_approver)
    judge = build_success_judge(enabled=judge_success, meter=meter)
    runner = Runner(stop_on_action_failure=not reflect, success_judge=judge)

    screenshots = out_dir / case.task.task_id / "screenshots"
    resolved_environment = _resolve_environment_kind(case, environment)
    screenshot_target = screenshots if resolved_environment is not EnvironmentKind.API else None
    with build_environment(
        case,
        environment,
        headless=headless,
        screenshot_dir=screenshot_target,
        appium_server=appium_server,
        appium_capabilities_file=appium_capabilities_file,
    ) as env:
        result = runner.run(case.task, chosen, env)
    if meter is not None:
        result = result.model_copy(
            update={"total_tokens": meter.total_tokens, "cost_usd": meter.cost_usd}
        )

    trace_path = write_trace_jsonl(result, out_dir / f"{case.task.task_id}.jsonl")

    table = Table(title=f"Run: {case.task.task_id}")
    table.add_column("field")
    table.add_column("value", justify="right")
    table.add_row("status", result.status.value)
    table.add_row("failure_category", result.failure_category.value)
    table.add_row("steps", str(result.step_count))
    table.add_row("retries", str(result.total_retries))
    table.add_row("duration_s", f"{result.duration_seconds:.2f}")
    if meter is not None:
        table.add_row("tokens", str(result.total_tokens))
        table.add_row("cost_usd", f"{result.cost_usd:.6f}")
    console.print(table)
    console.print(f"Wrote {trace_path}")
    if not result.succeeded:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
