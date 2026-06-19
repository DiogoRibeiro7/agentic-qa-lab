"""End-to-end demo against a real (local) page with a real browser.

Unlike ``simple_form_task.py`` (which targets example.com), this drives the
bundled ``pages/login.html`` over a ``file://`` URL, so the whole loop —
Playwright environment, rule-based agent, runner, trace, screenshots — runs
locally with no network and no test server.

Prerequisites
-------------
Install the Chromium binary once::

    playwright install chromium

Then run::

    python examples/local_login_demo.py

Outputs the trace at ``artifacts/runs/local-login.jsonl`` and step screenshots
under ``artifacts/runs/local-login/screenshots/``.
"""

from __future__ import annotations

from pathlib import Path

from agentic_qa_lab.agents import RuleBasedAgent, Runner, write_trace_jsonl
from agentic_qa_lab.domain import AgentAction, TaskSpec
from agentic_qa_lab.environments import PlaywrightEnvironment

PAGE = Path(__file__).resolve().parent / "pages" / "login.html"

TASK = TaskSpec(
    task_id="local-login",
    goal="Fill in the login form and reach the welcome screen.",
    start_url=PAGE.as_uri(),
    success_selector="Welcome",
    max_steps=8,
)

PLAN: list[AgentAction] = [
    AgentAction.type_text("alice", selector="#username"),
    AgentAction.type_text("s3cret", selector="#password"),
    AgentAction.click("#submit"),
    AgentAction.finish("Submitted the login form."),
]


def main() -> None:
    """Run the local login task end-to-end and report the outcome."""
    out_dir = Path("artifacts") / "runs"
    screenshots = out_dir / TASK.task_id / "screenshots"
    with PlaywrightEnvironment.launch(headless=True, screenshot_dir=screenshots) as env:
        result = Runner().run(TASK, RuleBasedAgent(PLAN), env)

    trace_path = write_trace_jsonl(result, out_dir / f"{TASK.task_id}.jsonl")
    print(f"status        : {result.status.value}")
    print(f"steps         : {result.step_count}")
    print(f"duration (s)  : {result.duration_seconds:.2f}")
    print(f"trace         : {trace_path}")
    print(f"screenshots   : {screenshots}")


if __name__ == "__main__":
    main()
