"""End-to-end test against the bundled local page with a real browser.

Skips automatically when Playwright's Chromium binary is not installed, so the
unit suite (and CI without ``playwright install``) stays green while the test
still runs locally once browsers are present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentic_qa_lab.agents import RuleBasedAgent, Runner, write_trace_jsonl
from agentic_qa_lab.domain import AgentAction, RunStatus, TaskSpec

PAGE = Path(__file__).resolve().parent.parent / "examples" / "pages" / "login.html"


def _launch_or_skip(screenshot_dir: Path) -> Any:
    """Launch Chromium, or skip the test if its binary is unavailable."""
    pytest.importorskip("playwright")
    from agentic_qa_lab.environments import PlaywrightEnvironment

    try:
        return PlaywrightEnvironment.launch(headless=True, screenshot_dir=screenshot_dir)
    except Exception as exc:  # noqa: BLE001 - any launch failure means "no browser here"
        pytest.skip(f"Chromium not available: {exc}")


def _plan() -> list[AgentAction]:
    return [
        AgentAction.type_text("alice", selector="#username"),
        AgentAction.type_text("s3cret", selector="#password"),
        AgentAction.click("#submit"),
        AgentAction.finish("submitted"),
    ]


def test_local_login_succeeds(tmp_path: Path) -> None:
    assert PAGE.exists(), "bundled login page is missing"
    env = _launch_or_skip(tmp_path / "shots")

    task = TaskSpec(
        task_id="local-login",
        goal="Log into the local demo page.",
        start_url=PAGE.as_uri(),
        success_selector="Welcome",
        max_steps=8,
    )
    with env:
        run = Runner().run(task, RuleBasedAgent(_plan()), env)

    assert run.status is RunStatus.SUCCESS
    # The success banner text must have been captured in a DOM snapshot.
    assert any("Welcome" in (s.observation.dom_snapshot or "") for s in run.steps)

    trace = write_trace_jsonl(run, tmp_path / "local-login.jsonl")
    assert trace.exists()


def test_local_login_fails_without_credentials(tmp_path: Path) -> None:
    # Submitting empty fields must not reveal the welcome banner -> finish fails.
    env = _launch_or_skip(tmp_path / "shots")

    task = TaskSpec(
        task_id="local-login-empty",
        goal="Submit the form with no input.",
        start_url=PAGE.as_uri(),
        success_selector="Welcome",
        max_steps=4,
    )
    plan = [AgentAction.click("#submit"), AgentAction.finish("submitted")]
    with env:
        run = Runner().run(task, RuleBasedAgent(plan), env)

    assert run.status is RunStatus.FAILURE
