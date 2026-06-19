"""Validate the bundled real-site benchmark tasks.

Parsing/shape checks always run (no network). The live runs are opt-in: set
``AGENTIC_QA_RUN_NETWORK_TESTS=1`` and have Chromium installed. Otherwise they
skip, so CI without network/browsers stays green.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from agentic_qa_lab.agents import RuleBasedAgent, Runner
from agentic_qa_lab.domain import ActionType, RunStatus
from agentic_qa_lab.evaluation import BenchmarkCase, load_cases

REAL_GLOBS = ["tasks/real/*.yaml", "tasks/real/*.json"]
RUN_NETWORK = os.environ.get("AGENTIC_QA_RUN_NETWORK_TESTS") == "1"
os.environ.setdefault("AGENTIC_QA_HEROKUAPP_PASSWORD", "SuperSecretPassword!")


def _real_cases() -> list[BenchmarkCase]:
    return load_cases(REAL_GLOBS)


def test_real_task_dir_exists() -> None:
    assert Path("tasks/real").is_dir()


def test_real_tasks_parse_and_are_well_formed() -> None:
    cases = _real_cases()
    assert len(cases) >= 6, "expected an expanded real-site task pack"

    for case in cases:
        task = case.task
        assert task.start_url.startswith("https://"), task.task_id
        assert task.success_selector, f"{task.task_id} needs a success_selector"
        assert case.plan, f"{task.task_id} needs a baseline plan"
        # A well-formed baseline plan ends by finishing.
        assert case.plan[-1].type is ActionType.FINISH, task.task_id


def test_real_tasks_cover_multiple_categories_and_difficulties() -> None:
    cases = _real_cases()
    categories = {case.task.metadata.get("category") for case in cases}
    difficulties = {case.task.metadata.get("difficulty") for case in cases}

    assert {"auth", "dynamic", "keyboard"} <= categories
    assert {"easy", "medium"} <= difficulties


def test_real_task_ids_are_unique() -> None:
    ids = [c.task.task_id for c in _real_cases()]
    assert len(ids) == len(set(ids))


def _launch_or_skip(screenshot_dir: Path) -> Any:
    pytest.importorskip("playwright")
    from agentic_qa_lab.environments import PlaywrightEnvironment

    try:
        return PlaywrightEnvironment.launch(headless=True, screenshot_dir=screenshot_dir)
    except Exception as exc:  # noqa: BLE001 - any launch failure means "no browser here"
        pytest.skip(f"Chromium not available: {exc}")


@pytest.mark.skipif(
    not RUN_NETWORK,
    reason="set AGENTIC_QA_RUN_NETWORK_TESTS=1 to run live real-site tasks",
)
@pytest.mark.parametrize("case", _real_cases(), ids=lambda c: c.task.task_id)
def test_real_task_runs_live(case: BenchmarkCase, tmp_path: Path) -> None:
    env = _launch_or_skip(tmp_path / "shots")
    with env:
        run = Runner().run(case.task, RuleBasedAgent(case.plan), env)
    assert run.status is RunStatus.SUCCESS, f"{case.task.task_id}: {run.failure_category.value}"
