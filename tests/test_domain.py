from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_qa_lab.domain import (
    ActionResult,
    ActionType,
    AgentAction,
    FailureCategory,
    Observation,
    RunResult,
    RunStatus,
    TaskSpec,
    TraceStep,
)


# --------------------------------------------------------------------------- #
# AgentAction
# --------------------------------------------------------------------------- #
def test_click_requires_selector_or_coordinates() -> None:
    with pytest.raises(ValidationError):
        AgentAction(type=ActionType.CLICK)

    assert AgentAction.click("#submit").selector == "#submit"
    assert AgentAction.click(x=10, y=20).x == 10


def test_type_text_requires_non_empty_text() -> None:
    with pytest.raises(ValidationError):
        AgentAction(type=ActionType.TYPE_TEXT, selector="#name", text="")

    action = AgentAction.type_text("hello", selector="#name")
    assert action.text == "hello"


def test_press_key_requires_key() -> None:
    with pytest.raises(ValidationError):
        AgentAction(type=ActionType.PRESS_KEY, selector="#name")

    assert AgentAction.press_key("Enter", selector="#name").key == "Enter"


def test_wait_requires_positive_duration() -> None:
    with pytest.raises(ValidationError):
        AgentAction(type=ActionType.WAIT, duration_ms=0)

    assert AgentAction.wait(500).duration_ms == 500


def test_negative_coordinates_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentAction(type=ActionType.CLICK, x=-1, y=5)


def test_terminal_actions_flagged() -> None:
    assert AgentAction.finish("done").is_terminal
    assert AgentAction.fail("nope").is_terminal
    assert not AgentAction.click("#a").is_terminal


# --------------------------------------------------------------------------- #
# TaskSpec
# --------------------------------------------------------------------------- #
def test_task_spec_rejects_bad_url() -> None:
    with pytest.raises(ValidationError):
        TaskSpec(task_id="t1", goal="do it", start_url="ftp://example.com")


def test_task_spec_defaults() -> None:
    task = TaskSpec(task_id="t1", goal="do it", start_url="https://example.com")
    assert task.max_steps == 25
    assert task.max_retries == 2
    assert task.timeout_seconds > 0
    assert task.success_judge is None


def test_task_spec_step_bounds() -> None:
    with pytest.raises(ValidationError):
        TaskSpec(task_id="t1", goal="g", start_url="https://e.com", max_steps=0)


# --------------------------------------------------------------------------- #
# Observation
# --------------------------------------------------------------------------- #
def test_observation_timestamp_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Observation(step=0, url="https://e.com", timestamp=0)


def test_observation_viewport_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Observation(step=0, url="https://e.com", timestamp=1.0, viewport=(0, 600))

    obs = Observation(
        step=1,
        url="https://e.com",
        timestamp=1.0,
        screenshot_path="shots/s.png",
        viewport=(800, 600),
    )
    assert obs.has_visual is True


# --------------------------------------------------------------------------- #
# ActionResult
# --------------------------------------------------------------------------- #
def test_action_result_helpers() -> None:
    ok = ActionResult.ok(duration_ms=12.0)
    assert ok.success and ok.failure_category is FailureCategory.NONE

    bad = ActionResult.failed("missing", category=FailureCategory.ELEMENT_NOT_FOUND, retries=2)
    assert not bad.success
    assert bad.failure_category is FailureCategory.ELEMENT_NOT_FOUND
    assert bad.retries == 2


# --------------------------------------------------------------------------- #
# RunResult
# --------------------------------------------------------------------------- #
def _step() -> TraceStep:
    return TraceStep(
        index=0,
        observation=Observation(step=0, url="https://e.com", timestamp=1.0),
        action=AgentAction.finish("done"),
        result=ActionResult.ok(),
    )


def test_run_result_success_must_have_no_failure_category() -> None:
    with pytest.raises(ValidationError):
        RunResult(
            task_id="t1",
            status=RunStatus.SUCCESS,
            failure_category=FailureCategory.TIMEOUT,
            started_at=1.0,
            ended_at=2.0,
        )


def test_run_result_failure_requires_category() -> None:
    with pytest.raises(ValidationError):
        RunResult(
            task_id="t1",
            status=RunStatus.FAILURE,
            failure_category=FailureCategory.NONE,
            started_at=1.0,
            ended_at=2.0,
        )


def test_run_result_timestamps_ordered() -> None:
    with pytest.raises(ValidationError):
        RunResult(task_id="t1", status=RunStatus.SUCCESS, started_at=2.0, ended_at=1.0)


def test_run_result_happy_path() -> None:
    run = RunResult(
        task_id="t1",
        status=RunStatus.SUCCESS,
        steps=[_step()],
        started_at=1.0,
        ended_at=2.5,
        duration_seconds=1.5,
    )
    assert run.succeeded
    assert run.step_count == 1
