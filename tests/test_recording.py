from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentic_qa_lab.domain import ActionType, AgentAction, TaskSpec
from agentic_qa_lab.evaluation import BenchmarkCase, build_recorded_plan, dump_case, record_case
from agentic_qa_lab.evaluation import recording as recording_module


class _FakePage:
    def __init__(self) -> None:
        self.binding_name: str | None = None
        self.binding: Any = None
        self.init_script: str | None = None
        self.goto_url: str | None = None

    def expose_binding(self, name: str, callback: Any) -> None:
        self.binding_name = name
        self.binding = callback

    def add_init_script(self, script: str) -> None:
        self.init_script = script

    def goto(self, url: str) -> None:
        self.goto_url = url


class _FakeBrowser:
    def __init__(self, page: _FakePage) -> None:
        self.page = page
        self.closed = False
        self.viewport: dict[str, int] | None = None

    def new_page(self, *, viewport: dict[str, int]) -> _FakePage:
        self.viewport = viewport
        return self.page

    def close(self) -> None:
        self.closed = True


class _FakeChromium:
    def __init__(self, browser: _FakeBrowser) -> None:
        self.browser = browser
        self.headless: bool | None = None

    def launch(self, *, headless: bool) -> _FakeBrowser:
        self.headless = headless
        return self.browser


class _FakePlaywright:
    def __init__(self, chromium: _FakeChromium) -> None:
        self.chromium = chromium


class _FakeSyncPlaywright:
    def __init__(self, playwright: _FakePlaywright) -> None:
        self.playwright = playwright

    def __enter__(self) -> _FakePlaywright:
        return self.playwright

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


def test_build_recorded_plan_collapses_repeated_type_text() -> None:
    plan = build_recorded_plan(
        [
            {"type": "click", "selector": "#user"},
            {"type": "type_text", "selector": "#user", "text": "a"},
            {"type": "type_text", "selector": "#user", "text": "alice"},
            {"type": "press_key", "selector": "#user", "key": "Enter"},
        ],
        finish_reason="done",
    )

    assert [action.type for action in plan] == [
        ActionType.CLICK,
        ActionType.TYPE_TEXT,
        ActionType.PRESS_KEY,
        ActionType.FINISH,
    ]
    assert plan[1].text == "alice"
    assert plan[-1].reason == "done"


def test_build_recorded_plan_skips_invalid_events() -> None:
    plan = build_recorded_plan(
        [
            {"type": "click"},
            {"type": "type_text", "selector": "#user", "text": ""},
            {"type": "press_key", "selector": "#user", "key": "Enter"},
        ]
    )

    assert plan == [
        AgentAction.press_key("Enter", selector="#user"),
        AgentAction.finish("recorded session complete"),
    ]


def test_dump_case_writes_yaml_and_json(tmp_path: Path) -> None:
    case = BenchmarkCase(
        task=TaskSpec(task_id="login", goal="Log in", start_url="https://e.com/"),
        plan=[AgentAction.click("#submit"), AgentAction.finish("done")],
    )

    yaml_path = dump_case(case, tmp_path / "case.yaml")
    json_path = dump_case(case, tmp_path / "case.json")

    yaml_text = yaml_path.read_text(encoding="utf-8")
    assert "task_id: login" in yaml_text
    assert "plan:" in yaml_text

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    assert raw["task_id"] == "login"
    assert raw["plan"][-1]["type"] == "finish"


def test_dump_case_can_replace_recorded_text_with_env_refs(tmp_path: Path) -> None:
    case = BenchmarkCase(
        task=TaskSpec(task_id="login", goal="Log in", start_url="https://e.com/"),
        plan=[
            AgentAction.type_text("alice", selector="#username"),
            AgentAction.type_text("super-secret", selector="#password"),
            AgentAction.finish("done"),
        ],
    )

    yaml_path = dump_case(
        case,
        tmp_path / "case.yaml",
        text_env_overrides={"#password": "AGENTIC_QA_PASSWORD"},
    )

    text = yaml_path.read_text(encoding="utf-8")
    assert "text: alice" in text
    assert "AGENTIC_QA_PASSWORD" in text
    assert "super-secret" not in text


def test_record_case_captures_events_and_closes_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage()
    browser = _FakeBrowser(page)
    chromium = _FakeChromium(browser)

    monkeypatch.setattr(
        "playwright.sync_api.sync_playwright",
        lambda: _FakeSyncPlaywright(_FakePlaywright(chromium)),
    )

    def wait_for_finish() -> None:
        assert page.binding is not None
        page.binding(None, {"type": "click", "selector": "#submit"})
        page.binding(None, {"type": "type_text", "selector": "#user", "text": "alice"})
        page.binding(None, "ignore non-dict payloads")

    task = TaskSpec(task_id="recorded", goal="Submit", start_url="https://e.com/")
    case = record_case(
        task,
        finish_reason="captured",
        headless=True,
        viewport=(800, 600),
        wait_for_finish=wait_for_finish,
    )

    assert case.task == task
    assert [action.type for action in case.plan] == [
        ActionType.CLICK,
        ActionType.TYPE_TEXT,
        ActionType.FINISH,
    ]
    assert case.plan[-1].reason == "captured"
    assert chromium.headless is True
    assert browser.viewport == {"width": 800, "height": 600}
    assert page.binding_name == "__agenticQaRecord"
    assert page.goto_url == "https://e.com/"
    assert page.init_script == recording_module._RECORDER_SCRIPT
    assert browser.closed is True


def test_record_case_closes_browser_on_wait_error(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage()
    browser = _FakeBrowser(page)

    monkeypatch.setattr(
        "playwright.sync_api.sync_playwright",
        lambda: _FakeSyncPlaywright(_FakePlaywright(_FakeChromium(browser))),
    )

    def wait_for_finish() -> None:
        raise RuntimeError("stop recording")

    task = TaskSpec(task_id="recorded", goal="Submit", start_url="https://e.com/")

    with pytest.raises(RuntimeError, match="stop recording"):
        record_case(task, wait_for_finish=wait_for_finish)

    assert browser.closed is True
