from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_qa_lab.domain import AgentAction, FailureCategory, TaskSpec
from agentic_qa_lab.environments import AppiumEnvironment


class FakeTimeoutError(Exception):
    """Stand-in mimicking Appium TimeoutException by class name."""


FakeTimeoutError.__name__ = "TimeoutException"


class FakeElement:
    def __init__(self, *, exc: Exception | None = None) -> None:
        self.exc = exc
        self.clicks = 0
        self.sent: list[Any] = []
        self.cleared = 0

    def click(self) -> None:
        self.clicks += 1
        if self.exc is not None:
            raise self.exc

    def clear(self) -> None:
        self.cleared += 1
        if self.exc is not None:
            raise self.exc

    def send_keys(self, value: Any) -> None:
        self.sent.append(value)
        if self.exc is not None:
            raise self.exc


class FakeDriver:
    def __init__(self, *, exc: Exception | None = None, has_url: bool = False) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.closed = False
        self.saved: list[str] = []
        self.exc = exc
        self.current_url = "https://example.com/" if has_url else ""
        self.current_package = "com.example.app"
        self.current_activity = ".MainActivity"
        self.title = ""
        self.page_source = "<hierarchy>visible text</hierarchy>"
        self.element = FakeElement(exc=exc)

    def get(self, url: str) -> None:
        self.calls.append(("get", (url,)))
        if self.exc is not None:
            raise self.exc
        self.current_url = url

    def save_screenshot(self, path: str) -> None:
        self.calls.append(("save_screenshot", (path,)))
        self.saved.append(path)

    def tap(self, coords: list[tuple[int, int]]) -> None:
        self.calls.append(("tap", tuple(coords)))
        if self.exc is not None:
            raise self.exc

    def find_element(self, by: str, value: str) -> FakeElement:
        self.calls.append(("find_element", (by, value)))
        if self.exc is not None:
            raise self.exc
        return self.element

    def execute_script(self, script: str, payload: dict[str, str]) -> str:
        self.calls.append(("execute_script", (script, payload)))
        if self.exc is not None:
            raise self.exc
        return "Visible text"

    def get_window_size(self) -> dict[str, int]:
        return {"width": 390, "height": 844}

    def quit(self) -> None:
        self.closed = True


def test_task_spec_allows_appium_start_url() -> None:
    task = TaskSpec(task_id="mobile", goal="Open app", start_url="appium://session")
    assert task.start_url == "appium://session"


def test_open_native_session_returns_observation() -> None:
    driver = FakeDriver()
    env = AppiumEnvironment(driver)

    obs = env.open("appium://session")

    assert obs.url == "appium://com.example.app/.MainActivity"
    assert obs.viewport == (390, 844)
    assert not any(name == "get" for name, _args in driver.calls)


def test_open_web_url_uses_get() -> None:
    driver = FakeDriver(has_url=True)
    env = AppiumEnvironment(driver)

    obs = env.open("https://example.com")

    assert obs.url == "https://example.com"
    assert ("get", ("https://example.com",)) in driver.calls


def test_execute_click_type_press_and_wait() -> None:
    driver = FakeDriver()
    env = AppiumEnvironment(driver)

    assert env.execute(AgentAction.click("id=login")).success
    assert env.execute(AgentAction.type_text("alice", selector="accessibility_id=username")).success
    assert env.execute(AgentAction.press_key("Enter", selector="xpath=//input[1]")).success
    assert env.execute(AgentAction.wait(1)).success

    assert ("find_element", ("id", "login")) in driver.calls
    assert ("find_element", ("accessibility id", "username")) in driver.calls
    assert ("find_element", ("xpath", "//input[1]")) in driver.calls


def test_coordinate_click_uses_tap() -> None:
    driver = FakeDriver()
    env = AppiumEnvironment(driver)

    assert env.execute(AgentAction.click(x=10, y=20)).success
    assert ("tap", ((10, 20),)) in driver.calls


def test_timeout_is_categorized() -> None:
    driver = FakeDriver(exc=FakeTimeoutError("timed out"))
    env = AppiumEnvironment(driver)

    result = env.execute(AgentAction.click("id=missing"))

    assert not result.success
    assert result.failure_category is FailureCategory.TIMEOUT


def test_element_not_found_is_categorized() -> None:
    driver = FakeDriver(exc=RuntimeError("no such element"))
    env = AppiumEnvironment(driver)

    result = env.execute(AgentAction.click("id=missing"))

    assert not result.success
    assert result.failure_category is FailureCategory.ELEMENT_NOT_FOUND


def test_terminal_actions_are_noops() -> None:
    driver = FakeDriver()
    env = AppiumEnvironment(driver)

    assert env.execute(AgentAction.finish("done")).success
    assert env.execute(AgentAction.fail("nope")).success
    assert driver.calls == []


def test_screenshot_written_when_dir_set(tmp_path: Path) -> None:
    driver = FakeDriver()
    env = AppiumEnvironment(driver, screenshot_dir=tmp_path)

    obs = env.observe()

    assert obs.screenshot_path is not None
    assert driver.saved


def test_context_manager_closes_driver() -> None:
    driver = FakeDriver()
    with AppiumEnvironment(driver) as env:
        env.observe()
    assert driver.closed is True
