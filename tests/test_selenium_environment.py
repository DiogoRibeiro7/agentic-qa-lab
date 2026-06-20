from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_qa_lab.domain import AgentAction, FailureCategory
from agentic_qa_lab.environments import SeleniumEnvironment


class FakeTimeoutException(Exception):
    """Stand-in mimicking Selenium TimeoutException by class name."""


FakeTimeoutException.__name__ = "TimeoutException"


class FakeElement:
    def __init__(self, *, text: str = "", raise_on: str | None = None, exc: Exception | None = None) -> None:
        self.text = text
        self.raise_on = raise_on
        self.exc = exc or RuntimeError("boom")
        self.clicks = 0
        self.sent: list[Any] = []
        self.cleared = 0

    def click(self) -> None:
        self.clicks += 1
        if self.raise_on == "click":
            raise self.exc

    def clear(self) -> None:
        self.cleared += 1
        if self.raise_on == "clear":
            raise self.exc

    def send_keys(self, value: Any) -> None:
        self.sent.append(value)
        if self.raise_on == "send_keys":
            raise self.exc


class FakeSwitchTo:
    def __init__(self, active_element: FakeElement) -> None:
        self.active_element = active_element


class FakeDriver:
    def __init__(self, *, element_exc: Exception | None = None) -> None:
        self.current_url = "https://example.com/"
        self.title = "Example"
        self.page_source = "<html><body>ok</body></html>"
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.closed = False
        self.saved: list[str] = []
        self.element_exc = element_exc
        self.body = FakeElement(text="visible text")
        self.default = FakeElement()
        self.active = FakeElement()
        self.switch_to = FakeSwitchTo(self.active)

    def get(self, url: str) -> None:
        self.calls.append(("get", (url,)))
        self.current_url = url

    def save_screenshot(self, path: str) -> None:
        self.calls.append(("save_screenshot", (path,)))
        self.saved.append(path)

    def execute_script(self, script: str, *args: Any) -> None:
        self.calls.append(("execute_script", (script, *args)))
        if self.element_exc is not None:
            raise self.element_exc

    def find_element(self, by: Any, value: str) -> FakeElement:
        self.calls.append(("find_element", (by, value)))
        if value == "body":
            return self.body
        if self.element_exc is not None:
            raise self.element_exc
        return self.default

    def quit(self) -> None:
        self.closed = True


def test_open_returns_observation() -> None:
    driver = FakeDriver()
    env = SeleniumEnvironment(driver)

    obs = env.open("https://example.com/login")

    assert obs.url == "https://example.com/login"
    assert obs.title == "Example"
    assert obs.visible_text == "visible text"
    assert obs.step == 0


def test_execute_click_success_and_step_increment() -> None:
    driver = FakeDriver()
    env = SeleniumEnvironment(driver)

    result = env.execute(AgentAction.click("#submit"))

    assert result.success
    assert driver.default.clicks == 1
    assert env.observe().step == 1


def test_execute_type_press_and_wait() -> None:
    driver = FakeDriver()
    env = SeleniumEnvironment(driver)

    assert env.execute(AgentAction.type_text("alice", selector="#user")).success
    assert env.execute(AgentAction.press_key("Enter", selector="#user")).success
    assert env.execute(AgentAction.press_key("Escape", selector="#user")).success
    assert env.execute(AgentAction.wait(1)).success

    assert driver.default.cleared >= 1
    assert driver.default.sent


def test_coordinate_click_uses_script() -> None:
    driver = FakeDriver()
    env = SeleniumEnvironment(driver)

    assert env.execute(AgentAction.click(x=10, y=20)).success
    assert any(name == "execute_script" for name, _args in driver.calls)


def test_timeout_is_categorized() -> None:
    driver = FakeDriver(element_exc=FakeTimeoutException("timed out"))
    env = SeleniumEnvironment(driver)

    result = env.execute(AgentAction.click("#missing"))

    assert not result.success
    assert result.failure_category is FailureCategory.TIMEOUT


def test_element_not_found_is_categorized() -> None:
    driver = FakeDriver(element_exc=RuntimeError("no such element: Unable to locate element"))
    env = SeleniumEnvironment(driver)

    result = env.execute(AgentAction.click("#missing"))

    assert not result.success
    assert result.failure_category is FailureCategory.ELEMENT_NOT_FOUND


def test_terminal_actions_are_noops() -> None:
    driver = FakeDriver()
    env = SeleniumEnvironment(driver)

    assert env.execute(AgentAction.finish("done")).success
    assert env.execute(AgentAction.fail("nope")).success
    assert driver.calls == []


def test_screenshot_written_when_dir_set(tmp_path: Path) -> None:
    driver = FakeDriver()
    env = SeleniumEnvironment(driver, screenshot_dir=tmp_path)

    obs = env.observe()

    assert obs.screenshot_path is not None
    assert driver.saved


def test_context_manager_closes_driver() -> None:
    driver = FakeDriver()
    with SeleniumEnvironment(driver) as env:
        env.observe()
    assert driver.closed is True


def test_close_is_idempotent() -> None:
    driver = FakeDriver()
    env = SeleniumEnvironment(driver)
    env.close()
    env.close()
    assert driver.closed is True
