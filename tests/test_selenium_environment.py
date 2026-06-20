from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentic_qa_lab.domain import AgentAction, FailureCategory
from agentic_qa_lab.environments import SeleniumEnvironment


class FakeTimeoutError(Exception):
    """Stand-in mimicking Selenium TimeoutException by class name."""


FakeTimeoutError.__name__ = "TimeoutException"


class FakeElement:
    def __init__(
        self, *, text: str = "", raise_on: str | None = None, exc: Exception | None = None
    ) -> None:
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


class BrokenAttrDriver(FakeDriver):
    @property
    def title(self) -> str:
        raise RuntimeError("no title")

    @title.setter
    def title(self, value: str) -> None:
        self._title = value

    @property
    def page_source(self) -> str:
        raise RuntimeError("no source")

    @page_source.setter
    def page_source(self, value: str) -> None:
        self._page_source = value


class QuitFailsDriver(FakeDriver):
    def quit(self) -> None:
        self.closed = True
        raise RuntimeError("quit failed")


class BodyFailsDriver(FakeDriver):
    def find_element(self, by: Any, value: str) -> FakeElement:
        self.calls.append(("find_element", (by, value)))
        if value == "body":
            raise RuntimeError("body lookup failed")
        return super().find_element(by, value)


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


def test_active_element_returns_driver_active_element() -> None:
    driver = FakeDriver()
    env = SeleniumEnvironment(driver)

    assert env._active_element() is driver.active  # noqa: SLF001


def test_coordinate_click_uses_script() -> None:
    driver = FakeDriver()
    env = SeleniumEnvironment(driver)

    assert env.execute(AgentAction.click(x=10, y=20)).success
    assert any(name == "execute_script" for name, _args in driver.calls)


def test_timeout_is_categorized() -> None:
    driver = FakeDriver(element_exc=FakeTimeoutError("timed out"))
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


def test_navigation_error_is_categorized() -> None:
    driver = FakeDriver(element_exc=RuntimeError("navigation failed: invalid argument"))
    env = SeleniumEnvironment(driver)

    result = env.execute(AgentAction.click("#bad"))

    assert not result.success
    assert result.failure_category is FailureCategory.NAVIGATION_ERROR


def test_unknown_error_is_categorized() -> None:
    driver = FakeDriver(element_exc=RuntimeError("mystery failure"))
    env = SeleniumEnvironment(driver)

    result = env.execute(AgentAction.click("#bad"))

    assert not result.success
    assert result.failure_category is FailureCategory.UNKNOWN


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


def test_observe_handles_driver_metadata_failures() -> None:
    driver = BrokenAttrDriver()
    env = SeleniumEnvironment(driver)

    obs = env.observe()

    assert obs.title is None
    assert obs.dom_snapshot is None
    assert obs.visible_text == "visible text"


def test_observe_handles_visible_text_failure() -> None:
    driver = BodyFailsDriver()
    env = SeleniumEnvironment(driver)

    obs = env.observe()

    assert obs.visible_text is None


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


def test_close_suppresses_driver_quit_error() -> None:
    driver = QuitFailsDriver()
    env = SeleniumEnvironment(driver)

    env.close()

    assert driver.closed is True


def test_map_key_falls_back_to_literal_when_keys_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = __import__

    def fake_import(
        name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0
    ) -> Any:
        if name == "selenium.webdriver.common.keys":
            raise ImportError("no selenium keys")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    assert SeleniumEnvironment._map_key("Enter") == "Enter"  # noqa: SLF001
