from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

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


class BrokenMetaDriver(FakeDriver):
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

    @property
    def current_url(self) -> str:
        raise RuntimeError("no url")

    @current_url.setter
    def current_url(self, value: str) -> None:
        self._current_url = value

    @property
    def current_package(self) -> str:
        raise RuntimeError("no package")

    @current_package.setter
    def current_package(self, value: str) -> None:
        self._current_package = value


class FailingViewportDriver(FakeDriver):
    def get_window_size(self) -> dict[str, int]:
        raise RuntimeError("no viewport")


class QuitFailsDriver(FakeDriver):
    def quit(self) -> None:
        self.closed = True
        raise RuntimeError("quit failed")


class FakeRemoteFactory:
    def __init__(self) -> None:
        self.driver = FakeDriver()
        self.command_executor: str | None = None
        self.options: Any = None

    def __call__(self, command_executor: str, options: Any) -> FakeDriver:
        self.command_executor = command_executor
        self.options = options
        return self.driver


class FakeAppiumOptions:
    def __init__(self) -> None:
        self.loaded: dict[str, Any] | None = None

    def load_capabilities(self, capabilities: dict[str, Any]) -> FakeAppiumOptions:
        self.loaded = dict(capabilities)
        return self


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


def test_launch_builds_remote_session(monkeypatch: pytest.MonkeyPatch) -> None:
    remote = FakeRemoteFactory()
    options = FakeAppiumOptions()
    fake_webdriver = types.SimpleNamespace(Remote=remote)
    fake_options_module = types.SimpleNamespace(AppiumOptions=lambda: options)

    monkeypatch.setitem(sys.modules, "appium", types.SimpleNamespace(webdriver=fake_webdriver))
    monkeypatch.setitem(sys.modules, "appium.options.common", fake_options_module)

    env = AppiumEnvironment.launch(
        command_executor="http://127.0.0.1:4725",
        capabilities={"platformName": "Android"},
        default_timeout_ms=1500,
    )

    assert remote.command_executor == "http://127.0.0.1:4725"
    assert options.loaded == {"platformName": "Android"}
    assert env._driver is remote.driver  # noqa: SLF001


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


def test_navigation_error_is_categorized() -> None:
    driver = FakeDriver(exc=RuntimeError("app activity navigation failed"))
    env = AppiumEnvironment(driver)

    result = env.execute(AgentAction.click("id=missing"))

    assert not result.success
    assert result.failure_category is FailureCategory.NAVIGATION_ERROR


def test_unknown_error_is_categorized() -> None:
    driver = FakeDriver(exc=RuntimeError("mystery failure"))
    env = AppiumEnvironment(driver)

    result = env.execute(AgentAction.click("id=missing"))

    assert not result.success
    assert result.failure_category is FailureCategory.UNKNOWN


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


def test_observe_handles_best_effort_metadata_failures() -> None:
    driver = BrokenMetaDriver()
    env = AppiumEnvironment(driver)

    obs = env.observe()

    assert obs.url == "appium://session"
    assert obs.title == ".MainActivity"
    assert obs.dom_snapshot is None
    assert obs.visible_text == "Visible text"


def test_observe_falls_back_to_page_source_for_visible_text() -> None:
    driver = FakeDriver()

    def fail_mobile_source(script: str, payload: dict[str, str]) -> str:
        raise RuntimeError("no mobile source")

    driver.execute_script = fail_mobile_source  # type: ignore[method-assign]
    env = AppiumEnvironment(driver)

    obs = env.observe()

    assert obs.visible_text == "<hierarchy>visible text</hierarchy>"


def test_observe_uses_none_when_viewport_unavailable() -> None:
    driver = FailingViewportDriver()
    env = AppiumEnvironment(driver)

    obs = env.observe()

    assert obs.viewport is None


def test_context_manager_closes_driver() -> None:
    driver = FakeDriver()
    with AppiumEnvironment(driver) as env:
        env.observe()
    assert driver.closed is True


def test_close_is_idempotent() -> None:
    driver = FakeDriver()
    env = AppiumEnvironment(driver)

    env.close()
    env.close()

    assert driver.closed is True


def test_close_suppresses_driver_quit_error() -> None:
    driver = QuitFailsDriver()
    env = AppiumEnvironment(driver)

    env.close()

    assert driver.closed is True


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("id=username", ("id", "username")),
        ("xpath=//input[1]", ("xpath", "//input[1]")),
        ("accessibility_id=login", ("accessibility id", "login")),
        ("css=.btn", ("css selector", ".btn")),
        ("plain-id", ("id", "plain-id")),
    ],
)
def test_selector_strategy(selector: str, expected: tuple[str, str]) -> None:
    assert AppiumEnvironment._selector_strategy(selector) == expected  # noqa: SLF001
