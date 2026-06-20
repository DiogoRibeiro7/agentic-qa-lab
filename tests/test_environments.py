from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from agentic_qa_lab.domain import AgentAction, FailureCategory
from agentic_qa_lab.environments import PlaywrightEnvironment


class FakeTimeoutError(Exception):
    """Stand-in mimicking Playwright's TimeoutError by class name."""


# Rename so _is_timeout_error matches on the "TimeoutError" suffix.
FakeTimeoutError.__name__ = "TimeoutError"


class FakeMouse:
    def __init__(self) -> None:
        self.clicks: list[tuple[int, int]] = []

    def click(self, x: int, y: int) -> None:
        self.clicks.append((x, y))


class FakeKeyboard:
    def __init__(self) -> None:
        self.presses: list[str] = []

    def press(self, key: str) -> None:
        self.presses.append(key)


class FakePage:
    """Minimal Playwright Page surface for unit tests."""

    def __init__(self, *, raise_on: str | None = None, exc: Exception | None = None) -> None:
        self.url = "https://example.com/"
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.mouse = FakeMouse()
        self.keyboard = FakeKeyboard()
        self._raise_on = raise_on
        self._exc = exc or RuntimeError("boom")
        self.closed = False

    def _record(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))
        if self._raise_on == name:
            raise self._exc

    def goto(self, url: str, timeout: int | None = None) -> None:
        self._record("goto", url)
        self.url = url

    def title(self) -> str:
        return "Example"

    def content(self) -> str:
        return "<html><body>ok</body></html>"

    def click(self, selector: str, timeout: int | None = None) -> None:
        self._record("click", selector)

    def fill(self, selector: str, text: str, timeout: int | None = None) -> None:
        self._record("fill", selector, text)

    def press(self, selector: str, key: str, timeout: int | None = None) -> None:
        self._record("press", selector, key)

    def wait_for_timeout(self, ms: int) -> None:
        self._record("wait_for_timeout", ms)

    def screenshot(self, path: str) -> None:
        self._record("screenshot", path)

    def close(self) -> None:
        self.closed = True


class BrokenMetaPage(FakePage):
    def title(self) -> str:
        raise RuntimeError("no title")

    def content(self) -> str:
        raise RuntimeError("no content")

    def inner_text(self, selector: str) -> str:
        raise RuntimeError("no visible text")


class FakeBrowser:
    def __init__(self) -> None:
        self.page = FakePage()
        self.closed = False
        self.launch_headless: bool | None = None
        self.viewport: dict[str, int] | None = None

    def new_page(self, *, viewport: dict[str, int]) -> FakePage:
        self.viewport = viewport
        return self.page

    def close(self) -> None:
        self.closed = True


class FakeChromium:
    def __init__(self, browser: FakeBrowser) -> None:
        self.browser = browser

    def launch(self, *, headless: bool) -> FakeBrowser:
        self.browser.launch_headless = headless
        return self.browser


class FakePlaywrightHandle:
    def __init__(self, browser: FakeBrowser) -> None:
        self.chromium = FakeChromium(browser)
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class FakeSyncPlaywright:
    def __init__(self, handle: FakePlaywrightHandle) -> None:
        self.handle = handle
        self.started = False

    def start(self) -> FakePlaywrightHandle:
        self.started = True
        return self.handle


def test_open_returns_observation() -> None:
    page = FakePage()
    env = PlaywrightEnvironment(page)
    obs = env.open("https://example.com/login")

    assert obs.url == "https://example.com/login"
    assert obs.title == "Example"
    assert obs.dom_snapshot is not None
    assert obs.step == 0


def test_execute_click_success_and_step_increment() -> None:
    page = FakePage()
    env = PlaywrightEnvironment(page)

    result = env.execute(AgentAction.click("#submit"))

    assert result.success
    assert ("click", ("#submit",)) in page.calls
    assert env.observe().step == 1


def test_execute_type_and_press_and_wait() -> None:
    page = FakePage()
    env = PlaywrightEnvironment(page)

    assert env.execute(AgentAction.type_text("alice", selector="#user")).success
    assert env.execute(AgentAction.press_key("Enter", selector="#user")).success
    assert env.execute(AgentAction.wait(50)).success

    assert ("fill", ("#user", "alice")) in page.calls
    assert ("press", ("#user", "Enter")) in page.calls
    assert ("wait_for_timeout", (50,)) in page.calls


def test_coordinate_click_uses_mouse() -> None:
    page = FakePage()
    env = PlaywrightEnvironment(page)

    assert env.execute(AgentAction.click(x=10, y=20)).success
    assert page.mouse.clicks == [(10, 20)]


def test_launch_builds_page_with_timeout_and_viewport(monkeypatch: pytest.MonkeyPatch) -> None:
    browser = FakeBrowser()
    sync_playwright = FakeSyncPlaywright(FakePlaywrightHandle(browser))
    fake_module = types.SimpleNamespace(sync_playwright=lambda: sync_playwright)

    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_module)

    def set_default_timeout(ms: int) -> None:
        browser.page.calls.append(("set_default_timeout", (ms,)))

    browser.page.set_default_timeout = set_default_timeout  # type: ignore[attr-defined]

    env = PlaywrightEnvironment.launch(headless=False, viewport=(800, 600), default_timeout_ms=1234)

    assert sync_playwright.started is True
    assert browser.launch_headless is False
    assert browser.viewport == {"width": 800, "height": 600}
    assert ("set_default_timeout", (1234,)) in browser.page.calls
    env.close()


def test_timeout_is_categorized() -> None:
    page = FakePage(raise_on="click", exc=FakeTimeoutError("timed out"))
    env = PlaywrightEnvironment(page)

    result = env.execute(AgentAction.click("#missing"))

    assert not result.success
    assert result.failure_category is FailureCategory.TIMEOUT


def test_element_not_found_is_categorized() -> None:
    page = FakePage(raise_on="click", exc=RuntimeError("no node found for selector #x"))
    env = PlaywrightEnvironment(page)

    result = env.execute(AgentAction.click("#x"))

    assert not result.success
    assert result.failure_category is FailureCategory.ELEMENT_NOT_FOUND


def test_strict_mode_violation_is_not_element_not_found() -> None:
    # An ambiguous selector that matched several nodes is not a missing element;
    # it must not be bucketed as ELEMENT_NOT_FOUND (which would trigger healing).
    assert (
        PlaywrightEnvironment._categorize(  # noqa: SLF001
            RuntimeError("strict mode violation: locator resolved to 3 elements")
        )
        is FailureCategory.INVALID_ACTION
    )


def test_navigation_error_is_categorized() -> None:
    assert (
        PlaywrightEnvironment._categorize(  # noqa: SLF001
            RuntimeError("navigation failed: net::ERR_ABORTED")
        )
        is FailureCategory.NAVIGATION_ERROR
    )


def test_unknown_error_is_categorized() -> None:
    assert (
        PlaywrightEnvironment._categorize(RuntimeError("mystery failure"))  # noqa: SLF001
        is FailureCategory.UNKNOWN
    )


def test_terminal_actions_are_noops() -> None:
    page = FakePage()
    env = PlaywrightEnvironment(page)

    assert env.execute(AgentAction.finish("done")).success
    assert env.execute(AgentAction.fail("nope")).success
    # No interaction calls recorded for terminal actions.
    assert page.calls == []


def test_screenshot_written_when_dir_set(tmp_path: Any) -> None:
    page = FakePage()
    env = PlaywrightEnvironment(page, screenshot_dir=tmp_path)

    obs = env.observe()

    assert obs.screenshot_path is not None
    assert any(name == "screenshot" for name, _ in page.calls)


def test_observe_handles_best_effort_metadata_failures() -> None:
    page = BrokenMetaPage()
    env = PlaywrightEnvironment(page)

    obs = env.observe()

    assert obs.title is None
    assert obs.dom_snapshot is None
    assert obs.visible_text is None


def test_context_manager_closes_page() -> None:
    page = FakePage()
    with PlaywrightEnvironment(page) as env:
        env.observe()
    assert page.closed is True


def test_close_is_idempotent() -> None:
    page = FakePage()
    env = PlaywrightEnvironment(page)
    env.close()
    env.close()
    assert page.closed is True


def test_close_also_closes_browser_and_stops_playwright() -> None:
    page = FakePage()
    browser = FakeBrowser()
    browser.page = page
    playwright = FakePlaywrightHandle(browser)
    env = PlaywrightEnvironment(page, browser=browser, playwright=playwright)

    env.close()

    assert page.closed is True
    assert browser.closed is True
    assert playwright.stopped is True
