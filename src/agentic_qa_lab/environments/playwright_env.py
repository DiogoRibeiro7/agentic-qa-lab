"""Playwright-backed :class:`BrowserEnvironment` implementation.

The Playwright import is performed lazily so the module (and its unit tests,
which inject a fake page) can be imported in environments where Playwright's
browser binaries are not installed.
"""

from __future__ import annotations

import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from ..domain import (
    ActionResult,
    ActionType,
    AgentAction,
    FailureCategory,
    Observation,
)
from .base import BrowserEnvironment

#: Default per-action timeout in milliseconds.
DEFAULT_TIMEOUT_MS = 10_000


def _is_timeout_error(exc: BaseException) -> bool:
    """Return ``True`` when ``exc`` looks like a Playwright timeout."""
    return type(exc).__name__ == "TimeoutError" or "Timeout" in type(exc).__name__


class PlaywrightEnvironment(BrowserEnvironment):
    """Drive a single Playwright ``Page`` as an agent environment.

    Parameters
    ----------
    page:
        A Playwright ``Page`` (or any object exposing the same surface). Inject
        a fake here in tests.
    screenshot_dir:
        Directory where step screenshots are written. When ``None`` no
        screenshots are captured.
    default_timeout_ms:
        Per-action timeout applied to navigation and element interactions.
    viewport:
        Reported ``(width, height)`` for observations.
    playwright:
        Optional handle to the running Playwright instance, stopped on
        :meth:`close`. Set by :meth:`launch`.
    browser:
        Optional browser handle closed on :meth:`close`. Set by :meth:`launch`.
    """

    def __init__(
        self,
        page: Any,
        *,
        screenshot_dir: Path | None = None,
        default_timeout_ms: int = DEFAULT_TIMEOUT_MS,
        viewport: tuple[int, int] = (1280, 720),
        playwright: Any | None = None,
        browser: Any | None = None,
    ) -> None:
        self._page = page
        self._screenshot_dir = screenshot_dir
        self._timeout_ms = default_timeout_ms
        self._viewport = viewport
        self._playwright = playwright
        self._browser = browser
        self._step = 0
        self._closed = False
        if screenshot_dir is not None:
            screenshot_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def launch(
        cls,
        *,
        headless: bool = True,
        screenshot_dir: Path | None = None,
        default_timeout_ms: int = DEFAULT_TIMEOUT_MS,
        viewport: tuple[int, int] = (1280, 720),
    ) -> PlaywrightEnvironment:
        """Launch a real Chromium browser and wrap a fresh page.

        Requires Playwright browser binaries (``playwright install chromium``).
        """
        from playwright.sync_api import sync_playwright

        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=headless)
        page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
        page.set_default_timeout(default_timeout_ms)
        return cls(
            page,
            screenshot_dir=screenshot_dir,
            default_timeout_ms=default_timeout_ms,
            viewport=viewport,
            playwright=playwright,
            browser=browser,
        )

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def open(self, url: str) -> Observation:
        """Navigate to ``url`` and return the first observation."""
        self._page.goto(url, timeout=self._timeout_ms)
        return self.observe()

    def observe(self) -> Observation:
        """Capture URL, title, DOM, and (optionally) a screenshot."""
        start = time.perf_counter()
        screenshot_path: str | None = None
        if self._screenshot_dir is not None:
            target = self._screenshot_dir / f"step_{self._step:04d}.png"
            self._page.screenshot(path=str(target))
            screenshot_path = str(target)

        url = self._page.url
        title = self._safe_title()
        dom = self._safe_content()
        visible = self._safe_visible_text()
        capture_ms = (time.perf_counter() - start) * 1000.0

        observation = Observation(
            step=self._step,
            url=url,
            title=title,
            dom_snapshot=dom,
            visible_text=visible,
            screenshot_path=screenshot_path,
            timestamp=time.time(),
            capture_ms=capture_ms,
            viewport=self._viewport,
        )
        return observation

    def _safe_title(self) -> str | None:
        try:
            return str(self._page.title())
        except Exception:  # noqa: BLE001 - title is best-effort metadata
            return None

    def _safe_content(self) -> str | None:
        try:
            return str(self._page.content())
        except Exception:  # noqa: BLE001 - DOM snapshot is best-effort
            return None

    def _safe_visible_text(self) -> str | None:
        # inner_text returns only rendered, visible text — no script source,
        # comments, or display:none nodes — which is what success markers key on.
        try:
            return str(self._page.inner_text("body"))
        except Exception:  # noqa: BLE001 - visible text is best-effort
            return None

    # ------------------------------------------------------------------ #
    # Action execution
    # ------------------------------------------------------------------ #
    def execute(self, action: AgentAction) -> ActionResult:
        """Dispatch ``action`` to the matching Playwright call.

        Returns a structured :class:`ActionResult`; timeouts and missing
        elements are mapped to :class:`FailureCategory` buckets rather than
        raising.
        """
        start = time.perf_counter()
        try:
            self._dispatch(action)
        except Exception as exc:  # noqa: BLE001 - converted to ActionResult
            return ActionResult.failed(
                str(exc),
                category=self._categorize(exc),
                duration_ms=(time.perf_counter() - start) * 1000.0,
            )
        else:
            self._step += 1
            return ActionResult.ok(duration_ms=(time.perf_counter() - start) * 1000.0)

    def _dispatch(self, action: AgentAction) -> None:
        """Translate an :class:`AgentAction` into a Playwright call."""
        if action.type is ActionType.CLICK:
            if action.selector is not None:
                self._page.click(action.selector, timeout=self._timeout_ms)
            else:
                self._page.mouse.click(action.x, action.y)
        elif action.type is ActionType.TYPE_TEXT:
            self._page.fill(action.selector, action.text, timeout=self._timeout_ms)
        elif action.type is ActionType.PRESS_KEY:
            if action.selector is not None:
                self._page.press(action.selector, action.key, timeout=self._timeout_ms)
            else:
                self._page.keyboard.press(action.key)
        elif action.type is ActionType.WAIT:
            self._page.wait_for_timeout(action.duration_ms)
        elif action.type in {ActionType.FINISH, ActionType.FAIL}:
            return  # terminal actions are no-ops for the environment
        else:  # pragma: no cover - exhaustive guard
            raise ValueError(f"Unsupported action type: {action.type}")

    @staticmethod
    def _categorize(exc: BaseException) -> FailureCategory:
        """Map an exception to a :class:`FailureCategory`."""
        if _is_timeout_error(exc):
            return FailureCategory.TIMEOUT
        name = type(exc).__name__
        message = str(exc).lower()
        # A strict-mode violation ("resolved to N elements") is an ambiguous
        # selector, not a missing element — keep it out of ELEMENT_NOT_FOUND so
        # SelfHealingAgent doesn't try to repair a selector that did match.
        if "strict mode violation" in message or "resolved to" in message:
            return FailureCategory.INVALID_ACTION
        if "no node found" in message or "not found" in message or name == "ElementNotFound":
            return FailureCategory.ELEMENT_NOT_FOUND
        if "navigation" in message or "net::" in message:
            return FailureCategory.NAVIGATION_ERROR
        return FailureCategory.UNKNOWN

    # ------------------------------------------------------------------ #
    # Teardown
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Close the page/browser and stop Playwright. Idempotent."""
        if self._closed:
            return
        self._closed = True
        teardown = (
            (self._page, "close"),
            (self._browser, "close"),
            (self._playwright, "stop"),
        )
        for handle, method in teardown:
            if handle is None:
                continue
            with suppress(Exception):  # teardown is best-effort
                getattr(handle, method)()
