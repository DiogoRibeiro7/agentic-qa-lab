"""Selenium-backed :class:`BrowserEnvironment` implementation."""

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


def _is_timeout_error(exc: BaseException) -> bool:
    """Return ``True`` when ``exc`` looks like a Selenium timeout."""
    return type(exc).__name__ == "TimeoutException" or "timeout" in str(exc).lower()


class SeleniumEnvironment(BrowserEnvironment):
    """Drive a Selenium WebDriver as an agent environment."""

    def __init__(
        self,
        driver: Any,
        *,
        screenshot_dir: Path | None = None,
        default_timeout_ms: int = 10_000,
        viewport: tuple[int, int] = (1280, 720),
    ) -> None:
        self._driver = driver
        self._screenshot_dir = screenshot_dir
        self._timeout_ms = default_timeout_ms
        self._viewport = viewport
        self._step = 0
        self._closed = False
        if screenshot_dir is not None:
            screenshot_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def launch(
        cls,
        *,
        headless: bool = True,
        screenshot_dir: Path | None = None,
        default_timeout_ms: int = 10_000,
        viewport: tuple[int, int] = (1280, 720),
    ) -> SeleniumEnvironment:
        """Launch a Chromium WebDriver and wrap it as an environment."""
        from selenium import webdriver

        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument(f"--window-size={viewport[0]},{viewport[1]}")
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(default_timeout_ms / 1000.0)
        with suppress(Exception):
            driver.implicitly_wait(default_timeout_ms / 1000.0)
        return cls(
            driver,
            screenshot_dir=screenshot_dir,
            default_timeout_ms=default_timeout_ms,
            viewport=viewport,
        )

    def open(self, url: str) -> Observation:
        """Navigate to ``url`` and return the first observation."""
        self._driver.get(url)
        return self.observe()

    def observe(self) -> Observation:
        """Capture URL, title, DOM, visible text, and an optional screenshot."""
        start = time.perf_counter()
        screenshot_path: str | None = None
        if self._screenshot_dir is not None:
            target = self._screenshot_dir / f"step_{self._step:04d}.png"
            self._driver.save_screenshot(str(target))
            screenshot_path = str(target)

        observation = Observation(
            step=self._step,
            url=str(self._driver.current_url),
            title=self._safe_title(),
            dom_snapshot=self._safe_page_source(),
            visible_text=self._safe_visible_text(),
            screenshot_path=screenshot_path,
            timestamp=time.time(),
            capture_ms=(time.perf_counter() - start) * 1000.0,
            viewport=self._viewport,
        )
        return observation

    def execute(self, action: AgentAction) -> ActionResult:
        """Dispatch an action to Selenium and return a structured result."""
        start = time.perf_counter()
        try:
            self._dispatch(action)
        except Exception as exc:  # noqa: BLE001 - converted to ActionResult
            return ActionResult.failed(
                str(exc),
                category=self._categorize(exc),
                duration_ms=(time.perf_counter() - start) * 1000.0,
            )
        self._step += 1
        return ActionResult.ok(duration_ms=(time.perf_counter() - start) * 1000.0)

    def close(self) -> None:
        """Quit the driver. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        with suppress(Exception):
            self._driver.quit()

    def _dispatch(self, action: AgentAction) -> None:
        """Translate an :class:`AgentAction` into Selenium calls."""
        if action.type is ActionType.CLICK:
            if action.selector is not None:
                self._find(action.selector).click()
                return
            self._driver.execute_script(
                "document.elementFromPoint(arguments[0], arguments[1]).click();",
                action.x,
                action.y,
            )
            return

        if action.type is ActionType.TYPE_TEXT:
            assert action.selector is not None  # noqa: S101 - guaranteed by AgentAction validator
            element = self._find(action.selector)
            with suppress(Exception):
                element.clear()
            element.send_keys(action.text)
            return

        if action.type is ActionType.PRESS_KEY:
            assert action.key is not None  # noqa: S101 - guaranteed by AgentAction validator
            target = (
                self._find(action.selector)
                if action.selector is not None
                else self._active_element()
            )
            target.send_keys(self._map_key(action.key))
            return

        if action.type is ActionType.WAIT:
            assert action.duration_ms is not None  # noqa: S101 - validator guarantees positive
            time.sleep(action.duration_ms / 1000.0)
            return

        if action.type in {ActionType.FINISH, ActionType.FAIL}:
            return

        raise ValueError(f"Unsupported action type: {action.type}")

    def _find(self, selector: str) -> Any:
        from selenium.webdriver.common.by import By

        return self._driver.find_element(By.CSS_SELECTOR, selector)

    def _active_element(self) -> Any:
        return self._driver.switch_to.active_element

    def _safe_title(self) -> str | None:
        try:
            return str(self._driver.title)
        except Exception:  # noqa: BLE001
            return None

    def _safe_page_source(self) -> str | None:
        try:
            return str(self._driver.page_source)
        except Exception:  # noqa: BLE001
            return None

    def _safe_visible_text(self) -> str | None:
        try:
            from selenium.webdriver.common.by import By

            return str(self._driver.find_element(By.TAG_NAME, "body").text)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _map_key(key: str) -> Any:
        """Map key names onto Selenium Keys constants when available."""
        with suppress(Exception):
            from selenium.webdriver.common.keys import Keys

            mapped = getattr(Keys, key.upper(), None)
            if mapped is not None:
                return mapped
        return key

    @staticmethod
    def _categorize(exc: BaseException) -> FailureCategory:
        """Map Selenium exceptions onto the shared failure taxonomy."""
        if _is_timeout_error(exc):
            return FailureCategory.TIMEOUT
        name = type(exc).__name__
        message = str(exc).lower()
        if name in {"NoSuchElementException", "ElementNotInteractableException"}:
            return FailureCategory.ELEMENT_NOT_FOUND
        if "no such element" in message or "unable to locate element" in message:
            return FailureCategory.ELEMENT_NOT_FOUND
        if "navigation" in message or "invalid argument" in message:
            return FailureCategory.NAVIGATION_ERROR
        return FailureCategory.UNKNOWN
