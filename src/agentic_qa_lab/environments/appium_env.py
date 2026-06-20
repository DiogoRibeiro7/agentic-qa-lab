"""Appium-backed :class:`BrowserEnvironment` implementation."""

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
    """Return ``True`` when ``exc`` looks like an Appium timeout."""
    return type(exc).__name__ == "TimeoutException" or "timeout" in str(exc).lower()


class AppiumEnvironment(BrowserEnvironment):
    """Drive an Appium session behind the shared browser environment contract."""

    def __init__(
        self,
        driver: Any,
        *,
        screenshot_dir: Path | None = None,
        default_timeout_ms: int = 10_000,
        viewport: tuple[int, int] | None = None,
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
        command_executor: str = "http://127.0.0.1:4723",
        capabilities: dict[str, Any] | None = None,
        screenshot_dir: Path | None = None,
        default_timeout_ms: int = 10_000,
        viewport: tuple[int, int] | None = None,
    ) -> AppiumEnvironment:
        """Start an Appium remote session and wrap it as an environment."""
        from appium import webdriver
        from appium.options.common import AppiumOptions

        options = AppiumOptions().load_capabilities(capabilities or {})
        driver = webdriver.Remote(command_executor, options=options)
        with suppress(Exception):
            driver.implicitly_wait(default_timeout_ms / 1000.0)
        return cls(
            driver,
            screenshot_dir=screenshot_dir,
            default_timeout_ms=default_timeout_ms,
            viewport=viewport,
        )

    def open(self, url: str) -> Observation:
        """Open a web URL or attach to the current native-app screen."""
        if url.startswith(("http://", "https://", "file://", "about:")):
            self._driver.get(url)
        return self.observe()

    def observe(self) -> Observation:
        """Capture current page/app state, visible text, and an optional screenshot."""
        start = time.perf_counter()
        screenshot_path: str | None = None
        if self._screenshot_dir is not None:
            target = self._screenshot_dir / f"step_{self._step:04d}.png"
            self._driver.save_screenshot(str(target))
            screenshot_path = str(target)

        observation = Observation(
            step=self._step,
            url=self._current_target(),
            title=self._safe_title(),
            dom_snapshot=self._safe_page_source(),
            visible_text=self._safe_visible_text(),
            screenshot_path=screenshot_path,
            timestamp=time.time(),
            capture_ms=(time.perf_counter() - start) * 1000.0,
            viewport=self._viewport or self._safe_viewport(),
        )
        return observation

    def execute(self, action: AgentAction) -> ActionResult:
        """Dispatch an action to Appium and return a structured result."""
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
        """Quit the Appium session. Safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        with suppress(Exception):
            self._driver.quit()

    def _dispatch(self, action: AgentAction) -> None:
        """Translate an :class:`AgentAction` into Appium driver calls."""
        if action.type is ActionType.CLICK:
            if action.selector is not None:
                self._find(action.selector).click()
                return
            self._driver.tap([(action.x, action.y)])
            return

        if action.type is ActionType.TYPE_TEXT:
            element = self._find(action.selector)
            with suppress(Exception):
                element.clear()
            element.send_keys(action.text)
            return

        if action.type is ActionType.PRESS_KEY:
            element = self._find(action.selector)
            element.send_keys(action.key)
            return

        if action.type is ActionType.WAIT:
            time.sleep(action.duration_ms / 1000.0)
            return

        if action.type in {ActionType.FINISH, ActionType.FAIL}:
            return

        raise ValueError(f"Unsupported action type: {action.type}")

    def _find(self, selector: str) -> Any:
        by, value = self._selector_strategy(selector)
        return self._driver.find_element(by, value)

    @staticmethod
    def _selector_strategy(selector: str) -> tuple[str, str]:
        """Map an agent selector string onto an Appium locator strategy."""
        prefixes = {
            "id=": "id",
            "xpath=": "xpath",
            "accessibility_id=": "accessibility id",
            "css=": "css selector",
        }
        for prefix, strategy in prefixes.items():
            if selector.startswith(prefix):
                return strategy, selector[len(prefix) :]
        return "id", selector

    def _current_target(self) -> str:
        try:
            current_url = str(self._driver.current_url)
            if current_url:
                return current_url
        except Exception:  # noqa: BLE001
            pass
        with suppress(Exception):
            package = str(self._driver.current_package)
            activity = str(getattr(self._driver, "current_activity", ""))
            suffix = f"/{activity}" if activity else ""
            return f"appium://{package}{suffix}"
        return "appium://session"

    def _safe_title(self) -> str | None:
        with suppress(Exception):
            title = str(self._driver.title)
            return title or None
        with suppress(Exception):
            activity = str(getattr(self._driver, "current_activity", ""))
            return activity or None
        return None

    def _safe_page_source(self) -> str | None:
        with suppress(Exception):
            return str(self._driver.page_source)
        return None

    def _safe_visible_text(self) -> str | None:
        with suppress(Exception):
            return str(
                self._driver.execute_script(
                    "mobile: source",
                    {"format": "description"},
                )
            )
        with suppress(Exception):
            return str(self._driver.page_source)
        return None

    def _safe_viewport(self) -> tuple[int, int] | None:
        with suppress(Exception):
            size = self._driver.get_window_size()
            width = int(size["width"])
            height = int(size["height"])
            return (width, height)
        return None

    @staticmethod
    def _categorize(exc: BaseException) -> FailureCategory:
        """Map Appium/WebDriver failures onto the shared taxonomy."""
        if _is_timeout_error(exc):
            return FailureCategory.TIMEOUT
        name = type(exc).__name__
        message = str(exc).lower()
        if name in {"NoSuchElementException", "ElementNotInteractableException"}:
            return FailureCategory.ELEMENT_NOT_FOUND
        if "no such element" in message or "unable to locate element" in message:
            return FailureCategory.ELEMENT_NOT_FOUND
        if "app" in message or "activity" in message or "navigation" in message:
            return FailureCategory.NAVIGATION_ERROR
        return FailureCategory.UNKNOWN
