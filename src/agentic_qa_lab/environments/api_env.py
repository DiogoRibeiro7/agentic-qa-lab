"""HTTP API environment behind the BrowserEnvironment interface.

This adapter is intended for non-UI flows where the same agent/runner loop is
useful but the side effect happens over HTTP rather than in a browser. It maps
the existing `AgentAction` primitives onto a small request-builder protocol:

- `type_text(..., selector="#method")` sets the HTTP method
- `type_text(..., selector="#path")` sets the request path
- `type_text(..., selector="#body")` sets the request body
- `type_text(..., selector="#query:<name>")` sets one query parameter
- `type_text(..., selector="#header:<name>")` sets one header
- `click("#send")` dispatches the prepared request

The observation renders the last request/response pair as text so rule-based or
LLM agents can reason over API responses the same way they reason over page
state in browser environments.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

from ..domain import ActionResult, ActionType, AgentAction, FailureCategory, Observation
from .base import BrowserEnvironment


class APIEnvironment(BrowserEnvironment):
    """HTTP request environment using the BrowserEnvironment contract."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        default_method: str = "GET",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._step = 0
        self._closed = False
        self._method = default_method.upper()
        self._path = "/"
        self._body = ""
        self._headers: dict[str, str] = {}
        self._query: dict[str, str] = {}
        self._last_status: int | None = None
        self._last_response_body = ""
        self._last_error: str | None = None

    def open(self, url: str) -> Observation:
        """Set the base URL for the API session and return the initial observation."""
        self._base_url = url.rstrip("/")
        return self.observe()

    def observe(self) -> Observation:
        """Render the current request builder state and last response."""
        summary = self._render_summary()
        return Observation(
            step=self._step,
            url=self._current_url(),
            title=f"API {self._method} {self._path}",
            dom_snapshot=summary,
            visible_text=summary,
            timestamp=time.time(),
        )

    def execute(self, action: AgentAction) -> ActionResult:
        """Apply a request-builder action or send the configured request."""
        start = time.perf_counter()
        try:
            if action.is_terminal:
                return ActionResult.ok(duration_ms=self._elapsed_ms(start))
            if action.type is ActionType.TYPE_TEXT:
                self._apply_type_text(action)
                return ActionResult.ok(duration_ms=self._elapsed_ms(start))
            if action.type is ActionType.CLICK and action.selector == "#send":
                return self._send(duration_start=start)
            if action.type is ActionType.WAIT:
                time.sleep(action.duration_ms / 1000.0)
                return ActionResult.ok(duration_ms=self._elapsed_ms(start))
            return ActionResult.failed(
                f"Unsupported API action: {action.type.value} on {action.selector!r}",
                category=FailureCategory.INVALID_ACTION,
                duration_ms=self._elapsed_ms(start),
            )
        except ValueError as exc:
            return ActionResult.failed(
                str(exc),
                category=FailureCategory.INVALID_ACTION,
                duration_ms=self._elapsed_ms(start),
            )

    def close(self) -> None:
        """Mark the environment closed. No network resources are held open."""
        self._closed = True

    def _apply_type_text(self, action: AgentAction) -> None:
        selector = action.selector or ""
        text = action.text or ""
        if selector == "#method":
            self._method = text.upper()
            return
        if selector == "#path":
            self._path = text or "/"
            return
        if selector == "#body":
            self._body = text
            return
        if selector.startswith("#header:"):
            self._headers[selector.removeprefix("#header:")] = text
            return
        if selector.startswith("#query:"):
            self._query[selector.removeprefix("#query:")] = text
            return
        raise ValueError(
            "APIEnvironment type_text selectors must be one of "
            "#method, #path, #body, #header:<name>, or #query:<name>."
        )

    def _send(self, *, duration_start: float) -> ActionResult:
        self._step += 1
        self._last_error = None
        url = self._current_url()
        payload = self._body.encode("utf-8") if self._method in {"POST", "PUT", "PATCH"} else None
        request = urllib.request.Request(  # noqa: S310 - operator-supplied endpoint
            url=url,
            data=payload,
            method=self._method,
            headers=self._headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                body = response.read().decode("utf-8")
                self._last_status = response.status
                self._last_response_body = body
                return ActionResult.ok(duration_ms=self._elapsed_ms(duration_start))
        except urllib.error.HTTPError as exc:
            self._last_status = exc.code
            self._last_response_body = exc.read().decode("utf-8", errors="replace")
            self._last_error = f"HTTP {exc.code}"
            return ActionResult.failed(
                self._last_error,
                category=FailureCategory.UNKNOWN,
                duration_ms=self._elapsed_ms(duration_start),
            )
        except urllib.error.URLError as exc:
            self._last_status = None
            self._last_response_body = ""
            self._last_error = str(exc.reason)
            return ActionResult.failed(
                self._last_error,
                category=FailureCategory.NAVIGATION_ERROR,
                duration_ms=self._elapsed_ms(duration_start),
            )

    def _current_url(self) -> str:
        path = self._path if self._path.startswith("/") else f"/{self._path}"
        query = urllib.parse.urlencode(self._query)
        return f"{self._base_url}{path}" + (f"?{query}" if query else "")

    def _render_summary(self) -> str:
        payload: dict[str, object] = {
            "request": {
                "method": self._method,
                "url": self._current_url(),
                "headers": self._headers,
                "body": self._body,
            },
            "last_response": {
                "status": self._last_status,
                "error": self._last_error,
                "body": self._last_response_body,
            },
        }
        return json.dumps(payload, indent=2)

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000.0
