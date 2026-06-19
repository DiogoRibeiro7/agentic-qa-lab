"""Record a manual browser session into a benchmark case plan."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..domain import ActionType, AgentAction, TaskSpec
from .tasks import BenchmarkCase

_RECORDER_SCRIPT = """
(() => {
  if (window.__agenticQaRecorderInstalled) {
    return;
  }
  window.__agenticQaRecorderInstalled = true;

  const cssEscape = (value) => {
    if (globalThis.CSS && typeof globalThis.CSS.escape === "function") {
      return globalThis.CSS.escape(value);
    }
    return String(value).replace(/[^a-zA-Z0-9_-]/g, "\\\\$&");
  };

  const textValue = (element) =>
    (element.innerText || element.textContent || "").replace(/\\s+/g, " ").trim();

  const selectorFor = (element) => {
    if (!(element instanceof Element)) {
      return null;
    }
    if (element.id) {
      return `#${cssEscape(element.id)}`;
    }
    const testId = element.getAttribute("data-testid");
    if (testId) {
      return `[data-testid="${testId.replace(/"/g, '\\"')}"]`;
    }
    const name = element.getAttribute("name");
    if (name) {
      return `[name="${name.replace(/"/g, '\\"')}"]`;
    }
    const text = textValue(element);
    if (text) {
      return `text=${text}`;
    }
    return element.tagName.toLowerCase();
  };

  const record = (payload) => {
    if (typeof window.__agenticQaRecord === "function") {
      window.__agenticQaRecord(payload);
    }
  };

  document.addEventListener(
    "click",
    (event) => {
      const target = event.target instanceof Element ? event.target.closest("*") : null;
      if (!(target instanceof Element)) {
        return;
      }
      const tag = target.tagName.toLowerCase();
      const inputType = (target.getAttribute("type") || "").toLowerCase();
      if (tag === "input" && !["button", "checkbox", "radio", "submit"].includes(inputType)) {
        return;
      }
      const selector = selectorFor(target);
      if (!selector) {
        return;
      }
      record({ type: "click", selector });
    },
    true,
  );

  document.addEventListener(
    "change",
    (event) => {
      const target = event.target;
      if (
        !(
          target instanceof HTMLInputElement ||
          target instanceof HTMLTextAreaElement ||
          target instanceof HTMLSelectElement
        )
      ) {
        return;
      }
      const selector = selectorFor(target);
      if (!selector) {
        return;
      }
      record({ type: "type_text", selector, text: target.value });
    },
    true,
  );

  document.addEventListener(
    "keydown",
    (event) => {
      if (!["Enter", "Tab", "Escape"].includes(event.key)) {
        return;
      }
      const target = event.target instanceof Element ? event.target : document.body;
      const selector = selectorFor(target);
      if (!selector) {
        return;
      }
      record({ type: "press_key", selector, key: event.key });
    },
    true,
  );
})();
"""


def build_recorded_plan(
    events: Sequence[Mapping[str, Any]], *, finish_reason: str = "recorded session complete"
) -> list[AgentAction]:
    """Normalize recorded DOM events into a deterministic baseline plan."""
    plan: list[AgentAction] = []
    for event in events:
        action = _event_to_action(event)
        if action is None:
            continue
        if (
            action.type is ActionType.TYPE_TEXT
            and plan
            and plan[-1].type is ActionType.TYPE_TEXT
            and plan[-1].selector == action.selector
        ):
            plan[-1] = action
            continue
        plan.append(action)
    plan.append(AgentAction.finish(finish_reason))
    return plan


def record_case(
    task: TaskSpec,
    *,
    finish_reason: str = "recorded session complete",
    headless: bool = False,
    viewport: tuple[int, int] = (1280, 720),
    wait_for_finish: Callable[[], None] | None = None,
) -> BenchmarkCase:
    """Launch a browser, capture manual actions, and return a benchmark case."""
    from playwright.sync_api import sync_playwright

    events: list[dict[str, Any]] = []

    def _record(_source: Any, payload: Any) -> None:
        if isinstance(payload, dict):
            events.append(payload)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
        try:
            page.expose_binding("__agenticQaRecord", _record)
            page.add_init_script(_RECORDER_SCRIPT)
            page.goto(task.start_url)
            if wait_for_finish is not None:
                wait_for_finish()
        finally:
            browser.close()

    return BenchmarkCase(task=task, plan=build_recorded_plan(events, finish_reason=finish_reason))


def _event_to_action(event: Mapping[str, Any]) -> AgentAction | None:
    """Convert one recorded DOM event into an :class:`AgentAction`."""
    kind = str(event.get("type") or "").strip().lower()
    selector = _string_value(event.get("selector"))
    if kind == "click":
        return AgentAction.click(selector) if selector else None
    if kind == "type_text":
        text = _string_value(event.get("text"))
        return AgentAction.type_text(text, selector=selector) if selector and text else None
    if kind == "press_key":
        key = _string_value(event.get("key"))
        return AgentAction.press_key(key, selector=selector) if selector and key else None
    return None


def _string_value(value: Any) -> str | None:
    """Return a stripped string value or ``None`` when empty."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
