from __future__ import annotations

import base64
from pathlib import Path

from agentic_qa_lab.agents import LLMMessage, LLMPlannerAgent, ObservationMode
from agentic_qa_lab.domain import Observation, TaskSpec


class RecordingLLM:
    """Captures the messages it is asked to complete."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.last: list[LLMMessage] = []

    def complete(self, messages: list[LLMMessage]) -> str:
        self.last = messages
        return self._reply


def _task() -> TaskSpec:
    return TaskSpec(task_id="t", goal="Click go", start_url="https://e.com/")


def _obs(screenshot: str | None) -> Observation:
    return Observation(
        step=0,
        url="https://e.com/",
        title="Demo",
        dom_snapshot="<button id='go'>Go</button>",
        screenshot_path=screenshot,
        timestamp=1.0,
    )


# --------------------------------------------------------------------------- #
# Image attachment on LLMMessage
# --------------------------------------------------------------------------- #
def test_message_without_images_is_plain_string() -> None:
    msg = LLMMessage(role="user", content="hi")
    assert msg.as_dict() == {"role": "user", "content": "hi"}


def test_message_with_image_uses_multimodal_parts(tmp_path: Path) -> None:
    png = tmp_path / "shot.png"
    png.write_bytes(b"\x89PNG\r\n_fake_")
    msg = LLMMessage(role="user", content="look", images=(str(png),))

    payload = msg.as_dict()
    parts = payload["content"]
    assert parts[0] == {"type": "text", "text": "look"}
    assert parts[1]["type"] == "image_url"
    uri = parts[1]["image_url"]["url"]
    assert uri.startswith("data:image/png;base64,")
    decoded = base64.b64decode(uri.split(",", 1)[1])
    assert decoded == b"\x89PNG\r\n_fake_"


# --------------------------------------------------------------------------- #
# ObservationMode flags
# --------------------------------------------------------------------------- #
def test_mode_flags() -> None:
    assert ObservationMode.DOM_ONLY.uses_dom
    assert not ObservationMode.DOM_ONLY.uses_screenshot
    assert ObservationMode.SCREENSHOT_ONLY.uses_screenshot
    assert not ObservationMode.SCREENSHOT_ONLY.uses_dom
    assert ObservationMode.COMBINED.uses_dom and ObservationMode.COMBINED.uses_screenshot


# --------------------------------------------------------------------------- #
# Planner prompt construction per mode
# --------------------------------------------------------------------------- #
def test_dom_only_omits_screenshot(tmp_path: Path) -> None:
    png = tmp_path / "s.png"
    png.write_bytes(b"x")
    llm = RecordingLLM('{"type": "finish"}')
    LLMPlannerAgent(llm, observation_mode=ObservationMode.DOM_ONLY).next_action(
        _task(), _obs(str(png)), []
    )

    system, user = llm.last
    assert "button id=go" in user.content  # DOM summary present
    assert user.images == ()  # no screenshot attached
    assert "SCREENSHOT" not in system.content


def test_screenshot_only_attaches_image_and_drops_dom(tmp_path: Path) -> None:
    png = tmp_path / "s.png"
    png.write_bytes(b"x")
    llm = RecordingLLM('{"type": "finish"}')
    LLMPlannerAgent(llm, observation_mode=ObservationMode.SCREENSHOT_ONLY).next_action(
        _task(), _obs(str(png)), []
    )

    system, user = llm.last
    assert "id='go'" not in user.content  # DOM omitted
    assert user.images == (str(png),)  # screenshot attached
    assert "SCREENSHOT is attached" in system.content or "SCREENSHOT" in system.content


def test_combined_includes_both(tmp_path: Path) -> None:
    png = tmp_path / "s.png"
    png.write_bytes(b"x")
    llm = RecordingLLM('{"type": "finish"}')
    LLMPlannerAgent(llm, observation_mode=ObservationMode.COMBINED).next_action(
        _task(), _obs(str(png)), []
    )

    _system, user = llm.last
    assert "button id=go" in user.content
    assert user.images == (str(png),)


def test_screenshot_mode_without_screenshot_does_not_attach() -> None:
    llm = RecordingLLM('{"type": "finish"}')
    LLMPlannerAgent(llm, observation_mode=ObservationMode.SCREENSHOT_ONLY).next_action(
        _task(), _obs(None), []
    )

    _system, user = llm.last
    assert user.images == ()
    assert "unavailable" in user.content
