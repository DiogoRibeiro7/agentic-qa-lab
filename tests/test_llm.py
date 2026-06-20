from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from agentic_qa_lab.agents.llm import (
    LLMConfigError,
    LLMMessage,
    OpenAICompatibleClient,
    Usage,
    _extract_completion_content,
    _image_data_uri,
    _usage_from_response,
)
from agentic_qa_lab.config import LLMSettings


class _FakeResponse:
    def __init__(self, body: object) -> None:
        self._body = body

    def read(self) -> bytes:
        return json.dumps(self._body).encode("utf-8")

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


def test_openai_client_complete_json_sends_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(
            {
                "choices": [{"message": {"content": '{"type":"click","selector":"#go"}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = OpenAICompatibleClient(timeout=12.5)
    reply = client.complete_json(
        [LLMMessage(role="user", content="Click go")],
        schema_name="agent_action",
        schema={"type": "object", "required": ["type"]},
    )

    assert reply["selector"] == "#go"
    assert captured["timeout"] == 12.5
    assert captured["payload"]["response_format"]["type"] == "json_schema"
    assert captured["payload"]["response_format"]["json_schema"]["name"] == "agent_action"
    assert client.last_usage is not None
    assert client.last_usage.input_tokens == 12


def test_openai_client_uses_explicit_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    settings = LLMSettings(
        api_key="sk-explicit",
        base_url="https://example.com/v1",
        model="demo-model",
        timeout=12.0,
        temperature=0.5,
    )
    client = OpenAICompatibleClient(settings)

    assert client._api_key == "sk-explicit"  # noqa: SLF001 - constructor wiring test
    assert client._base_url == "https://example.com/v1"  # noqa: SLF001
    assert client._model == "demo-model"  # noqa: SLF001


def test_usage_from_response_handles_missing_and_partial_usage() -> None:
    assert _usage_from_response({}) is None
    assert _usage_from_response({"usage": "bad"}) is None
    assert _usage_from_response({"usage": {"prompt_tokens": 4}}) is None


def test_usage_from_response_parses_numeric_values() -> None:
    usage = _usage_from_response({"usage": {"prompt_tokens": "4", "completion_tokens": 3}})

    assert usage == Usage(input_tokens=4, output_tokens=3)


def test_image_data_uri_uses_file_suffix_and_default(tmp_path: Path) -> None:
    png = tmp_path / "shot.png"
    png.write_bytes(b"png-bytes")
    no_suffix = tmp_path / "shot"
    no_suffix.write_bytes(b"raw")

    assert _image_data_uri(png) == (
        "data:image/png;base64," + base64.b64encode(b"png-bytes").decode("ascii")
    )
    assert _image_data_uri(no_suffix) == (
        "data:image/png;base64," + base64.b64encode(b"raw").decode("ascii")
    )


def test_llm_message_as_dict_handles_text_and_images(tmp_path: Path) -> None:
    image = tmp_path / "screen.jpg"
    image.write_bytes(b"jpeg-data")

    assert LLMMessage(role="user", content="hello").as_dict() == {
        "role": "user",
        "content": "hello",
    }

    multimodal = LLMMessage(role="user", content="look", images=(str(image),)).as_dict()

    assert multimodal["role"] == "user"
    assert multimodal["content"][0] == {"type": "text", "text": "look"}
    assert multimodal["content"][1]["image_url"]["url"] == (
        "data:image/jpg;base64," + base64.b64encode(b"jpeg-data").decode("ascii")
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("bad", "Completion response must be a JSON object."),
        ({}, "Completion response is missing a non-empty 'choices' list."),
        ({"choices": []}, "Completion response is missing a non-empty 'choices' list."),
        ({"choices": ["bad"]}, "Completion choice must be a JSON object."),
        ({"choices": [{}]}, "Completion choice is missing 'message'."),
        (
            {"choices": [{"message": {"content": ["bad"]}}]},
            "Completion content must be a string when present.",
        ),
    ],
)
def test_extract_completion_content_rejects_invalid_shapes(body: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _extract_completion_content(body)


def test_extract_completion_content_allows_missing_content() -> None:
    assert _extract_completion_content({"choices": [{"message": {}}]}) == ""


def test_openai_client_complete_returns_text_and_tracks_absent_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-test")

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        assert timeout == 7.0
        payload = json.loads(request.data.decode("utf-8"))
        assert "response_format" not in payload
        assert request.full_url == "https://api.openai.com/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer sk-test"
        return _FakeResponse({"choices": [{"message": {"content": "done"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = OpenAICompatibleClient(timeout=7.0)

    assert client.complete([LLMMessage(role="user", content="Hi")]) == "done"
    assert client.last_usage is None


def test_openai_client_complete_json_rejects_non_object_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-test")

    def fake_urlopen(_request: Any, timeout: float) -> _FakeResponse:
        assert timeout == 5.0
        return _FakeResponse({"choices": [{"message": {"content": "[]"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = OpenAICompatibleClient(timeout=5.0)

    with pytest.raises(ValueError, match="Structured completion JSON must be an object."):
        client.complete_json(
            [LLMMessage(role="user", content="Click")],
            schema_name="agent_action",
            schema={"type": "object"},
        )


def test_post_chat_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    client = OpenAICompatibleClient(
        LLMSettings(
            api_key="",
            base_url="https://example.com/v1",
            model="demo",
            timeout=5.0,
            temperature=0.1,
        )
    )

    with pytest.raises(LLMConfigError, match="LLM_API_KEY environment variable is not set."):
        client.complete([LLMMessage(role="user", content="hello")])


def test_post_chat_rejects_non_object_response_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setattr("agentic_qa_lab.agents.llm._usage_from_response", lambda body: None)

    def fake_urlopen(_request: Any, timeout: float) -> _FakeResponse:
        assert timeout == 4.0
        return _FakeResponse(["not", "a", "dict"])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = OpenAICompatibleClient(timeout=4.0)

    with pytest.raises(ValueError, match="Completion response must be a JSON object."):
        client._post_chat([LLMMessage(role="user", content="hello")])  # noqa: SLF001
