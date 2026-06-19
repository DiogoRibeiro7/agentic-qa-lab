from __future__ import annotations

import json
from typing import Any

import pytest

from agentic_qa_lab.agents.llm import LLMMessage, OpenAICompatibleClient


class _FakeResponse:
    def __init__(self, body: dict[str, Any]) -> None:
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
