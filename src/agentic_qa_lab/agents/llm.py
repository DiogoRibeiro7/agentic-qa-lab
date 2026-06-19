"""LLM client abstraction.

The core never imports a vendor SDK. It depends only on the :class:`LLMClient`
protocol — a single ``complete(messages)`` call returning text. The bundled
:class:`OpenAICompatibleClient` talks to any OpenAI-style ``/chat/completions``
endpoint using only the standard library, configured entirely through
environment variables.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Usage:
    """Token usage reported by a provider for a single completion."""

    input_tokens: int
    output_tokens: int


def _usage_from_response(body: dict[str, Any]) -> Usage | None:
    """Extract an OpenAI-style ``usage`` block, or ``None`` if absent."""
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if prompt is None or completion is None:
        return None
    return Usage(input_tokens=int(prompt), output_tokens=int(completion))


def _image_data_uri(path: str | Path) -> str:
    """Read a local image file and return a base64-encoded data URI.

    This is used for OpenAI multimodal ``image_url`` content parts, so the
    image can be attached to a chat message without a separate hosting step.
    """
    raw = Path(path).read_bytes()
    suffix = Path(path).suffix.lower().lstrip(".") or "png"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:image/{suffix};base64,{encoded}"


@dataclass(frozen=True)
class LLMMessage:
    """A single chat message, optionally carrying image attachments.

    Attributes
    ----------
    role:
        One of ``system``, ``user``, or ``assistant``.
    content:
        The message text.
    images:
        Paths to image files attached to the message. When present the message
        is serialized using the OpenAI multimodal ``content`` parts format.
    """

    role: Role
    content: str
    images: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        """Return the message in OpenAI ``/chat/completions`` shape.

        Text-only messages use a plain string ``content``; messages with images
        use the list-of-parts (``text`` + ``image_url``) multimodal format.
        """
        if not self.images:
            return {"role": self.role, "content": self.content}
        parts: list[dict[str, Any]] = [{"type": "text", "text": self.content}]
        for image in self.images:
            parts.append({"type": "image_url", "image_url": {"url": _image_data_uri(image)}})
        return {"role": self.role, "content": parts}


def _extract_completion_content(body: object) -> str:
    """Extract assistant content from a chat-completions API response.

    Args:
        body: Decoded JSON response object.

    Returns:
        Assistant content as text.

    Raises:
        ValueError: If the response does not match the expected schema.
    """
    if not isinstance(body, dict):
        raise ValueError("Completion response must be a JSON object.")

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Completion response is missing a non-empty 'choices' list.")

    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("Completion choice must be a JSON object.")

    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("Completion choice is missing 'message'.")

    content = message.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ValueError("Completion content must be a string when present.")
    return content


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat-completion interface used by planner agents."""

    def complete(self, messages: list[LLMMessage]) -> str:
        """Return the assistant's text completion for ``messages``."""
        ...


@runtime_checkable
class StructuredLLMClient(Protocol):
    """Optional JSON-schema completion capability for chat clients."""

    def complete_json(
        self,
        messages: list[LLMMessage],
        *,
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a JSON object conforming to ``schema``."""
        ...


class LLMConfigError(RuntimeError):
    """Raised when required LLM configuration is missing."""


class OpenAICompatibleClient:
    """Chat client for any OpenAI-compatible ``/chat/completions`` endpoint.

    Configuration is read from the environment so no secret is ever passed
    through code:

    ============== ====================================== ==================
    Variable       Meaning                                Default
    ============== ====================================== ==================
    ``LLM_API_KEY``   Bearer token (required)              —
    ``LLM_BASE_URL``  API root                             ``https://api.openai.com/v1``
    ``LLM_MODEL``     Model name                           ``gpt-4o-mini``
    ============== ====================================== ==================

    Parameters
    ----------
    temperature:
        Sampling temperature.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(self, *, temperature: float = 0.0, timeout: float = 30.0) -> None:
        self._api_key = os.environ.get("LLM_API_KEY")
        self._base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self._model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self._temperature = temperature
        self._timeout = timeout
        #: Token usage reported by the most recent :meth:`complete` call, when
        #: the provider returned a ``usage`` block (``None`` otherwise).
        self.last_usage: Usage | None = None

    def complete(self, messages: list[LLMMessage]) -> str:
        """POST ``messages`` to the chat endpoint and return the reply text.

        Raises
        ------
        LLMConfigError
            If ``LLM_API_KEY`` is not set.
        """
        body = self._post_chat(messages)
        return _extract_completion_content(body)

    def complete_json(
        self,
        messages: list[LLMMessage],
        *,
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a JSON object using OpenAI ``response_format=json_schema``."""
        body = self._post_chat(
            messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        raw = _extract_completion_content(body)
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Structured completion JSON must be an object.")
        return data

    def _post_chat(
        self,
        messages: list[LLMMessage],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST ``messages`` to the chat endpoint and return the decoded body."""
        if not self._api_key:
            raise LLMConfigError("LLM_API_KEY environment variable is not set.")

        payload_data: dict[str, Any] = {
            "model": self._model,
            "messages": [m.as_dict() for m in messages],
            "temperature": self._temperature,
        }
        if response_format is not None:
            payload_data["response_format"] = response_format
        payload = json.dumps(payload_data).encode("utf-8")

        request = urllib.request.Request(  # noqa: S310 - URL is operator-configured
            url=f"{self._base_url}/chat/completions",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310
            body = json.loads(response.read().decode("utf-8"))
        self.last_usage = _usage_from_response(body)
        if not isinstance(body, dict):
            raise ValueError("Completion response must be a JSON object.")
        return body
