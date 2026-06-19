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


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat-completion interface used by planner agents."""

    def complete(self, messages: list[LLMMessage]) -> str:
        """Return the assistant's text completion for ``messages``."""
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

    def complete(self, messages: list[LLMMessage]) -> str:
        """POST ``messages`` to the chat endpoint and return the reply text.

        Raises
        ------
        LLMConfigError
            If ``LLM_API_KEY`` is not set.
        """
        if not self._api_key:
            raise LLMConfigError("LLM_API_KEY environment variable is not set.")

        payload = json.dumps(
            {
                "model": self._model,
                "messages": [m.as_dict() for m in messages],
                "temperature": self._temperature,
            }
        ).encode("utf-8")

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
        return str(body["choices"][0]["message"]["content"])
