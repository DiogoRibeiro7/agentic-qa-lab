"""Token and cost metering for LLM clients.

Cost lives at the client boundary, so :class:`MeteredClient` wraps any
:class:`~agentic_qa_lab.agents.llm.LLMClient` and records token usage into a
:class:`TokenMeter` on every call. Token counts are estimated from text length
(no tokenizer dependency); pass real prices to turn counts into a USD estimate.
The wrapped client's behaviour is otherwise unchanged, so metering composes with
the planner transparently.
"""

from __future__ import annotations

from .llm import LLMClient, LLMMessage, Usage

#: Rough characters-per-token ratio used to estimate token counts.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` from its length."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


class TokenMeter:
    """Accumulate token usage and translate it into a USD cost estimate.

    Parameters
    ----------
    price_per_1k_input:
        USD per 1,000 input (prompt) tokens.
    price_per_1k_output:
        USD per 1,000 output (completion) tokens.
    """

    def __init__(
        self,
        *,
        price_per_1k_input: float = 0.0,
        price_per_1k_output: float = 0.0,
    ) -> None:
        self.price_per_1k_input = price_per_1k_input
        self.price_per_1k_output = price_per_1k_output
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def record(self, input_tokens: int, output_tokens: int) -> None:
        """Add one call's input/output token counts."""
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    @property
    def total_tokens(self) -> int:
        """Total input + output tokens recorded."""
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        """Estimated cost in USD from the recorded tokens and prices."""
        return (
            self.input_tokens / 1000 * self.price_per_1k_input
            + self.output_tokens / 1000 * self.price_per_1k_output
        )


class MeteredClient:
    """Wrap an :class:`LLMClient`, recording token usage on each completion.

    Parameters
    ----------
    inner:
        The client whose calls are metered.
    meter:
        Accumulator updated on every :meth:`complete` call.
    """

    def __init__(self, inner: LLMClient, meter: TokenMeter) -> None:
        self._inner = inner
        self._meter = meter

    def complete(self, messages: list[LLMMessage]) -> str:
        """Delegate to the inner client and record token usage.

        Prefers the real ``usage`` the inner client exposes via a ``last_usage``
        attribute (set by :class:`OpenAICompatibleClient`); falls back to a
        length-based estimate when the provider does not report usage.
        """
        reply = self._inner.complete(messages)
        usage: Usage | None = getattr(self._inner, "last_usage", None)
        if usage is not None:
            self._meter.record(usage.input_tokens, usage.output_tokens)
        else:
            input_tokens = sum(estimate_tokens(m.content) for m in messages)
            self._meter.record(input_tokens, estimate_tokens(reply))
        return reply
