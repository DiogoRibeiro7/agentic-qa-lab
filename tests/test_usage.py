from __future__ import annotations

from typing import Any

from agentic_qa_lab.agents import LLMMessage, MeteredClient, TokenMeter, Usage, estimate_tokens
from agentic_qa_lab.domain import (
    ActionResult,
    AgentAction,
    FailureCategory,
    Observation,
    RunResult,
    RunStatus,
    TraceStep,
)
from agentic_qa_lab.evaluation import compute_summary


class EchoLLM:
    """Returns a fixed reply and counts calls."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls = 0

    def complete(self, messages: list[LLMMessage]) -> str:
        self.calls += 1
        return self._reply


# --------------------------------------------------------------------------- #
# estimate_tokens / TokenMeter
# --------------------------------------------------------------------------- #
def test_estimate_tokens_scales_with_length() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_token_meter_accumulates_and_costs() -> None:
    meter = TokenMeter(price_per_1k_input=1.0, price_per_1k_output=2.0)
    meter.record(1000, 500)
    meter.record(1000, 500)

    assert meter.calls == 2
    assert meter.input_tokens == 2000
    assert meter.output_tokens == 1000
    assert meter.total_tokens == 3000
    # 2000/1000*1.0 + 1000/1000*2.0 = 2.0 + 2.0
    assert meter.cost_usd == 4.0


def test_zero_price_meter_still_counts_tokens() -> None:
    meter = TokenMeter()
    meter.record(123, 45)
    assert meter.total_tokens == 168
    assert meter.cost_usd == 0.0


# --------------------------------------------------------------------------- #
# MeteredClient
# --------------------------------------------------------------------------- #
def test_metered_client_records_usage_and_passes_through() -> None:
    meter = TokenMeter(price_per_1k_input=3.0, price_per_1k_output=6.0)
    inner = EchoLLM("ok")  # 2 chars -> 1 output token (max(1, ...))
    client = MeteredClient(inner, meter)

    reply = client.complete([LLMMessage(role="user", content="a" * 400)])  # 100 input tokens

    assert reply == "ok"
    assert inner.calls == 1
    assert meter.input_tokens == 100
    assert meter.output_tokens == 1
    assert meter.calls == 1


class UsageReportingLLM:
    """Inner client that reports real usage via ``last_usage``."""

    def __init__(self, reply: str, usage: Usage) -> None:
        self._reply = reply
        self.last_usage = usage

    def complete(self, messages: list[LLMMessage]) -> str:
        return self._reply


class StructuredEchoLLM:
    """Structured-capable stub returning a fixed JSON object."""

    def __init__(self, reply: dict[str, Any]) -> None:
        self._reply = reply
        self.calls = 0

    def complete(self, messages: list[LLMMessage]) -> str:
        raise AssertionError("plain completion should not be used")

    def complete_json(
        self,
        messages: list[LLMMessage],
        *,
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls += 1
        return self._reply


def test_metered_client_prefers_real_usage_over_estimate() -> None:
    meter = TokenMeter(price_per_1k_input=1.0, price_per_1k_output=1.0)
    # Real usage is wildly different from what length estimation would give,
    # so a match proves the real numbers were used.
    inner = UsageReportingLLM("reply", Usage(input_tokens=4242, output_tokens=99))
    client = MeteredClient(inner, meter)

    client.complete([LLMMessage(role="user", content="a" * 400)])

    assert meter.input_tokens == 4242
    assert meter.output_tokens == 99


def test_metered_client_falls_back_to_estimate_without_usage() -> None:
    meter = TokenMeter()
    inner = EchoLLM("ok")  # no last_usage attribute
    client = MeteredClient(inner, meter)

    client.complete([LLMMessage(role="user", content="a" * 400)])

    assert meter.input_tokens == 100  # estimated from length
    assert meter.output_tokens == 1


def test_metered_client_records_structured_completion() -> None:
    meter = TokenMeter()
    inner = StructuredEchoLLM({"type": "click", "selector": "#go"})
    client = MeteredClient(inner, meter)

    reply = client.complete_json(
        [LLMMessage(role="user", content="a" * 400)],
        schema_name="agent_action",
        schema={"type": "object"},
    )

    assert reply["type"] == "click"
    assert inner.calls == 1
    assert meter.input_tokens == 100
    assert meter.output_tokens >= 1


def test_metered_client_structured_completion_prefers_real_usage() -> None:
    meter = TokenMeter()
    inner = StructuredEchoLLM({"type": "click"})
    inner.last_usage = Usage(input_tokens=321, output_tokens=12)
    client = MeteredClient(inner, meter)

    reply = client.complete_json(
        [LLMMessage(role="user", content="a" * 400)],
        schema_name="agent_action",
        schema={"type": "object"},
    )

    assert reply["type"] == "click"
    assert meter.input_tokens == 321
    assert meter.output_tokens == 12


def test_metered_client_complete_json_rejects_non_structured_inner() -> None:
    meter = TokenMeter()
    inner = EchoLLM("ok")
    client = MeteredClient(inner, meter)

    try:
        client.complete_json(
            [LLMMessage(role="user", content="hello")],
            schema_name="agent_action",
            schema={"type": "object"},
        )
    except TypeError as exc:
        assert "structured completion" in str(exc).lower()
    else:
        raise AssertionError("Expected complete_json to reject non-structured inner client")


# --------------------------------------------------------------------------- #
# Metrics: latency + cost aggregation
# --------------------------------------------------------------------------- #
def _run_with_latencies(
    latencies: list[float],
    *,
    tokens: int = 0,
    cost: float = 0.0,
    obs_ms: float = 0.0,
) -> RunResult:
    steps = [
        TraceStep(
            index=i,
            observation=Observation(
                step=i, url="https://e.com/", timestamp=1.0 + i, capture_ms=obs_ms
            ),
            action=AgentAction.click("#x"),
            result=ActionResult.ok(duration_ms=ms),
        )
        for i, ms in enumerate(latencies)
    ]
    return RunResult(
        task_id="t",
        status=RunStatus.SUCCESS,
        failure_category=FailureCategory.NONE,
        steps=steps,
        started_at=1.0,
        ended_at=2.0,
        duration_seconds=1.0,
        total_tokens=tokens,
        cost_usd=cost,
    )


def test_summary_includes_latency_and_cost() -> None:
    results = [
        _run_with_latencies([10.0, 20.0, 30.0], tokens=100, cost=0.5, obs_ms=5.0),
        _run_with_latencies([40.0, 100.0], tokens=50, cost=0.25, obs_ms=15.0),
    ]
    summary = compute_summary(results)

    # latencies pooled across all steps: [10,20,30,40,100]
    assert summary.mean_step_latency_ms == 40.0
    assert summary.p95_step_latency_ms == 100.0
    # observation latencies pooled: [5,5,5,15,15] -> mean 9
    assert summary.mean_observation_latency_ms == 9.0
    assert summary.total_tokens == 150
    assert summary.total_cost_usd == 0.75


def test_summary_latency_zero_without_steps() -> None:
    run = RunResult(
        task_id="t",
        status=RunStatus.SUCCESS,
        started_at=1.0,
        ended_at=2.0,
    )
    summary = compute_summary([run])
    assert summary.mean_step_latency_ms == 0.0
    assert summary.p95_step_latency_ms == 0.0
    assert summary.total_cost_usd == 0.0
