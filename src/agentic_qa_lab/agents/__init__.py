"""Agent implementations and the execution runner.

Agents are pure decision functions (see :class:`Agent`); the :class:`Runner`
owns the loop, safeguards, and trace aggregation.
"""

from __future__ import annotations

from .approval import (
    ApprovalAgent,
    Approver,
    RiskPolicy,
    allow_all,
    deny_all,
)
from .base import Agent
from .llm import (
    LLMClient,
    LLMConfigError,
    LLMMessage,
    OpenAICompatibleClient,
)
from .memory import FailedTarget, MemorySummary, summarize_trace
from .planner import LLMPlannerAgent, ObservationMode
from .reflective import ReflectiveAgent
from .rule_based import RuleBasedAgent
from .runner import Runner, write_trace_jsonl
from .usage import MeteredClient, TokenMeter, estimate_tokens

__all__ = [
    "Agent",
    "ApprovalAgent",
    "Approver",
    "FailedTarget",
    "LLMClient",
    "LLMConfigError",
    "LLMMessage",
    "LLMPlannerAgent",
    "MemorySummary",
    "MeteredClient",
    "ObservationMode",
    "OpenAICompatibleClient",
    "ReflectiveAgent",
    "RiskPolicy",
    "RuleBasedAgent",
    "Runner",
    "TokenMeter",
    "allow_all",
    "deny_all",
    "estimate_tokens",
    "summarize_trace",
    "write_trace_jsonl",
]
