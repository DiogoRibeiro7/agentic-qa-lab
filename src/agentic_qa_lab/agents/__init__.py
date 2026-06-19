"""Agent implementations and the execution runner.

Agents are pure decision functions (see :class:`Agent`); the :class:`Runner`
owns the loop, safeguards, and trace aggregation.
"""

from __future__ import annotations

from .base import Agent
from .llm import (
    LLMClient,
    LLMConfigError,
    LLMMessage,
    OpenAICompatibleClient,
)
from .planner import LLMPlannerAgent, ObservationMode
from .reflective import ReflectiveAgent
from .rule_based import RuleBasedAgent
from .runner import Runner, write_trace_jsonl

__all__ = [
    "Agent",
    "LLMClient",
    "LLMConfigError",
    "LLMMessage",
    "LLMPlannerAgent",
    "ObservationMode",
    "OpenAICompatibleClient",
    "ReflectiveAgent",
    "RuleBasedAgent",
    "Runner",
    "write_trace_jsonl",
]
