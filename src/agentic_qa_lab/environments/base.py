"""Abstract browser-environment interface.

An environment is the only component allowed to touch the outside world (a real
browser). Agents never call Playwright directly; they emit
:class:`~agentic_qa_lab.domain.action.AgentAction` objects that the environment
executes, returning structured
:class:`~agentic_qa_lab.domain.result.ActionResult` and
:class:`~agentic_qa_lab.domain.observation.Observation` values.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType

from ..domain import ActionResult, AgentAction, Observation


class BrowserEnvironment(ABC):
    """Interface every browser adapter must implement.

    Implementations are context managers so callers can guarantee the
    underlying browser is released even on error.
    """

    @abstractmethod
    def open(self, url: str) -> Observation:
        """Navigate to ``url`` and return the initial observation."""

    @abstractmethod
    def observe(self) -> Observation:
        """Capture the current environment state as an :class:`Observation`."""

    @abstractmethod
    def execute(self, action: AgentAction) -> ActionResult:
        """Apply ``action`` to the environment and return its result."""

    @abstractmethod
    def close(self) -> None:
        """Release all browser resources. Must be idempotent."""

    def __enter__(self) -> BrowserEnvironment:
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the environment on context exit."""
        self.close()
