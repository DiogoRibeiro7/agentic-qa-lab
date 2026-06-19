"""Agent action types and their validation rules.

An :class:`AgentAction` is the unit of intent an agent emits on every step of a
run. Actions are intentionally small and explicit so that environments can
execute them deterministically and traces remain easy to audit.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class ActionType(StrEnum):
    """Enumeration of the actions an agent may take."""

    CLICK = "click"
    TYPE_TEXT = "type_text"
    PRESS_KEY = "press_key"
    WAIT = "wait"
    FINISH = "finish"
    FAIL = "fail"


#: Action types that terminate a run.
TERMINAL_ACTIONS: frozenset[ActionType] = frozenset({ActionType.FINISH, ActionType.FAIL})


class AgentAction(BaseModel):
    """A single action emitted by an agent.

    Only the fields relevant to the chosen :attr:`type` should be populated;
    cross-field validation enforces that contract so malformed actions are
    rejected at construction time rather than during execution.

    Attributes
    ----------
    type:
        The kind of action to perform.
    selector:
        CSS/XPath selector or element label. Required for ``click``,
        ``type_text`` and ``press_key``.
    x, y:
        Optional absolute click coordinates. Must be non-negative when given.
    text:
        Text payload for ``type_text``.
    key:
        Key name (for example ``"Enter"``) for ``press_key``.
    duration_ms:
        Wait duration in milliseconds for ``wait``. Must be positive.
    reason:
        Human-readable rationale, recommended for ``finish`` and ``fail``.
    """

    type: ActionType
    selector: str | None = Field(default=None)
    x: int | None = Field(default=None)
    y: int | None = Field(default=None)
    text: str | None = Field(default=None)
    key: str | None = Field(default=None)
    duration_ms: int | None = Field(default=None)
    reason: str | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_payload(self) -> AgentAction:
        """Validate that the payload matches the action type.

        Returns
        -------
        AgentAction
            The validated action.

        Raises
        ------
        ValueError
            If required fields are missing or coordinates/duration are invalid.
        """
        if self.x is not None and self.x < 0:
            raise ValueError("x coordinate must be non-negative.")
        if self.y is not None and self.y < 0:
            raise ValueError("y coordinate must be non-negative.")

        if self.type in {ActionType.CLICK, ActionType.TYPE_TEXT, ActionType.PRESS_KEY}:
            has_target = self.selector is not None or (self.x is not None and self.y is not None)
            if not has_target:
                raise ValueError(f"Action '{self.type}' requires a selector or (x, y) coordinates.")
        if self.type is ActionType.TYPE_TEXT and not self.text:
            raise ValueError("Action 'type_text' requires a non-empty text payload.")
        if self.type is ActionType.PRESS_KEY and not self.key:
            raise ValueError("Action 'press_key' requires a key name.")
        if self.type is ActionType.WAIT and (self.duration_ms is None or self.duration_ms <= 0):
            raise ValueError("Action 'wait' requires a positive duration_ms.")
        return self

    @property
    def is_terminal(self) -> bool:
        """Return ``True`` if this action ends the run."""
        return self.type in TERMINAL_ACTIONS

    @classmethod
    def click(
        cls, selector: str | None = None, *, x: int | None = None, y: int | None = None
    ) -> AgentAction:
        """Construct a ``click`` action."""
        return cls(type=ActionType.CLICK, selector=selector, x=x, y=y)

    @classmethod
    def type_text(cls, text: str, *, selector: str | None = None) -> AgentAction:
        """Construct a ``type_text`` action."""
        return cls(type=ActionType.TYPE_TEXT, selector=selector, text=text)

    @classmethod
    def press_key(cls, key: str, *, selector: str | None = None) -> AgentAction:
        """Construct a ``press_key`` action."""
        return cls(type=ActionType.PRESS_KEY, selector=selector, key=key)

    @classmethod
    def wait(cls, duration_ms: int) -> AgentAction:
        """Construct a ``wait`` action."""
        return cls(type=ActionType.WAIT, duration_ms=duration_ms)

    @classmethod
    def finish(cls, reason: str | None = None) -> AgentAction:
        """Construct a terminal ``finish`` action."""
        return cls(type=ActionType.FINISH, reason=reason)

    @classmethod
    def fail(cls, reason: str | None = None) -> AgentAction:
        """Construct a terminal ``fail`` action."""
        return cls(type=ActionType.FAIL, reason=reason)
