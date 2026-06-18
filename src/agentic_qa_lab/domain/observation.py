"""Observation produced by an environment after each step."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Observation(BaseModel):
    """A snapshot of environment state presented to the agent.

    Observations are multimodal: they may carry a DOM snapshot, a screenshot
    reference, or both. Agents decide which channels to consume.

    Attributes
    ----------
    step:
        Zero-based index of the step that produced this observation.
    url:
        Current page URL.
    title:
        Current page title, when available.
    dom_snapshot:
        Serialized DOM/accessibility text, when captured.
    screenshot_path:
        Filesystem path to a screenshot, when captured.
    timestamp:
        Epoch seconds when the observation was captured.
    viewport:
        ``(width, height)`` of the viewport in pixels.
    """

    step: int = Field(ge=0)
    url: str = Field(min_length=1)
    title: str | None = Field(default=None)
    dom_snapshot: str | None = Field(default=None)
    screenshot_path: str | None = Field(default=None)
    timestamp: float = Field(gt=0)
    viewport: tuple[int, int] | None = Field(default=None)

    @field_validator("viewport")
    @classmethod
    def _validate_viewport(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        """Reject non-positive viewport dimensions."""
        if value is not None and (value[0] <= 0 or value[1] <= 0):
            raise ValueError("viewport dimensions must be positive.")
        return value

    @property
    def has_visual(self) -> bool:
        """Return ``True`` when a screenshot is attached."""
        return self.screenshot_path is not None
