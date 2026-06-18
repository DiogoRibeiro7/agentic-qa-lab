"""Shared project configuration models."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class ProjectPaths(BaseModel):
    """Filesystem paths used by the project.

    The model keeps path validation explicit so CLI and services fail early when
    a required directory is missing or incorrectly configured.
    """

    data_dir: Path = Field(default=Path("data"), description="Base data directory.")
    artifact_dir: Path = Field(default=Path("artifacts"), description="Generated artifacts.")
    model_dir: Path = Field(default=Path("models"), description="Trained model storage.")

    @field_validator("data_dir", "artifact_dir", "model_dir")
    @classmethod
    def ensure_relative_or_absolute_path(cls, value: Path) -> Path:
        """Validate path-like fields.

        Parameters
        ----------
        value:
            Candidate path.

        Returns
        -------
        Path
            The validated path.
        """
        if not isinstance(value, Path):
            raise TypeError("Expected pathlib.Path instance.")
        return value


class RuntimeSettings(BaseModel):
    """Runtime settings shared by notebooks, CLI, and API components."""

    environment: str = Field(default="local")
    random_seed: int = Field(default=42, ge=0)
    debug: bool = Field(default=False)
    paths: ProjectPaths = Field(default_factory=ProjectPaths)
