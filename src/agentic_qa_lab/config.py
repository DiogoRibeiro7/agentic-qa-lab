"""Shared project configuration models."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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


class LLMSettings(BaseSettings):
    """Typed LLM configuration loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_API_KEY", "AGENTIC_QA_LLM_API_KEY"),
    )
    base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "AGENTIC_QA_LLM_BASE_URL"),
    )
    model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("LLM_MODEL", "AGENTIC_QA_LLM_MODEL"),
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout: float = Field(default=30.0, gt=0.0)

    @field_validator("base_url")
    @classmethod
    def strip_trailing_slash(cls, value: str) -> str:
        """Normalize the API root so callers can append endpoint paths safely."""
        return value.rstrip("/")


class APISettings(BaseSettings):
    """Typed API service settings loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    store_dir: Path = Field(
        default=Path("artifacts/runs"),
        validation_alias=AliasChoices("AGENTIC_QA_STORE_DIR"),
    )


class RuntimeSettings(BaseSettings):
    """Runtime settings shared by notebooks, CLI, and API components."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AGENTIC_QA_",
        env_nested_delimiter="__",
        extra="ignore",
        populate_by_name=True,
    )

    environment: str = Field(default="local")
    random_seed: int = Field(default=42, ge=0)
    debug: bool = Field(default=False)
    paths: ProjectPaths = Field(default_factory=ProjectPaths)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    api: APISettings = Field(default_factory=APISettings)
