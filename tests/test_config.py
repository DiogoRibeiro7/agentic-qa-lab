from __future__ import annotations

from pathlib import Path

import pytest

from agentic_qa_lab.config import APISettings, LLMSettings, ProjectPaths, RuntimeSettings


def test_runtime_settings_defaults_are_valid() -> None:
    settings = RuntimeSettings()

    assert settings.environment == "local"
    assert settings.random_seed >= 0
    assert isinstance(settings.paths, ProjectPaths)
    assert isinstance(settings.llm, LLMSettings)
    assert isinstance(settings.api, APISettings)


def test_project_paths_accept_path_objects() -> None:
    paths = ProjectPaths(
        data_dir=Path("data"), artifact_dir=Path("artifacts"), model_dir=Path("models")
    )

    assert paths.data_dir == Path("data")
    assert paths.artifact_dir == Path("artifacts")
    assert paths.model_dir == Path("models")


def test_llm_settings_read_dotenv_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_API_KEY=sk-test\n" "LLM_BASE_URL=https://example.com/v1/\n" "LLM_MODEL=test-model\n",
        encoding="utf-8",
    )

    settings = LLMSettings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.api_key == "sk-test"
    assert settings.base_url == "https://example.com/v1"
    assert settings.model == "test-model"


def test_runtime_settings_read_nested_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTIC_QA_ENVIRONMENT", "ci")
    monkeypatch.setenv("AGENTIC_QA_PATHS__ARTIFACT_DIR", "build-artifacts")

    settings = RuntimeSettings()

    assert settings.environment == "ci"
    assert settings.paths.artifact_dir == Path("build-artifacts")
