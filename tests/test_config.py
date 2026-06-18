from __future__ import annotations

from pathlib import Path

from agentic_qa_lab.config import ProjectPaths, RuntimeSettings


def test_runtime_settings_defaults_are_valid() -> None:
    settings = RuntimeSettings()

    assert settings.environment == "local"
    assert settings.random_seed >= 0
    assert isinstance(settings.paths, ProjectPaths)


def test_project_paths_accept_path_objects() -> None:
    paths = ProjectPaths(data_dir=Path("data"), artifact_dir=Path("artifacts"), model_dir=Path("models"))

    assert paths.data_dir == Path("data")
    assert paths.artifact_dir == Path("artifacts")
    assert paths.model_dir == Path("models")
