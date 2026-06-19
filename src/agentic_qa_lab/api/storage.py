"""Filesystem-backed store for completed runs.

Each run is persisted as a single JSON file named by its generated ``run_id``.
Listing scans the directory, so the store is process-restart safe and needs no
database — appropriate for a local lab. The ``id_factory`` is injectable so
tests can make ids deterministic.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from ..domain import RunResult, TraceStep


class RunRecord(BaseModel):
    """A stored run: its id plus the full :class:`RunResult`."""

    run_id: str
    run: RunResult


class RunSummary(BaseModel):
    """Compact run description used for list views."""

    run_id: str
    task_id: str
    status: str
    failure_category: str
    steps: int
    total_retries: int
    duration_seconds: float


class RunNotFoundError(KeyError):
    """Raised when a ``run_id`` is not present in the store."""


class RunStore:
    """Persist and retrieve :class:`RunResult` records as JSON files.

    Parameters
    ----------
    root:
        Directory holding one ``<run_id>.json`` file per run. Created if absent.
    id_factory:
        Callable producing a new unique id. Defaults to a uuid4 hex string.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._id_factory = id_factory

    def _path(self, run_id: str) -> Path:
        """Return the filesystem path for a stored run id."""
        return self._root / f"{run_id}.json"

    def create(self, run: RunResult) -> RunRecord:
        """Store ``run`` under a fresh id and return the resulting record."""
        record = RunRecord(run_id=self._id_factory(), run=run)
        self._path(record.run_id).write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return record

    def get(self, run_id: str) -> RunRecord:
        """Return the stored record for ``run_id``.

        Raises
        ------
        RunNotFoundError
            If no run with that id exists.
        """
        path = self._path(run_id)
        if not path.exists():
            raise RunNotFoundError(run_id)
        return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def get_trace(self, run_id: str) -> list[TraceStep]:
        """Return the trace steps for ``run_id``."""
        return self.get(run_id).run.steps

    def list(self) -> list[RunSummary]:
        """Return summaries of all stored runs, newest file first."""
        summaries: list[RunSummary] = []
        for path in self._root.glob("*.json"):
            try:
                record = RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except (ValueError, json.JSONDecodeError):
                continue  # skip unrelated/corrupt files
            run = record.run
            summaries.append(
                RunSummary(
                    run_id=record.run_id,
                    task_id=run.task_id,
                    status=run.status.value,
                    failure_category=run.failure_category.value,
                    steps=run.step_count,
                    total_retries=run.total_retries,
                    duration_seconds=run.duration_seconds,
                )
            )
        summaries.sort(key=lambda s: s.run_id)
        return summaries
