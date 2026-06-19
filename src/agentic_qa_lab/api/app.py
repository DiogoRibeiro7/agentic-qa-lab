"""FastAPI service exposing stored runs and their traces.

Design note: the API ingests *completed* runs rather than launching browsers
itself. The :class:`Runner`/benchmark produce :class:`RunResult` objects; this
service stores them and serves them to the dashboard. That keeps the web tier
stateless and free of browser dependencies.

Endpoints
---------
``GET  /health``            liveness probe
``POST /runs``              store a RunResult, returns its ``run_id``
``GET  /runs``              list run summaries
``GET  /runs/{run_id}``     full RunResult
``GET  /runs/{run_id}/trace`` trace steps for a run
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from ..config import APISettings
from ..domain import RunResult, TraceStep
from .storage import RunRecord, RunStore, RunSummary


def create_app(store: RunStore | None = None) -> FastAPI:
    """Build the FastAPI app, optionally with an injected :class:`RunStore`.

    When ``store`` is omitted the directory is taken from
    :class:`~agentic_qa_lab.config.APISettings`.
    """
    if store is None:
        store = RunStore(APISettings().store_dir)

    app = FastAPI(title="agentic-qa-lab", version="0.1.0")
    app.state.store = store

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    @app.post("/runs", response_model=RunRecord, status_code=201)
    def create_run(run: RunResult) -> RunRecord:
        """Store a completed run and return its record."""
        return store.create(run)

    @app.get("/runs", response_model=list[RunSummary])
    def list_runs() -> list[RunSummary]:
        """List summaries of all stored runs."""
        return store.list()

    @app.get("/runs/{run_id}", response_model=RunResult)
    def get_run(run_id: str) -> RunResult:
        """Return the full run for ``run_id``."""
        return _lookup(store, run_id).run

    @app.get("/runs/{run_id}/trace", response_model=list[TraceStep])
    def get_trace(run_id: str) -> list[TraceStep]:
        """Return the trace steps for ``run_id``."""
        return _lookup(store, run_id).run.steps

    return app


def _lookup(store: RunStore, run_id: str) -> RunRecord:
    """Fetch a record or raise a 404."""
    from .storage import RunNotFoundError

    try:
        return store.get(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found") from exc


#: Module-level app for ``uvicorn agentic_qa_lab.api.app:app``.
app = create_app()
