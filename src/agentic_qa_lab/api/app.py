"""FastAPI service exposing stored runs, traces, and queued executions.

The API still accepts completed :class:`RunResult` payloads through ``POST
/runs``, but it can also launch fresh task-file executions locally through a
lightweight background worker used by the dashboard.

Endpoints
---------
``GET  /health``              liveness probe
``POST /runs/execute``        queue a task file for execution
``GET  /executions``          list queued/completed executions
``GET  /executions/{id}``     execution status/details
``POST /runs``                store a RunResult, returns its ``run_id``
``GET  /runs``                list run summaries
``GET  /runs/{run_id}``       full RunResult
``GET  /runs/{run_id}/trace`` trace steps for a run
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from ..config import APISettings
from ..domain import RunResult, TraceStep
from .execution import (
    ExecutionNotFoundError,
    RunExecutionManager,
    RunExecutionRecord,
    RunExecutionRequest,
)
from .storage import RunRecord, RunStore, RunSummary


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Close background execution resources when the app shuts down."""
    try:
        yield
    finally:
        app.state.execution_manager.close()


def create_app(
    store: RunStore | None = None,
    execution_manager: RunExecutionManager | None = None,
) -> FastAPI:
    """Build the FastAPI app, optionally with an injected :class:`RunStore`.

    When ``store`` is omitted the directory is taken from
    :class:`~agentic_qa_lab.config.APISettings`.
    """
    settings = APISettings()
    if store is None:
        store = RunStore(settings.store_dir)
    if execution_manager is None:
        execution_manager = RunExecutionManager(store, artifact_dir=settings.store_dir)

    app = FastAPI(title="agentic-qa-lab", version="0.1.0", lifespan=_lifespan)
    app.state.store = store
    app.state.execution_manager = execution_manager

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    @app.post("/runs", response_model=RunRecord, status_code=201)
    def create_run(run: RunResult) -> RunRecord:
        """Store a completed run and return its record."""
        return store.create(run)

    @app.post("/runs/execute", response_model=RunExecutionRecord, status_code=202)
    def execute_run(request: RunExecutionRequest) -> RunExecutionRecord:
        """Queue one task file for execution."""
        return execution_manager.submit(request)

    @app.get("/executions", response_model=list[RunExecutionRecord])
    def list_executions() -> list[RunExecutionRecord]:
        """List queued and completed executions."""
        return execution_manager.list()

    @app.get("/executions/{execution_id}", response_model=RunExecutionRecord)
    def get_execution(execution_id: str) -> RunExecutionRecord:
        """Return one execution record."""
        try:
            return execution_manager.get(execution_id)
        except ExecutionNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"execution '{execution_id}' not found",
            ) from exc

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
