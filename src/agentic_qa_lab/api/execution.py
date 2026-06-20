"""Queued task execution for the HTTP API.

The API's storage layer persists completed runs, but some clients need to
launch fresh runs from task files instead of posting completed `RunResult`
objects. This module provides a small in-memory execution queue with a single
background worker thread. It is intentionally lightweight and appropriate for a
local lab, not a distributed job system.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from queue import Queue

from pydantic import BaseModel, Field

from ..agents import ObservationMode, Runner, TokenMeter, write_trace_jsonl
from ..domain import RunResult
from ..evaluation import BenchmarkCase, load_case
from .storage import RunStore


class ExecutionStatus(StrEnum):
    """Lifecycle states for an API-triggered task execution."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExecutionAgent(StrEnum):
    """Agents exposed through the run-execution API."""

    RULE = "rule"
    LLM = "llm"


class RunExecutionRequest(BaseModel):
    """Execution request submitted through `POST /runs/execute`."""

    task_path: Path
    agent: ExecutionAgent = Field(default=ExecutionAgent.RULE)
    mode: ObservationMode = Field(default=ObservationMode.DOM_ONLY)
    reflect: bool = Field(default=False)
    environment: str = Field(default="auto")
    judge_success: bool = Field(default=False)
    headless: bool = Field(default=True)
    appium_server: str = Field(default="http://127.0.0.1:4723")
    appium_capabilities_file: Path | None = None
    price_in: float = Field(default=0.0, ge=0.0)
    price_out: float = Field(default=0.0, ge=0.0)


class RunExecutionRecord(BaseModel):
    """Status record for one queued or completed execution."""

    execution_id: str
    status: ExecutionStatus
    task_path: str
    agent: ExecutionAgent
    mode: ObservationMode
    reflect: bool
    environment: str
    judge_success: bool
    headless: bool
    appium_server: str
    appium_capabilities_file: str | None = None
    run_id: str | None = None
    error: str | None = None
    submitted_at: float = Field(gt=0)
    started_at: float | None = None
    ended_at: float | None = None


RunFactory = Callable[[RunExecutionRequest], RunResult]


class ExecutionNotFoundError(KeyError):
    """Raised when an execution id is not present in the manager."""


class RunExecutionManager:
    """In-memory queue + worker thread for API-triggered runs."""

    def __init__(
        self,
        store: RunStore,
        *,
        artifact_dir: str | Path,
        run_factory: RunFactory | None = None,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._store = store
        self._artifact_dir = Path(artifact_dir)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._run_factory = run_factory or self._default_run_factory
        self._id_factory = id_factory
        self._records: dict[str, RunExecutionRecord] = {}
        self._queue: Queue[tuple[str, RunExecutionRequest] | None] = Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._work, name="api-run-worker", daemon=True)
        self._worker.start()

    def submit(self, request: RunExecutionRequest) -> RunExecutionRecord:
        """Queue one execution request and return its initial record."""
        execution_id = self._id_factory()
        record = RunExecutionRecord(
            execution_id=execution_id,
            status=ExecutionStatus.QUEUED,
            task_path=str(request.task_path),
            agent=request.agent,
            mode=request.mode,
            reflect=request.reflect,
            environment=request.environment,
            judge_success=request.judge_success,
            headless=request.headless,
            appium_server=request.appium_server,
            appium_capabilities_file=(
                str(request.appium_capabilities_file)
                if request.appium_capabilities_file is not None
                else None
            ),
            submitted_at=time.time(),
        )
        with self._lock:
            self._records[execution_id] = record
        self._queue.put((execution_id, request))
        return record

    def get(self, execution_id: str) -> RunExecutionRecord:
        """Return one execution record or raise if missing."""
        with self._lock:
            record = self._records.get(execution_id)
        if record is None:
            raise ExecutionNotFoundError(execution_id)
        return record

    def list(self) -> list[RunExecutionRecord]:
        """Return all execution records, newest first."""
        with self._lock:
            records = list(self._records.values())
        return sorted(records, key=lambda record: record.submitted_at, reverse=True)

    def close(self) -> None:
        """Stop the worker thread."""
        self._queue.put(None)
        self._worker.join(timeout=5.0)

    def _work(self) -> None:
        """Worker loop consuming queued execution requests."""
        while True:
            item = self._queue.get()
            if item is None:
                return
            execution_id, request = item
            self._update(
                execution_id,
                status=ExecutionStatus.RUNNING,
                started_at=time.time(),
                error=None,
            )
            try:
                result = self._run_factory(request)
                stored = self._store.create(result)
                self._update(
                    execution_id,
                    status=ExecutionStatus.COMPLETED,
                    run_id=stored.run_id,
                    ended_at=time.time(),
                    error=None,
                )
            except Exception as exc:  # noqa: BLE001 - captured into execution record
                self._update(
                    execution_id,
                    status=ExecutionStatus.FAILED,
                    ended_at=time.time(),
                    error=str(exc),
                )

    def _update(self, execution_id: str, **changes: object) -> None:
        """Replace one execution record with an updated copy."""
        with self._lock:
            record = self._records[execution_id]
            self._records[execution_id] = record.model_copy(update=changes)

    def _default_run_factory(self, request: RunExecutionRequest) -> RunResult:
        """Execute one task file using the same runtime path as the CLI."""
        from ..agents import ReflectiveAgent
        from ..cli import (
            AgentKind,
            EnvironmentKind,
            build_agent,
            build_environment,
            build_success_judge,
        )

        case: BenchmarkCase = load_case(request.task_path)
        meter = (
            TokenMeter(
                price_per_1k_input=request.price_in,
                price_per_1k_output=request.price_out,
            )
            if request.agent is ExecutionAgent.LLM
            else None
        )
        agent = build_agent(case, AgentKind(request.agent.value), request.mode, meter=meter)
        if request.reflect:
            agent = ReflectiveAgent(agent)
        judge = build_success_judge(enabled=request.judge_success, meter=meter)
        runner = Runner(stop_on_action_failure=not request.reflect, success_judge=judge)
        screenshots = self._artifact_dir / case.task.task_id / "screenshots"
        resolved_environment = EnvironmentKind(request.environment)
        screenshot_target = screenshots if resolved_environment is not EnvironmentKind.API else None
        with build_environment(
            case,
            resolved_environment,
            headless=request.headless,
            screenshot_dir=screenshot_target,
            appium_server=request.appium_server,
            appium_capabilities_file=request.appium_capabilities_file,
        ) as env:
            result = runner.run(case.task, agent, env)
        if meter is not None:
            result = result.model_copy(
                update={"total_tokens": meter.total_tokens, "cost_usd": meter.cost_usd}
            )
        write_trace_jsonl(result, self._artifact_dir / f"{case.task.task_id}.jsonl")
        return result
