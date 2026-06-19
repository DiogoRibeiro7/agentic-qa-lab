# API

The FastAPI service stores and serves completed `RunResult` records, and it can
also queue task-file executions through a lightweight local worker.

## Endpoints

### `GET /health`

Simple liveness probe.

Response:

```json
{"status": "ok"}
```

### `POST /runs`

Stores one completed `RunResult`.

Returns a `RunRecord` with a generated `run_id`.

### `POST /runs/execute`

Queues one task file for execution.

Request body:

```json
{
  "task_path": "tasks/example_login.yaml",
  "agent": "rule",
  "mode": "dom_only",
  "reflect": false,
  "headless": true
}
```

Returns a `RunExecutionRecord` in `queued` state (or `running` quickly after).

### `GET /executions`

Lists queued, running, completed, and failed execution records.

### `GET /executions/{execution_id}`

Returns one execution record, including its eventual `run_id` on success.

### `GET /runs`

Lists stored run summaries. Each summary includes:

- `run_id`
- `task_id`
- `status`
- `failure_category`
- `steps`
- `total_retries`
- `duration_seconds`

### `GET /runs/{run_id}`

Returns the full stored `RunResult` for one run.

### `GET /runs/{run_id}/trace`

Returns the ordered list of `TraceStep` items for the run.

## Persistence

Runs are stored as one JSON file per run under `AGENTIC_QA_STORE_DIR`, which
defaults to `artifacts/runs`. The same root also holds trace JSONL files and
execution screenshots for API-triggered runs.

## App construction

For embedded or test use, construct the app with an injected store:

```python
from pathlib import Path

from agentic_qa_lab.api import RunStore, create_app

store = RunStore(Path("artifacts/runs"))
app = create_app(store)
```
