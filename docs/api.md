# API

The FastAPI service stores and serves completed `RunResult` records. It does
not launch browsers directly; execution still happens through the runner and
benchmark layers.

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
defaults to `artifacts/runs`. The path is configured through `APISettings`.

## App construction

For embedded or test use, construct the app with an injected store:

```python
from pathlib import Path

from agentic_qa_lab.api import RunStore, create_app

store = RunStore(Path("artifacts/runs"))
app = create_app(store)
```
