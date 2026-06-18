# agentic-qa-lab

[![CI](https://github.com/DiogoRibeiro7/agentic-qa-lab/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DiogoRibeiro7/agentic-qa-lab/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB)
![Poetry](https://img.shields.io/badge/dependencies-poetry-60A5FA)

Autonomous UI and game-testing agent using vision-language reasoning, browser control, action planning, failure recovery, and evaluation.

This repository is designed as a portfolio project for founding AI engineer, agentic AI, AI/ML engineer, and applied AI research roles.

## Goal

Build an agent that can interact with a browser-based game or UI workflow. The agent observes state, plans actions, executes clicks or keyboard events, recovers from failure, and produces structured test reports.

## Core capabilities

- Browser automation with Playwright.
- Agent planner with tool-calling interface.
- Optional vision-language model support through screenshot observations.
- Stateful execution loop with retries, timeouts, and recovery policies.
- Evaluation metrics: task success rate, steps to completion, retries, timeout rate, and failure taxonomy.
- FastAPI service and Streamlit dashboard for inspecting test runs.
- Dockerized local execution.

## Suggested implementation stack

- Python 3.11
- Poetry
- Playwright
- FastAPI
- Pydantic
- LangGraph or a custom state machine
- OpenAI-compatible LLM client abstraction
- SQLite/PostgreSQL for trace storage
- MLflow or local JSONL for experiment tracking

## Domain model

The core domain layer (`agentic_qa_lab.domain`) defines the provider- and
environment-agnostic types that flow through every run. They are plain Pydantic
models with strict validation and no I/O, so they serialize cleanly and are easy
to test.

- **`TaskSpec`** — declarative description of a task: goal, `start_url`, an
  optional `success_selector`, and run safeguards (`max_steps`, `max_retries`,
  `timeout_seconds`).
- **`Observation`** — a multimodal snapshot of environment state: URL, title,
  DOM snapshot, optional screenshot path, timestamp, and viewport.
- **`AgentAction`** — the unit of agent intent. Supported `ActionType` values
  are `click`, `type_text`, `press_key`, `wait`, `finish`, and `fail`.
  Cross-field validation enforces that, e.g., `click` has a selector or
  coordinates, `type_text` carries non-empty text, and `wait` has a positive
  duration. `finish`/`fail` are terminal.
- **`ActionResult`** — outcome of executing one action, including a
  `FailureCategory` taxonomy bucket, retry count, and duration.
- **`TraceStep`** — one `(observation, action, result)` tuple.
- **`RunResult`** — aggregated run outcome with a terminal `RunStatus`
  (`success`, `failure`, `timeout`, `max_steps`, `error`), the ordered trace,
  total retries, and timing. Validation keeps the status, failure category, and
  timestamps mutually consistent.

```python
from agentic_qa_lab.domain import AgentAction, TaskSpec

task = TaskSpec(task_id="login", goal="Log in", start_url="https://example.com")
action = AgentAction.type_text("alice", selector="#username")
```

Run the domain tests with `pytest tests/test_domain.py`.

## Browser environment

`agentic_qa_lab.environments` keeps all browser I/O behind the
`BrowserEnvironment` interface so agents only ever speak in domain types.
`PlaywrightEnvironment` is the reference adapter:

- `open(url)` navigates and returns the first `Observation`.
- `observe()` captures URL, title, DOM snapshot, and an optional screenshot.
- `execute(action)` dispatches a `click`, `type_text`, `press_key`, or `wait`
  to Playwright and returns a structured `ActionResult`. Timeouts and missing
  elements are mapped to `FailureCategory` buckets instead of raising.
- It is a context manager, so the browser is always released.

The adapter accepts an injected `page`, which makes it fully unit-testable with
a fake — see `tests/test_environments.py` (no browser binaries required).

```bash
playwright install chromium     # one-time, for real runs
python examples/simple_form_task.py
```

## Portfolio signal

This project shows that you can build agents that act in real software environments, not only generate text.

## Quickstart

### Prerequisites

- Python 3.11+
- Poetry
- Make (optional, but recommended)

### Setup

```bash
poetry install --with dev
poetry run playwright install
```

### Run locally

```bash
make run
```

## Development workflow

Use the Makefile commands for a consistent local loop:

```bash
make lint
make typecheck
make test
make format
make precommit
```

Install and use pre-commit hooks:

```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
```

## Project structure

```text
src/agentic_qa_lab/
  agents/         # Agent orchestration and decision logic
  domain/         # Domain entities and business rules
  environments/   # Browser/game environment adapters
  evaluation/     # Metrics and reporting utilities
  cli.py          # Main CLI entrypoint
  config.py       # Configuration and settings
tests/            # Unit and integration tests
docs/             # Architecture and supporting documentation
examples/         # Usage examples and scripts
notebooks/        # Exploration and experiment notebooks
```

## Quality gates

Pull requests are expected to pass:

- Ruff linting and formatting
- Mypy strict type checks
- Pytest test suite
- Pre-commit hook set

The CI workflow runs these checks on pushes and pull requests to main and develop.

## Contributing

1. Create a feature branch.
2. Keep changes scoped and include tests where behavior changes.
3. Run local quality checks before opening a pull request.
4. Open a PR with clear context, validation notes, and follow-up items.

Pull requests and issues should use the repository templates in .github to keep reports actionable and consistent.

## Roadmap

Planned milestones are tracked in ROADMAP.md.
