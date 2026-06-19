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

## Agents and the runner

`agentic_qa_lab.agents` separates *deciding* from *doing*. An `Agent` is a pure
function — `next_action(task, observation, trace) -> AgentAction` — that performs
no I/O, so agents are trivial to test and swap.

- **`RuleBasedAgent`** — the deterministic baseline. It replays a fixed action
  plan, finishes early if the task's `success_selector` appears in the DOM, and
  finishes cleanly once the plan is exhausted.
- **`Runner`** — owns the observe → decide → act loop and the safeguards:
  `max_steps`, `max_retries` (failed actions are retried up to the budget), and
  a wall-clock `timeout_seconds`. It returns an aggregated `RunResult` whose
  terminal `RunStatus` is one of `success`, `failure`, `timeout`, `max_steps`,
  or `error`. Agent exceptions are caught and surface as `error` /
  `AGENT_ERROR` rather than crashing the run.
- **`write_trace_jsonl(run, path)`** — persists a run as JSONL: one `step`
  record per trace step plus a final `summary` record.

```python
from agentic_qa_lab.agents import RuleBasedAgent, Runner, write_trace_jsonl
from agentic_qa_lab.domain import AgentAction, TaskSpec

task = TaskSpec(task_id="demo", goal="Submit", start_url="https://example.com/")
agent = RuleBasedAgent([AgentAction.click("#submit"), AgentAction.finish()])
# run = Runner().run(task, agent, env)   # env is any BrowserEnvironment
# write_trace_jsonl(run, "artifacts/demo.jsonl")
```

### Failure modes

- A non-terminal action that keeps failing past `max_retries` ends the run as
  `failure` with the environment's `FailureCategory`.
- Exceeding `max_steps` ends the run as `max_steps`.
- Exceeding `timeout_seconds` ends the run as `timeout`.
- A `finish` action with an unmet `success_selector` is treated as `failure`.

## LLM planner

The planning layer is provider-agnostic. The core depends only on the
`LLMClient` protocol — a single `complete(messages) -> str` call — so any
backend can be plugged in without touching the agent logic.

- **`OpenAICompatibleClient`** — talks to any OpenAI-style
  `/chat/completions` endpoint using only the standard library. It is
  configured entirely through environment variables: `LLM_API_KEY` (required),
  `LLM_BASE_URL` (default `https://api.openai.com/v1`), and `LLM_MODEL`
  (default `gpt-4o-mini`).
- **`LLMPlannerAgent`** — renders the goal, the current observation (URL,
  title, truncated DOM), and a short action history into a chat prompt, then
  parses the reply into a strictly-validated `AgentAction`. JSON inside a
  ```` ```json ```` fence is supported. Invalid replies trigger a correction
  re-prompt up to `max_parse_retries`; if the model still fails, the agent
  emits a terminal `fail` so the run ends cleanly.

```bash
export LLM_API_KEY=sk-...           # any OpenAI-compatible provider
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini
```

```python
from agentic_qa_lab.agents import LLMPlannerAgent, OpenAICompatibleClient

agent = LLMPlannerAgent(OpenAICompatibleClient())
# run = Runner().run(task, agent, env)
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
