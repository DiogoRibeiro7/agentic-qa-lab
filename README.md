# agentic-qa-lab

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

## Roadmap

Planned milestones are tracked in ROADMAP.md.
