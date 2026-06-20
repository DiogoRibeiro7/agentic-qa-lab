# Contributing to agentic-qa-lab

Thanks for your interest in contributing. This project values small, well-tested
changes that keep the quality gates green.

## Development setup

```bash
poetry install --with dev
poetry run playwright install
poetry run pre-commit install
```

## Local quality loop

Run these before opening a pull request — CI runs the same checks on Python
3.11, 3.12, and 3.13:

```bash
make lint        # ruff check
make typecheck   # mypy --strict
make test        # pytest with --cov-fail-under=90
make format      # ruff format
make precommit   # all pre-commit hooks
```

Or run the tools directly:

```bash
poetry run ruff check .
poetry run mypy
poetry run pytest -q
poetry run pre-commit run --all-files
```

## Pull request guidelines

1. Create a feature branch off `main`.
2. Keep changes scoped; include tests for any behavior change.
3. Ensure `ruff`, `mypy`, and `pytest` all pass locally.
4. Maintain test coverage at or above the 90% floor.
5. Update `README.md`, `ROADMAP.md`, and `CHANGELOG.md` when behavior or scope
   changes.
6. Open a PR using the template in `.github/`, with clear context and validation
   notes.

## Conventions

- Domain models live in `src/agentic_qa_lab/domain` and stay free of I/O.
- Environment adapters implement the `BrowserEnvironment` contract so agents
  only ever speak in domain types.
- Agents are pure `next_action(...)` functions; side effects belong to the
  `Runner` and environments.
- Network/browser tests must skip cleanly when binaries or connectivity are
  unavailable (see `tests/test_e2e.py`, `tests/test_real_tasks.py`).

## Reporting bugs and requesting features

Use the issue templates under `.github/ISSUE_TEMPLATE/`. For security issues,
follow [SECURITY.md](SECURITY.md) instead of opening a public issue.
