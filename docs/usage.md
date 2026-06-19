# Usage

## Setup

```bash
poetry install --with dev
poetry run playwright install chromium
```

Environment-backed configuration can be supplied through shell variables or a
local `.env` file. The main LLM settings are:

```bash
export LLM_API_KEY=sk-...
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini
```

## Run one task

```bash
agentic-qa run --task tasks/example_login.yaml
```

Useful variants:

- `--agent llm` to use the planner instead of the rule baseline.
- `--mode combined` to attach both DOM-derived context and screenshots.
- `--reflect` to enable the settle-and-retry repair loop.
- `--require-approval` to prompt before risky actions.

Example:

```bash
agentic-qa run --task tasks/example_login.yaml --agent llm --mode combined --reflect
```

## Run a benchmark

```bash
agentic-qa benchmark --tasks "tasks/*.yaml" --tasks "tasks/*.json" --out-dir artifacts/benchmark
```

To run multiple cases concurrently:

```bash
agentic-qa benchmark --tasks "tasks/real/*.yaml" --workers 2
```

The benchmark command writes:

- `benchmark_summary.csv` with one row per run.
- `benchmark_summary.json` with aggregate summary plus per-run detail.

## API and dashboard

Run the API and dashboard locally:

```bash
uvicorn agentic_qa_lab.api.app:app --reload
streamlit run apps/dashboard/app.py
```

Or bring up both via Docker Compose:

```bash
docker compose up --build
```

## Local docs

Build the docs site:

```bash
poetry run mkdocs build --strict
```

Serve it locally with live reload:

```bash
poetry run mkdocs serve
```
