# Roadmap

## Phase 1 — Project foundation

- Define domain models for tasks, observations, actions, traces, and evaluation results.
- Add CLI commands for running a single task and a batch benchmark.
- Add deterministic test environment with one simple browser task.

## Phase 2 — Browser environment

- Implement Playwright adapter.
- Add screenshot capture, DOM snapshot extraction, click/type/wait tools.
- Add timeout and retry controls.

## Phase 3 — Agent loop

- Implement planner interface.
- Add rule-based baseline agent.
- Add LLM planner.
- Add memory of previous actions and failures.

## Phase 4 — Evaluation

- Define benchmark tasks.
- Track success, retries, steps, latency, cost, and failure category.
- Store traces as JSONL and export summary reports.

## Phase 5 — Product layer

- Add FastAPI endpoints for run creation and trace inspection.
- Add dashboard for comparing runs.
- Add Docker Compose for local execution.

## Phase 6 — Advanced research

- Add vision-language screenshot reasoning.
- Add self-reflection and repair prompts.
- Add human approval for risky actions.
- Add regression tests for agent behavior.
