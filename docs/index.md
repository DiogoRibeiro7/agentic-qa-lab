# agentic-qa-lab

`agentic-qa-lab` is a browser automation and evaluation lab for agentic QA
workflows. It combines browser control, planner abstractions, multimodal
observations, and benchmark/export tooling so you can compare deterministic and
LLM-driven agents on the same tasks.

## What it includes

- A domain layer for tasks, observations, actions, traces, and results.
- A Playwright-backed browser environment with screenshots and visible-text capture.
- Rule-based, reflective, approval-gated, and LLM-backed agents.
- Benchmark execution, summary metrics, and JSONL / CSV / JSON exports.
- A FastAPI ingestion API and a Streamlit inspection dashboard.

## Main workflows

- Run one task through the CLI and inspect its trace.
- Benchmark a task suite with the rule baseline or the LLM planner.
- Compare outcomes with latency, retries, failure categories, and token/cost metrics.
- Store completed runs in the API and inspect them through the dashboard.

## Quick links

- [Usage](usage.md) for local setup, CLI commands, and common workflows.
- [API](api.md) for the FastAPI endpoints and persistence model.
- [Architecture](architecture.md) for the execution model and component boundaries.
