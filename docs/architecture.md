# Architecture

```text
TaskSpec
  -> BrowserEnvironment
  -> Observation
  -> Agent.next_action(...)
  -> AgentAction
  -> BrowserEnvironment.apply(action)
  -> ActionResult
  -> TraceStep
  -> Evaluator
```

## Main design idea

The project separates decision-making from execution. Agents choose actions. Environments execute them. Evaluators score the final trace. This separation makes it possible to compare rule-based agents, LLM agents, and vision-language agents on the same tasks.

## Main components

- `agentic_qa_lab.domain`: transport-safe Pydantic models for tasks, actions, observations, traces, and run results.
- `agentic_qa_lab.environments`: browser adapters behind the `BrowserEnvironment` interface.
- `agentic_qa_lab.agents`: planners, wrappers, and the `Runner` loop.
- `agentic_qa_lab.evaluation`: task loading, benchmark execution, summary metrics, and exports.
- `agentic_qa_lab.api`: FastAPI ingestion and retrieval service for completed runs.

## Execution flow

1. A `TaskSpec` defines the goal, start URL, and termination constraints.
2. An environment opens the page and emits an initial `Observation`.
3. An agent turns the current observation plus trace into an `AgentAction`.
4. The environment executes the action and returns an `ActionResult`.
5. The runner appends a `TraceStep` and repeats until success or a terminal failure state.
6. Evaluation utilities aggregate run outcomes into benchmark metrics and exports.

## Why the boundaries matter

- Agents stay pure and testable because they do not perform I/O directly.
- Environments can be swapped without changing planner logic.
- Benchmarks can compare multiple agent strategies on identical tasks.
- The API and dashboard can remain browser-free because they only ingest completed runs.
