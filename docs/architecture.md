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
