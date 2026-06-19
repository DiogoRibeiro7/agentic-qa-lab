# Roadmap

Status legend: ✅ done · 🟡 in progress · ⬜ planned

## Delivered

### Phase 1 — Project foundation ✅

- ✅ Domain models for tasks, observations, actions, traces, and results.
- ✅ CLI commands for a single task (`run`) and a batch benchmark (`benchmark`).
- ✅ Deterministic test environment and a local browser task.

### Phase 2 — Browser environment ✅

- ✅ Playwright adapter behind a `BrowserEnvironment` interface.
- ✅ Screenshot capture, DOM snapshot extraction, click/type/press/wait.
- ✅ Timeout handling and `FailureCategory` mapping.

### Phase 3 — Agent loop ✅

- ✅ `Agent` planner protocol and a deterministic `RuleBasedAgent` baseline.
- ✅ Provider-agnostic `LLMPlannerAgent` + `OpenAICompatibleClient`.
- ✅ Run memory of previous actions and failures (`summarize_trace`).

### Phase 4 — Evaluation ✅

- ✅ YAML/JSON benchmark tasks and a `BenchmarkRunner`.
- ✅ Metrics: success rate, steps, retries, latency, cost, failure categories.
- ✅ JSONL traces and CSV/JSON summary export.

### Phase 5 — Product layer ✅

- ✅ FastAPI endpoints for run ingestion and trace inspection.
- ✅ Streamlit dashboard for run comparison and trace viewing.
- ✅ Docker Compose for local execution.

### Phase 6 — Advanced ✅

- ✅ Vision-language screenshot reasoning with observation modes.
- ✅ Self-reflection / repair (`ReflectiveAgent`).
- ✅ Human approval for risky actions (`ApprovalAgent` + `RiskPolicy`).
- ✅ Regression tests for agent behaviour + real-browser e2e tests.

### Extras ✅

- ✅ Per-step latency and token/cost metering (`TokenMeter`, `MeteredClient`).
- ✅ Real local demo page and richer real-site benchmark tasks (`tasks/real/`).
- ✅ Green CI (Python 3.11): pre-commit, ruff, mypy, pytest.

---

## Known issues & bug fixes ⬜

- ✅ **Substring success detection is fragile.** Success now matches against an
  `Observation.visible_text` channel (`body.inner_text()`) that excludes
  `<script>` source, comments, and hidden nodes, falling back to `dom_snapshot`
  only when visible text is unavailable.
- ⬜ **Python version mismatch.** `pyproject.toml` pins `>=3.11,<3.13` while the
  project is developed/tested on 3.13. Widen the bound and add a CI matrix
  (3.11–3.13) to test what we actually support.
- ⬜ **No committed `poetry.lock`.** CI re-resolves dependencies on every run,
  so a transitive release can break the build (as happened with click 8.2).
  Commit a lock file and update it deliberately.
- ⬜ **Token counts are estimated, not measured.** `MeteredClient` approximates
  tokens by character count. Read real `usage` from the provider response when
  available and fall back to the estimate otherwise.
- ⬜ **Per-action latency excludes observation cost.** `duration_ms` times only
  `execute`; screenshot/DOM capture in `observe` is unmeasured. Add an
  observation-latency field.
- ⬜ **Approval gate is per-step, not per-session.** `ApprovalAgent` re-prompts
  for each risky action with no "approve all / remember" option.
- ⬜ **CRLF noise on Windows.** Add a `.gitattributes` enforcing LF to silence
  the line-ending warnings on commit.

## Improvements ⬜

- ⬜ **Reproducibility:** commit `poetry.lock`; pin pre-commit and CI tool
  versions to the dev dependencies so local and CI agree exactly.
- ⬜ **Parallel benchmarking:** run independent cases concurrently (process or
  async pool) with a configurable worker count; today `BenchmarkRunner` is
  sequential.
- ⬜ **Richer prompts:** trim/condense DOM more intelligently (accessibility
  tree instead of raw HTML), and cap memory/history by token budget.
- ⬜ **Structured LLM output:** use tool/function-calling or JSON-schema mode
  instead of free-text JSON parsing to cut retry churn.
- ⬜ **Coverage gate:** add a `--cov-fail-under` threshold in CI now that
  coverage sits around 90%.
- ⬜ **Typed config via env:** load `RuntimeSettings`/LLM config from `.env`
  with `pydantic-settings` rather than ad-hoc `os.environ` reads.
- ⬜ **Docs:** publish API/usage docs (mkdocs) and add a README CI badge.

## New features ⬜

- ⬜ **Run execution API.** Today the API only ingests completed runs. Add a
  `POST /runs/execute` that queues a task and a worker that runs it, so the
  dashboard can launch runs, not just view them.
- ⬜ **Dashboard upgrades.** Show step screenshots inline, a trace timeline, and
  side-by-side diffing of two runs (DOM-only vs combined, etc.).
- ⬜ **More environments.** Selenium and Appium/mobile adapters behind the same
  `BrowserEnvironment` interface; an API-testing environment for non-UI flows.
- ⬜ **Self-healing selectors.** On `element_not_found`, propose alternative
  selectors (nearby text/role) and retry before failing.
- ⬜ **Record & replay.** Capture a manual session into a `TaskSpec` + plan to
  seed the rule-based baseline.
- ⬜ **Secrets handling for tasks.** Reference credentials via env/secret refs in
  task files instead of inline plaintext (`SuperSecretPassword!`).
- ⬜ **Report exporters.** JUnit/Allure output so benchmark results plug into
  existing CI dashboards.
- ⬜ **Task pack.** Expand `tasks/real/` into a curated suite (forms, tables,
  dynamic content, auth) with difficulty tiers for cross-agent comparison.
- ⬜ **LLM judge.** Add an optional model-graded success check for tasks whose
  outcome is hard to assert via selectors.
