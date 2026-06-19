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
- ✅ **Python version mismatch.** Widened to `>=3.11,<3.14`, dropped the unused
  numpy dependency, bumped pandas to a 3.13-compatible release, and added a CI
  matrix (3.11 / 3.12 / 3.13).
- ✅ **No committed `poetry.lock`.** A lock file now pins the full dependency
  graph so CI installs are deterministic and a transitive release cannot
  silently break the build.
- ✅ **Token counts are estimated, not measured.** `OpenAICompatibleClient` now
  records the provider's `usage` block, and `MeteredClient` records those real
  token counts when present, falling back to the length-based estimate only
  when the provider reports none.
- ✅ **Per-action latency excludes observation cost.** `Observation.capture_ms`
  now records the time spent producing each observation (screenshot + DOM +
  visible text); `RunResult.observation_latency_ms` and a
  `mean_observation_latency_ms` summary metric expose it.
- ✅ **Approval gate is per-step, not per-session.** `ApprovalAgent` now accepts
  session-scoped approvals (`ApprovalDecision.ALLOW_SESSION`), and the CLI
  prompt offers `approve all for this run`.
- ✅ **CRLF noise on Windows.** Added a `.gitattributes` enforcing LF so commits
  no longer warn about line-ending conversion.

## Improvements ⬜

- ✅ **Reproducibility:** `poetry.lock` is committed, dev toolchain versions in
  `pyproject.toml` are exact, pre-commit hook revisions are pinned, and CI now
  installs a fixed Poetry version instead of floating to latest.
- ✅ **Parallel benchmarking:** `BenchmarkRunner.run(..., workers=N)` now runs
  independent cases concurrently with a configurable thread-pool worker count;
  the CLI exposes it as `agentic-qa benchmark --workers N`.
- ✅ **Richer prompts:** `LLMPlannerAgent` now prefers visible page text plus a
  compact interactive-element summary instead of raw HTML dumps, and caps
  history/memory blocks by approximate token budget.
- ✅ **Structured LLM output:** clients can now expose
  `StructuredLLMClient.complete_json(...)`; `OpenAICompatibleClient` uses
  OpenAI `response_format=json_schema`, and `LLMPlannerAgent` prefers that path
  before falling back to free-text JSON parsing.
- ✅ **Coverage gate:** pytest now enforces `--cov-fail-under=90`, so CI fails
  when total coverage regresses below the agreed floor.
- ✅ **Typed config via env:** `RuntimeSettings`, `LLMSettings`, and
  `APISettings` now load from environment variables or `.env` via
  `pydantic-settings`; runtime code no longer reads `os.environ` ad hoc for LLM
  or API store configuration.
- ✅ **Docs:** added a MkDocs site (`mkdocs.yml`, usage/API/architecture
  pages), CI now builds docs, GitHub Pages deployment is wired, and README has
  docs/CI badges.

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
