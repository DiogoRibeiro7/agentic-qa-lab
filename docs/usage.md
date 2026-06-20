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
- `--environment selenium` (or `playwright`, `appium`, `api`, `auto`) to choose the backend.
- `--mode combined` to attach both DOM-derived context and screenshots.
- `--reflect` to enable the settle-and-retry repair loop.
- `--self-heal` to retry `element_not_found` actions with DOM-derived selector alternatives.
- `--judge-success` to enable model-graded success checks for tasks that define `success_judge`.
- `--require-approval` to prompt before risky actions.

Example:

```bash
agentic-qa run --task tasks/example_login.yaml --agent llm --mode combined --reflect
```

Selenium example:

```bash
agentic-qa run --task tasks/example_login.yaml --environment selenium
```

## Record a task

Capture a manual browser session into a reusable task file:

```bash
agentic-qa record --task-id example-login --goal "Log in" --start-url https://example.com/login --out-file tasks/example_login.yaml
agentic-qa record --task-id example-login --goal "Log in" --start-url https://example.com/login --out-file tasks/example_login.yaml --secret-field "#password=AGENTIC_QA_EXAMPLE_LOGIN_PASSWORD"
```

The recorder launches a browser, logs clicks, field edits, and supported key
presses, then writes a `TaskSpec` plus baseline `plan` to YAML or JSON.

## Secret values in task files

Task actions can resolve sensitive text from the environment instead of storing
plaintext in source control:

```yaml
plan:
  - type: type_text
    selector: "#password"
    text: {env: AGENTIC_QA_EXAMPLE_LOGIN_PASSWORD}
```

`load_case` and the CLI resolve `{env: VAR_NAME}` before validating the
`AgentAction`. If the variable is missing, task loading fails fast.

## Semantic success judging

For tasks whose completion is not a simple text/selector match, add a natural-
language success rubric:

```yaml
task_id: semantic-demo
goal: Confirm the report was published.
start_url: https://example.com/reports
success_judge: The page should clearly indicate that the report is published and visible to the user.
plan:
  - {type: finish, reason: Publication flow completed.}
```

Then enable judging at runtime:

```bash
agentic-qa run --task tasks/semantic_demo.yaml --judge-success
agentic-qa benchmark --tasks "tasks/*.yaml" --judge-success
```

## API environment

For non-UI flows, `APIEnvironment` lets the same runner/agent loop drive HTTP
requests instead of browser actions.

Request-building conventions:

- `type_text(..., selector="#method")`
- `type_text(..., selector="#path")`
- `type_text(..., selector="#body")`
- `type_text(..., selector="#query:<name>")`
- `type_text(..., selector="#header:<name>")`
- `click("#send")`

## Selenium environment

For a Selenium-backed browser adapter, use `SeleniumEnvironment` from
`agentic_qa_lab.environments`. It mirrors the Playwright environment contract,
including structured observations, screenshots, and shared failure taxonomy.

## Appium environment

For mobile or native-app automation, use `AppiumEnvironment` from
`agentic_qa_lab.environments`.

- Use `appium://...` as a task `start_url` for native-app sessions.
- Use selectors such as `id=login_button`, `accessibility_id=Username`, or
  `xpath=//android.widget.TextView[@text="Battery"]`.
- `AppiumEnvironment.launch(...)` expects an Appium server plus desired
  capabilities for the target device/session.

CLI example:

```bash
agentic-qa run --task tasks/mobile/login.yaml --environment appium --appium-capabilities-file capabilities/android.yaml
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
- `junit.xml` for JUnit-compatible CI report ingestion.
- `allure-results/` with one Allure result JSON file per run.

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

The dashboard supports:

- queuing runs through the API
- viewing the execution queue
- comparing two runs side by side
- stepping through a trace timeline with inline screenshots

## Local docs

Build the docs site:

```bash
poetry run mkdocs build --strict
```

Serve it locally with live reload:

```bash
poetry run mkdocs serve
```
