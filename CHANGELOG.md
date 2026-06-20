# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `LICENSE` (MIT) and project metadata (`license`, `homepage`, `repository`,
  `documentation`, `keywords`) in `pyproject.toml`.
- `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`, and `CITATION.cff`.
- `tasks/api/httpbin_get.yaml` — a runnable example for the HTTP `APIEnvironment`.
- `FailureCategory.AGENT_FAILED` and `FailureCategory.SUCCESS_UNCONFIRMED` to
  distinguish a deliberate agent `fail`, and a `finish` whose success criterion
  was not met, from the catch-all `UNKNOWN` bucket.

### Fixed

- **Type checking:** resolved all `mypy --strict` errors across `src` and
  `tests`, so the documented type-check gate passes again.
- **`p95_step_latency_ms`:** replaced banker's-rounding nearest-rank with a
  deterministic `ceil`-based rank, so the tail metric no longer silently
  collapses onto the maximum for small sample sizes.
- **Token/cost metering:** `TokenMeter.record` is now thread-safe, fixing lost
  increments when benchmarking with `--workers > 1`.
- **Failure taxonomy:** strict-mode selector violations are no longer
  mislabeled as `ELEMENT_NOT_FOUND` (which wrongly triggered self-healing).
- **Retries:** structurally invalid actions (`INVALID_ACTION`) are treated as
  permanent and no longer retried, matching the documented "transient" policy.

### Changed

- README: corrected the `tasks/api/` benchmark example, clarified that dashboard
  screenshots are not bundled, and the API module docstring now lists all
  endpoints.

## [0.1.0]

Initial portfolio release: domain models, Playwright/Selenium/Appium/API
environments, rule-based and LLM planner agents, reflective/approval/self-heal
wrappers, the evaluation benchmark with JSONL/CSV/JSON/JUnit/Allure exporters,
the FastAPI service and Streamlit dashboard, Docker Compose, and the MkDocs
site.

[Unreleased]: https://github.com/DiogoRibeiro7/agentic-qa-lab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/agentic-qa-lab/releases/tag/v0.1.0
