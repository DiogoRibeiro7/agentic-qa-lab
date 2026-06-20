# Security Policy

## Supported versions

This is a portfolio/research project under active development. Security fixes
are applied to the `main` branch only.

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities.

Instead, report privately via one of:

- GitHub's [private vulnerability reporting](https://github.com/DiogoRibeiro7/agentic-qa-lab/security/advisories/new)
- Email: hansolo.dj@gmail.com

Include a description, reproduction steps, and the impact you observed. You can
expect an acknowledgement within a few days.

## Handling secrets

This project automates browsers and HTTP requests and can drive authenticated
flows. To keep credentials out of source control:

- Task plans reference secrets with explicit env refs, e.g.
  `text: {env: AGENTIC_QA_EXAMPLE_LOGIN_PASSWORD}` — never inline plaintext.
- The recorder's `--secret-field "#password=ENV_VAR"` option stores an env
  reference instead of the typed value.
- LLM and API configuration (`LLM_API_KEY`, etc.) is loaded from environment
  variables or a local, untracked `.env` file via `pydantic-settings`.

Never commit real API keys, passwords, or `.env` files.

## Risky actions

The `ApprovalAgent` + `RiskPolicy` layer gates destructive actions (delete,
submit, pay, confirm, logout, ...) behind an approver callback. When automating
against real systems, prefer running with `--require-approval` or a custom
approver rather than `allow_all`.
