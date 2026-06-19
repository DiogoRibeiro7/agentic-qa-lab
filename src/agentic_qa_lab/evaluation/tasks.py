"""Load benchmark task definitions from YAML or JSON files.

A task file describes a :class:`TaskSpec` and, optionally, a ``plan``: a list of
actions for the deterministic :class:`RuleBasedAgent` baseline. Example::

    task_id: login
    goal: Log into the demo app
    start_url: https://example.com/login
    success_selector: Welcome
    max_steps: 10
    plan:
      - {type: type_text, selector: "#user", text: alice}
      - {type: click, selector: "#submit"}
      - {type: finish, reason: submitted}
"""

from __future__ import annotations

import json
from glob import glob
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ..domain import AgentAction, TaskSpec


class BenchmarkCase(BaseModel):
    """A task plus an optional baseline action plan.

    Attributes
    ----------
    task:
        The task specification.
    plan:
        Actions for the rule-based baseline agent. May be empty.
    """

    task: TaskSpec
    plan: list[AgentAction] = Field(default_factory=list)


def _load_raw(path: Path) -> dict[str, Any]:
    """Parse a single YAML/JSON file into a dict."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported task file extension: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"Task file {path} must contain a mapping at the top level.")
    return data


def load_case(path: str | Path) -> BenchmarkCase:
    """Load a single :class:`BenchmarkCase` from a file."""
    raw = _load_raw(Path(path))
    plan_raw = raw.pop("plan", []) or []
    plan = [AgentAction.model_validate(item) for item in plan_raw]
    task = TaskSpec.model_validate(raw)
    return BenchmarkCase(task=task, plan=plan)


def load_cases(patterns: list[str]) -> list[BenchmarkCase]:
    """Load and sort cases from one or more file paths or glob patterns.

    Globs are expanded internally so the command works the same whether or not
    the shell performs expansion (PowerShell does not). Duplicate paths are
    de-duplicated; results are sorted by ``task_id`` for stable reporting.
    """
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob(pattern)
        if matches:
            paths.extend(Path(m) for m in matches)
        elif Path(pattern).exists():
            paths.append(Path(pattern))
    unique = sorted({p.resolve() for p in paths})
    cases = [load_case(p) for p in unique]
    return sorted(cases, key=lambda c: c.task.task_id)


def dump_case(case: BenchmarkCase, path: str | Path) -> Path:
    """Write one :class:`BenchmarkCase` to YAML or JSON and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    raw = case.task.model_dump(mode="json")
    if case.plan:
        raw["plan"] = [action.model_dump(mode="json", exclude_none=True) for action in case.plan]

    if target.suffix.lower() in {".yaml", ".yml"}:
        text = yaml.safe_dump(raw, sort_keys=False, allow_unicode=False)
    elif target.suffix.lower() == ".json":
        text = json.dumps(raw, indent=2) + "\n"
    else:
        raise ValueError(f"Unsupported task file extension: {target.suffix}")

    target.write_text(text, encoding="utf-8")
    return target
