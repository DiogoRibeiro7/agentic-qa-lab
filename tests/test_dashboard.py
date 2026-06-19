from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


def _load_dashboard_module():
    path = Path("apps/dashboard/app.py")
    spec = importlib.util.spec_from_file_location("dashboard_app", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("streamlit", types.SimpleNamespace())
    spec.loader.exec_module(module)
    return module


def test_trace_rows_flatten_trace() -> None:
    dashboard = _load_dashboard_module()

    rows = dashboard._trace_rows(  # noqa: SLF001 - testing dashboard helper
        [
            {
                "index": 1,
                "action": {"type": "click", "selector": "#go"},
                "result": {"success": True, "failure_category": "none", "duration_ms": 12.5},
                "observation": {
                    "url": "https://example.com",
                    "title": "Demo",
                    "capture_ms": 8.0,
                    "screenshot_path": "artifacts/runs/demo.png",
                },
            }
        ]
    )

    assert rows == [
        {
            "step": 1,
            "action": "click",
            "selector": "#go",
            "success": True,
            "failure_category": "none",
            "duration_ms": 12.5,
            "capture_ms": 8.0,
            "url": "https://example.com",
            "title": "Demo",
            "screenshot_path": "artifacts/runs/demo.png",
        }
    ]


def test_run_diff_rows_marks_changed_fields() -> None:
    dashboard = _load_dashboard_module()

    rows = dashboard._run_diff_rows(  # noqa: SLF001 - testing dashboard helper
        {
            "task_id": "login",
            "status": "success",
            "failure_category": "none",
            "steps": 2,
            "total_retries": 0,
            "duration_seconds": 1.0,
            "total_tokens": 10,
            "cost_usd": 0.01,
        },
        {
            "task_id": "login",
            "status": "failure",
            "failure_category": "timeout",
            "steps": 3,
            "total_retries": 1,
            "duration_seconds": 2.0,
            "total_tokens": 20,
            "cost_usd": 0.02,
        },
    )

    status_row = next(row for row in rows if row["metric"] == "Status")
    task_row = next(row for row in rows if row["metric"] == "Task")

    assert status_row["different"] is True
    assert task_row["different"] is False
