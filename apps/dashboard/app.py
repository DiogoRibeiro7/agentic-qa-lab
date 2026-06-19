"""Streamlit dashboard for inspecting and comparing runs.

Reads from the FastAPI service (``AGENTIC_QA_API_URL``, default
``http://localhost:8000``). It offers two views:

* **Comparison** — a sortable table of all runs with status, steps, retries,
  and duration, plus a headline success-rate metric.
* **Trace viewer** — step-by-step observation/action/result for a selected run.

Run with::

    streamlit run apps/dashboard/app.py
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

API_URL = os.environ.get("AGENTIC_QA_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 15.0


def _get(path: str) -> Any:
    """GET ``path`` from the API and return decoded JSON."""
    with urllib.request.urlopen(f"{API_URL}{path}", timeout=TIMEOUT) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _post(path: str, payload: dict[str, Any]) -> Any:
    """POST JSON to the API and return the decoded response."""
    request = urllib.request.Request(  # noqa: S310 - operator-configured URL
        url=f"{API_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _trace_rows(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten trace steps into dashboard-friendly table rows."""
    rows: list[dict[str, Any]] = []
    for step in trace:
        action = step["action"]
        result = step["result"]
        observation = step["observation"]
        rows.append(
            {
                "step": step["index"],
                "action": action["type"],
                "selector": action.get("selector"),
                "success": result["success"],
                "failure_category": result["failure_category"],
                "duration_ms": result.get("duration_ms", 0.0),
                "capture_ms": observation.get("capture_ms", 0.0),
                "url": observation["url"],
                "title": observation.get("title"),
                "screenshot_path": observation.get("screenshot_path"),
            }
        )
    return rows


def _run_diff_rows(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a compact side-by-side run comparison."""
    metrics = [
        ("task_id", "Task"),
        ("status", "Status"),
        ("failure_category", "Failure category"),
        ("steps", "Steps"),
        ("total_retries", "Retries"),
        ("duration_seconds", "Duration (s)"),
        ("total_tokens", "Tokens"),
        ("cost_usd", "Cost (USD)"),
    ]
    rows: list[dict[str, Any]] = []
    for key, label in metrics:
        rows.append(
            {
                "metric": label,
                "left": left.get(key),
                "right": right.get(key),
                "different": left.get(key) != right.get(key),
            }
        )
    return rows


def _show_screenshot(path_value: str | None, *, caption: str) -> None:
    """Render a screenshot when the file exists locally."""
    if not path_value:
        st.caption("No screenshot captured for this step.")
        return
    path = Path(path_value)
    if not path.exists():
        st.caption(f"Screenshot path not available locally: {path_value}")
        return
    st.image(str(path), caption=caption, use_container_width=True)


def render() -> None:
    """Render the dashboard."""
    st.set_page_config(page_title="agentic-qa-lab", layout="wide")
    st.title("agentic-qa-lab — run dashboard")
    st.caption(f"API: {API_URL}")

    try:
        executions = _get("/executions")
        runs = _get("/runs")
    except OSError as exc:
        st.error(f"Could not reach the API at {API_URL}: {exc}")
        st.stop()

    st.header("Launch run")
    with st.form("execute-run"):
        task_path = st.text_input("Task path", value="tasks/example_login.yaml")
        col1, col2 = st.columns(2)
        agent = col1.selectbox("Agent", ["rule", "llm"])
        mode = col2.selectbox("Observation mode", ["dom_only", "screenshot_only", "combined"])
        col3, col4 = st.columns(2)
        reflect = col3.checkbox("Reflect", value=False)
        headless = col4.checkbox("Headless", value=True)
        submitted = st.form_submit_button("Queue run")
    if submitted:
        try:
            record = _post(
                "/runs/execute",
                {
                    "task_path": task_path,
                    "agent": agent,
                    "mode": mode,
                    "reflect": reflect,
                    "headless": headless,
                },
            )
        except OSError as exc:
            st.error(f"Failed to queue run: {exc}")
        else:
            st.success(f"Queued execution {record['execution_id']}")

    st.header("Execution queue")
    if executions:
        st.dataframe(pd.DataFrame(executions), use_container_width=True)
    else:
        st.info("No queued or completed executions yet.")

    if not runs:
        st.info("No runs stored yet. Queue one above or POST a RunResult to /runs.")
        st.stop()

    frame = pd.DataFrame(runs)

    st.header("Run comparison")
    success_rate = (frame["status"] == "success").mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("Runs", len(frame))
    col2.metric("Success rate", f"{success_rate:.0%}")
    col3.metric("Failures", int((frame["status"] != "success").sum()))
    st.dataframe(frame, use_container_width=True)

    st.header("Run diff")
    diff_col1, diff_col2 = st.columns(2)
    left_run_id = diff_col1.selectbox("Left run", frame["run_id"].tolist(), key="left-run")
    right_choices = frame["run_id"].tolist()
    right_default = 1 if len(right_choices) > 1 else 0
    right_run_id = diff_col2.selectbox(
        "Right run", right_choices, index=right_default, key="right-run"
    )
    if left_run_id and right_run_id:
        left_run = _get(f"/runs/{left_run_id}")
        right_run = _get(f"/runs/{right_run_id}")
        diff_rows = _run_diff_rows(left_run, right_run)
        st.dataframe(pd.DataFrame(diff_rows), use_container_width=True)

    st.header("Trace timeline")
    run_id = st.selectbox("Select a run", frame["run_id"].tolist())
    if run_id:
        run = _get(f"/runs/{run_id}")
        trace = _get(f"/runs/{run_id}/trace")
        if not trace:
            st.write("This run has no recorded steps.")
            return
        st.caption(
            f"Run {run_id}: {run['status']} in {run['duration_seconds']:.2f}s, "
            f"{len(trace)} step(s), {run['total_retries']} retrie(s)."
        )
        rows = _trace_rows(trace)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        for step in trace:
            action = step["action"]
            result = step["result"]
            observation = step["observation"]
            title = (
                f"Step {step['index']} · {action['type']} · "
                f"{'ok' if result['success'] else result['failure_category']}"
            )
            with st.expander(title):
                meta1, meta2, meta3 = st.columns(3)
                meta1.metric("Action ms", f"{result.get('duration_ms', 0.0):.1f}")
                meta2.metric("Capture ms", f"{observation.get('capture_ms', 0.0):.1f}")
                meta3.metric("URL", observation["url"])
                if action.get("selector"):
                    st.code(action["selector"], language="text")
                _show_screenshot(
                    observation.get("screenshot_path"),
                    caption=f"Step {step['index']} screenshot",
                )
                st.json(
                    {
                        "observation": observation,
                        "action": action,
                        "result": result,
                    }
                )
        with st.expander("Raw trace JSON"):
            st.json(trace)


if __name__ == "__main__":
    render()
