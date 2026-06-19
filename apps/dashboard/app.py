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
    col1, col2 = st.columns(2)
    col1.metric("Runs", len(frame))
    col2.metric("Success rate", f"{success_rate:.0%}")
    st.dataframe(frame, use_container_width=True)

    st.header("Trace viewer")
    run_id = st.selectbox("Select a run", frame["run_id"].tolist())
    if run_id:
        trace = _get(f"/runs/{run_id}/trace")
        if not trace:
            st.write("This run has no recorded steps.")
            return
        rows = [
            {
                "step": step["index"],
                "action": step["action"]["type"],
                "selector": step["action"].get("selector"),
                "success": step["result"]["success"],
                "failure_category": step["result"]["failure_category"],
                "url": step["observation"]["url"],
            }
            for step in trace
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        with st.expander("Raw trace JSON"):
            st.json(trace)


if __name__ == "__main__":
    render()
