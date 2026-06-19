from __future__ import annotations

import json
from pathlib import Path

from agentic_qa_lab.domain import ActionType, AgentAction, TaskSpec
from agentic_qa_lab.evaluation import BenchmarkCase, build_recorded_plan, dump_case


def test_build_recorded_plan_collapses_repeated_type_text() -> None:
    plan = build_recorded_plan(
        [
            {"type": "click", "selector": "#user"},
            {"type": "type_text", "selector": "#user", "text": "a"},
            {"type": "type_text", "selector": "#user", "text": "alice"},
            {"type": "press_key", "selector": "#user", "key": "Enter"},
        ],
        finish_reason="done",
    )

    assert [action.type for action in plan] == [
        ActionType.CLICK,
        ActionType.TYPE_TEXT,
        ActionType.PRESS_KEY,
        ActionType.FINISH,
    ]
    assert plan[1].text == "alice"
    assert plan[-1].reason == "done"


def test_build_recorded_plan_skips_invalid_events() -> None:
    plan = build_recorded_plan(
        [
            {"type": "click"},
            {"type": "type_text", "selector": "#user", "text": ""},
            {"type": "press_key", "selector": "#user", "key": "Enter"},
        ]
    )

    assert plan == [
        AgentAction.press_key("Enter", selector="#user"),
        AgentAction.finish("recorded session complete"),
    ]


def test_dump_case_writes_yaml_and_json(tmp_path: Path) -> None:
    case = BenchmarkCase(
        task=TaskSpec(task_id="login", goal="Log in", start_url="https://e.com/"),
        plan=[AgentAction.click("#submit"), AgentAction.finish("done")],
    )

    yaml_path = dump_case(case, tmp_path / "case.yaml")
    json_path = dump_case(case, tmp_path / "case.json")

    yaml_text = yaml_path.read_text(encoding="utf-8")
    assert "task_id: login" in yaml_text
    assert "plan:" in yaml_text

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    assert raw["task_id"] == "login"
    assert raw["plan"][-1]["type"] == "finish"
