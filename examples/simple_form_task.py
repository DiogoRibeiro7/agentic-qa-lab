"""Run a tiny scripted login-form interaction with the Playwright environment.

This example shows how the domain actions and the
:class:`~agentic_qa_lab.environments.PlaywrightEnvironment` fit together without
any agent or LLM in the loop — a fixed action script is executed directly.

Prerequisites
-------------
Install the Chromium binary once::

    playwright install chromium

Then run::

    python examples/simple_form_task.py
"""

from __future__ import annotations

from pathlib import Path

from agentic_qa_lab.domain import AgentAction, TaskSpec
from agentic_qa_lab.environments import PlaywrightEnvironment

TASK = TaskSpec(
    task_id="login-demo",
    goal="Fill the login form and submit it.",
    start_url="https://example.com/",
    success_selector="text=Welcome",
)

#: Fixed action script standing in for an agent's decisions.
SCRIPT: list[AgentAction] = [
    AgentAction.type_text("alice", selector="#username"),
    AgentAction.type_text("s3cret", selector="#password"),
    AgentAction.click("#submit"),
    AgentAction.finish("Submitted the form."),
]


def main() -> None:
    """Execute the scripted task and print each step's result."""
    screenshots = Path("artifacts") / "simple_form_task"
    with PlaywrightEnvironment.launch(headless=True, screenshot_dir=screenshots) as env:
        observation = env.open(TASK.start_url)
        print(f"opened {observation.url!r} (title={observation.title!r})")

        for action in SCRIPT:
            result = env.execute(action)
            status = "ok" if result.success else f"FAILED [{result.failure_category}]"
            print(f"  {action.type:<10} -> {status}")
            if action.is_terminal or not result.success:
                break


if __name__ == "__main__":
    main()
