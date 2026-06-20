"""The environment/agent execution loop.

``Runner`` drives a single :class:`TaskSpec` to completion against a
:class:`BrowserEnvironment` using an :class:`Agent`, applying the task's
``max_steps``, ``max_retries``, and ``timeout_seconds`` safeguards, and
returning an aggregated :class:`RunResult`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from ..domain import (
    ActionResult,
    ActionType,
    AgentAction,
    FailureCategory,
    Observation,
    RunResult,
    RunStatus,
    TaskSpec,
    TraceStep,
)
from ..environments import BrowserEnvironment
from .base import Agent
from .judge import JudgeVerdict, SuccessJudge


class Runner:
    """Execute the observe → decide → act loop for one task.

    Parameters
    ----------
    monotonic:
        Source of monotonic seconds used for the wall-clock timeout. Injectable
        for deterministic tests.
    wall_clock:
        Source of epoch seconds used for run timestamps.
    stop_on_action_failure:
        When ``True`` (default) a non-terminal action that fails after its
        retries ends the run as ``failure``. Set ``False`` to let the agent
        recover — the failure stays in the trace and the loop continues, so a
        self-repairing agent (see :class:`ReflectiveAgent`) can react to it.
    """

    def __init__(
        self,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        stop_on_action_failure: bool = True,
        success_judge: SuccessJudge | None = None,
    ) -> None:
        self._monotonic = monotonic
        self._wall_clock = wall_clock
        self._stop_on_action_failure = stop_on_action_failure
        self._success_judge = success_judge

    def run(self, task: TaskSpec, agent: Agent, env: BrowserEnvironment) -> RunResult:
        """Run ``agent`` against ``env`` for ``task`` and return the result."""
        started_at = self._wall_clock()
        start_mono = self._monotonic()
        steps: list[TraceStep] = []
        total_retries = 0

        status = RunStatus.MAX_STEPS
        failure_category = FailureCategory.MAX_STEPS_EXCEEDED

        observation = env.open(task.start_url)

        for index in range(task.max_steps):
            if self._monotonic() - start_mono > task.timeout_seconds:
                status, failure_category = RunStatus.TIMEOUT, FailureCategory.TIMEOUT
                break

            try:
                action = agent.next_action(task, observation, steps)
            except Exception as exc:  # noqa: BLE001 - converted to a terminal result
                steps.append(self._error_step(index, observation, str(exc)))
                status, failure_category = RunStatus.ERROR, FailureCategory.AGENT_ERROR
                break

            result, retries = self._execute_with_retries(env, action, task.max_retries)
            total_retries += retries
            action, terminal = self._resolve_terminal(task, action, result, observation, steps)
            steps.append(
                TraceStep(index=index, observation=observation, action=action, result=result)
            )
            if terminal is None and not result.success and self._stop_on_action_failure:
                terminal = (RunStatus.FAILURE, result.failure_category)
            if terminal is not None:
                status, failure_category = terminal
                break

            observation = env.observe()
        # end for

        ended_at = self._wall_clock()
        return RunResult(
            task_id=task.task_id,
            status=status,
            failure_category=failure_category,
            steps=steps,
            total_retries=total_retries,
            duration_seconds=max(0.0, self._monotonic() - start_mono),
            started_at=started_at,
            ended_at=ended_at,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _execute_with_retries(
        self,
        env: BrowserEnvironment,
        action: AgentAction,
        max_retries: int,
    ) -> tuple[ActionResult, int]:
        """Execute ``action``, retrying transient failures up to ``max_retries``."""
        result = env.execute(action)
        retries = 0
        while not result.success and retries < max_retries:
            retries += 1
            result = env.execute(action)
        # Surface the accumulated retry count on the returned result.
        return result.model_copy(update={"retries": retries}), retries

    def _resolve_terminal(
        self,
        task: TaskSpec,
        action: AgentAction,
        result: ActionResult,
        observation: Observation,
        steps: list[TraceStep],
    ) -> tuple[AgentAction, tuple[RunStatus, FailureCategory] | None]:
        """Map a step outcome to a terminal status, or ``None`` to continue.

        A failed non-terminal action is *not* resolved here; that policy lives
        in :meth:`run` so it can honour ``stop_on_action_failure``.
        """
        if action.type is ActionType.FAIL:
            return action, (RunStatus.FAILURE, FailureCategory.UNKNOWN)

        if action.type is ActionType.FINISH:
            if task.success_selector is None:
                if task.success_judge and self._success_judge is not None:
                    verdict = self._success_judge.evaluate(task, observation, steps)
                    action = self._apply_judge_reason(action, verdict)
                    if verdict.success:
                        return action, (RunStatus.SUCCESS, FailureCategory.NONE)
                    return action, (RunStatus.FAILURE, FailureCategory.JUDGE_REJECTED)
                return action, (RunStatus.SUCCESS, FailureCategory.NONE)
            if observation.contains_marker(task.success_selector):
                return action, (RunStatus.SUCCESS, FailureCategory.NONE)
            if task.success_judge and self._success_judge is not None:
                verdict = self._success_judge.evaluate(task, observation, steps)
                action = self._apply_judge_reason(action, verdict)
                if verdict.success:
                    return action, (RunStatus.SUCCESS, FailureCategory.NONE)
                return action, (RunStatus.FAILURE, FailureCategory.JUDGE_REJECTED)
            return action, (RunStatus.FAILURE, FailureCategory.UNKNOWN)

        return action, None

    @staticmethod
    def _apply_judge_reason(action: AgentAction, verdict: JudgeVerdict) -> AgentAction:
        """Attach the judge rationale to a terminal action's reason text."""
        base = action.reason or "finish"
        return action.model_copy(update={"reason": f"{base} | judge: {verdict.reason}"})

    def _error_step(self, index: int, observation: Observation, message: str) -> TraceStep:
        """Build a trace step recording an agent-side exception."""
        return TraceStep(
            index=index,
            observation=observation,
            action=AgentAction.fail(f"Agent error: {message}"),
            result=ActionResult.failed(message, category=FailureCategory.AGENT_ERROR),
        )


def write_trace_jsonl(run: RunResult, path: str | Path) -> Path:
    """Write a run's trace to a JSONL file.

    Each trace step is serialized as one JSON line, followed by a final summary
    line tagged ``{"record": "summary", ...}``. The parent directory is created
    if needed.

    Returns
    -------
    Path
        The path that was written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for step in run.steps:
            line = {"record": "step", **step.model_dump(mode="json")}
            handle.write(json.dumps(line) + "\n")
        summary = {
            "record": "summary",
            "task_id": run.task_id,
            "status": run.status.value,
            "failure_category": run.failure_category.value,
            "step_count": run.step_count,
            "total_retries": run.total_retries,
            "duration_seconds": run.duration_seconds,
        }
        handle.write(json.dumps(summary) + "\n")
    return target
