"""Human-in-the-loop approval for risky actions.

Some UI actions are irreversible — deleting a record, submitting a payment,
logging out. ``ApprovalAgent`` wraps any inner agent and gates such actions
through an *approver* callback before they ever reach the environment. A denied
action is replaced with a terminal ``fail`` so the run stops safely rather than
performing the risky operation.

Risk classification lives in :class:`RiskPolicy` (keyword-based by default) and
is fully overridable. Approver callbacks are plain ``Callable[[AgentAction],
bool]`` so the gate can be wired to a console prompt, an allow-list, or — in
tests — a deterministic stub. The default approver is :func:`deny_all`, i.e.
risky actions are blocked unless something explicitly approves them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from ..domain import ActionType, AgentAction, Observation, TaskSpec, TraceStep
from .base import Agent

#: Substrings that, when found in an action's target/text/reason, mark it risky.
DEFAULT_RISKY_KEYWORDS: frozenset[str] = frozenset(
    {
        "delete",
        "remove",
        "destroy",
        "drop",
        "purge",
        "submit",
        "pay",
        "buy",
        "purchase",
        "checkout",
        "order",
        "confirm",
        "logout",
        "sign out",
        "deactivate",
        "unsubscribe",
        "close account",
    }
)

#: Approver callback signature: return ``True`` to allow the action.
Approver = Callable[[AgentAction], bool]


def allow_all(action: AgentAction) -> bool:
    """Approver that permits every action (use only in trusted automation)."""
    return True


def deny_all(action: AgentAction) -> bool:
    """Approver that blocks every risky action — the safe default."""
    return False


class RiskPolicy:
    """Decide whether an action is risky.

    The default policy flags interactive actions (``click``/``press_key``) whose
    selector, text, key, or reason contains a risky keyword. ``type_text`` and
    ``wait`` are treated as safe.

    Parameters
    ----------
    keywords:
        Case-insensitive substrings that mark an action risky.
    risky_types:
        Action types eligible for risk classification.
    """

    def __init__(
        self,
        keywords: Iterable[str] = DEFAULT_RISKY_KEYWORDS,
        risky_types: Iterable[ActionType] = (ActionType.CLICK, ActionType.PRESS_KEY),
    ) -> None:
        self._keywords = frozenset(k.lower() for k in keywords)
        self._risky_types = frozenset(risky_types)

    def is_risky(self, action: AgentAction) -> bool:
        """Return ``True`` when ``action`` matches the risk policy."""
        if action.type not in self._risky_types:
            return False
        haystack = " ".join(
            part for part in (action.selector, action.text, action.key, action.reason) if part
        ).lower()
        return any(keyword in haystack for keyword in self._keywords)


class ApprovalAgent:
    """Gate risky actions from an inner agent behind an approver.

    Parameters
    ----------
    inner:
        The supervised agent.
    approver:
        Callback consulted for risky actions. Defaults to :func:`deny_all`.
    policy:
        Risk classifier. A default :class:`RiskPolicy` is used when omitted.
    """

    def __init__(
        self,
        inner: Agent,
        *,
        approver: Approver = deny_all,
        policy: RiskPolicy | None = None,
    ) -> None:
        self._inner = inner
        self._approver = approver
        self._policy = policy or RiskPolicy()

    def next_action(
        self,
        task: TaskSpec,
        observation: Observation,
        trace: list[TraceStep],
    ) -> AgentAction:
        """Return the inner action, blocking risky ones that are not approved.

        See :class:`~agentic_qa_lab.agents.base.Agent` for the contract.
        """
        proposed = self._inner.next_action(task, observation, trace)
        if proposed.is_terminal or not self._policy.is_risky(proposed):
            return proposed
        if self._approver(proposed):
            return proposed
        target = proposed.selector or (proposed.x, proposed.y)
        return AgentAction.fail(
            f"Risky action '{proposed.type.value}' on {target} was not approved."
        )
