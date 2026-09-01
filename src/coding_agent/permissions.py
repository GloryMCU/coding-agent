"""Permission classification, rules, and user approval for local tools."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatchcase
from typing import Callable, Iterable, Protocol, TextIO


class PermissionKind(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"


class PermissionRuleDecision(str, Enum):
    """A rule outcome, ordered separately by :class:`PermissionRuleEngine`."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class ApprovalDecision(str, Enum):
    """A human decision for one request or its bounded task scope."""

    DENY = "deny"
    ALLOW_ONCE = "allow_once"
    ALLOW_TASK = "allow_task"


@dataclass(frozen=True, slots=True)
class PermissionRequest:
    """A fully rendered operation presented to an approval policy."""

    tool_name: str
    kind: PermissionKind
    description: str
    resource: str | None = None
    command: tuple[str, ...] = ()
    sandboxed: bool | None = None
    task_scope: str | None = None

    @property
    def effective_task_scope(self) -> str:
        """Return the narrow reusable scope for a task-level approval."""

        return self.task_scope or f"{self.kind.value}:{self.tool_name}"


ApprovalResult = bool | ApprovalDecision


class ApprovalPolicy(Protocol):
    def approve(self, request: PermissionRequest) -> ApprovalResult:
        ...


@dataclass(frozen=True, slots=True)
class PermissionRule:
    """Match a permission request and contribute an allow/ask/deny decision."""

    decision: PermissionRuleDecision
    kinds: frozenset[PermissionKind] | None = None
    tool_pattern: str = "*"
    resource_pattern: str | None = None
    command_prefix: tuple[str, ...] = ()
    sandboxed: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, PermissionRuleDecision):
            raise TypeError("decision must be a PermissionRuleDecision")
        if not self.tool_pattern:
            raise ValueError("tool_pattern must not be empty")
        if self.resource_pattern == "":
            raise ValueError("resource_pattern must not be empty")
        if any(not token for token in self.command_prefix):
            raise ValueError("command_prefix tokens must not be empty")

    def matches(self, request: PermissionRequest) -> bool:
        if self.kinds is not None and request.kind not in self.kinds:
            return False
        if not fnmatchcase(request.tool_name, self.tool_pattern):
            return False
        if self.resource_pattern is not None:
            resource = (
                request.resource.replace("\\", "/")
                if request.resource is not None
                else None
            )
            pattern = self.resource_pattern.replace("\\", "/")
            if resource is None or not fnmatchcase(resource, pattern):
                return False
        if self.command_prefix:
            prefix_length = len(self.command_prefix)
            if request.command[:prefix_length] != self.command_prefix:
                return False
        if self.sandboxed is not None and request.sandboxed is not self.sandboxed:
            return False
        return True


class PermissionRuleEngine:
    """Evaluate matching rules with DENY > ASK > ALLOW precedence."""

    _PRECEDENCE = {
        PermissionRuleDecision.ALLOW: 1,
        PermissionRuleDecision.ASK: 2,
        PermissionRuleDecision.DENY: 3,
    }

    def __init__(
        self,
        rules: Iterable[PermissionRule] = (),
        *,
        default: PermissionRuleDecision = PermissionRuleDecision.ASK,
    ) -> None:
        self.rules = tuple(rules)
        self.default = default

    def evaluate(self, request: PermissionRequest) -> PermissionRuleDecision:
        matches = [rule.decision for rule in self.rules if rule.matches(request)]
        if not matches:
            return self.default
        return max(matches, key=self._PRECEDENCE.__getitem__)


class RuleBasedApprovalPolicy:
    """Apply permission rules and cache explicit approvals for one agent task."""

    def __init__(
        self,
        engine: PermissionRuleEngine,
        *,
        reviewer: ApprovalPolicy | None = None,
    ) -> None:
        self.engine = engine
        self.reviewer = reviewer
        self._task_scopes: set[str] = set()

    def begin_task(self) -> None:
        """Clear approvals that were granted only for the previous task."""

        self._task_scopes.clear()

    def approve(self, request: PermissionRequest) -> bool:
        rule_decision = self.engine.evaluate(request)
        if rule_decision is PermissionRuleDecision.DENY:
            return False
        if request.effective_task_scope in self._task_scopes:
            return True
        if rule_decision is PermissionRuleDecision.ALLOW:
            return True
        if self.reviewer is None:
            return False

        decision = normalize_approval_decision(self.reviewer.approve(request))
        if decision is ApprovalDecision.ALLOW_TASK:
            self._task_scopes.add(request.effective_task_scope)
        return decision is not ApprovalDecision.DENY


class AllowAllApprovalPolicy:
    """Approve all requests. Intended for trusted embedding and tests."""

    def approve(self, request: PermissionRequest) -> bool:
        return True


class DenyApprovalPolicy:
    """Deny every operation that requires approval."""

    def approve(self, request: PermissionRequest) -> bool:
        return False


class InteractiveApprovalPolicy:
    """Prompt on a terminal and support once or task-scoped approval."""

    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] = input,
        output: TextIO | None = None,
    ) -> None:
        self._input = input_fn
        self._output = output or sys.stderr

    def approve(self, request: PermissionRequest) -> ApprovalDecision:
        print(
            f"Approval required [{request.kind.value}] {request.tool_name}: "
            f"{request.description}",
            file=self._output,
        )
        try:
            answer = self._input("Allow [y] once, allow for [t]ask, or [N] deny? ")
        except (EOFError, KeyboardInterrupt):
            return ApprovalDecision.DENY
        normalized = answer.strip().casefold()
        if normalized in {"y", "yes"}:
            return ApprovalDecision.ALLOW_ONCE
        if normalized in {"t", "task"}:
            return ApprovalDecision.ALLOW_TASK
        return ApprovalDecision.DENY


def normalize_approval_decision(result: ApprovalResult) -> ApprovalDecision:
    """Adapt legacy boolean policies to the scoped decision model."""

    if isinstance(result, ApprovalDecision):
        return result
    if isinstance(result, bool):
        return ApprovalDecision.ALLOW_ONCE if result else ApprovalDecision.DENY
    raise TypeError("approval policies must return bool or ApprovalDecision")


def approval_result_is_allowed(result: ApprovalResult) -> bool:
    return normalize_approval_decision(result) is not ApprovalDecision.DENY


def create_approval_policy(
    mode: str,
    *,
    reviewer: ApprovalPolicy | None = None,
    rules: Iterable[PermissionRule] = (),
) -> RuleBasedApprovalPolicy:
    """Build a compatible top-level policy for a CLI or TUI approval mode."""

    if mode not in {"workspace", "ask", "deny", "allow"}:
        raise ValueError("approval mode must be workspace, ask, deny, or allow")

    base_rules: list[PermissionRule] = []
    default = PermissionRuleDecision.ASK
    if mode == "workspace":
        base_rules.extend(
            [
                PermissionRule(
                    decision=PermissionRuleDecision.ALLOW,
                    kinds=frozenset({PermissionKind.WRITE}),
                ),
                PermissionRule(
                    decision=PermissionRuleDecision.ALLOW,
                    kinds=frozenset({PermissionKind.EXECUTE}),
                    sandboxed=True,
                ),
            ]
        )
    elif mode == "deny":
        base_rules.append(PermissionRule(PermissionRuleDecision.DENY))
    elif mode == "allow":
        default = PermissionRuleDecision.ALLOW

    return RuleBasedApprovalPolicy(
        PermissionRuleEngine((*base_rules, *rules), default=default),
        reviewer=reviewer,
    )
