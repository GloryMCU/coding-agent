"""Permission classification and user approval for local tool execution."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, TextIO


class PermissionKind(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"


@dataclass(frozen=True, slots=True)
class PermissionRequest:
    """A fully rendered operation presented to an approval policy."""

    tool_name: str
    kind: PermissionKind
    description: str


class ApprovalPolicy(Protocol):
    def approve(self, request: PermissionRequest) -> bool:
        ...


class AllowAllApprovalPolicy:
    """Approve all requests. Intended for trusted embedding and tests."""

    def approve(self, request: PermissionRequest) -> bool:
        return True


class DenyApprovalPolicy:
    """Deny every operation that requires approval."""

    def approve(self, request: PermissionRequest) -> bool:
        return False


class InteractiveApprovalPolicy:
    """Prompt on a terminal for each state-changing or executable action."""

    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] = input,
        output: TextIO | None = None,
    ) -> None:
        self._input = input_fn
        self._output = output or sys.stderr

    def approve(self, request: PermissionRequest) -> bool:
        print(
            f"Approval required [{request.kind.value}] {request.tool_name}: "
            f"{request.description}",
            file=self._output,
        )
        try:
            answer = self._input("Allow once? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().casefold() in {"y", "yes"}
