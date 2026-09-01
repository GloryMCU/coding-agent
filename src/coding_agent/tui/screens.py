"""Modal screens used by the terminal interface."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from ..permissions import ApprovalDecision, PermissionRequest


class ApprovalScreen(ModalScreen[ApprovalDecision]):
    """Ask the user to deny, allow once, or allow a bounded task scope."""

    BINDINGS = [
        Binding("y", "allow", "Allow", show=False),
        Binding("t", "allow_task", "Allow for task", show=False),
        Binding("n", "deny", "Deny", show=False),
        Binding("escape", "deny", "Deny", show=False),
    ]

    DEFAULT_CSS = """
    ApprovalScreen {
        align: center middle;
        background: $background 60%;
    }

    ApprovalScreen > Vertical {
        width: 92%;
        max-width: 76;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        border: round $warning;
        background: $surface;
    }

    ApprovalScreen .approval-title {
        color: $warning;
        text-style: bold;
        margin-bottom: 1;
    }

    ApprovalScreen .approval-operation {
        margin: 1 0;
        padding: 1;
        background: $boost;
        overflow-y: auto;
        max-height: 12;
    }

    ApprovalScreen Horizontal {
        width: 100%;
        height: auto;
        align-horizontal: right;
        margin-top: 1;
    }

    ApprovalScreen Button {
        margin-left: 1;
    }
    """

    def __init__(self, request: PermissionRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Approval required", classes="approval-title")
            yield Label(
                f"{self.request.kind.value.upper()} · {self.request.tool_name}"
            )
            yield Static(
                Text(self.request.description),
                classes="approval-operation",
            )
            yield Label("Y allow once · T allow for task · N/Esc deny")
            with Horizontal():
                yield Button("Deny", id="deny", variant="error")
                yield Button("Allow once", id="allow", variant="success")
                yield Button("Allow for task", id="allow-task", variant="primary")

    def action_allow(self) -> None:
        self.dismiss(ApprovalDecision.ALLOW_ONCE)

    def action_allow_task(self) -> None:
        self.dismiss(ApprovalDecision.ALLOW_TASK)

    def action_deny(self) -> None:
        self.dismiss(ApprovalDecision.DENY)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        decisions = {
            "allow": ApprovalDecision.ALLOW_ONCE,
            "allow-task": ApprovalDecision.ALLOW_TASK,
        }
        self.dismiss(decisions.get(event.button.id, ApprovalDecision.DENY))
