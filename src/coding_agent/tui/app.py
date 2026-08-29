"""Textual application for interactive coding-agent sessions."""

from __future__ import annotations

import json
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, TypeAlias

from rich.markdown import Markdown
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, RichLog, Static, TextArea
from textual.worker import Worker

from ..agent import Agent, AgentResult
from ..events import EventSink
from ..permissions import (
    AllowAllApprovalPolicy,
    ApprovalPolicy,
    DenyApprovalPolicy,
    PermissionRequest,
)
from .bridge import TuiApprovalPolicy, TuiEventSink
from .screens import ApprovalScreen


AgentFactory: TypeAlias = Callable[[EventSink, ApprovalPolicy], Agent]


class CodingAgentApp(App[None]):
    """A deliberately small TUI shell around the existing agent core."""

    TITLE = "coding-agent"
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("ctrl+enter", "submit_prompt", "Send", priority=True),
        Binding("ctrl+s", "submit_prompt", "Send", priority=True),
        Binding("ctrl+l", "clear_conversation", "Clear", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    CSS = """
    Screen {
        layout: vertical;
        background: $background;
    }

    Header {
        height: 1;
    }

    #context-bar {
        height: 3;
        padding: 0 2;
        color: $text-muted;
        border-bottom: solid $primary-background;
        content-align: left middle;
    }

    #conversation {
        height: 1fr;
        padding: 1 2;
        scrollbar-size: 1 1;
    }

    #activity {
        height: 3;
        padding: 0 2;
        color: $text-muted;
        border-top: solid $primary-background;
        content-align: left middle;
    }

    #prompt {
        height: 6;
        margin: 0 1 1 1;
        border: round $primary;
        background: $surface;
    }

    #prompt:focus {
        border: round $accent;
    }

    Footer {
        height: 1;
    }
    """

    def __init__(
        self,
        *,
        agent_factory: AgentFactory,
        workspace: str | Path,
        model_name: str,
        approval_mode: str = "ask",
        session_id: str | None = None,
        initial_prompt: str | None = None,
    ) -> None:
        super().__init__()
        if approval_mode not in {"ask", "deny", "allow"}:
            raise ValueError("approval_mode must be ask, deny, or allow")
        self.agent_factory = agent_factory
        self.workspace = Path(workspace).resolve()
        self.model_name = model_name
        self.approval_mode = approval_mode
        self.session_id = session_id
        self.initial_prompt = initial_prompt
        self.agent: Agent | None = None
        self.agent_worker: Worker[AgentResult] | None = None
        self.tui_approval: TuiApprovalPolicy | None = None
        self.busy = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="context-bar")
        yield RichLog(
            id="conversation",
            wrap=True,
            highlight=False,
            markup=False,
            auto_scroll=True,
        )
        yield Static("Idle", id="activity")
        yield TextArea(
            "",
            id="prompt",
            show_line_numbers=False,
            soft_wrap=True,
        )
        yield Footer()

    def on_mount(self) -> None:
        event_sink = TuiEventSink(self)
        approval_policy: ApprovalPolicy
        if self.approval_mode == "ask":
            self.tui_approval = TuiApprovalPolicy(self)
            approval_policy = self.tui_approval
        elif self.approval_mode == "deny":
            approval_policy = DenyApprovalPolicy()
        else:
            approval_policy = AllowAllApprovalPolicy()
        self.agent = self.agent_factory(event_sink, approval_policy)
        self._refresh_context()
        self._conversation().write(
            Text.from_markup(
                "[bold cyan]coding-agent[/] is ready. "
                "Type a task, or [bold]/help[/] for local commands."
            )
        )
        self._prompt().focus()
        if self.initial_prompt:
            self.call_after_refresh(self.submit_prompt, self.initial_prompt)

    def on_unmount(self) -> None:
        if self.tui_approval is not None:
            self.tui_approval.deny_pending()

    def action_submit_prompt(self) -> None:
        self.submit_prompt(self._prompt().text)

    def submit_prompt(self, raw_prompt: str) -> None:
        prompt = raw_prompt.strip()
        if not prompt:
            return
        if self.busy:
            self.notify("The agent is already running a task.", severity="warning")
            return
        if prompt.startswith("/") and self._handle_local_command(prompt):
            self._prompt().text = ""
            return
        if self.agent is None:
            self.notify("The agent is not ready.", severity="error")
            return

        self._prompt().text = ""
        self._set_busy(True, "Thinking…")
        self.agent_worker = self.run_agent(prompt)

    @work(thread=True, exclusive=True, group="agent")
    def run_agent(self, prompt: str) -> AgentResult:
        assert self.agent is not None
        try:
            result = self.agent.run(prompt, session_id=self.session_id)
        except Exception as exc:
            self.call_from_thread(self._turn_failed, exc)
            raise
        self.call_from_thread(self._turn_finished, result)
        return result

    def post_agent_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Render one core event; future event types can be added without rewiring."""

        event_session = payload.get("session_id")
        if isinstance(event_session, str):
            self.session_id = event_session
            self._refresh_context()

        if event_type == "user_message":
            self._write_message("You", str(payload.get("content", "")), "cyan")
            return

        if event_type == "model_response":
            assistant_message = payload.get("assistant_message") or {}
            reasoning = assistant_message.get("reasoning_content")
            if reasoning:
                excerpt = _one_line(str(reasoning), limit=240)
                self._conversation().write(
                    Text.assemble(("  reasoning  ", "dim magenta"), (excerpt, "dim"))
                )
            for call in payload.get("tool_calls", []):
                name = str(call.get("name", "tool"))
                arguments = _compact_json(call.get("arguments", {}), limit=260)
                self._conversation().write(
                    Text.assemble(
                        ("○ ", "yellow"),
                        (name, "bold yellow"),
                        (f"  {arguments}", "dim"),
                    )
                )
                self._set_activity(f"Running tool · {name}")
            text = payload.get("text")
            if text:
                self._conversation().write(Text("Agent", style="bold green"))
                self._conversation().write(Markdown(str(text)))
            return

        if event_type == "tool_result":
            ok = bool(payload.get("ok"))
            marker = "✓" if ok else "✗"
            style = "green" if ok else "red"
            name = str(payload.get("name", "tool"))
            detail = (
                _summarize_output(payload.get("output"))
                if ok
                else _one_line(str(payload.get("error", "failed")), limit=300)
            )
            self._conversation().write(
                Text.assemble(
                    (f"{marker} ", style),
                    (name, f"bold {style}"),
                    (f"  {detail}" if detail else "", "dim"),
                )
            )
            self._set_activity("Thinking…")
            return

        if event_type == "model_request_error":
            self._conversation().write(
                Text(
                    f"Model request failed; retry {payload.get('attempt')}: "
                    f"{payload.get('error')}",
                    style="red",
                )
            )
            return

        if event_type == "tool_calls_recovered":
            self._conversation().write(
                Text(
                    f"Recovered {payload.get('count', 0)} interrupted tool call(s).",
                    style="yellow",
                )
            )
            return

        if event_type == "agent_terminated":
            self._set_activity(
                f"Finishing · {payload.get('reason', 'completed')}"
            )

    def show_approval(
        self,
        request: PermissionRequest,
        future: Future[bool],
    ) -> None:
        self._set_activity(f"Waiting for approval · {request.tool_name}")

        def resolved(approved: bool | None) -> None:
            decision = bool(approved)
            if not future.done():
                future.set_result(decision)
            self._conversation().write(
                Text(
                    f"{'✓ Allowed' if decision else '✗ Denied'} · "
                    f"{request.tool_name}",
                    style="green" if decision else "red",
                )
            )
            self._set_activity("Thinking…")

        self.push_screen(ApprovalScreen(request), resolved)

    def action_clear_conversation(self) -> None:
        self._conversation().clear()
        self._conversation().write(Text("Conversation view cleared.", style="dim"))

    def _handle_local_command(self, prompt: str) -> bool:
        command = prompt.casefold()
        if command in {"/quit", "/exit"}:
            self.exit()
            return True
        if command == "/clear":
            self.action_clear_conversation()
            return True
        if command == "/new":
            self.session_id = None
            self.action_clear_conversation()
            self._refresh_context()
            self._conversation().write(Text("Started a new session.", style="green"))
            return True
        if command == "/help":
            self._conversation().write(
                Text.from_markup(
                    "[bold]/new[/] new session  ·  [bold]/clear[/] clear view  ·  "
                    "[bold]/exit[/] quit\n"
                    "[dim]Ctrl+Enter or Ctrl+S sends the prompt.[/]"
                )
            )
            return True
        self.notify(f"Unknown local command: {prompt}", severity="warning")
        return True

    def _turn_finished(self, result: AgentResult) -> None:
        self.session_id = result.session_id or self.session_id
        self._set_busy(False, f"Ready · {result.steps} model step(s)")
        self._refresh_context()
        self._prompt().focus()

    def _turn_failed(self, exc: Exception) -> None:
        self._conversation().write(
            Text(f"Agent failed: {type(exc).__name__}: {exc}", style="bold red")
        )
        self._set_busy(False, "Failed · ready for another task")
        self._prompt().focus()

    def _set_busy(self, busy: bool, activity: str) -> None:
        self.busy = busy
        if self.is_mounted:
            self._prompt().disabled = busy
            self._set_activity(activity)

    def _set_activity(self, activity: str) -> None:
        self.query_one("#activity", Static).update(activity)

    def _refresh_context(self) -> None:
        session = self.session_id[:8] if self.session_id else "new"
        self.query_one("#context-bar", Static).update(
            f"Workspace  {self.workspace}\n"
            f"Model  {self.model_name}   ·   Approval  {self.approval_mode}   ·   "
            f"Session  {session}"
        )

    def _write_message(self, role: str, content: str, color: str) -> None:
        self._conversation().write(Text(role, style=f"bold {color}"))
        self._conversation().write(Text(content))

    def _conversation(self) -> RichLog:
        return self.query_one("#conversation", RichLog)

    def _prompt(self) -> TextArea:
        return self.query_one("#prompt", TextArea)


def _compact_json(value: Any, *, limit: int) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        rendered = repr(value)
    return _one_line(rendered, limit=limit)


def _one_line(value: str, *, limit: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)] + "…"


def _summarize_output(output: Any) -> str:
    if not isinstance(output, dict):
        return _one_line(str(output or ""), limit=300)
    preferred = (
        "path",
        "count",
        "match_count",
        "exit_code",
        "duration_ms",
        "created",
        "overwritten",
        "deleted",
        "replacements",
    )
    summary = {key: output[key] for key in preferred if key in output}
    if not summary:
        summary = {key: output[key] for key in tuple(output)[:3]}
    return _compact_json(summary, limit=300)
