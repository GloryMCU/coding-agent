"""Textual application for interactive coding-agent sessions."""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, TypeAlias

from rich.markdown import Markdown
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Header, RichLog, Static, TextArea
from textual.worker import Worker

from ..agent import Agent, AgentResult
from ..errors import CodingAgentError
from ..events import EventSink
from ..permissions import (
    ApprovalDecision,
    ApprovalPolicy,
    PermissionRequest,
    create_approval_policy,
)
from .bridge import TuiApprovalPolicy, TuiEventSink
from .screens import ApprovalScreen


AgentFactory: TypeAlias = Callable[[EventSink, ApprovalPolicy], Agent]


class PromptTextArea(TextArea):
    """Composer that sends on Enter while retaining Shift+Enter for new lines."""

    BINDINGS = [
        Binding("enter", "submit", "Send", show=False, priority=True),
        Binding("shift+enter", "insert_newline", "New line", show=False),
    ]

    def action_submit(self) -> None:
        if not self.disabled:
            self.app.action_submit_prompt()

    def action_insert_newline(self) -> None:
        start, end = self.selection
        self.replace("\n", start, end, maintain_selection_offset=False)


class CodingAgentApp(App[None]):
    """A deliberately small TUI shell around the existing agent core."""

    TITLE = "coding-agent"
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("ctrl+enter", "submit_prompt", "Send", priority=True),
        Binding("ctrl+y", "copy_last_response", "Copy answer", priority=True),
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
        background: $panel;
    }

    #context-bar {
        height: 2;
        padding: 0 3;
        color: $text-muted;
        background: $panel;
        border-bottom: solid $primary-background-lighten-1;
        content-align: left middle;
    }

    #conversation {
        height: 1fr;
        padding: 1 3 2 3;
        scrollbar-size: 1 1;
        background: $background;
    }

    #composer {
        height: auto;
        min-height: 7;
        padding: 0 2 1 2;
        background: $panel;
        border-top: solid $primary-background-lighten-1;
    }

    #activity {
        height: 2;
        padding: 0 1;
        color: $text-muted;
        content-align: left middle;
    }

    #prompt {
        height: 4;
        margin: 0;
        padding: 0 1;
        border: round $primary;
        background: $surface;
    }

    #prompt:focus {
        border: round $accent;
    }

    Footer {
        height: 1;
        background: $panel;
    }
    """

    def __init__(
        self,
        *,
        agent_factory: AgentFactory,
        workspace: str | Path,
        model_name: str,
        approval_mode: str = "workspace",
        session_id: str | None = None,
        initial_prompt: str | None = None,
    ) -> None:
        super().__init__()
        if approval_mode not in {"workspace", "ask", "deny", "allow"}:
            raise ValueError(
                "approval_mode must be workspace, ask, deny, or allow"
            )
        self.agent_factory = agent_factory
        self.workspace = Path(workspace).resolve()
        self.model_name = model_name
        self.approval_mode = approval_mode
        self.session_id = session_id
        self.initial_prompt = initial_prompt
        self.agent: Agent | None = None
        self.agent_worker: Worker[AgentResult | None] | None = None
        self.tui_approval: TuiApprovalPolicy | None = None
        self.busy = False
        self.completed_tools = 0
        self.last_assistant_text: str | None = None

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
        with Vertical(id="composer"):
            yield Static("Ready  ·  Enter send  ·  Shift+Enter new line", id="activity")
            yield PromptTextArea(
                "",
                id="prompt",
                show_line_numbers=False,
                soft_wrap=True,
                highlight_cursor_line=False,
                placeholder="Ask coding-agent to inspect, change, or explain…",
            )
        yield Footer()

    def on_mount(self) -> None:
        event_sink = TuiEventSink(self)
        reviewer: ApprovalPolicy | None = None
        if self.approval_mode in {"workspace", "ask"}:
            self.tui_approval = TuiApprovalPolicy(self)
            reviewer = self.tui_approval
        approval_policy = create_approval_policy(
            self.approval_mode,
            reviewer=reviewer,
        )
        self.agent = self.agent_factory(event_sink, approval_policy)
        self._refresh_context()
        self._conversation().write(
            Text.from_markup(
                "[bold cyan]coding-agent[/] is ready  ·  "
                "[dim]Type [bold]/help[/] for local commands.[/]"
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
        self.completed_tools = 0
        self._set_busy(True, "Thinking…")
        self.agent_worker = self.run_agent(prompt)

    @work(thread=True, exclusive=True, group="agent")
    def run_agent(self, prompt: str) -> AgentResult | None:
        assert self.agent is not None
        try:
            result = self.agent.run(prompt, session_id=self.session_id)
        except CodingAgentError as exc:
            self.call_from_thread(self._turn_failed, exc)
            return None
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
            if payload.get("provisional"):
                self._set_activity("Verifying project…")
                return
            tool_calls = payload.get("tool_calls", [])
            if tool_calls:
                names = [str(call.get("name", "tool")) for call in tool_calls]
                self._set_activity(_tool_activity(names, self.completed_tools))
                return
            text = payload.get("text")
            if text:
                self.last_assistant_text = str(text)
                self._conversation().write(
                    Text.assemble(("◆ ", "bold green"), ("Agent", "bold green"))
                )
                self._conversation().write(Markdown(self.last_assistant_text))
                self._set_activity("Finishing…")
            return

        if event_type == "tool_result":
            ok = bool(payload.get("ok"))
            name = str(payload.get("name", "tool"))
            self.completed_tools += 1
            if ok:
                self._set_activity(
                    f"Working…  ·  {self.completed_tools} tool"
                    f"{'s' if self.completed_tools != 1 else ''} completed"
                )
            else:
                self._set_activity(f"Tool failed · {_friendly_tool_name(name)}")
            return

        if event_type == "model_request_error":
            attempt = payload.get("attempt", "?")
            if payload.get("will_retry"):
                self._set_activity(
                    f"Model request failed · retrying after attempt {attempt}…"
                )
            else:
                self._set_activity(
                    f"Model request failed · stopped after attempt {attempt}"
                )
            return

        if event_type == "tool_calls_recovered":
            self._set_activity(
                f"Recovered {payload.get('count', 0)} interrupted tool call(s)"
            )
            return

        if event_type == "agent_terminated":
            self._set_activity(
                f"Finishing · {payload.get('reason', 'completed')}"
            )

    def show_approval(
        self,
        request: PermissionRequest,
        future: Future[ApprovalDecision],
    ) -> None:
        self._set_activity(f"Waiting for approval · {request.tool_name}")

        def resolved(decision: ApprovalDecision | None) -> None:
            decision = decision or ApprovalDecision.DENY
            if not future.done():
                future.set_result(decision)
            self._set_activity(
                f"{'Denied' if decision is ApprovalDecision.DENY else 'Allowed'} · "
                f"{_friendly_tool_name(request.tool_name)}"
            )

        self.push_screen(ApprovalScreen(request), resolved)

    def action_clear_conversation(self) -> None:
        self._conversation().clear()
        self._conversation().write(Text("Conversation view cleared.", style="dim"))

    def action_copy_last_response(self) -> None:
        if self.last_assistant_text is None:
            self.notify("No agent response is available to copy.", severity="warning")
            return
        self.copy_to_clipboard(self.last_assistant_text)
        self.notify("Copied the latest agent response.")

    def _handle_local_command(self, prompt: str) -> bool:
        command = prompt.casefold()
        if command in {"/quit", "/exit"}:
            self.exit()
            return True
        if command == "/clear":
            self.action_clear_conversation()
            return True
        if command == "/copy":
            self.action_copy_last_response()
            return True
        if command == "/new":
            self.session_id = None
            self.last_assistant_text = None
            self.action_clear_conversation()
            self._refresh_context()
            self._conversation().write(Text("Started a new session.", style="green"))
            return True
        if command == "/help":
            self._conversation().write(
                Text.from_markup(
                    "[bold]/new[/] new session  ·  [bold]/clear[/] clear view  ·  "
                    "[bold]/copy[/] copy latest answer  ·  [bold]/exit[/] quit\n"
                    "[dim]Enter sends · Shift+Enter adds a new line · "
                    "Ctrl+Y copies the latest answer · select text then Ctrl+C.[/]"
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
        if isinstance(exc, CodingAgentError):
            message = f"Agent stopped: {exc}"
        else:
            message = f"Unexpected agent error ({type(exc).__name__}): {exc}"
        self._conversation().write(
            Text(message, style="bold red")
        )
        self._set_busy(False, "Failed · ready for another task")
        self._prompt().focus()

    def _set_busy(self, busy: bool, activity: str) -> None:
        self.busy = busy
        if self.is_mounted:
            self._prompt().disabled = busy
            if busy:
                self._set_activity(activity)
            else:
                self._set_activity(
                    f"{activity}  ·  Enter send  ·  Shift+Enter new line"
                )

    def _set_activity(self, activity: str) -> None:
        self.query_one("#activity", Static).update(activity)

    def _refresh_context(self) -> None:
        session = self.session_id[:8] if self.session_id else "new"
        self.query_one("#context-bar", Static).update(
            f"{self.workspace}   ·   {self.model_name}   ·   "
            f"approval {self.approval_mode}   ·   session {session}"
        )

    def _write_message(self, role: str, content: str, color: str) -> None:
        marker = "›" if role == "You" else "◆"
        self._conversation().write(
            Text.assemble((f"{marker} ", f"bold {color}"), (role, f"bold {color}"))
        )
        self._conversation().write(Text(content))

    def _conversation(self) -> RichLog:
        return self.query_one("#conversation", RichLog)

    def _prompt(self) -> PromptTextArea:
        return self.query_one("#prompt", PromptTextArea)


def _friendly_tool_name(name: str) -> str:
    labels = {
        "read_file": "Reading file",
        "list_files": "Listing files",
        "glob_files": "Finding files",
        "search_text": "Searching code",
        "web_search": "Searching the web",
        "fetch_webpage": "Reading web page",
        "git_status": "Checking Git status",
        "git_diff": "Reviewing changes",
        "git_log": "Reading Git history",
        "write_file": "Writing file",
        "apply_patch": "Applying patch",
        "delete_file": "Deleting file",
        "run_command": "Running command",
        "verify_project": "Verifying project",
    }
    return labels.get(name, name.replace("_", " ").strip().capitalize() or "Using tool")


def _tool_activity(names: list[str], completed: int) -> str:
    if len(names) == 1:
        current = _friendly_tool_name(names[0])
    else:
        current = f"Using {len(names)} tools"
    prefix = f"{completed} completed  ·  " if completed else ""
    return f"Working…  ·  {prefix}{current}"
