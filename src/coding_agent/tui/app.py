"""Textual application for interactive coding-agent sessions."""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from time import monotonic
from typing import Any, Callable, TypeAlias

from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import CodeBlock, Markdown
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
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


class CopyableCodeBlock(CodeBlock):
    """Markdown code block with a clickable copy control."""

    @classmethod
    def create(cls, markdown: Markdown, token: Any) -> CopyableCodeBlock:
        node_info = token.info or ""
        lexer_name = node_info.partition(" ")[0]
        code_id = str(token.meta.get("copy_code_id", ""))
        return cls(lexer_name or "text", markdown.code_theme, code_id)

    def __init__(self, lexer_name: str, theme: str, code_id: str) -> None:
        super().__init__(lexer_name, theme)
        self.code_id = code_id

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        code = str(self.text).rstrip()
        header = Table.grid(expand=True, padding=(0, 1))
        header.style = "on #202020"
        header.add_column(ratio=1)
        header.add_column(justify="right")
        copy_control = Text(
            "⧉ Copy",
            style=Style(
                color="bright_cyan",
                bold=True,
                meta={"@click": f"app.copy_code('{self.code_id}')"},
            ),
        )
        header.add_row(Text(self.lexer_name, style="dim"), copy_control)
        yield header
        yield Syntax(
            code,
            self.lexer_name,
            theme=self.theme,
            word_wrap=True,
            padding=1,
        )


class CopyableMarkdown(Markdown):
    """Rich Markdown renderer that registers each fenced code block."""

    elements = {
        **Markdown.elements,
        "fence": CopyableCodeBlock,
        "code_block": CopyableCodeBlock,
    }

    def __init__(
        self,
        markup: str,
        register_code: Callable[[str], str],
    ) -> None:
        super().__init__(markup)
        for token in self.parsed:
            if token.type in {"fence", "code_block"}:
                token.meta["copy_code_id"] = register_code(
                    token.content.rstrip("\n")
                )


class PromptTextArea(TextArea):
    """Multiline composer with an explicit Shift+Enter newline binding."""

    BINDINGS = [
        Binding(
            "shift+enter",
            "insert_newline",
            "New line",
            show=False,
            priority=True,
        ),
    ]

    def action_insert_newline(self) -> None:
        start, end = self.selection
        self.replace("\n", start, end, maintain_selection_offset=False)


class CodingAgentApp(App[None]):
    """A deliberately small TUI shell around the existing agent core."""

    TITLE = "coding-agent"
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding(
            "ctrl+enter",
            "submit_prompt",
            "Send",
            show=False,
            priority=True,
        ),
        Binding("ctrl+s", "submit_prompt", "Send", priority=True),
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
        approval_timeout_s: float = 300,
        session_id: str | None = None,
        initial_prompt: str | None = None,
    ) -> None:
        super().__init__()
        if approval_mode not in {"workspace", "ask", "deny", "allow"}:
            raise ValueError(
                "approval_mode must be workspace, ask, deny, or allow"
            )
        if approval_timeout_s <= 0:
            raise ValueError("approval_timeout_s must be positive")
        self.agent_factory = agent_factory
        self.workspace = Path(workspace).resolve()
        self.model_name = model_name
        self.approval_mode = approval_mode
        self.approval_timeout_s = approval_timeout_s
        self.session_id = session_id
        self.initial_prompt = initial_prompt
        self.agent: Agent | None = None
        self.agent_worker: Worker[AgentResult | None] | None = None
        self.tui_approval: TuiApprovalPolicy | None = None
        self.busy = False
        self.completed_tools = 0
        self.last_assistant_text: str | None = None
        self._code_blocks: dict[str, str] = {}
        self._next_code_block_id = 1
        self._activity_label = "Ready"
        self._activity_started_at: float | None = None

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
            yield Static(
                "Ready  ·  Enter new line  ·  Ctrl+S send",
                id="activity",
            )
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
            self.tui_approval = TuiApprovalPolicy(
                self,
                timeout_s=self.approval_timeout_s,
            )
            reviewer = self.tui_approval
        approval_policy = create_approval_policy(
            self.approval_mode,
            reviewer=reviewer,
        )
        self.agent = self.agent_factory(event_sink, approval_policy)
        self.set_interval(1.0, self._refresh_activity_elapsed)
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

        if event_type == "model_request_started":
            step = payload.get("step", "?")
            max_steps = payload.get("max_steps")
            max_total_steps = payload.get("max_total_steps")
            step_label = f"Model step {step}"
            if max_steps is not None:
                step_label += f"/{max_steps}"
            if (
                max_total_steps is not None
                and max_total_steps != max_steps
            ):
                step_label += f" (hard limit {max_total_steps})"
            phase = (
                "finalizing response"
                if payload.get("finalizing")
                else "waiting for model"
            )
            self._set_activity(
                f"{step_label} · {phase}",
                track_elapsed=True,
            )
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
                self._conversation().write(
                    CopyableMarkdown(
                        self.last_assistant_text,
                        self._register_code_block,
                    )
                )
                self._set_activity("Finishing…")
            return

        if event_type == "tool_result":
            ok = bool(payload.get("ok"))
            name = str(payload.get("name", "tool"))
            self.completed_tools += 1
            duration = _format_duration(payload.get("duration_ms"))
            if ok:
                self._set_activity(
                    f"Working…  ·  {self.completed_tools} tool"
                    f"{'s' if self.completed_tools != 1 else ''} completed"
                    f" · last took {duration}"
                )
            else:
                self._set_activity(
                    f"Tool failed · {_friendly_tool_name(name)} · after {duration}"
                )
            return

        if event_type == "tool_started":
            name = str(payload.get("name", "tool"))
            step = payload.get("step", "?")
            completed = (
                f" · {self.completed_tools} completed" if self.completed_tools else ""
            )
            self._set_activity(
                f"Model step {step} · {_friendly_tool_name(name)}{completed}",
                track_elapsed=True,
            )
            return

        if event_type == "model_request_error":
            attempt = payload.get("attempt", "?")
            if payload.get("will_retry"):
                self._set_activity(
                    f"Model request failed · retrying after attempt {attempt}…",
                    track_elapsed=True,
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

        if event_type == "no_progress_detected":
            self._set_activity("No progress detected · preparing partial handoff")
            return

        if event_type == "agent_terminated":
            self._set_activity(
                f"Finishing · {payload.get('status', 'completed')} · "
                f"{payload.get('reason', 'completed')}"
            )

    def show_approval(
        self,
        request: PermissionRequest,
        future: Future[ApprovalDecision],
    ) -> None:
        self._set_activity(
            f"Waiting for approval · {_friendly_tool_name(request.tool_name)}",
            track_elapsed=True,
        )

        def resolved(decision: ApprovalDecision | None) -> None:
            decision = decision or ApprovalDecision.DENY
            if not future.done():
                future.set_result(decision)
            self._set_activity(
                f"{'Denied' if decision is ApprovalDecision.DENY else 'Allowed'} · "
                f"{_friendly_tool_name(request.tool_name)}"
            )

        self.push_screen(ApprovalScreen(request), resolved)

    def expire_approval(self, request: PermissionRequest) -> None:
        if isinstance(self.screen, ApprovalScreen):
            self.screen.dismiss(ApprovalDecision.DENY)
        self._set_activity(
            f"Approval timed out · {_friendly_tool_name(request.tool_name)}"
        )
        self.notify("Approval timed out and was denied.", severity="warning")

    def action_clear_conversation(self) -> None:
        self._conversation().clear()
        self._code_blocks.clear()
        self._conversation().write(Text("Conversation view cleared.", style="dim"))

    def action_copy_last_response(self) -> None:
        if self.last_assistant_text is None:
            self.notify("No agent response is available to copy.", severity="warning")
            return
        self.copy_to_clipboard(self.last_assistant_text)
        self.notify("Copied the latest agent response.")

    def action_copy_code(self, code_id: str) -> None:
        code = self._code_blocks.get(code_id)
        if code is None:
            self.notify("That code block is no longer available.", severity="warning")
            return
        self.copy_to_clipboard(code)
        self.notify("Copied code block.")

    def _register_code_block(self, code: str) -> str:
        code_id = f"code-{self._next_code_block_id}"
        self._next_code_block_id += 1
        self._code_blocks[code_id] = code
        return code_id

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
                    "[dim]Enter adds a new line · Ctrl+S sends · "
                    "Ctrl+Y copies the latest answer · select text then Ctrl+C.[/]"
                )
            )
            return True
        self.notify(f"Unknown local command: {prompt}", severity="warning")
        return True

    def _turn_finished(self, result: AgentResult) -> None:
        self.session_id = result.session_id or self.session_id
        label = {
            "completed": "Completed",
            "partial": "Partial · ready to continue",
            "blocked": "Blocked · user action needed",
            "interrupted": "Interrupted",
            "failed": "Failed",
        }.get(result.status.value, result.status.value)
        self._set_busy(False, f"{label} · {result.steps} model step(s)")
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
                self._set_activity(activity, track_elapsed=True)
            else:
                self._set_activity(
                    f"{activity}  ·  Enter new line  ·  Ctrl+S send"
                )

    def _set_activity(self, activity: str, *, track_elapsed: bool = False) -> None:
        self._activity_label = activity
        self._activity_started_at = monotonic() if track_elapsed else None
        self.query_one("#activity", Static).update(activity)

    def _refresh_activity_elapsed(self) -> None:
        if self._activity_started_at is None:
            return
        elapsed_seconds = max(0, int(monotonic() - self._activity_started_at))
        self.query_one("#activity", Static).update(
            f"{self._activity_label} · {_format_elapsed(elapsed_seconds)} elapsed"
        )

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


def _format_elapsed(seconds: int) -> str:
    minutes, seconds = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _format_duration(duration_ms: Any) -> str:
    if not isinstance(duration_ms, (int, float)) or isinstance(duration_ms, bool):
        return "unknown time"
    if duration_ms < 1000:
        return f"{max(0, round(duration_ms))}ms"
    return f"{duration_ms / 1000:.1f}s"
