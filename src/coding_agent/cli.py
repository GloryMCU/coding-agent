"""Plain and interactive terminal entry points for the coding agent."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .agent import Agent, AgentConfig
from .events import CompositeEventSink, EventSink, JsonlEventSink, NullEventSink
from .model import DeepSeekV4ProClient
from .permissions import (
    AllowAllApprovalPolicy,
    ApprovalPolicy,
    DenyApprovalPolicy,
    InteractiveApprovalPolicy,
)
from .storage import SqliteConversationStore
from .tools import create_workspace_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the framework-free coding agent")
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Task for one-shot mode, or the initial task with --interactive",
    )
    interface = parser.add_mutually_exclusive_group()
    interface.add_argument(
        "--interactive",
        action="store_true",
        help="Open the terminal UI, optionally with the positional initial task",
    )
    interface.add_argument(
        "--plain",
        action="store_true",
        help="Force one-shot text output (requires a prompt)",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root exposed to local tools (default: current directory)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("DEEPSEEK_BASE_URL", DeepSeekV4ProClient.BASE_URL),
        help="DeepSeek API or compatible gateway base URL",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=sorted(DeepSeekV4ProClient.REASONING_EFFORTS),
        default="high",
        help="DeepSeek V4 Pro reasoning effort (default: high)",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable thinking mode (default: enabled)",
    )
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--approval-mode",
        choices=("ask", "deny", "allow"),
        default="ask",
        help=(
            "Policy for file changes and command execution: ask, deny, or allow "
            "(default: ask)"
        ),
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=24_000,
        help="Approximate model context budget for durable sessions (default: 24000)",
    )
    parser.add_argument(
        "--context-summary-tokens",
        type=int,
        default=2_000,
        help="Maximum share of context used by compacted history (default: 2000)",
    )
    parser.add_argument(
        "--history-search-limit",
        type=int,
        default=5,
        help="FTS5 matches recalled into compacted context (default: 5; 0 disables)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=(
            "SQLite history database "
            "(default: <workspace>/.coding-agent/history.sqlite3)"
        ),
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Continue an existing durable session instead of creating one",
    )
    parser.add_argument(
        "--event-log",
        type=Path,
        default=None,
        help=(
            "Append-only local execution log "
            "(default: <workspace>/.coding-agent/events.jsonl)"
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    interactive = args.interactive or (args.prompt is None and not args.plain)
    if args.plain and args.prompt is None:
        parser.error("--plain requires a prompt")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("missing API key: set DEEPSEEK_API_KEY")

    workspace = args.workspace.resolve(strict=True)
    database_path = (
        args.db.resolve()
        if args.db is not None
        else workspace / ".coding-agent" / "history.sqlite3"
    )
    event_log_path = (
        args.event_log.resolve()
        if args.event_log is not None
        else workspace / ".coding-agent" / "events.jsonl"
    )
    model = DeepSeekV4ProClient(
        api_key=api_key,
        base_url=args.base_url,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
    )
    audit_events = JsonlEventSink(event_log_path)

    def make_agent(events: EventSink, approval_policy: ApprovalPolicy) -> Agent:
        return Agent(
            model=model,
            tools=create_workspace_registry(
                workspace, approval_policy=approval_policy
            ),
            config=AgentConfig(
                max_steps=args.max_steps,
                max_context_tokens=args.max_context_tokens,
                context_summary_tokens=args.context_summary_tokens,
                history_search_limit=args.history_search_limit,
            ),
            events=CompositeEventSink(audit_events, events),
            store=SqliteConversationStore(database_path),
            workspace=workspace,
            model_name=DeepSeekV4ProClient.MODEL,
        )

    if interactive:
        try:
            from .tui import CodingAgentApp
        except ModuleNotFoundError as exc:
            if exc.name in {"textual", "rich"}:
                raise SystemExit(
                    'interactive mode requires: python -m pip install -e ".[tui]"'
                ) from exc
            raise
        app = CodingAgentApp(
            agent_factory=make_agent,
            workspace=workspace,
            model_name=DeepSeekV4ProClient.MODEL,
            approval_mode=args.approval_mode,
            session_id=args.session_id,
            initial_prompt=args.prompt,
        )
        app.run()
        return 0

    approval_policy = {
        "ask": InteractiveApprovalPolicy,
        "deny": DenyApprovalPolicy,
        "allow": AllowAllApprovalPolicy,
    }[args.approval_mode]()
    agent = make_agent(NullEventSink(), approval_policy)
    assert args.prompt is not None
    result = agent.run(args.prompt, session_id=args.session_id)
    print(result.text)
    print(f"session_id: {result.session_id}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
