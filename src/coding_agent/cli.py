"""One-shot command line entry point for the MVP."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .agent import Agent, AgentConfig
from .events import JsonlEventSink
from .model import DeepSeekV4ProClient
from .storage import SqliteConversationStore
from .tools import create_read_only_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the framework-free coding agent")
    parser.add_argument("prompt", help="Task for the coding agent")
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
    args = build_parser().parse_args()
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
    agent = Agent(
        model=model,
        tools=create_read_only_registry(workspace),
        config=AgentConfig(max_steps=args.max_steps),
        events=JsonlEventSink(event_log_path),
        store=SqliteConversationStore(database_path),
        workspace=workspace,
        model_name=DeepSeekV4ProClient.MODEL,
    )
    result = agent.run(args.prompt, session_id=args.session_id)
    print(result.text)
    print(f"session_id: {result.session_id}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
