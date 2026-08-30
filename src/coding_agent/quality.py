"""Offline operational quality summaries for JSONL agent event logs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def summarize_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    event_counts: Counter[str] = Counter()
    termination_reasons: Counter[str] = Counter()
    usage_totals: Counter[str] = Counter()
    sessions: set[str] = set()
    model_duration_ms = 0
    tool_duration_ms = 0
    tool_failures = 0

    for event in events:
        event_type = str(event.get("type", "unknown"))
        payload = event.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        event_counts[event_type] += 1

        session_id = payload.get("session_id")
        if isinstance(session_id, str) and session_id:
            sessions.add(session_id)
        duration = payload.get("duration_ms")
        if isinstance(duration, int) and duration >= 0:
            if event_type == "model_response":
                model_duration_ms += duration
            elif event_type == "tool_result":
                tool_duration_ms += duration
        if event_type == "tool_result" and payload.get("ok") is False:
            tool_failures += 1
        if event_type == "agent_terminated":
            termination_reasons[str(payload.get("reason", "unknown"))] += 1

        usage = payload.get("usage")
        if isinstance(usage, dict):
            for name, value in usage.items():
                if (
                    isinstance(name, str)
                    and name.endswith("_tokens")
                    and isinstance(value, int)
                    and value >= 0
                ):
                    usage_totals[name] += value

    return {
        "events": sum(event_counts.values()),
        "sessions_observed": len(sessions),
        "model_responses": event_counts["model_response"],
        "model_request_errors": event_counts["model_request_error"],
        "tool_calls_completed": event_counts["tool_result"],
        "tool_failures": tool_failures,
        "model_duration_ms": model_duration_ms,
        "tool_duration_ms": tool_duration_ms,
        "usage": dict(sorted(usage_totals.items())),
        "termination_reasons": dict(sorted(termination_reasons.items())),
        "event_counts": dict(sorted(event_counts.items())),
    }


def summarize_event_log(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    events: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON event at {source}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(event, dict):
                raise ValueError(
                    f"event at {source}:{line_number} must be a JSON object"
                )
            events.append(event)
    return summarize_events(events)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize an offline coding-agent JSONL event log"
    )
    parser.add_argument(
        "event_log",
        nargs="?",
        type=Path,
        default=Path(".coding-agent/events.jsonl"),
    )
    args = parser.parse_args(argv)
    print(json.dumps(summarize_event_log(args.event_log), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
