"""Append-only JSONL event logging for debugging and recovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


class EventSink(Protocol):
    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        ...


class NullEventSink:
    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        return None


@dataclass(frozen=True, slots=True)
class CompositeEventSink:
    """Fan out each event to multiple presentation and audit sinks."""

    sinks: tuple[EventSink, ...]

    def __init__(self, *sinks: EventSink) -> None:
        object.__setattr__(self, "sinks", tuple(sinks))

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        for sink in self.sinks:
            sink.emit(event_type, payload)


@dataclass(slots=True)
class JsonlEventSink:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

