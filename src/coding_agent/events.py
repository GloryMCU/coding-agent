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

