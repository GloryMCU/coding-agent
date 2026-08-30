"""Append-only JSONL event logging for debugging and recovery."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "authorization",
        "cookie",
        "credential",
        "password",
        "refresh_token",
        "reasoning_content",
        "secret",
        "token",
    }
)
_SENSITIVE_KEY_SUFFIXES = ("_api_key", "_credential", "_password", "_secret")
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)\b(api[_-]?key|authorization|password|secret|token)"
        r"(\s*[:=]\s*)([^\s,;]+)"
    ),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)


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
    redact_sensitive: bool = True

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        safe_payload = _redact_value(payload) if self.redact_sensitive else payload
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "payload": safe_payload,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def _redact_value(value: Any, *, key: str | None = None) -> Any:
    """Return a JSON-compatible copy with common credentials removed.

    Audit logs are long-lived and often attached to bug reports. Redaction is
    deliberately applied only at the JSONL boundary so the live UI and agent
    loop still receive complete tool results.
    """

    if key is not None:
        normalized = key.casefold().replace("-", "_")
        if normalized in _SENSITIVE_KEYS or normalized.endswith(
            _SENSITIVE_KEY_SUFFIXES
        ):
            return "[REDACTED]"
    if isinstance(value, dict):
        return {
            str(child_key): _redact_value(child, key=str(child_key))
            for child_key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_value(child) for child in value]
    if isinstance(value, str):
        redacted = value
        for pattern in _SECRET_PATTERNS:
            if pattern.groups == 1:
                redacted = pattern.sub(r"\1[REDACTED]", redacted)
            elif pattern.groups == 3:
                redacted = pattern.sub(r"\1\2[REDACTED]", redacted)
            else:
                redacted = pattern.sub("[REDACTED]", redacted)
        return redacted
    return value

