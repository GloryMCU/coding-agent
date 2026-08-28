"""Projection of durable records into provider-compatible model messages."""

from __future__ import annotations

import json
from typing import Any

from .conversation import Message
from .storage import SqliteConversationStore, StoredMessage, StoredPart


class ContextBuilder:
    def __init__(self, store: SqliteConversationStore) -> None:
        self._store = store

    def build(self, session_id: str) -> list[Message]:
        session = self._store.get_session(session_id)
        messages: list[Message] = []
        if session.system_prompt.strip():
            messages.append({"role": "system", "content": session.system_prompt})
        for stored in self._store.load_messages(session_id):
            if stored.role == "user":
                messages.append(
                    {
                        "role": "user",
                        "content": "".join(
                            part.data.get("text", "")
                            for part in stored.parts
                            if part.type == "text"
                        ),
                    }
                )
            else:
                messages.extend(self._project_assistant(stored))
        return messages

    @staticmethod
    def _project_assistant(stored: StoredMessage) -> list[Message]:
        text_parts = [part for part in stored.parts if part.type == "text"]
        text = "".join(part.data.get("text", "") for part in text_parts)
        reasoning = "".join(
            part.data.get("text", "")
            for part in stored.parts
            if part.type == "reasoning"
        )
        tool_parts = [part for part in stored.parts if part.type == "tool"]
        assistant: Message = {
            "role": "assistant",
            "content": text if text_parts else None,
        }
        if reasoning:
            assistant["reasoning_content"] = reasoning
        if tool_parts:
            assistant["tool_calls"] = [
                {
                    "id": part.call_id,
                    "type": "function",
                    "function": {
                        "name": part.tool_name,
                        "arguments": json.dumps(
                            part.data["arguments"], ensure_ascii=False
                        ),
                    },
                }
                for part in tool_parts
            ]

        projected: list[Message] = [assistant]
        projected.extend(ContextBuilder._project_tool_result(part) for part in tool_parts)
        return projected

    @staticmethod
    def _project_tool_result(part: StoredPart) -> Message:
        payload: dict[str, Any]
        if part.status == "completed":
            payload = {"ok": True, "output": part.data.get("output")}
        elif part.status in {"error", "interrupted"}:
            payload = {
                "ok": False,
                "error": part.data.get("error") or f"tool call {part.status}",
            }
        else:
            payload = {
                "ok": False,
                "error": "tool execution has no durable result",
            }
        return {
            "role": "tool",
            "tool_call_id": part.call_id,
            "name": part.tool_name,
            "content": json.dumps(payload, ensure_ascii=False),
        }
