"""Local conversation state owned by the application."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


Message = dict[str, Any]


@dataclass(slots=True)
class ConversationState:
    """Stores the complete local message history for a single task."""

    _messages: list[Message] = field(default_factory=list)

    @classmethod
    def with_system_prompt(cls, system_prompt: str) -> "ConversationState":
        state = cls()
        if system_prompt.strip():
            state._messages.append({"role": "system", "content": system_prompt})
        return state

    @classmethod
    def from_messages(cls, messages: list[Message]) -> "ConversationState":
        return cls(_messages=deepcopy(messages))

    @property
    def messages(self) -> list[Message]:
        """Return a defensive copy suitable for a model request."""
        return deepcopy(self._messages)

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, message: Message) -> None:
        if message.get("role") != "assistant":
            raise ValueError("assistant message must have role='assistant'")
        self._messages.append(deepcopy(message))

    def add_tool(self, message: Message) -> None:
        if message.get("role") != "tool":
            raise ValueError("tool message must have role='tool'")
        self._messages.append(deepcopy(message))

    def __len__(self) -> int:
        return len(self._messages)
