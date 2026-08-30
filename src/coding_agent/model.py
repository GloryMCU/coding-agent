"""Model protocol and an optional OpenAI-compatible adapter."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .conversation import Message
from .errors import ModelProtocolError, ModelRequestError


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def as_openai_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str | None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    assistant_message: Message = field(default_factory=dict)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    response_id: str | None = None

    @classmethod
    def from_parts(
        cls,
        *,
        text: str | None = None,
        tool_calls: Sequence[ToolCall] = (),
        reasoning_content: str | None = None,
        finish_reason: str | None = None,
        usage: dict[str, Any] | None = None,
        response_id: str | None = None,
    ) -> "ModelResponse":
        calls = tuple(tool_calls)
        message: Message = {"role": "assistant", "content": text}
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        if calls:
            message["tool_calls"] = [call.as_openai_dict() for call in calls]
        return cls(
            text=text,
            tool_calls=calls,
            assistant_message=message,
            finish_reason=finish_reason,
            usage=deepcopy(usage),
            response_id=response_id,
        )


class ModelClient(Protocol):
    """Small interface that keeps the agent loop vendor independent."""

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        timeout_s: float,
    ) -> ModelResponse:
        ...


class OpenAIChatClient:
    """Adapter for OpenAI and OpenAI-compatible Chat Completions APIs.

    The official ``openai`` package is deliberately an optional dependency;
    the agent core and its tests do not depend on a network client.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI adapter requires the optional dependency: "
                "pip install -e '.[openai]'"
            ) from exc

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._extra_body = deepcopy(extra_body)

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        timeout_s: float,
    ) -> ModelResponse:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "timeout": timeout_s,
        }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        if self._reasoning_effort:
            request["reasoning_effort"] = self._reasoning_effort
        if self._extra_body:
            request["extra_body"] = deepcopy(self._extra_body)

        try:
            completion = self._client.chat.completions.create(**request)
        except Exception as exc:  # Vendor exception types vary by client version.
            raise ModelRequestError(f"model request failed: {exc}") from exc

        if not completion.choices:
            raise ModelProtocolError("model response contains no choices")

        choice = completion.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason in {"length", "content_filter"}:
            raise ModelProtocolError(
                f"model response did not complete normally: {finish_reason}"
            )
        if finish_reason not in {None, "stop", "tool_calls"}:
            raise ModelProtocolError(
                f"unsupported model finish reason: {finish_reason}"
            )

        response = parse_openai_message(
            choice.message,
            finish_reason=finish_reason,
            usage=_model_dump(getattr(completion, "usage", None)),
            response_id=getattr(completion, "id", None),
        )
        if finish_reason == "tool_calls" and not response.tool_calls:
            raise ModelProtocolError(
                "model reported tool_calls completion without any tool call"
            )
        if finish_reason == "stop" and response.tool_calls:
            raise ModelProtocolError(
                "model reported stop completion while returning tool calls"
            )
        return response


class DeepSeekV4ProClient(OpenAIChatClient):
    """DeepSeek V4 Pro over its OpenAI-compatible Chat Completions API."""

    MODEL = "deepseek-v4-pro"
    BASE_URL = "https://api.deepseek.com"
    REASONING_EFFORTS = frozenset({"high", "max"})

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = BASE_URL,
        thinking: bool = True,
        reasoning_effort: str = "high",
    ) -> None:
        if reasoning_effort not in self.REASONING_EFFORTS:
            choices = ", ".join(sorted(self.REASONING_EFFORTS))
            raise ValueError(f"DeepSeek V4 Pro reasoning_effort must be one of: {choices}")
        super().__init__(
            model=self.MODEL,
            api_key=api_key,
            base_url=base_url,
            reasoning_effort=reasoning_effort if thinking else None,
            extra_body={
                "thinking": {"type": "enabled" if thinking else "disabled"}
            },
        )


def parse_openai_message(
    message: Any,
    *,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
    response_id: str | None = None,
) -> ModelResponse:
    """Parse the SDK response shape without trusting its tool arguments."""

    content = getattr(message, "content", None)
    if content is not None and not isinstance(content, str):
        raise ModelProtocolError("assistant content must be text or null")

    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        raise ModelProtocolError("assistant reasoning_content must be text or null")

    parsed_calls: list[ToolCall] = []
    for call in getattr(message, "tool_calls", None) or []:
        call_id = getattr(call, "id", None)
        function = getattr(call, "function", None)
        name = getattr(function, "name", None)
        raw_arguments = getattr(function, "arguments", None)
        if not isinstance(call_id, str) or not call_id:
            raise ModelProtocolError("tool call id must be a non-empty string")
        if not isinstance(name, str) or not name:
            raise ModelProtocolError("tool name must be a non-empty string")
        if raw_arguments is not None and not isinstance(raw_arguments, str):
            raise ModelProtocolError(f"tool {name!r} arguments must be JSON text")
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ModelProtocolError(
                f"tool {name!r} returned invalid JSON arguments"
            ) from exc
        if not isinstance(arguments, dict):
            raise ModelProtocolError(
                f"tool {name!r} arguments must be a JSON object"
            )
        parsed_calls.append(ToolCall(id=call_id, name=name, arguments=arguments))

    return ModelResponse.from_parts(
        text=content,
        tool_calls=parsed_calls,
        reasoning_content=reasoning_content,
        finish_reason=finish_reason,
        usage=usage,
        response_id=response_id,
    )


def _model_dump(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return deepcopy(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        dumped = dump()
        return deepcopy(dumped) if isinstance(dumped, dict) else None
    result = {
        name: getattr(value, name)
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
        )
        if getattr(value, name, None) is not None
    }
    return result or None
