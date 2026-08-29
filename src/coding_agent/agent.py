"""The model/tool loop, termination conditions, and retry policy."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from .conversation import ConversationState, Message
from .context import ContextBuilder, ContextConfig
from .errors import (
    LoopDetected,
    MaxStepsExceeded,
    ModelProtocolError,
    ModelRequestError,
)
from .events import EventSink, NullEventSink
from .model import ModelClient, ModelResponse, ToolCall
from .storage import SqliteConversationStore
from .tools import ToolRegistry


DEFAULT_SYSTEM_PROMPT = """You are a coding agent operating on a local workspace.
Use the available tools when you need evidence from the repository.
Never invent file contents. Tool errors are observations: correct the arguments or explain the limitation.
Prefer apply_patch for focused changes and use write_file overwrite only for intentional full replacements.
Delete files only when the user explicitly requests deletion or it is an unavoidable part of their requested change.
When the task is complete, respond with a concise final answer and do not call another tool."""


@dataclass(frozen=True, slots=True)
class AgentConfig:
    max_steps: int = 10
    model_timeout_s: float = 60.0
    max_model_retries: int = 2
    retry_base_delay_s: float = 0.5
    repeated_tool_call_limit: int = 3
    max_context_tokens: int = 24_000
    context_summary_tokens: int = 2_000
    history_search_limit: int = 5

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.model_timeout_s <= 0:
            raise ValueError("model_timeout_s must be > 0")
        if self.max_model_retries < 0:
            raise ValueError("max_model_retries must be >= 0")
        if self.repeated_tool_call_limit < 1:
            raise ValueError("repeated_tool_call_limit must be >= 1")
        ContextConfig(
            max_tokens=self.max_context_tokens,
            summary_max_tokens=self.context_summary_tokens,
            history_search_limit=self.history_search_limit,
        )


@dataclass(frozen=True, slots=True)
class AgentResult:
    text: str
    steps: int
    termination_reason: str
    conversation: ConversationState
    session_id: str | None = None


class Agent:
    def __init__(
        self,
        *,
        model: ModelClient,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        events: EventSink | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        store: SqliteConversationStore | None = None,
        workspace: str | Path | None = None,
        model_name: str | None = None,
    ) -> None:
        self._model = model
        self._tools = tools
        self._config = config or AgentConfig()
        self._events = events or NullEventSink()
        self._system_prompt = system_prompt
        self._store = store
        self._context = (
            ContextBuilder(
                store,
                ContextConfig(
                    max_tokens=self._config.max_context_tokens,
                    summary_max_tokens=self._config.context_summary_tokens,
                    history_search_limit=self._config.history_search_limit,
                ),
            )
            if store is not None
            else None
        )
        self._workspace = Path(workspace).resolve() if workspace is not None else None
        self._model_name = model_name or type(model).__name__

    def run(
        self,
        user_input: str,
        *,
        conversation: ConversationState | None = None,
        session_id: str | None = None,
    ) -> AgentResult:
        if not user_input.strip():
            raise ValueError("user_input must not be empty")
        if self._store is not None:
            if conversation is not None:
                raise ValueError(
                    "conversation cannot be supplied when a durable store is configured"
                )
            return self._run_persisted(user_input, session_id=session_id)
        if session_id is not None:
            raise ValueError("session_id requires a durable store")

        return self._run_in_memory(user_input, conversation=conversation)

    def _run_in_memory(
        self,
        user_input: str,
        *,
        conversation: ConversationState | None,
    ) -> AgentResult:

        state = conversation or ConversationState.with_system_prompt(self._system_prompt)
        state.add_user(user_input)
        self._events.emit("user_message", {"content": user_input})

        last_fingerprint: str | None = None
        consecutive_repeats = 0
        for step in range(1, self._config.max_steps + 1):
            response = self._request_model(state.messages)
            self._validate_response(response)
            state.add_assistant(response.assistant_message)
            self._events.emit(
                "model_response",
                {
                    "step": step,
                    "assistant_message": response.assistant_message,
                    "text": response.text,
                    "tool_calls": [
                        {"id": call.id, "name": call.name, "arguments": call.arguments}
                        for call in response.tool_calls
                    ],
                },
            )

            if not response.tool_calls:
                text = response.text or ""
                self._events.emit(
                    "agent_terminated",
                    {"step": step, "reason": "final_response"},
                )
                return AgentResult(
                    text=text,
                    steps=step,
                    termination_reason="final_response",
                    conversation=state,
                )

            for call in response.tool_calls:
                fingerprint = self._tool_fingerprint(call)
                if fingerprint == last_fingerprint:
                    consecutive_repeats += 1
                else:
                    last_fingerprint = fingerprint
                    consecutive_repeats = 1
                if consecutive_repeats > self._config.repeated_tool_call_limit:
                    self._events.emit(
                        "agent_terminated",
                        {
                            "step": step,
                            "reason": "repeated_tool_call",
                            "tool": call.name,
                        },
                    )
                    raise LoopDetected(
                        f"tool {call.name!r} repeated with identical arguments more than "
                        f"{self._config.repeated_tool_call_limit} times"
                    )

                result = self._tools.execute(call)
                state.add_tool(result.to_message())
                self._events.emit(
                    "tool_result",
                    {
                        "step": step,
                        "tool_call_id": call.id,
                        "name": call.name,
                        "ok": result.ok,
                        "output": result.output,
                        "error": result.error,
                    },
                )

        self._events.emit(
            "agent_terminated",
            {"step": self._config.max_steps, "reason": "max_steps"},
        )
        raise MaxStepsExceeded(
            f"agent exceeded maximum of {self._config.max_steps} model steps"
        )

    def _run_persisted(
        self, user_input: str, *, session_id: str | None
    ) -> AgentResult:
        assert self._store is not None
        assert self._context is not None

        if session_id is None:
            session_id = self._store.create_session(
                workspace=self._workspace or Path.cwd(),
                model=self._model_name,
                system_prompt=self._system_prompt,
            )
        else:
            session = self._store.get_session(session_id)
            if (
                self._workspace is not None
                and Path(session.workspace).resolve() != self._workspace
            ):
                raise ValueError(
                    "the stored session belongs to a different workspace: "
                    f"{session.workspace}"
                )

        recovered = self._store.recover_interrupted_calls(session_id)
        if recovered:
            self._events.emit(
                "tool_calls_recovered",
                {"session_id": session_id, "count": recovered},
            )

        try:
            self._store.append_user(session_id, user_input)
            self._events.emit(
                "user_message",
                {"session_id": session_id, "content": user_input},
            )

            last_fingerprint: str | None = None
            consecutive_repeats = 0
            for step in range(1, self._config.max_steps + 1):
                response = self._request_model(self._context.build(session_id))
                self._validate_response(response)
                self._store.append_assistant(session_id, response)
                self._events.emit(
                    "model_response",
                    {
                        "session_id": session_id,
                        "step": step,
                        "assistant_message": response.assistant_message,
                        "text": response.text,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "name": call.name,
                                "arguments": call.arguments,
                            }
                            for call in response.tool_calls
                        ],
                    },
                )

                if not response.tool_calls:
                    self._store.set_session_status(session_id, "completed")
                    self._events.emit(
                        "agent_terminated",
                        {
                            "session_id": session_id,
                            "step": step,
                            "reason": "final_response",
                        },
                    )
                    return AgentResult(
                        text=response.text or "",
                        steps=step,
                        termination_reason="final_response",
                        conversation=ConversationState.from_messages(
                            self._context.build(session_id)
                        ),
                        session_id=session_id,
                    )

                for call in response.tool_calls:
                    fingerprint = self._tool_fingerprint(call)
                    if fingerprint == last_fingerprint:
                        consecutive_repeats += 1
                    else:
                        last_fingerprint = fingerprint
                        consecutive_repeats = 1
                    if consecutive_repeats > self._config.repeated_tool_call_limit:
                        self._events.emit(
                            "agent_terminated",
                            {
                                "session_id": session_id,
                                "step": step,
                                "reason": "repeated_tool_call",
                                "tool": call.name,
                            },
                        )
                        raise LoopDetected(
                            f"tool {call.name!r} repeated with identical arguments more than "
                            f"{self._config.repeated_tool_call_limit} times"
                        )

                    claim = self._store.claim_tool_call(session_id, call.id)
                    if claim.execute:
                        result = self._tools.execute(claim.call)
                        self._store.finish_tool_call(session_id, result)
                    elif claim.result is not None:
                        result = claim.result
                    else:
                        raise RuntimeError(
                            f"tool call {call.id!r} is already {claim.status}"
                        )
                    self._events.emit(
                        "tool_result",
                        {
                            "session_id": session_id,
                            "step": step,
                            "tool_call_id": call.id,
                            "name": call.name,
                            "ok": result.ok,
                            "output": result.output,
                            "error": result.error,
                            "replayed": not claim.execute,
                        },
                    )

            self._events.emit(
                "agent_terminated",
                {
                    "session_id": session_id,
                    "step": self._config.max_steps,
                    "reason": "max_steps",
                },
            )
            raise MaxStepsExceeded(
                f"agent exceeded maximum of {self._config.max_steps} model steps"
            )
        except Exception as exc:
            try:
                self._store.set_session_status(session_id, "error", error=str(exc))
            except Exception:
                pass
            raise

    def _request_model(self, messages: list[Message]) -> ModelResponse:
        last_error: ModelRequestError | None = None
        attempts = self._config.max_model_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return self._model.generate(
                    messages,
                    self._tools.schemas(),
                    timeout_s=self._config.model_timeout_s,
                )
            except ModelRequestError as exc:
                last_error = exc
                self._events.emit(
                    "model_request_error",
                    {"attempt": attempt, "error": str(exc)},
                )
                if attempt < attempts:
                    time.sleep(self._config.retry_base_delay_s * (2 ** (attempt - 1)))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _validate_response(response: ModelResponse) -> None:
        if response.assistant_message.get("role") != "assistant":
            raise ModelProtocolError("model response lacks a valid assistant message")
        if not response.tool_calls and not (response.text and response.text.strip()):
            raise ModelProtocolError("model returned neither text nor tool calls")
        call_ids = [call.id for call in response.tool_calls]
        if any(not call_id for call_id in call_ids):
            raise ModelProtocolError("tool call id must not be empty")
        if len(call_ids) != len(set(call_ids)):
            raise ModelProtocolError("tool call ids must be unique within a response")

    @staticmethod
    def _tool_fingerprint(call: ToolCall) -> str:
        canonical = json.dumps(
            {"name": call.name, "arguments": call.arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
