from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from coding_agent.agent import Agent, AgentConfig, AgentStatus
from coding_agent.conversation import Message
from coding_agent.errors import (
    ModelRequestError,
    VerificationRequiredError,
)
from coding_agent.execution import ControlledCommandRunner
from coding_agent.events import CompositeEventSink, JsonlEventSink
from coding_agent.model import ModelResponse, ToolCall
from coding_agent.permissions import PermissionRequest
from coding_agent.storage import SqliteConversationStore
from coding_agent.tools import (
    ToolDefinition,
    ToolRegistry,
    create_read_only_registry,
    create_workspace_registry,
)


class FakeModel:
    def __init__(self, responses: list[ModelResponse | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[list[Message]] = []
        self.tool_requests: list[list[dict[str, Any]]] = []

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        timeout_s: float,
    ) -> ModelResponse:
        self.requests.append(messages)
        self.tool_requests.append(tools)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, payload))


class PassingCommandRunner(ControlledCommandRunner):
    def run(
        self,
        argv: list[str] | tuple[str, ...],
        *,
        cwd: str = ".",
        timeout_s: int = 120,
    ) -> dict[str, Any]:
        return {
            "argv": list(argv),
            "cwd": cwd,
            "exit_code": 0,
            "timed_out": False,
            "duration_ms": 0,
            "stdout": "",
            "stderr": "",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "sandboxed": False,
        }


def create_verification_registry(
    verification_outputs: list[dict[str, Any]],
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="change_file",
            description="Simulate a successful workspace mutation.",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=lambda arguments: {"changed": True},
            requires_verification=True,
        )
    )

    outputs = iter(verification_outputs)
    registry.register(
        ToolDefinition(
            name="verify_project",
            description="Return the next deterministic verification result.",
            parameters={
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["test", "build", "format_check", "all"],
                    }
                },
                "additionalProperties": False,
            },
            handler=lambda arguments: next(outputs),
        )
    )
    return registry


class AgentLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        (self.workspace / "README.md").write_text(
            "# Example\nA tiny project.\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_tool_call_then_final_response(self) -> None:
        model = FakeModel(
            [
                ModelResponse.from_parts(
                    tool_calls=[
                        ToolCall(id="call-1", name="read_file", arguments={"path": "README.md"})
                    ],
                    reasoning_content="I should inspect the requested file.",
                ),
                ModelResponse.from_parts(text="This is a tiny example project."),
            ]
        )
        agent = Agent(model=model, tools=create_read_only_registry(self.workspace))

        result = agent.run("Read README.md and describe the project")

        self.assertEqual(result.termination_reason, "final_response")
        self.assertEqual(result.steps, 2)
        second_request = model.requests[1]
        assistant_message = second_request[-2]
        tool_message = second_request[-1]
        self.assertEqual(
            assistant_message["reasoning_content"],
            "I should inspect the requested file.",
        )
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "call-1")
        self.assertIn("A tiny project", tool_message["content"])

    def test_each_agent_run_starts_a_new_approval_task(self) -> None:
        class TaskAwarePolicy:
            def __init__(self) -> None:
                self.tasks_started = 0

            def begin_task(self) -> None:
                self.tasks_started += 1

            def approve(self, request: PermissionRequest) -> bool:
                return True

        policy = TaskAwarePolicy()
        model = FakeModel(
            [
                ModelResponse.from_parts(text="First complete"),
                ModelResponse.from_parts(text="Second complete"),
            ]
        )
        agent = Agent(model=model, tools=ToolRegistry(approval_policy=policy))

        agent.run("First task")
        agent.run("Second task")

        self.assertEqual(policy.tasks_started, 2)

    def test_agent_can_create_a_file_with_workspace_tools(self) -> None:
        (self.workspace / "pyproject.toml").write_text(
            "[build-system]\nrequires = []\nbuild-backend = 'setuptools.build_meta'\n",
            encoding="utf-8",
        )
        (self.workspace / "tests").mkdir()
        (self.workspace / "tests" / "test_sample.py").write_text(
            "import unittest\n\n"
            "class Sample(unittest.TestCase):\n"
            "    def test_ok(self):\n"
            "        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        model = FakeModel(
            [
                ModelResponse.from_parts(
                    tool_calls=[
                        ToolCall(
                            id="write-1",
                            name="write_file",
                            arguments={"path": "created.py", "content": "value = 42\n"},
                        )
                    ]
                ),
                ModelResponse.from_parts(text="Created created.py."),
            ]
        )
        agent = Agent(
            model=model,
            tools=create_workspace_registry(
                self.workspace,
                command_runner=PassingCommandRunner(self.workspace),
            ),
        )

        result = agent.run("Create created.py")

        self.assertEqual(result.text, "Created created.py.")
        self.assertEqual(
            (self.workspace / "created.py").read_text(encoding="utf-8"),
            "value = 42\n",
        )
        tool_payload = json.loads(model.requests[1][-1]["content"])
        self.assertTrue(tool_payload["ok"])
        self.assertTrue(tool_payload["output"]["created"])

    def test_final_response_is_blocked_until_automatic_verification_passes(
        self,
    ) -> None:
        model = FakeModel(
            [
                ModelResponse.from_parts(
                    tool_calls=[
                        ToolCall(id="change-1", name="change_file", arguments={})
                    ]
                ),
                ModelResponse.from_parts(text="Changes are complete."),
            ]
        )
        events = RecordingEventSink()
        agent = Agent(
            model=model,
            tools=create_verification_registry(
                [{"ok": True, "skipped": False, "results": []}]
            ),
            events=events,
        )

        result = agent.run("Change the project")

        self.assertEqual(result.text, "Changes are complete.")
        self.assertEqual(
            [message["role"] for message in result.conversation.messages],
            ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"],
        )
        verification_call = result.conversation.messages[-3]
        self.assertEqual(
            verification_call["tool_calls"][0]["function"]["name"],
            "verify_project",
        )
        automatic_results = [
            payload
            for event_type, payload in events.events
            if event_type == "tool_result" and payload.get("automatic")
        ]
        self.assertEqual(len(automatic_results), 1)

    def test_finalization_step_proactively_verifies_pending_changes(self) -> None:
        model = FakeModel(
            [
                ModelResponse.from_parts(
                    tool_calls=[
                        ToolCall(id="change-1", name="change_file", arguments={})
                    ]
                ),
                ModelResponse.from_parts(
                    text="Budget reached after the verified change; follow-up remains."
                ),
            ]
        )
        agent = Agent(
            model=model,
            tools=create_verification_registry(
                [{"ok": True, "skipped": False, "results": []}]
            ),
            config=AgentConfig(max_steps=2),
        )

        result = agent.run("Change the project and keep working")

        self.assertEqual(result.status, AgentStatus.PARTIAL)
        self.assertEqual(result.termination_reason, "max_steps")
        self.assertEqual(model.tool_requests[1], [])
        self.assertEqual(model.requests[1][-1]["name"], "verify_project")
        verification_result = json.loads(model.requests[1][-1]["content"])
        self.assertTrue(verification_result["output"]["ok"])
        self.assertIn("verification passed", model.requests[1][0]["content"])

    def test_finalization_reports_blocked_when_verification_is_unavailable(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="change_file",
                description="Simulate a mutation.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=lambda arguments: {"changed": True},
                requires_verification=True,
            )
        )
        model = FakeModel(
            [
                ModelResponse.from_parts(
                    tool_calls=[
                        ToolCall(id="change-1", name="change_file", arguments={})
                    ]
                ),
                ModelResponse.from_parts(
                    text="Blocked because project verification is unavailable."
                ),
            ]
        )
        agent = Agent(
            model=model,
            tools=registry,
            config=AgentConfig(max_steps=2),
        )

        result = agent.run("Change the project")

        self.assertEqual(result.status, AgentStatus.BLOCKED)
        self.assertEqual(result.termination_reason, "max_steps")
        self.assertEqual(model.tool_requests[-1], [])
        self.assertIn("blocked or unavailable", model.requests[-1][0]["content"])

    def test_failed_verification_is_returned_to_model_before_retrying_final(
        self,
    ) -> None:
        model = FakeModel(
            [
                ModelResponse.from_parts(
                    tool_calls=[
                        ToolCall(id="change-1", name="change_file", arguments={})
                    ]
                ),
                ModelResponse.from_parts(text="Premature final answer."),
                ModelResponse.from_parts(text="Verified final answer."),
            ]
        )
        events = RecordingEventSink()
        agent = Agent(
            model=model,
            tools=create_verification_registry(
                [
                    {"ok": False, "skipped": False, "results": [{"exit_code": 1}]},
                    {"ok": True, "skipped": False, "results": [{"exit_code": 0}]},
                ]
            ),
            events=events,
        )

        result = agent.run("Change and fix the project")

        self.assertEqual(result.text, "Verified final answer.")
        self.assertEqual(result.steps, 3)
        failed_result = json.loads(model.requests[2][-1]["content"])
        self.assertFalse(failed_result["output"]["ok"])
        assistant_texts = [
            message.get("content")
            for message in result.conversation.messages
            if message["role"] == "assistant"
        ]
        self.assertNotIn("Premature final answer.", assistant_texts)
        self.assertTrue(
            any(
                event_type == "verification_gate_blocked"
                and payload["reason"] == "tests_failed"
                for event_type, payload in events.events
            )
        )

    def test_verification_unavailable_raises_instead_of_accepting_final(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="change_file",
                description="Simulate a mutation.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=lambda arguments: {"changed": True},
                requires_verification=True,
            )
        )
        model = FakeModel(
            [
                ModelResponse.from_parts(
                    tool_calls=[
                        ToolCall(id="change-1", name="change_file", arguments={})
                    ]
                ),
                ModelResponse.from_parts(text="Must not be accepted."),
            ]
        )
        agent = Agent(model=model, tools=registry)

        with self.assertRaisesRegex(
            VerificationRequiredError, "unknown tool: verify_project"
        ):
            agent.run("Change the project")

    def test_model_request_is_retried(self) -> None:
        model = FakeModel(
            [
                ModelRequestError("temporary"),
                ModelResponse.from_parts(text="Recovered"),
            ]
        )
        agent = Agent(
            model=model,
            tools=create_read_only_registry(self.workspace),
            config=AgentConfig(max_model_retries=1, retry_base_delay_s=0),
        )

        result = agent.run("Hello")

        self.assertEqual(result.text, "Recovered")
        self.assertEqual(len(model.requests), 2)

    def test_permanent_model_request_error_is_not_retried(self) -> None:
        events = RecordingEventSink()
        error = ModelRequestError(
            "model authentication failed",
            retryable=False,
            status_code=401,
        )
        model = FakeModel([error])
        agent = Agent(
            model=model,
            tools=create_read_only_registry(self.workspace),
            config=AgentConfig(max_model_retries=5, retry_base_delay_s=0),
            events=events,
        )

        with self.assertRaises(ModelRequestError) as raised:
            agent.run("Hello")

        self.assertIs(raised.exception, error)
        self.assertEqual(len(model.requests), 1)
        request_errors = [
            payload
            for event_type, payload in events.events
            if event_type == "model_request_error"
        ]
        self.assertEqual(len(request_errors), 1)
        self.assertFalse(request_errors[0]["retryable"])
        self.assertFalse(request_errors[0]["will_retry"])
        self.assertEqual(request_errors[0]["status_code"], 401)

    def test_max_steps_reserves_a_text_only_partial_handoff(self) -> None:
        model = FakeModel(
            [
                ModelResponse.from_parts(
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="read_file",
                            arguments={"path": "README.md"},
                        )
                    ]
                ),
                ModelResponse.from_parts(text="Partial handoff with remaining work."),
            ]
        )
        agent = Agent(
            model=model,
            tools=create_read_only_registry(self.workspace),
            config=AgentConfig(max_steps=2),
        )

        result = agent.run("Never finish")

        self.assertEqual(result.status, AgentStatus.PARTIAL)
        self.assertEqual(result.termination_reason, "max_steps")
        self.assertEqual(result.steps, 2)
        self.assertTrue(model.tool_requests[0])
        self.assertEqual(model.tool_requests[1], [])
        self.assertIn("maximum model steps", model.requests[1][0]["content"])

    def test_persisted_max_steps_records_partial_session_status(self) -> None:
        model = FakeModel(
            [
                ModelResponse.from_parts(
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="read_file",
                            arguments={"path": "README.md"},
                        )
                    ]
                ),
                ModelResponse.from_parts(text="Persisted partial handoff."),
            ]
        )
        store = SqliteConversationStore(
            self.workspace / ".coding-agent" / "history.sqlite3"
        )
        agent = Agent(
            model=model,
            tools=create_read_only_registry(self.workspace),
            config=AgentConfig(max_steps=2),
            store=store,
            workspace=self.workspace,
            model_name="fake-model",
        )

        result = agent.run("Never finish")

        self.assertEqual(result.status, AgentStatus.PARTIAL)
        self.assertIsNotNone(result.session_id)
        assert result.session_id is not None
        self.assertEqual(store.get_session(result.session_id).status, "partial")

    def test_consecutive_tool_errors_trigger_no_progress_handoff(self) -> None:
        model = FakeModel(
            [
                ModelResponse.from_parts(
                    tool_calls=[
                        ToolCall(
                            id=f"bad-{index}",
                            name=f"missing_tool_{index}",
                            arguments={},
                        )
                    ]
                )
                for index in range(3)
            ]
            + [ModelResponse.from_parts(text="Stopped after repeated failures.")]
        )
        agent = Agent(
            model=model,
            tools=ToolRegistry(),
            config=AgentConfig(
                max_steps=8,
                max_consecutive_tool_errors=3,
            ),
        )

        result = agent.run("Keep trying missing tools")

        self.assertEqual(result.termination_reason, "no_progress")
        self.assertEqual(result.status, AgentStatus.PARTIAL)
        self.assertEqual(result.steps, 4)

    def test_repeated_tool_call_returns_a_partial_handoff(self) -> None:
        call_responses = [
            ModelResponse.from_parts(
                tool_calls=[
                    ToolCall(
                        id=f"call-{index}",
                        name="read_file",
                        arguments={"path": "README.md"},
                    )
                ]
            )
            for index in range(3)
        ]
        model = FakeModel(
            [
                *call_responses,
                ModelResponse.from_parts(text="Stopped after detecting no progress."),
            ]
        )
        agent = Agent(
            model=model,
            tools=create_read_only_registry(self.workspace),
            config=AgentConfig(max_steps=5, repeated_tool_call_limit=2),
        )

        result = agent.run("Repeat forever")

        self.assertEqual(result.status, AgentStatus.PARTIAL)
        self.assertEqual(result.termination_reason, "no_progress")
        self.assertEqual(result.steps, 4)
        self.assertEqual(model.tool_requests[-1], [])
        tool_messages = [
            message for message in result.conversation.messages if message["role"] == "tool"
        ]
        self.assertIn("not executed", tool_messages[-1]["content"])

    def test_alternating_tool_cycle_is_detected(self) -> None:
        calls = [
            ModelResponse.from_parts(
                tool_calls=[
                    ToolCall(
                        id=f"call-{index}",
                        name="read_file",
                        arguments={
                            "path": "README.md",
                            "start_line": 1 if index % 2 == 0 else 2,
                        },
                    )
                ]
            )
            for index in range(6)
        ]
        model = FakeModel(
            [*calls, ModelResponse.from_parts(text="Alternating cycle stopped.")]
        )
        agent = Agent(
            model=model,
            tools=create_read_only_registry(self.workspace),
            config=AgentConfig(max_steps=10, repeated_tool_call_limit=2),
        )

        result = agent.run("Alternate forever")

        self.assertEqual(result.termination_reason, "no_progress")
        self.assertEqual(result.status, AgentStatus.PARTIAL)
        self.assertEqual(result.steps, 7)

    def test_jsonl_event_log_records_termination(self) -> None:
        log_path = self.workspace / ".coding-agent" / "events.jsonl"
        model = FakeModel(
            [
                ModelResponse.from_parts(
                    text="Done",
                    finish_reason="stop",
                    usage={"prompt_tokens": 7, "completion_tokens": 2},
                    response_id="response-test",
                )
            ]
        )
        agent = Agent(
            model=model,
            tools=create_read_only_registry(self.workspace),
            events=JsonlEventSink(log_path),
        )

        agent.run("Finish immediately")

        events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(
            [event["type"] for event in events],
            ["user_message", "model_response", "agent_terminated"],
        )
        self.assertEqual(events[-1]["payload"]["reason"], "final_response")
        model_event = events[1]["payload"]
        self.assertEqual(model_event["finish_reason"], "stop")
        self.assertEqual(model_event["usage"]["prompt_tokens"], 7)
        self.assertEqual(model_event["response_id"], "response-test")
        self.assertIsInstance(model_event["duration_ms"], int)

    def test_jsonl_event_log_redacts_credentials_and_reasoning(self) -> None:
        log_path = self.workspace / ".coding-agent" / "redacted.jsonl"
        sink = JsonlEventSink(log_path)

        sink.emit(
            "model_response",
            {
                "assistant_message": {
                    "reasoning_content": "private chain of thought",
                    "content": "Authorization: Bearer abcdefghijklmnop",
                },
                "tool": {"api_key": "sk-abcdefghijklmnop"},
            },
        )

        raw = log_path.read_text(encoding="utf-8")
        self.assertNotIn("private chain of thought", raw)
        self.assertNotIn("abcdefghijklmnop", raw)
        self.assertGreaterEqual(raw.count("[REDACTED]"), 3)

    def test_composite_event_sink_fans_out_in_order(self) -> None:
        first = RecordingEventSink()
        second = RecordingEventSink()
        sink = CompositeEventSink(first, second)

        sink.emit("example", {"value": 42})

        self.assertEqual(first.events, [("example", {"value": 42})])
        self.assertEqual(second.events, first.events)


if __name__ == "__main__":
    unittest.main()
