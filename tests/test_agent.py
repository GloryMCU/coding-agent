from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from coding_agent.agent import Agent, AgentConfig
from coding_agent.conversation import Message
from coding_agent.errors import LoopDetected, MaxStepsExceeded, ModelRequestError
from coding_agent.events import CompositeEventSink, JsonlEventSink
from coding_agent.model import ModelResponse, ToolCall
from coding_agent.tools import create_read_only_registry, create_workspace_registry


class FakeModel:
    def __init__(self, responses: list[ModelResponse | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[list[Message]] = []

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        timeout_s: float,
    ) -> ModelResponse:
        self.requests.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, payload))


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

    def test_agent_can_create_a_file_with_workspace_tools(self) -> None:
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
        agent = Agent(model=model, tools=create_workspace_registry(self.workspace))

        result = agent.run("Create created.py")

        self.assertEqual(result.text, "Created created.py.")
        self.assertEqual(
            (self.workspace / "created.py").read_text(encoding="utf-8"),
            "value = 42\n",
        )
        tool_payload = json.loads(model.requests[1][-1]["content"])
        self.assertTrue(tool_payload["ok"])
        self.assertTrue(tool_payload["output"]["created"])

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

    def test_max_steps_terminates_agent(self) -> None:
        model = FakeModel(
            [
                ModelResponse.from_parts(
                    tool_calls=[
                        ToolCall(
                            id=f"call-{index}",
                            name="read_file",
                            arguments={"path": "README.md", "start_line": index + 1},
                        )
                    ]
                )
                for index in range(2)
            ]
        )
        agent = Agent(
            model=model,
            tools=create_read_only_registry(self.workspace),
            config=AgentConfig(max_steps=2),
        )

        with self.assertRaises(MaxStepsExceeded):
            agent.run("Never finish")

    def test_repeated_tool_call_is_detected(self) -> None:
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
        model = FakeModel(call_responses)
        agent = Agent(
            model=model,
            tools=create_read_only_registry(self.workspace),
            config=AgentConfig(max_steps=5, repeated_tool_call_limit=2),
        )

        with self.assertRaises(LoopDetected):
            agent.run("Repeat forever")

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
