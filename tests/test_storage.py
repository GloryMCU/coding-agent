from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from coding_agent.agent import Agent
from coding_agent.context import ContextBuilder
from coding_agent.conversation import Message
from coding_agent.model import ModelResponse, ToolCall
from coding_agent.storage import SqliteConversationStore
from coding_agent.tools import ToolExecutionResult, create_read_only_registry


class FakeModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
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
        return self.responses.pop(0)


class SqliteConversationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.database = self.workspace / ".coding-agent" / "history.sqlite3"
        (self.workspace / "README.md").write_text(
            "# Durable project\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def create_store_and_session(self) -> tuple[SqliteConversationStore, str]:
        store = SqliteConversationStore(self.database)
        session_id = store.create_session(
            workspace=self.workspace,
            model="fake-model",
            system_prompt="system prompt",
        )
        return store, session_id

    def test_round_trip_projects_provider_messages_after_reopen(self) -> None:
        store, session_id = self.create_store_and_session()
        store.append_user(session_id, "inspect README")
        store.append_assistant(
            session_id,
            ModelResponse.from_parts(
                reasoning_content="I should read it.",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="read_file",
                        arguments={"path": "README.md"},
                    )
                ],
            ),
        )
        claim = store.claim_tool_call(session_id, "call-1")
        self.assertTrue(claim.execute)
        store.finish_tool_call(
            session_id,
            ToolExecutionResult(
                tool_call_id="call-1",
                name="read_file",
                ok=True,
                output={"content": "# Durable project"},
            ),
        )
        store.append_assistant(
            session_id, ModelResponse.from_parts(text="It is durable.")
        )

        reopened = SqliteConversationStore(self.database)
        messages = ContextBuilder(reopened).build(session_id)

        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(messages[2]["reasoning_content"], "I should read it.")
        self.assertEqual(messages[2]["tool_calls"][0]["id"], "call-1")
        self.assertIn("Durable project", messages[3]["content"])

    def test_completed_tool_call_is_not_executed_twice(self) -> None:
        store, session_id = self.create_store_and_session()
        store.append_assistant(
            session_id,
            ModelResponse.from_parts(
                tool_calls=[
                    ToolCall(id="call-1", name="example", arguments={"value": 1})
                ]
            ),
        )
        first = store.claim_tool_call(session_id, "call-1")
        self.assertTrue(first.execute)
        store.finish_tool_call(
            session_id,
            ToolExecutionResult(
                tool_call_id="call-1", name="example", ok=True, output="done"
            ),
        )

        second = store.claim_tool_call(session_id, "call-1")

        self.assertFalse(second.execute)
        self.assertIsNotNone(second.result)
        self.assertEqual(second.result.output, "done")

    def test_pending_and_running_calls_become_interrupted(self) -> None:
        store, session_id = self.create_store_and_session()
        store.append_assistant(
            session_id,
            ModelResponse.from_parts(
                tool_calls=[
                    ToolCall(id="call-1", name="first", arguments={}),
                    ToolCall(id="call-2", name="second", arguments={}),
                ]
            ),
        )
        store.claim_tool_call(session_id, "call-1")

        recovered = store.recover_interrupted_calls(session_id)

        self.assertEqual(recovered, 2)
        tool_parts = [
            part
            for message in store.load_messages(session_id)
            for part in message.parts
            if part.type == "tool"
        ]
        self.assertEqual([part.status for part in tool_parts], ["interrupted"] * 2)
        projected = ContextBuilder(store).build(session_id)
        self.assertEqual(projected[-2]["role"], "tool")
        self.assertIn("interrupted", projected[-2]["content"])

    def test_agent_can_continue_session_after_process_restart(self) -> None:
        store = SqliteConversationStore(self.database)
        first_model = FakeModel([ModelResponse.from_parts(text="First answer")])
        first_agent = Agent(
            model=first_model,
            tools=create_read_only_registry(self.workspace),
            store=store,
            workspace=self.workspace,
            model_name="fake-model",
        )
        first_result = first_agent.run("First question")

        reopened = SqliteConversationStore(self.database)
        second_model = FakeModel([ModelResponse.from_parts(text="Second answer")])
        second_agent = Agent(
            model=second_model,
            tools=create_read_only_registry(self.workspace),
            store=reopened,
            workspace=self.workspace,
            model_name="fake-model",
        )
        second_result = second_agent.run(
            "Second question", session_id=first_result.session_id
        )

        self.assertEqual(second_result.session_id, first_result.session_id)
        request = second_model.requests[0]
        self.assertEqual(
            [message["role"] for message in request],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(request[-2]["content"], "First answer")


if __name__ == "__main__":
    unittest.main()
