from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from typing import Any

from coding_agent.agent import Agent
from coding_agent.context import (
    ContextBuilder,
    ContextConfig,
    estimate_context_tokens,
)
from coding_agent.conversation import Message
from coding_agent.model import ModelResponse, ToolCall
from coding_agent.storage import SqliteConversationStore
from coding_agent.tools import (
    ToolDefinition,
    ToolExecutionResult,
    ToolRegistry,
    create_read_only_registry,
)


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


def create_verification_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="change_file",
            description="Simulate a workspace mutation.",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=lambda arguments: {"changed": True},
            requires_verification=True,
        )
    )
    registry.register(
        ToolDefinition(
            name="verify_project",
            description="Simulate successful verification.",
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
            handler=lambda arguments: {"ok": True, "skipped": False},
        )
    )
    return registry


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

    def test_agent_restores_pending_verification_after_process_restart(self) -> None:
        store, session_id = self.create_store_and_session()
        registry = create_verification_registry()
        store.append_assistant(
            session_id,
            ModelResponse.from_parts(
                tool_calls=[
                    ToolCall(id="change-1", name="change_file", arguments={})
                ]
            ),
        )
        claim = store.claim_tool_call(session_id, "change-1")
        self.assertTrue(claim.execute)
        store.finish_tool_call(
            session_id,
            ToolExecutionResult(
                tool_call_id="change-1",
                name="change_file",
                ok=True,
                output={"changed": True},
            ),
        )
        model = FakeModel([ModelResponse.from_parts(text="Verified after restart.")])
        agent = Agent(
            model=model,
            tools=registry,
            store=SqliteConversationStore(self.database),
            workspace=self.workspace,
            model_name="fake-model",
        )

        result = agent.run("Continue", session_id=session_id)

        self.assertEqual(result.text, "Verified after restart.")
        stored_tool_names = [
            part.tool_name
            for message in store.load_messages(session_id)
            for part in message.parts
            if part.type == "tool"
        ]
        self.assertEqual(stored_tool_names, ["change_file", "verify_project"])
        self.assertEqual(store.get_session(session_id).status, "completed")

    def test_interrupted_mutation_still_requires_verification_after_restart(self) -> None:
        store, session_id = self.create_store_and_session()
        registry = create_verification_registry()
        store.append_assistant(
            session_id,
            ModelResponse.from_parts(
                tool_calls=[
                    ToolCall(id="change-crashed", name="change_file", arguments={})
                ]
            ),
        )
        claim = store.claim_tool_call(session_id, "change-crashed")
        self.assertTrue(claim.execute)
        (self.workspace / "changed-before-crash.py").write_text(
            "changed = True\n", encoding="utf-8"
        )

        model = FakeModel([ModelResponse.from_parts(text="Verified after crash.")])
        agent = Agent(
            model=model,
            tools=registry,
            store=SqliteConversationStore(self.database),
            workspace=self.workspace,
            model_name="fake-model",
        )

        result = agent.run("Resume after crash", session_id=session_id)

        self.assertEqual(result.text, "Verified after crash.")
        tool_parts = [
            part
            for message in store.load_messages(session_id)
            for part in message.parts
            if part.type == "tool"
        ]
        self.assertEqual(
            [(part.tool_name, part.status) for part in tool_parts],
            [("change_file", "interrupted"), ("verify_project", "completed")],
        )

    def test_all_numbered_schema_migrations_are_applied(self) -> None:
        self.create_store_and_session()

        with closing(sqlite3.connect(self.database)) as connection:
            versions = [
                row[0]
                for row in connection.execute(
                    "SELECT version FROM schema_migration ORDER BY version"
                )
            ]
            summary_table = connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name = 'context_summary'
                """
            ).fetchone()
            fts_table = connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name = 'history_fts'
                """
            ).fetchone()

        self.assertEqual(versions, [1, 2, 3])
        self.assertIsNotNone(summary_table)
        self.assertIsNotNone(fts_table)

    def test_existing_version_one_database_is_upgraded(self) -> None:
        self.database.parent.mkdir(parents=True)
        migration = (
            Path(__file__).parents[1]
            / "src"
            / "coding_agent"
            / "migrations"
            / "001_initial.sql"
        ).read_text(encoding="utf-8")
        with closing(sqlite3.connect(self.database)) as connection:
            connection.executescript(migration)
            connection.execute(
                """
                INSERT INTO session(
                    id, workspace, model, system_prompt, status,
                    created_at, updated_at
                ) VALUES ('legacy-session', ?, 'legacy-model', '', 'completed', ?, ?)
                """,
                (str(self.workspace), "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
            )
            connection.execute(
                """
                INSERT INTO message(
                    id, session_id, seq, role, created_at
                ) VALUES ('legacy-message', 'legacy-session', 1, 'user', ?)
                """,
                ("2026-01-01T00:00:00Z",),
            )
            connection.execute(
                """
                INSERT INTO part(
                    id, session_id, message_id, seq, type, data_json,
                    created_at, updated_at
                ) VALUES (
                    'legacy-part', 'legacy-session', 'legacy-message', 1,
                    'text', '{"text":"LegacyBackfillMarker"}', ?, ?
                )
                """,
                ("2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
            )
            connection.commit()

        upgraded = SqliteConversationStore(self.database)

        with closing(sqlite3.connect(self.database)) as connection:
            versions = [
                row[0]
                for row in connection.execute(
                    "SELECT version FROM schema_migration ORDER BY version"
                )
            ]
        self.assertEqual(versions, [1, 2, 3])
        matches = upgraded.search_history(
            "LegacyBackfillMarker", session_id="legacy-session"
        )
        self.assertEqual(len(matches), 1)

    def test_context_budget_compacts_old_turns_and_persists_summary(self) -> None:
        store, session_id = self.create_store_and_session()
        for index in range(5):
            store.append_user(session_id, f"question-{index} " + "u" * 500)
            store.append_assistant(
                session_id,
                ModelResponse.from_parts(
                    text=f"answer-{index} " + "a" * 500
                ),
            )

        messages = ContextBuilder(
            store,
            ContextConfig(max_tokens=700, summary_max_tokens=160),
        ).build(session_id)

        self.assertLessEqual(estimate_context_tokens(messages), 700)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("UNTRUSTED HISTORICAL RECORD", messages[1]["content"])
        self.assertIn("question-4", messages[-2]["content"])
        self.assertIn("answer-4", messages[-1]["content"])
        self.assertNotIn(
            "question-0",
            "\n".join(str(message.get("content")) for message in messages[2:]),
        )

        reopened = SqliteConversationStore(self.database)
        summary = reopened.get_context_summary(session_id)
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertGreaterEqual(summary.through_seq, 2)
        self.assertEqual(summary.data["schema_version"], 2)

    def test_fts_search_indexes_text_and_completed_tool_output(self) -> None:
        store, session_id = self.create_store_and_session()
        store.append_user(session_id, "Investigate DurableWidget initialization")
        store.append_assistant(
            session_id,
            ModelResponse.from_parts(
                tool_calls=[
                    ToolCall(
                        id="call-search",
                        name="read_file",
                        arguments={"path": "README.md"},
                    )
                ],
            ),
        )
        store.claim_tool_call(session_id, "call-search")
        store.finish_tool_call(
            session_id,
            ToolExecutionResult(
                tool_call_id="call-search",
                name="read_file",
                ok=True,
                output={"content": "UniqueToolObservation"},
            ),
        )

        text_matches = store.search_history(
            "DurableWidget", session_id=session_id
        )
        tool_matches = store.search_history(
            "UniqueToolObservation", session_id=session_id
        )

        self.assertEqual(len(text_matches), 1)
        self.assertEqual(text_matches[0].role, "user")
        self.assertIn("[DurableWidget]", text_matches[0].snippet)
        self.assertEqual(len(tool_matches), 1)
        self.assertEqual(tool_matches[0].part_type, "tool")
        self.assertIn("[UniqueToolObservation]", tool_matches[0].snippet)

    def test_fts_search_filters_session_and_sequence(self) -> None:
        store, first_session = self.create_store_and_session()
        store.append_user(first_session, "sharedmarker first")
        store.append_assistant(
            first_session, ModelResponse.from_parts(text="sharedmarker second")
        )
        second_session = store.create_session(
            workspace=self.workspace,
            model="fake-model",
            system_prompt="system prompt",
        )
        store.append_user(second_session, "sharedmarker other session")

        matches = store.search_history(
            "sharedmarker",
            session_id=first_session,
            before_seq=1,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].session_id, first_session)
        self.assertEqual(matches[0].message_seq, 1)

    def test_fts_trigram_search_matches_chinese_substrings(self) -> None:
        store, session_id = self.create_store_and_session()
        store.append_user(session_id, "数据库迁移方案需要保持向后兼容")

        matches = store.search_history("数据库迁移", session_id=session_id)

        self.assertEqual(len(matches), 1)
        self.assertIn("[数据库迁移]", matches[0].snippet)

    def test_context_summary_recalls_relevant_fts_match(self) -> None:
        store, session_id = self.create_store_and_session()
        store.append_user(
            session_id,
            "rare_architecture_marker means the cache must remain local",
        )
        store.append_assistant(
            session_id,
            ModelResponse.from_parts(text="Recorded the architecture decision."),
        )
        for index in range(5):
            store.append_user(session_id, f"filler-{index} " + "u" * 450)
            store.append_assistant(
                session_id,
                ModelResponse.from_parts(text=f"filler answer {index} " + "a" * 450),
            )
        store.append_user(session_id, "Revisit rare_architecture_marker")
        store.append_assistant(session_id, ModelResponse.from_parts(text="Checking."))

        ContextBuilder(
            store,
            ContextConfig(
                max_tokens=650,
                summary_max_tokens=220,
                history_search_limit=3,
            ),
        ).build(session_id)

        summary = store.get_context_summary(session_id)
        self.assertIsNotNone(summary)
        assert summary is not None
        retrieved = summary.data["retrieved_matches"]
        self.assertTrue(retrieved)
        self.assertTrue(
            any("rare_architecture_marker" in match["snippet"] for match in retrieved)
        )

    def test_context_compaction_never_orphans_a_tool_result(self) -> None:
        store, session_id = self.create_store_and_session()
        store.append_user(session_id, "old request " + "x" * 900)
        store.append_assistant(
            session_id,
            ModelResponse.from_parts(
                tool_calls=[
                    ToolCall(
                        id="call-old",
                        name="read_file",
                        arguments={"path": "README.md"},
                    )
                ],
            ),
        )
        store.claim_tool_call(session_id, "call-old")
        store.finish_tool_call(
            session_id,
            ToolExecutionResult(
                tool_call_id="call-old",
                name="read_file",
                ok=True,
                output={"content": "z" * 900},
            ),
        )
        for index in range(3):
            store.append_user(session_id, f"recent-{index} " + "u" * 350)
            store.append_assistant(
                session_id,
                ModelResponse.from_parts(text=f"done-{index} " + "a" * 350),
            )
        store.append_user(session_id, "current tool request")
        store.append_assistant(
            session_id,
            ModelResponse.from_parts(
                tool_calls=[
                    ToolCall(
                        id="call-recent",
                        name="read_file",
                        arguments={"path": "README.md"},
                    )
                ],
            ),
        )
        store.claim_tool_call(session_id, "call-recent")
        store.finish_tool_call(
            session_id,
            ToolExecutionResult(
                tool_call_id="call-recent",
                name="read_file",
                ok=True,
                output={"content": "current result"},
            ),
        )

        messages = ContextBuilder(
            store,
            ContextConfig(max_tokens=600, summary_max_tokens=128),
        ).build(session_id)

        tool_messages = 0
        for index, message in enumerate(messages):
            if message["role"] == "tool":
                tool_messages += 1
                self.assertGreater(index, 0)
                previous = messages[index - 1]
                self.assertEqual(previous["role"], "assistant")
                self.assertTrue(previous.get("tool_calls"))
        self.assertEqual(tool_messages, 1)


if __name__ == "__main__":
    unittest.main()
