"""SQLite-backed conversation and tool execution state."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from .model import ModelResponse, ToolCall
from .tools import ToolExecutionResult


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_load(value: str) -> Any:
    return json.loads(value)


@dataclass(frozen=True, slots=True)
class SessionRecord:
    id: str
    workspace: str
    model: str
    system_prompt: str
    status: str
    error: str | None


@dataclass(frozen=True, slots=True)
class StoredPart:
    id: str
    message_id: str
    seq: int
    type: str
    call_id: str | None
    tool_name: str | None
    status: str | None
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StoredMessage:
    id: str
    seq: int
    role: str
    provider_message: dict[str, Any] | None
    parts: tuple[StoredPart, ...]


@dataclass(frozen=True, slots=True)
class ToolCallClaim:
    execute: bool
    status: str
    call: ToolCall
    result: ToolExecutionResult | None = None


class SqliteConversationStore:
    """The durable source of truth for sessions, messages, and tool states."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def _initialize(self) -> None:
        migration = (
            resources.files("coding_agent.migrations")
            .joinpath("001_initial.sql")
            .read_text(encoding="utf-8")
        )
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(migration)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def create_session(
        self,
        *,
        workspace: str | Path,
        model: str,
        system_prompt: str,
        session_id: str | None = None,
    ) -> str:
        identifier = session_id or uuid4().hex
        timestamp = _now()
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO session(
                    id, workspace, model, system_prompt, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'active', ?, ?)
                """,
                (
                    identifier,
                    str(Path(workspace).resolve()),
                    model,
                    system_prompt,
                    timestamp,
                    timestamp,
                ),
            )
        return identifier

    def get_session(self, session_id: str) -> SessionRecord:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM session WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown session: {session_id}")
        return SessionRecord(
            id=row["id"],
            workspace=row["workspace"],
            model=row["model"],
            system_prompt=row["system_prompt"],
            status=row["status"],
            error=row["error"],
        )

    @staticmethod
    def _next_message_seq(connection: sqlite3.Connection, session_id: str) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM message WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert row is not None
        return int(row["next_seq"])

    def append_user(self, session_id: str, content: str) -> str:
        message_id = uuid4().hex
        timestamp = _now()
        with self._transaction() as connection:
            seq = self._next_message_seq(connection, session_id)
            connection.execute(
                """
                INSERT INTO message(id, session_id, seq, role, created_at)
                VALUES (?, ?, ?, 'user', ?)
                """,
                (message_id, session_id, seq, timestamp),
            )
            connection.execute(
                """
                INSERT INTO part(
                    id, session_id, message_id, seq, type, data_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 1, 'text', ?, ?, ?)
                """,
                (
                    uuid4().hex,
                    session_id,
                    message_id,
                    _json_dump({"text": content}),
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                "UPDATE session SET status = 'active', error = NULL, updated_at = ? WHERE id = ?",
                (timestamp, session_id),
            )
        return message_id

    def append_assistant(self, session_id: str, response: ModelResponse) -> str:
        message_id = uuid4().hex
        timestamp = _now()
        with self._transaction() as connection:
            seq = self._next_message_seq(connection, session_id)
            connection.execute(
                """
                INSERT INTO message(
                    id, session_id, seq, role, provider_json, created_at
                ) VALUES (?, ?, ?, 'assistant', ?, ?)
                """,
                (
                    message_id,
                    session_id,
                    seq,
                    _json_dump(response.assistant_message),
                    timestamp,
                ),
            )
            part_seq = 0
            if response.text is not None:
                part_seq += 1
                self._insert_part(
                    connection,
                    session_id=session_id,
                    message_id=message_id,
                    seq=part_seq,
                    part_type="text",
                    data={"text": response.text},
                    timestamp=timestamp,
                )
            reasoning = response.assistant_message.get("reasoning_content")
            if isinstance(reasoning, str):
                part_seq += 1
                self._insert_part(
                    connection,
                    session_id=session_id,
                    message_id=message_id,
                    seq=part_seq,
                    part_type="reasoning",
                    data={"text": reasoning},
                    timestamp=timestamp,
                )
            for call in response.tool_calls:
                part_seq += 1
                self._insert_part(
                    connection,
                    session_id=session_id,
                    message_id=message_id,
                    seq=part_seq,
                    part_type="tool",
                    call_id=call.id,
                    tool_name=call.name,
                    status="pending",
                    data={
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                    timestamp=timestamp,
                )
            connection.execute(
                "UPDATE session SET updated_at = ? WHERE id = ?",
                (timestamp, session_id),
            )
        return message_id

    @staticmethod
    def _insert_part(
        connection: sqlite3.Connection,
        *,
        session_id: str,
        message_id: str,
        seq: int,
        part_type: str,
        data: dict[str, Any],
        timestamp: str,
        call_id: str | None = None,
        tool_name: str | None = None,
        status: str | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO part(
                id, session_id, message_id, seq, type, call_id, tool_name,
                status, data_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid4().hex,
                session_id,
                message_id,
                seq,
                part_type,
                call_id,
                tool_name,
                status,
                _json_dump(data),
                timestamp,
                timestamp,
            ),
        )

    def load_messages(self, session_id: str) -> list[StoredMessage]:
        with closing(self._connect()) as connection:
            message_rows = connection.execute(
                "SELECT * FROM message WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
            part_rows = connection.execute(
                "SELECT * FROM part WHERE session_id = ? ORDER BY message_id, seq",
                (session_id,),
            ).fetchall()

        parts_by_message: dict[str, list[StoredPart]] = {}
        for row in part_rows:
            parts_by_message.setdefault(row["message_id"], []).append(
                StoredPart(
                    id=row["id"],
                    message_id=row["message_id"],
                    seq=row["seq"],
                    type=row["type"],
                    call_id=row["call_id"],
                    tool_name=row["tool_name"],
                    status=row["status"],
                    data=_json_load(row["data_json"]),
                )
            )
        return [
            StoredMessage(
                id=row["id"],
                seq=row["seq"],
                role=row["role"],
                provider_message=(
                    _json_load(row["provider_json"])
                    if row["provider_json"] is not None
                    else None
                ),
                parts=tuple(parts_by_message.get(row["id"], ())),
            )
            for row in message_rows
        ]

    def claim_tool_call(self, session_id: str, call_id: str) -> ToolCallClaim:
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM part
                WHERE session_id = ? AND call_id = ? AND type = 'tool'
                """,
                (session_id, call_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown tool call: {call_id}")
            data = _json_load(row["data_json"])
            call = ToolCall(
                id=call_id,
                name=row["tool_name"],
                arguments=data["arguments"],
            )
            status = row["status"]
            if status == "pending":
                timestamp = _now()
                data["started_at"] = timestamp
                connection.execute(
                    """
                    UPDATE part SET status = 'running', data_json = ?, updated_at = ?
                    WHERE id = ? AND status = 'pending'
                    """,
                    (_json_dump(data), timestamp, row["id"]),
                )
                return ToolCallClaim(execute=True, status="running", call=call)

            result = self._result_from_tool_data(call, status, data)
            return ToolCallClaim(
                execute=False,
                status=status,
                call=call,
                result=result,
            )

    @staticmethod
    def _result_from_tool_data(
        call: ToolCall, status: str, data: dict[str, Any]
    ) -> ToolExecutionResult | None:
        if status == "completed":
            return ToolExecutionResult(
                tool_call_id=call.id,
                name=call.name,
                ok=True,
                output=data.get("output"),
            )
        if status in {"error", "interrupted"}:
            return ToolExecutionResult(
                tool_call_id=call.id,
                name=call.name,
                ok=False,
                error=data.get("error") or f"tool call {status}",
            )
        return None

    def finish_tool_call(
        self, session_id: str, result: ToolExecutionResult
    ) -> None:
        timestamp = _now()
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT id, status, data_json FROM part
                WHERE session_id = ? AND call_id = ? AND type = 'tool'
                """,
                (session_id, result.tool_call_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown tool call: {result.tool_call_id}")
            if row["status"] != "running":
                raise RuntimeError(
                    f"tool call {result.tool_call_id!r} is {row['status']}, not running"
                )
            data = _json_load(row["data_json"])
            data["finished_at"] = timestamp
            if result.ok:
                data["output"] = result.output
                status = "completed"
            else:
                data["error"] = result.error
                status = "error"
            connection.execute(
                """
                UPDATE part SET status = ?, data_json = ?, updated_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (status, _json_dump(data), timestamp, row["id"]),
            )
            connection.execute(
                "UPDATE session SET updated_at = ? WHERE id = ?",
                (timestamp, session_id),
            )

    def recover_interrupted_calls(self, session_id: str) -> int:
        """Close calls that could have been executing when the process stopped.

        They are not retried automatically because future tools may have side effects.
        """

        timestamp = _now()
        recovered = 0
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT id, status, data_json FROM part
                WHERE session_id = ? AND type = 'tool'
                  AND status IN ('pending', 'running')
                """,
                (session_id,),
            ).fetchall()
            for row in rows:
                data = _json_load(row["data_json"])
                data["finished_at"] = timestamp
                data["error"] = (
                    "tool execution was interrupted before a durable result was stored"
                )
                connection.execute(
                    """
                    UPDATE part
                    SET status = 'interrupted', data_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (_json_dump(data), timestamp, row["id"]),
                )
                recovered += 1
            if recovered:
                connection.execute(
                    "UPDATE session SET updated_at = ? WHERE id = ?",
                    (timestamp, session_id),
                )
        return recovered

    def set_session_status(
        self, session_id: str, status: str, *, error: str | None = None
    ) -> None:
        if status not in {"active", "completed", "error"}:
            raise ValueError(f"invalid session status: {status}")
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE session SET status = ?, error = ?, updated_at = ? WHERE id = ?
                """,
                (status, error, _now(), session_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown session: {session_id}")
