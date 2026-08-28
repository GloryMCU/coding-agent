from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from coding_agent.model import ToolCall
from coding_agent.tools import create_read_only_registry


class ReadFileToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        (self.workspace / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
        self.registry = create_read_only_registry(self.workspace)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_reads_selected_lines(self) -> None:
        result = self.registry.execute(
            ToolCall(
                id="call-1",
                name="read_file",
                arguments={"path": "notes.txt", "start_line": 2, "end_line": 3},
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.output["content"], "two\nthree")

    def test_rejects_path_traversal(self) -> None:
        result = self.registry.execute(
            ToolCall(
                id="call-2",
                name="read_file",
                arguments={"path": "../outside.txt"},
            )
        )

        self.assertFalse(result.ok)
        self.assertIn("escapes the workspace", result.error or "")
        message_payload = json.loads(result.to_message()["content"])
        self.assertFalse(message_payload["ok"])

    def test_rejects_unknown_arguments(self) -> None:
        result = self.registry.execute(
            ToolCall(
                id="call-3",
                name="read_file",
                arguments={"path": "notes.txt", "unexpected": True},
            )
        )

        self.assertFalse(result.ok)
        self.assertIn("unknown fields", result.error or "")

    def test_rejects_absolute_path(self) -> None:
        result = self.registry.execute(
            ToolCall(
                id="call-4",
                name="read_file",
                arguments={"path": str((self.workspace / "notes.txt").resolve())},
            )
        )

        self.assertFalse(result.ok)
        self.assertIn("absolute paths are not allowed", result.error or "")


if __name__ == "__main__":
    unittest.main()

