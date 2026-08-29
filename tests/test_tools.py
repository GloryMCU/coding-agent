from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from coding_agent.model import ToolCall
from coding_agent.tools import create_read_only_registry, create_workspace_registry


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


class SearchTextToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        (self.workspace / "src").mkdir()
        (self.workspace / "src" / "main.py").write_text(
            "def greet():\n    return 'Hello World'\n", encoding="utf-8"
        )
        (self.workspace / "src" / "other.txt").write_text(
            "before\nhello hello from text\nafter\n", encoding="utf-8"
        )
        (self.workspace / "src" / "generated.py").write_text(
            "def generated_value():\n    return 'hello'\n", encoding="utf-8"
        )
        (self.workspace / "README.md").write_text(
            "Say hello to the project.\n", encoding="utf-8"
        )
        (self.workspace / "binary.dat").write_bytes(b"hello\x00world")
        self.registry = create_read_only_registry(self.workspace)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def search(self, **arguments: object):
        return self.registry.execute(
            ToolCall(id="search-1", name="search_text", arguments=arguments)
        )

    def test_searches_recursively_with_locations(self) -> None:
        result = self.search(query="hello")

        self.assertTrue(result.ok)
        self.assertEqual(result.output["match_count"], 5)
        self.assertEqual(
            sorted({match["path"] for match in result.output["matches"]}),
            ["README.md", "src/generated.py", "src/main.py", "src/other.txt"],
        )
        main_match = next(
            match
            for match in result.output["matches"]
            if match["path"] == "src/main.py"
        )
        self.assertEqual(main_match["line_number"], 2)
        self.assertEqual(main_match["column_number"], 13)
        self.assertEqual(main_match["end_column_number"], 17)
        self.assertEqual(main_match["matched_text"], "Hello")
        self.assertEqual(result.output["skipped_binary_files"], 1)
        self.assertFalse(result.output["truncated"])

    def test_can_limit_path_glob_and_case(self) -> None:
        result = self.search(
            query="Hello",
            path="src",
            file_pattern="*.py",
            case_sensitive=True,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.output["match_count"], 1)
        self.assertEqual(result.output["matches"][0]["path"], "src/main.py")

    def test_supports_regex_multiple_patterns_exclusions_and_context(self) -> None:
        result = self.search(
            query=r"hello\s+hello",
            path="src",
            regex=True,
            include_patterns=["*.py", "*.txt"],
            exclude_patterns=["generated.py"],
            context_lines=1,
            case_sensitive=True,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.output["match_count"], 1)
        match = result.output["matches"][0]
        self.assertEqual(match["path"], "src/other.txt")
        self.assertEqual(match["matched_text"], "hello hello")
        self.assertEqual(
            [item["line_number"] for item in match["context"]], [1, 2, 3]
        )
        self.assertTrue(match["context"][1]["is_match"])

    def test_rejects_invalid_regex(self) -> None:
        result = self.search(query="(", regex=True)

        self.assertFalse(result.ok)
        self.assertIn("invalid regular expression", result.error or "")

    def test_honors_result_limit(self) -> None:
        result = self.search(query="hello", max_results=2)

        self.assertTrue(result.ok)
        self.assertEqual(result.output["match_count"], 2)
        self.assertTrue(result.output["truncated"])
        self.assertEqual(result.output["truncation_reasons"], ["result_limit"])

    def test_bounds_detailed_output_size(self) -> None:
        registry = create_read_only_registry(
            self.workspace, max_search_output_chars=300
        )
        result = registry.execute(
            ToolCall(
                id="bounded-search",
                name="search_text",
                arguments={"query": "hello", "context_lines": 2},
            )
        )

        self.assertTrue(result.ok)
        self.assertGreaterEqual(result.output["match_count"], 1)
        self.assertTrue(result.output["truncated"])
        self.assertIn("output_size_limit", result.output["truncation_reasons"])

    def test_rejects_empty_query_and_path_traversal(self) -> None:
        empty = self.search(query="")
        escaped = self.search(query="hello", path="../outside")

        self.assertFalse(empty.ok)
        self.assertIn("query must not be empty", empty.error or "")
        self.assertFalse(escaped.ok)
        self.assertIn("escapes the workspace", escaped.error or "")

    def test_schema_exposes_both_read_only_tools(self) -> None:
        names = [schema["function"]["name"] for schema in self.registry.schemas()]

        self.assertEqual(names, ["read_file", "search_text"])


class WorkspaceMutationToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        (self.workspace / "existing.py").write_text(
            "value = 1\nvalue = 1\n", encoding="utf-8"
        )
        self.registry = create_workspace_registry(self.workspace)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def execute(self, name: str, **arguments: object):
        return self.registry.execute(
            ToolCall(id=f"{name}-1", name=name, arguments=arguments)
        )

    def test_creates_file_and_parent_directories(self) -> None:
        result = self.execute(
            "write_file",
            path="new/package/module.py",
            content="answer = 42\n",
            create_parent_dirs=True,
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.output["created"])
        self.assertEqual(result.output["bytes_written"], 12)
        self.assertEqual(
            (self.workspace / "new/package/module.py").read_text(encoding="utf-8"),
            "answer = 42\n",
        )

    def test_requires_explicit_overwrite(self) -> None:
        rejected = self.execute(
            "write_file", path="existing.py", content="replacement\n"
        )

        self.assertFalse(rejected.ok)
        self.assertIn("overwrite=true", rejected.error or "")
        self.assertEqual(
            (self.workspace / "existing.py").read_text(encoding="utf-8"),
            "value = 1\nvalue = 1\n",
        )

        replaced = self.execute(
            "write_file",
            path="existing.py",
            content="replacement\n",
            overwrite=True,
        )
        self.assertTrue(replaced.ok)
        self.assertFalse(replaced.output["created"])

    def test_applies_exact_patch_and_rejects_ambiguous_patch(self) -> None:
        ambiguous = self.execute(
            "apply_patch",
            path="existing.py",
            old_text="value = 1",
            new_text="value = 2",
        )
        self.assertFalse(ambiguous.ok)
        self.assertIn("found 2", ambiguous.error or "")

        applied = self.execute(
            "apply_patch",
            path="existing.py",
            old_text="value = 1",
            new_text="value = 2",
            expected_replacements=2,
        )
        self.assertTrue(applied.ok)
        self.assertEqual(applied.output["replacements"], 2)
        self.assertEqual(
            (self.workspace / "existing.py").read_text(encoding="utf-8"),
            "value = 2\nvalue = 2\n",
        )

    def test_deletes_only_regular_files(self) -> None:
        deleted = self.execute("delete_file", path="existing.py")

        self.assertTrue(deleted.ok)
        self.assertTrue(deleted.output["deleted"])
        self.assertFalse((self.workspace / "existing.py").exists())

        (self.workspace / "directory").mkdir()
        rejected = self.execute("delete_file", path="directory")
        self.assertFalse(rejected.ok)
        self.assertIn("regular file", rejected.error or "")

    def test_mutations_reject_path_traversal(self) -> None:
        for tool_name, arguments in [
            ("write_file", {"path": "../new.py", "content": "x"}),
            (
                "apply_patch",
                {"path": "../old.py", "old_text": "x", "new_text": "y"},
            ),
            ("delete_file", {"path": "../old.py"}),
        ]:
            with self.subTest(tool=tool_name):
                result = self.execute(tool_name, **arguments)
                self.assertFalse(result.ok)
                self.assertIn("escapes the workspace", result.error or "")

    def test_protects_repository_metadata_and_agent_state(self) -> None:
        for protected_path in [".git/config", ".coding-agent/history.sqlite3"]:
            with self.subTest(path=protected_path):
                result = self.execute(
                    "write_file",
                    path=protected_path,
                    content="unsafe",
                    create_parent_dirs=True,
                )
                self.assertFalse(result.ok)
                self.assertIn("cannot be modified", result.error or "")

    def test_workspace_registry_exposes_mutation_tools(self) -> None:
        names = [schema["function"]["name"] for schema in self.registry.schemas()]

        self.assertEqual(
            names,
            ["read_file", "search_text", "write_file", "apply_patch", "delete_file"],
        )


if __name__ == "__main__":
    unittest.main()

