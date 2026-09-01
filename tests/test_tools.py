from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_agent.model import ToolCall
from coding_agent.permissions import DenyApprovalPolicy, PermissionRequest
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

    def test_reads_lines_beyond_the_output_byte_limit(self) -> None:
        large = self.workspace / "large.txt"
        large.write_text(
            "first " + "x" * 80_000 + "\nsecond\ntarget\n",
            encoding="utf-8",
        )
        registry = create_read_only_registry(self.workspace, max_file_bytes=64)

        result = registry.execute(
            ToolCall(
                id="call-large",
                name="read_file",
                arguments={"path": "large.txt", "start_line": 3, "end_line": 3},
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.output["content"], "target")
        self.assertFalse(result.output["truncated"])

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
        if result.output["search_backend"] == "python":
            self.assertEqual(result.output["skipped_binary_files"], 1)
            self.assertTrue(result.output["search_statistics_complete"])
        else:
            self.assertEqual(result.output["search_backend"], "ripgrep")
            self.assertFalse(result.output["search_statistics_complete"])
        self.assertFalse(result.output["truncated"])

    @unittest.skipUnless(shutil.which("rg"), "ripgrep is not installed")
    def test_prefers_ripgrep_and_respects_gitignore(self) -> None:
        subprocess.run(
            ["git", "init", "-q", str(self.workspace)],
            check=True,
            capture_output=True,
        )
        (self.workspace / ".gitignore").write_text(
            "ignored.txt\n", encoding="utf-8"
        )
        (self.workspace / "ignored.txt").write_text(
            "hello from ignored file\n", encoding="utf-8"
        )

        result = self.search(query="hello")

        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.output["search_backend"], "ripgrep")
        self.assertNotIn(
            "ignored.txt", {match["path"] for match in result.output["matches"]}
        )

    def test_falls_back_to_python_when_ripgrep_is_unavailable(self) -> None:
        with patch("coding_agent.tools.shutil.which", return_value=None):
            result = self.search(query="hello")

        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.output["search_backend"], "python")
        self.assertEqual(
            result.output["ripgrep_fallback_reason"], "ripgrep_not_found"
        )
        self.assertTrue(result.output["search_statistics_complete"])

    def test_refuses_a_workspace_local_ripgrep_executable(self) -> None:
        fake_ripgrep = self.workspace / "rg.exe"
        fake_ripgrep.write_bytes(b"not an executable")
        with patch(
            "coding_agent.tools.shutil.which", return_value=str(fake_ripgrep)
        ):
            result = self.search(query="hello")

        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.output["search_backend"], "python")
        self.assertEqual(
            result.output["ripgrep_fallback_reason"], "ripgrep_inside_workspace"
        )

    @unittest.skipUnless(shutil.which("rg"), "ripgrep is not installed")
    def test_falls_back_for_python_regex_unsupported_by_ripgrep(self) -> None:
        result = self.search(query=r"(?=hello)hello", regex=True)

        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.output["search_backend"], "python")
        self.assertEqual(
            result.output["ripgrep_fallback_reason"], "ripgrep_search_failed"
        )
        self.assertGreater(result.output["match_count"], 0)

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

    def test_reports_character_columns_for_unicode_text(self) -> None:
        (self.workspace / "unicode.txt").write_text(
            "前缀 hello\n", encoding="utf-8"
        )

        result = self.search(query="hello", path="unicode.txt")

        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.output["matches"][0]["column_number"], 4)
        self.assertEqual(result.output["matches"][0]["end_column_number"], 8)

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

    def test_schema_exposes_read_only_tools(self) -> None:
        names = [schema["function"]["name"] for schema in self.registry.schemas()]

        self.assertEqual(
            names,
            [
                "read_file",
                "list_files",
                "glob_files",
                "git_status",
                "git_diff",
                "git_log",
                "search_text",
            ],
        )


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
        for protected_path in [
            ".git/config",
            ".coding-agent/history.sqlite3",
            ".coding-agent-verification.toml",
        ]:
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
            [
                "read_file",
                "list_files",
                "glob_files",
                "git_status",
                "git_diff",
                "git_log",
                "search_text",
                "write_file",
                "apply_patch",
                "delete_file",
                "run_command",
                "verify_project",
            ],
        )
        for tool_name in ["write_file", "apply_patch", "delete_file", "run_command"]:
            with self.subTest(tool=tool_name):
                self.assertTrue(self.registry.requires_verification(tool_name))
        self.assertFalse(self.registry.requires_verification("read_file"))
        self.assertFalse(self.registry.requires_verification("verify_project"))


class FileDiscoveryToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        (self.workspace / ".gitignore").write_text(
            "*.log\nbuild/\n", encoding="utf-8"
        )
        (self.workspace / "visible.py").write_text("pass\n", encoding="utf-8")
        (self.workspace / "ignored.log").write_text("ignored\n", encoding="utf-8")
        (self.workspace / "src").mkdir()
        (self.workspace / "src" / "nested.py").write_text("pass\n", encoding="utf-8")
        (self.workspace / "build").mkdir()
        (self.workspace / "build" / "artifact.bin").write_bytes(b"ignored")
        self.registry = create_read_only_registry(self.workspace)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def execute(self, name: str, **arguments: object):
        return self.registry.execute(
            ToolCall(id=f"{name}-1", name=name, arguments=arguments)
        )

    def test_list_files_respects_gitignore_and_recursion(self) -> None:
        result = self.execute("list_files", recursive=True)

        self.assertTrue(result.ok)
        paths = [item["path"] for item in result.output["files"]]
        self.assertIn("visible.py", paths)
        self.assertIn("src/nested.py", paths)
        self.assertNotIn("ignored.log", paths)
        self.assertNotIn("build/artifact.bin", paths)

    def test_glob_files_matches_visible_files_only(self) -> None:
        result = self.execute("glob_files", patterns=["*.py"])

        self.assertTrue(result.ok)
        self.assertEqual(
            [item["path"] for item in result.output["files"]],
            ["src/nested.py", "visible.py"],
        )


class GitReadOnlyToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        subprocess.run(["git", "init", "-q", str(self.workspace)], check=True)
        subprocess.run(
            ["git", "-C", str(self.workspace), "config", "user.email", "test@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.workspace), "config", "user.name", "Test User"],
            check=True,
        )
        (self.workspace / "tracked.txt").write_text("before\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(self.workspace), "add", "tracked.txt"], check=True
        )
        subprocess.run(
            ["git", "-C", str(self.workspace), "commit", "-q", "-m", "initial"],
            check=True,
        )
        (self.workspace / "tracked.txt").write_text("after\n", encoding="utf-8")
        self.registry = create_read_only_registry(self.workspace)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def execute(self, name: str, **arguments: object):
        return self.registry.execute(ToolCall(id=name, name=name, arguments=arguments))

    def test_status_diff_and_log_are_bounded_read_only_operations(self) -> None:
        status = self.execute("git_status")
        diff = self.execute("git_diff", paths=["tracked.txt"])
        log = self.execute("git_log", max_count=1)

        self.assertTrue(status.ok)
        self.assertIn("tracked.txt", status.output["stdout"])
        self.assertTrue(diff.ok)
        self.assertIn("-before", diff.output["stdout"])
        self.assertTrue(log.ok)
        self.assertIn("initial", log.output["stdout"])

    def test_diff_and_log_accept_a_deleted_path(self) -> None:
        (self.workspace / "tracked.txt").unlink()

        diff = self.execute("git_diff", paths=["tracked.txt"])
        log = self.execute("git_log", path="tracked.txt", max_count=1)

        self.assertTrue(diff.ok)
        self.assertIn("deleted file mode", diff.output["stdout"])
        self.assertTrue(log.ok)
        self.assertIn("initial", log.output["stdout"])


class RecordingApprovalPolicy:
    def __init__(self, approved: bool) -> None:
        self.approved = approved
        self.requests: list[PermissionRequest] = []

    def approve(self, request: PermissionRequest) -> bool:
        self.requests.append(request)
        return self.approved


class ApprovalAndCommandToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        (self.workspace / "input.txt").write_text("safe\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_read_is_unprompted_but_write_requires_approval(self) -> None:
        policy = RecordingApprovalPolicy(approved=False)
        registry = create_workspace_registry(
            self.workspace, approval_policy=policy
        )

        read = registry.execute(
            ToolCall(id="read", name="read_file", arguments={"path": "input.txt"})
        )
        write = registry.execute(
            ToolCall(
                id="write",
                name="write_file",
                arguments={"path": "denied.txt", "content": "no\n"},
            )
        )

        self.assertTrue(read.ok)
        self.assertFalse(write.ok)
        self.assertIn("not approved", write.error or "")
        self.assertFalse((self.workspace / "denied.txt").exists())
        self.assertEqual([request.tool_name for request in policy.requests], ["write_file"])

    def test_controlled_command_uses_argv_and_captures_output(self) -> None:
        policy = RecordingApprovalPolicy(approved=True)
        registry = create_workspace_registry(
            self.workspace, approval_policy=policy
        )
        result = registry.execute(
            ToolCall(
                id="command",
                name="run_command",
                arguments={"argv": [sys.executable, "-c", "print('verified')"]},
            )
        )

        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.output["exit_code"], 0)
        self.assertEqual(result.output["stdout"], "verified\n")
        self.assertFalse(result.output["sandboxed"])
        self.assertEqual(policy.requests[0].tool_name, "run_command")

    def test_command_receives_a_minimal_environment_without_secrets(self) -> None:
        registry = create_workspace_registry(
            self.workspace, approval_policy=RecordingApprovalPolicy(approved=True)
        )
        code = (
            "import os; "
            "print(os.getenv('CODING_AGENT_TEST_SECRET', 'missing')); "
            "print(os.getenv('CI'))"
        )

        with patch.dict(os.environ, {"CODING_AGENT_TEST_SECRET": "do-not-leak"}):
            result = registry.execute(
                ToolCall(
                    id="environment",
                    name="run_command",
                    arguments={"argv": [sys.executable, "-c", code]},
                )
            )

        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.output["stdout"], "missing\n1\n")

    def test_command_timeout_terminates_the_process_group(self) -> None:
        registry = create_workspace_registry(
            self.workspace, approval_policy=RecordingApprovalPolicy(approved=True)
        )

        result = registry.execute(
            ToolCall(
                id="timeout",
                name="run_command",
                arguments={
                    "argv": [sys.executable, "-c", "import time; time.sleep(30)"],
                    "timeout_s": 1,
                },
            )
        )

        self.assertTrue(result.ok, result.error)
        self.assertTrue(result.output["timed_out"])
        self.assertIsNone(result.output["exit_code"])

    def test_command_denial_and_powershell_danger_filter(self) -> None:
        denied = create_workspace_registry(
            self.workspace, approval_policy=DenyApprovalPolicy()
        ).execute(
            ToolCall(
                id="denied",
                name="run_command",
                arguments={"argv": ["python", "--version"]},
            )
        )
        dangerous = create_workspace_registry(
            self.workspace, approval_policy=RecordingApprovalPolicy(True)
        ).execute(
            ToolCall(
                id="dangerous",
                name="run_command",
                arguments={
                    "argv": ["powershell", "-NoProfile", "-Command", "Remove-Item x"]
                },
            )
        )

        self.assertFalse(denied.ok)
        self.assertFalse(dangerous.ok)
        self.assertIn("denied operation", dangerous.error or "")

        git_bypass = create_workspace_registry(
            self.workspace, approval_policy=RecordingApprovalPolicy(True)
        ).execute(
            ToolCall(
                id="git-bypass",
                name="run_command",
                arguments={"argv": ["git", "-c", "color.ui=false", "push"]},
            )
        )
        self.assertFalse(git_bypass.ok)
        self.assertIn("read-only Git", git_bypass.error or "")

    def test_verify_project_runs_detected_tests(self) -> None:
        (self.workspace / "pyproject.toml").write_text(
            "[build-system]\nrequires = []\nbuild-backend = 'setuptools.build_meta'\n",
            encoding="utf-8",
        )
        (self.workspace / "tests").mkdir()
        (self.workspace / "tests" / "test_sample.py").write_text(
            "import unittest\n\nclass Sample(unittest.TestCase):\n"
            "    def test_ok(self):\n        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        registry = create_workspace_registry(
            self.workspace, approval_policy=RecordingApprovalPolicy(True)
        )

        result = registry.execute(
            ToolCall(
                id="verify", name="verify_project", arguments={"kind": "test"}
            )
        )

        self.assertTrue(result.ok, result.error)
        self.assertTrue(result.output["ok"], result.output)
        self.assertEqual(result.output["results"][0]["check"], "test")


if __name__ == "__main__":
    unittest.main()

