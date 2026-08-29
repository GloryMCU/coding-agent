"""Bounded process execution and repository verification strategies."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ToolArgumentsError
from .policy import WorkspacePolicy


ALLOWED_PROGRAMS = frozenset(
    {
        "python",
        "python3",
        "py",
        "pytest",
        "ruff",
        "black",
        "mypy",
        "npm",
        "npx",
        "pnpm",
        "yarn",
        "cargo",
        "rustfmt",
        "go",
        "gofmt",
        "dotnet",
        "mvn",
        "mvnw",
        "gradle",
        "gradlew",
        "git",
        "powershell",
        "pwsh",
    }
)

_POWERSHELL_DENIED_FRAGMENTS = (
    "remove-item",
    "clear-content",
    "format-volume",
    "stop-computer",
    "restart-computer",
    "invoke-expression",
    "invoke-webrequest",
    "invoke-restmethod",
    "start-process",
    "set-executionpolicy",
    " rm ",
    " del ",
    " rmdir ",
)


@dataclass(frozen=True, slots=True)
class CommandSpec:
    argv: tuple[str, ...]
    cwd: str = "."
    label: str = "command"


class ControlledCommandRunner:
    """Run approved programs without implicit shell parsing."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        max_output_chars: int = 64 * 1024,
    ) -> None:
        if max_output_chars < 1:
            raise ValueError("max_output_chars must be >= 1")
        self.policy = WorkspacePolicy(Path(workspace))
        self.max_output_chars = max_output_chars

    def run(
        self,
        argv: list[str] | tuple[str, ...],
        *,
        cwd: str = ".",
        timeout_s: int = 120,
    ) -> dict[str, Any]:
        normalized = self.validate(argv, cwd=cwd)
        working_directory = self.policy.resolve_existing_path(cwd)
        if not working_directory.is_dir():
            raise ToolArgumentsError("cwd must be a directory")
        started = time.monotonic()
        environment = os.environ.copy()
        for name in list(environment):
            upper = name.upper()
            if any(
                marker in upper
                for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
            ):
                environment.pop(name, None)
        environment.update(
            {
                "GIT_PAGER": "cat",
                "PAGER": "cat",
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_OPTIONAL_LOCKS": "0",
            }
        )
        try:
            completed = subprocess.run(
                normalized,
                cwd=working_directory,
                capture_output=True,
                check=False,
                timeout=timeout_s,
                text=True,
                errors="replace",
                env=environment,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            exit_code: int | None = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            stdout = _decode_timeout_output(exc.stdout)
            stderr = _decode_timeout_output(exc.stderr)
            exit_code = None
            timed_out = True
        stdout, stdout_truncated = _truncate_output(stdout, self.max_output_chars)
        stderr, stderr_truncated = _truncate_output(stderr, self.max_output_chars)
        return {
            "argv": normalized,
            "cwd": self.policy.display_path(working_directory),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_ms": round((time.monotonic() - started) * 1000),
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }

    def validate(
        self, argv: list[str] | tuple[str, ...], *, cwd: str = "."
    ) -> list[str]:
        if not argv:
            raise ToolArgumentsError("argv must contain a program")
        if len(argv) > 100:
            raise ToolArgumentsError("argv must contain at most 100 values")
        normalized = [str(value) for value in argv]
        if any(not value or "\x00" in value or len(value) > 4000 for value in normalized):
            raise ToolArgumentsError("argv values must be non-empty, bounded strings")
        raw_program = Path(normalized[0])
        is_current_python = False
        try:
            is_current_python = raw_program.resolve() == Path(sys.executable).resolve()
        except OSError:
            pass
        if raw_program.name != normalized[0] and not is_current_python:
            raise ToolArgumentsError("program must be an allowlisted name, not a path")
        program = raw_program.stem.casefold()
        if program not in ALLOWED_PROGRAMS:
            raise ToolArgumentsError(f"program is not allowlisted: {normalized[0]}")
        self.policy.resolve_existing_path(cwd)

        if program in {"powershell", "pwsh"}:
            rendered = " " + " ".join(normalized[1:]).casefold() + " "
            if any(fragment in rendered for fragment in _POWERSHELL_DENIED_FRAGMENTS):
                raise ToolArgumentsError("PowerShell command contains a denied operation")
        if program == "git":
            git_action = _git_subcommand(normalized)
            if git_action not in {
                "blame",
                "diff",
                "grep",
                "log",
                "ls-files",
                "rev-parse",
                "show",
                "status",
            }:
                raise ToolArgumentsError(
                    "only allowlisted read-only Git commands may use run_command"
                )
        return normalized


def discover_verification_plan(workspace: Path, kind: str) -> list[CommandSpec]:
    """Choose non-interactive checks from repository markers."""

    requested = {"test", "build", "format_check"} if kind == "all" else {kind}
    plan: list[CommandSpec] = []

    pyproject = workspace / "pyproject.toml"
    if pyproject.exists():
        config = pyproject.read_text(encoding="utf-8", errors="replace")
        if "test" in requested:
            if "[tool.pytest" in config:
                plan.append(CommandSpec((sys.executable, "-m", "pytest"), label="test"))
            elif (workspace / "tests").is_dir():
                plan.append(
                    CommandSpec(
                        (sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"),
                        label="test",
                    )
                )
        if "build" in requested:
            plan.append(CommandSpec((sys.executable, "-m", "build"), label="build"))
        if "format_check" in requested:
            if "[tool.ruff" in config:
                plan.append(CommandSpec(("ruff", "format", "--check", "."), label="format_check"))
            elif "[tool.black" in config:
                plan.append(CommandSpec(("black", "--check", "."), label="format_check"))

    package_json = workspace / "package.json"
    if package_json.exists():
        try:
            scripts = json.loads(package_json.read_text(encoding="utf-8")).get("scripts", {})
        except (OSError, ValueError, AttributeError):
            scripts = {}
        for check, script_names in {
            "test": ("test",),
            "build": ("build",),
            "format_check": ("format:check", "format-check"),
        }.items():
            if check not in requested:
                continue
            script = next((name for name in script_names if name in scripts), None)
            if script:
                plan.append(CommandSpec(("npm", "run", script), label=check))

    if (workspace / "Cargo.toml").exists():
        if "test" in requested:
            plan.append(CommandSpec(("cargo", "test"), label="test"))
        if "build" in requested:
            plan.append(CommandSpec(("cargo", "build"), label="build"))
        if "format_check" in requested:
            plan.append(
                CommandSpec(("cargo", "fmt", "--all", "--", "--check"), label="format_check")
            )

    if (workspace / "go.mod").exists():
        if "test" in requested:
            plan.append(CommandSpec(("go", "test", "./..."), label="test"))
        if "build" in requested:
            plan.append(CommandSpec(("go", "build", "./..."), label="build"))

    return plan


def run_verification_plan(
    runner: ControlledCommandRunner,
    workspace: Path,
    *,
    kind: str,
    timeout_s: int,
) -> dict[str, Any]:
    plan = discover_verification_plan(workspace, kind)
    requested_checks = (
        {"test", "build", "format_check"} if kind == "all" else {kind}
    )
    planned_checks = {command.label for command in plan}
    skipped_checks = sorted(requested_checks - planned_checks)
    results = []
    for command in plan:
        result = runner.run(command.argv, cwd=command.cwd, timeout_s=timeout_s)
        results.append({"check": command.label, **result})
        if result["timed_out"] or result["exit_code"] != 0:
            break
    ok = bool(plan) and len(results) == len(plan) and all(
        item["exit_code"] == 0 and not item["timed_out"] for item in results
    )
    return {
        "kind": kind,
        "commands": [list(item.argv) for item in plan],
        "results": results,
        "checks_planned": len(plan),
        "checks_run": len(results),
        "skipped_checks": skipped_checks,
        "ok": ok,
        "complete": ok and not skipped_checks,
        "skipped": not plan,
        "skip_reason": "no configured verification command was detected" if not plan else None,
    }


def _truncate_output(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[:limit], True


def _decode_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _git_subcommand(argv: list[str]) -> str | None:
    """Find the Git subcommand after supported global options."""

    options_with_value = {
        "-c",
        "-C",
        "--config-env",
        "--exec-path",
        "--git-dir",
        "--namespace",
        "--super-prefix",
        "--work-tree",
    }
    long_value_prefixes = tuple(f"{item}=" for item in options_with_value if item.startswith("--"))
    index = 1
    while index < len(argv):
        value = argv[index]
        if value in options_with_value:
            index += 2
            continue
        if value.startswith(long_value_prefixes) or value.startswith("--exec-path="):
            index += 1
            continue
        if value.startswith("-"):
            index += 1
            continue
        return value.casefold()
    return None
