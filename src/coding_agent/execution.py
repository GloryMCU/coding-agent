"""Bounded process execution and repository verification strategies."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import SandboxUnavailableError, ToolArgumentsError
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


_CONTAINER_IMAGE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@:+-]{0,254}\Z")
_MEMORY_LIMIT_PATTERN = re.compile(r"[1-9][0-9]*(?:[kKmMgG](?:[bB])?)?\Z")
_CONTAINER_PATH_PATTERN = re.compile(r"/[A-Za-z0-9._/-]+\Z")


@dataclass(frozen=True, slots=True)
class ContainerSandbox:
    """Run commands in a locked-down, credential-free OCI container."""

    runtime: str
    image: str
    memory: str = "2g"
    cpus: str = "2"
    pids_limit: int = 256
    workspace_target: str = "/workspace"

    def __post_init__(self) -> None:
        if not self.runtime or "\x00" in self.runtime:
            raise ValueError("container runtime must be a non-empty executable")
        if not _CONTAINER_IMAGE_PATTERN.fullmatch(self.image):
            raise ValueError("sandbox image is not a valid container image reference")
        if not _MEMORY_LIMIT_PATTERN.fullmatch(self.memory):
            raise ValueError("sandbox memory must be a positive container memory value")
        try:
            cpu_count = float(self.cpus)
        except ValueError as exc:
            raise ValueError("sandbox cpus must be numeric") from exc
        if not 0 < cpu_count <= 64 or not re.fullmatch(
            r"[0-9]+(?:\.[0-9]+)?", self.cpus
        ):
            raise ValueError("sandbox cpus must be between 0 and 64")
        if self.pids_limit < 16:
            raise ValueError("pids_limit must be >= 16")
        if not _CONTAINER_PATH_PATTERN.fullmatch(self.workspace_target):
            raise ValueError("workspace_target must be a simple absolute container path")

    @property
    def backend_name(self) -> str:
        return Path(self.runtime).stem.casefold()

    def build_argv(
        self,
        workspace: Path,
        working_directory: Path,
        command: list[str],
        *,
        container_name: str,
    ) -> list[str]:
        """Build runtime argv with all model-controlled values after the image."""

        workspace = workspace.resolve(strict=True)
        working_directory = working_directory.resolve(strict=True)
        relative_cwd = working_directory.relative_to(workspace)
        host_workspace = workspace.as_posix()
        if "," in host_workspace:
            raise SandboxUnavailableError(
                "container isolation does not support a workspace path containing ','"
            )
        container_cwd = self.workspace_target
        if relative_cwd.parts:
            container_cwd += "/" + relative_cwd.as_posix()

        container_command = list(command)
        if _is_current_python(container_command[0]):
            container_command[0] = "python"

        runtime_argv = [
            self.runtime,
            "run",
            "--rm",
            "--name",
            container_name,
            "--pull=never",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--pids-limit={self.pids_limit}",
            f"--memory={self.memory}",
            f"--memory-swap={self.memory}",
            f"--cpus={self.cpus}",
            "--ulimit=nofile=1024:1024",
            f"--user={_container_user()}",
            "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=512m,mode=1777",
            (
                f"--tmpfs={self.workspace_target}/.coding-agent:"
                "ro,nosuid,nodev,noexec,size=1m,mode=000"
            ),
            "--mount",
            f"type=bind,src={host_workspace},dst={self.workspace_target}",
        ]
        git_metadata = workspace / ".git"
        if git_metadata.is_symlink():
            raise SandboxUnavailableError(
                "container isolation refuses a symbolic-link .git entry"
            )
        if git_metadata.exists():
            git_source = git_metadata.absolute().as_posix()
            if "," in git_source:
                raise SandboxUnavailableError(
                    "container isolation does not support a Git path containing ','"
                )
            runtime_argv.extend(
                [
                    "--mount",
                    (
                        f"type=bind,src={git_source},"
                        f"dst={self.workspace_target}/.git,readonly"
                    ),
                ]
            )
        runtime_argv.extend(
            [
                "--workdir",
                container_cwd,
                "--env=CI=1",
                "--env=HOME=/tmp",
                "--env=GIT_PAGER=cat",
                "--env=PAGER=cat",
                "--env=GIT_TERMINAL_PROMPT=0",
                "--env=GIT_OPTIONAL_LOCKS=0",
                "--env=PIP_NO_INPUT=1",
                "--env=PYTHONNOUSERSITE=1",
                self.image,
                *container_command,
            ]
        )
        return runtime_argv

    def run(
        self,
        workspace: Path,
        working_directory: Path,
        command: list[str],
        *,
        timeout_s: int,
        max_output_chars: int,
    ) -> dict[str, Any]:
        container_name = f"coding-agent-{uuid.uuid4().hex}"
        runtime_argv = self.build_argv(
            workspace,
            working_directory,
            command,
            container_name=container_name,
        )
        started = time.monotonic()
        process_kwargs = _process_group_kwargs()
        process = subprocess.Popen(
            runtime_argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            env=_container_runtime_environment(),
            **process_kwargs,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
            exit_code: int | None = process.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            _force_remove_container(self.runtime, container_name)
            _terminate_process_tree(process)
            stdout, stderr = process.communicate()
            exit_code = None
            timed_out = True
        stdout, stdout_truncated = _truncate_output(stdout, max_output_chars)
        stderr, stderr_truncated = _truncate_output(stderr, max_output_chars)
        return {
            "argv": command,
            "cwd": working_directory.relative_to(workspace).as_posix() or ".",
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_ms": round((time.monotonic() - started) * 1000),
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "sandboxed": True,
            "sandbox_backend": self.backend_name,
            "network_access": False,
        }


def discover_container_sandbox(
    *,
    image: str | None,
    runtime: str = "auto",
) -> ContainerSandbox:
    """Resolve and verify a local OCI runtime and an already-present image."""

    if not image:
        raise SandboxUnavailableError(
            "OS sandbox is required; set CODING_AGENT_SANDBOX_IMAGE or "
            "pass --sandbox-image with a trusted, locally available image"
        )
    if runtime not in {"auto", "docker", "podman"}:
        raise SandboxUnavailableError("sandbox runtime must be auto, docker, or podman")
    candidates = ("docker", "podman") if runtime == "auto" else (runtime,)
    resolved = None
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved is not None:
            break
    if resolved is None:
        raise SandboxUnavailableError(
            "OS sandbox is required, but neither Docker nor Podman is available"
        )
    sandbox = ContainerSandbox(runtime=resolved, image=image)
    try:
        runtime_check = subprocess.run(
            [resolved, "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=10,
            check=False,
            env=_container_runtime_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SandboxUnavailableError(
            f"container runtime {sandbox.backend_name!r} is not ready: {exc}"
        ) from exc
    if runtime_check.returncode != 0:
        detail = (runtime_check.stderr or "runtime info failed").strip()[:500]
        raise SandboxUnavailableError(
            f"container runtime {sandbox.backend_name!r} is not ready: {detail}"
        )
    try:
        image_check = subprocess.run(
            [resolved, "image", "inspect", "--format", "{{.Os}}", image],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=15,
            check=False,
            env=_container_runtime_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SandboxUnavailableError(
            f"could not inspect sandbox image {image!r}: {exc}"
        ) from exc
    if image_check.returncode != 0:
        raise SandboxUnavailableError(
            f"sandbox image {image!r} is not available locally; images are never "
            "pulled automatically"
        )
    image_os = image_check.stdout.strip().casefold()
    if image_os != "linux":
        raise SandboxUnavailableError(
            f"sandbox image {image!r} must be a Linux image, got {image_os or 'unknown'}"
        )
    return sandbox


class ControlledCommandRunner:
    """Run approved programs without implicit shell parsing."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        max_output_chars: int = 64 * 1024,
        sandbox: ContainerSandbox | None = None,
    ) -> None:
        if max_output_chars < 1:
            raise ValueError("max_output_chars must be >= 1")
        self.policy = WorkspacePolicy(Path(workspace))
        self.max_output_chars = max_output_chars
        self.sandbox = sandbox

    @property
    def sandboxed(self) -> bool:
        return self.sandbox is not None

    def run(
        self,
        argv: list[str] | tuple[str, ...],
        *,
        cwd: str = ".",
        timeout_s: int = 120,
    ) -> dict[str, Any]:
        if not 1 <= timeout_s <= 300:
            raise ToolArgumentsError("timeout_s must be between 1 and 300")
        normalized = self.validate(argv, cwd=cwd)
        working_directory = self.policy.resolve_existing_path(cwd)
        if not working_directory.is_dir():
            raise ToolArgumentsError("cwd must be a directory")
        if self.sandbox is not None:
            return self.sandbox.run(
                self.policy.root,
                working_directory,
                normalized,
                timeout_s=timeout_s,
                max_output_chars=self.max_output_chars,
            )
        started = time.monotonic()
        environment = _sanitized_environment()
        process_kwargs = _process_group_kwargs()
        try:
            process = subprocess.Popen(
                normalized,
                cwd=working_directory,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                env=environment,
                **process_kwargs,
            )
            stdout, stderr = process.communicate(timeout=timeout_s)
            exit_code: int | None = process.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process)
            stdout, stderr = process.communicate()
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
            # The runner constrains argv, cwd, environment, duration, and output,
            # but does not claim filesystem or network isolation.
            "sandboxed": False,
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


def _process_group_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _container_user() -> str:
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        uid = os.getuid()
        gid = os.getgid()
        if uid != 0:
            return f"{uid}:{gid}"
    return "65532:65532"


def _is_current_python(value: str) -> bool:
    try:
        return Path(value).resolve() == Path(sys.executable).resolve()
    except OSError:
        return False


def _force_remove_container(runtime: str, container_name: str) -> None:
    """Best-effort cleanup when the client process exceeds its deadline."""

    try:
        subprocess.run(
            [runtime, "rm", "--force", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
            env=_container_runtime_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _container_runtime_environment() -> dict[str, str]:
    """Add only host-side settings required to reach a local/rootless runtime."""

    environment = _sanitized_environment()
    runtime_settings = {
        "CONTAINER_HOST",
        "CONTAINERS_CONF",
        "DOCKER_CERT_PATH",
        "DOCKER_CONFIG",
        "DOCKER_CONTEXT",
        "DOCKER_HOST",
        "DOCKER_TLS_VERIFY",
        "HOME",
        "USERPROFILE",
        "XDG_CONFIG_HOME",
        "XDG_RUNTIME_DIR",
    }
    environment.update(
        {
            name: value
            for name, value in os.environ.items()
            if name.upper() in runtime_settings
        }
    )
    return environment


def _sanitized_environment() -> dict[str, str]:
    """Build a minimal subprocess environment instead of copying all secrets."""

    allowed = {
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "NUMBER_OF_PROCESSORS",
        "OS",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TERM",
        "TMP",
        "WINDIR",
    }
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.upper() in allowed
    }
    environment.update(
        {
            "CI": "1",
            "GIT_PAGER": "cat",
            "PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Best-effort termination of the process group created for one command."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
                env=_sanitized_environment(),
            )
            if completed.returncode == 0:
                return
            process.kill()
            return
        except (OSError, subprocess.SubprocessError):
            process.kill()
            return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        process.kill()


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
