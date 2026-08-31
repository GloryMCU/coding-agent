from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_agent.errors import SandboxUnavailableError, ToolArgumentsError
from coding_agent.execution import (
    ContainerSandbox,
    ControlledCommandRunner,
    discover_container_sandbox,
    discover_verification_plan,
)
from coding_agent.policy import VERIFICATION_CONFIG_NAME


class ContainerSandboxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        (self.workspace / "src").mkdir()
        (self.workspace / ".git").mkdir()
        self.sandbox = ContainerSandbox(
            runtime="docker",
            image="local/coding-agent:test",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_builds_locked_down_container_invocation(self) -> None:
        (self.workspace / VERIFICATION_CONFIG_NAME).write_text(
            "version = 1\ncommands = []\n", encoding="utf-8"
        )
        argv = self.sandbox.build_argv(
            self.workspace,
            self.workspace / "src",
            ["python", "-m", "pytest"],
            container_name="coding-agent-test",
        )

        image_index = argv.index("local/coding-agent:test")
        self.assertEqual(argv[image_index + 1 :], ["python", "-m", "pytest"])
        self.assertIn("--pull=never", argv[:image_index])
        self.assertIn("--network=none", argv[:image_index])
        self.assertIn("--read-only", argv[:image_index])
        self.assertIn("--cap-drop=ALL", argv[:image_index])
        self.assertIn("--security-opt=no-new-privileges", argv[:image_index])
        self.assertIn("--pids-limit=256", argv[:image_index])
        self.assertIn("--memory=2g", argv[:image_index])
        self.assertIn("--memory-swap=2g", argv[:image_index])
        self.assertIn("--cpus=2", argv[:image_index])
        self.assertIn("--workdir", argv[:image_index])
        self.assertEqual(argv[argv.index("--workdir") + 1], "/workspace/src")
        mounts = [value for value in argv if value.startswith("type=bind")]
        self.assertTrue(any("dst=/workspace" in value for value in mounts))
        self.assertTrue(
            any("dst=/workspace/.git,readonly" in value for value in mounts)
        )
        self.assertTrue(
            any(value.startswith("--tmpfs=/workspace/.coding-agent:") for value in argv)
        )
        self.assertTrue(
            any(
                value.endswith(
                    f"dst=/workspace/{VERIFICATION_CONFIG_NAME},readonly"
                )
                for value in mounts
            )
        )

    def test_model_arguments_cannot_become_runtime_options(self) -> None:
        command = ["python", "-c", "print('x')", "--network=host"]
        argv = self.sandbox.build_argv(
            self.workspace,
            self.workspace,
            command,
            container_name="coding-agent-test",
        )

        image_index = argv.index("local/coding-agent:test")
        self.assertEqual(argv[image_index + 1 :], command)

    def test_maps_host_python_executable_to_container_python(self) -> None:
        argv = self.sandbox.build_argv(
            self.workspace,
            self.workspace,
            [sys.executable, "-m", "unittest"],
            container_name="coding-agent-test",
        )

        image_index = argv.index("local/coding-agent:test")
        self.assertEqual(argv[image_index + 1], "python")

    def test_rejects_image_option_injection(self) -> None:
        with self.assertRaises(ValueError):
            ContainerSandbox(runtime="docker", image="--privileged")
        with self.assertRaises(ValueError):
            ContainerSandbox(runtime="docker", image="image with spaces")

    def test_discovery_fails_closed_without_runtime(self) -> None:
        with patch("coding_agent.execution.shutil.which", return_value=None):
            with self.assertRaisesRegex(
                SandboxUnavailableError, "neither Docker nor Podman"
            ):
                discover_container_sandbox(image="local/agent:test")

    def test_discovery_requires_an_explicit_trusted_image(self) -> None:
        with self.assertRaisesRegex(SandboxUnavailableError, "SANDBOX_IMAGE"):
            discover_container_sandbox(image=None)

    def test_discovery_checks_runtime_and_local_linux_image_without_pull(self) -> None:
        completed = [
            subprocess.CompletedProcess(["docker", "info"], 0, "", ""),
            subprocess.CompletedProcess(
                ["docker", "image", "inspect"], 0, "linux\n", ""
            ),
        ]
        with (
            patch("coding_agent.execution.shutil.which", return_value="docker"),
            patch("coding_agent.execution.subprocess.run", side_effect=completed) as run,
        ):
            sandbox = discover_container_sandbox(
                image="local/agent:test", runtime="docker"
            )

        self.assertEqual(sandbox.image, "local/agent:test")
        inspect_argv = run.call_args_list[1].args[0]
        self.assertEqual(inspect_argv[:3], ["docker", "image", "inspect"])
        self.assertNotIn("pull", inspect_argv)

    def test_discovery_rejects_a_non_linux_image(self) -> None:
        completed = [
            subprocess.CompletedProcess(["docker", "info"], 0, "", ""),
            subprocess.CompletedProcess(
                ["docker", "image", "inspect"], 0, "windows\n", ""
            ),
        ]
        with (
            patch("coding_agent.execution.shutil.which", return_value="docker"),
            patch("coding_agent.execution.subprocess.run", side_effect=completed),
        ):
            with self.assertRaisesRegex(SandboxUnavailableError, "Linux image"):
                discover_container_sandbox(
                    image="local/agent:test", runtime="docker"
                )

    def test_runner_reports_that_container_backend_is_enabled(self) -> None:
        runner = ControlledCommandRunner(self.workspace, sandbox=self.sandbox)

        self.assertTrue(runner.sandboxed)

    def test_explicit_verification_config_is_authoritative(self) -> None:
        (self.workspace / VERIFICATION_CONFIG_NAME).write_text(
            """version = 1

[[commands]]
kind = "test"
argv = ["python", "-m", "unittest", "discover", "-s", "../tests", "-v"]
cwd = "src"
""",
            encoding="utf-8",
        )
        (self.workspace / "pyproject.toml").write_text(
            "[build-system]\nrequires = []\n",
            encoding="utf-8",
        )

        plan = discover_verification_plan(self.workspace, "all")

        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].label, "test")
        self.assertEqual(plan[0].cwd, "src")
        self.assertEqual(plan[0].argv[0], "python")

    def test_default_python_plan_uses_src_layout_without_implicit_build(self) -> None:
        (self.workspace / "tests").mkdir()
        (self.workspace / "pyproject.toml").write_text(
            "[build-system]\nrequires = []\n",
            encoding="utf-8",
        )

        plan = discover_verification_plan(self.workspace, "all")

        self.assertEqual([command.label for command in plan], ["test"])
        self.assertEqual(plan[0].cwd, "src")
        self.assertIn("../tests", plan[0].argv)

    def test_invalid_verification_config_is_rejected(self) -> None:
        (self.workspace / VERIFICATION_CONFIG_NAME).write_text(
            """version = 1

[[commands]]
kind = "test"
argv = ["sh", "-c", "exit 0"]
""",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ToolArgumentsError, "not allowlisted"):
            discover_verification_plan(self.workspace, "all")


if __name__ == "__main__":
    unittest.main()
