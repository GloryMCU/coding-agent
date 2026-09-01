from __future__ import annotations

import io
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from coding_agent.cli import _run_plain, build_parser
from coding_agent.errors import ModelRequestError


class CliDefaultsTests(unittest.TestCase):
    def test_defaults_to_deepseek_v4_pro_configuration(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            args = build_parser().parse_args(["Inspect README.md"])

        self.assertEqual(args.base_url, "https://api.deepseek.com")
        self.assertEqual(args.reasoning_effort, "high")
        self.assertTrue(args.thinking)
        self.assertEqual(args.model_timeout_s, 60.0)
        self.assertEqual(args.max_model_retries, 2)
        self.assertEqual(args.retry_base_delay_s, 0.5)
        self.assertEqual(args.max_context_tokens, 131_072)
        self.assertEqual(args.context_summary_tokens, 8_192)
        self.assertEqual(args.history_search_limit, 5)
        self.assertEqual(args.approval_mode, "workspace")
        self.assertEqual(args.sandbox, "required")
        self.assertEqual(args.sandbox_runtime, "auto")
        self.assertIsNone(args.sandbox_image)
        self.assertTrue(args.web_access)
        self.assertFalse(args.interactive)
        self.assertFalse(args.plain)

    def test_prompt_is_optional_for_interactive_mode(self) -> None:
        args = build_parser().parse_args(["--workspace", "."])

        self.assertIsNone(args.prompt)

    def test_interactive_mode_accepts_an_initial_prompt(self) -> None:
        args = build_parser().parse_args(["--interactive", "Inspect README.md"])

        self.assertTrue(args.interactive)
        self.assertEqual(args.prompt, "Inspect README.md")

    def test_sandbox_configuration_can_be_explicit(self) -> None:
        args = build_parser().parse_args(
            [
                "--sandbox",
                "off",
                "--sandbox-runtime",
                "podman",
                "--sandbox-image",
                "local/agent:test",
                "Inspect README.md",
            ]
        )

        self.assertEqual(args.sandbox, "off")
        self.assertEqual(args.sandbox_runtime, "podman")
        self.assertEqual(args.sandbox_image, "local/agent:test")

    def test_web_access_can_be_disabled(self) -> None:
        args = build_parser().parse_args(["--no-web-access", "Inspect README.md"])

        self.assertFalse(args.web_access)

    def test_model_retry_configuration_can_be_explicit(self) -> None:
        args = build_parser().parse_args(
            [
                "--model-timeout-s",
                "180",
                "--max-model-retries",
                "5",
                "--retry-base-delay-s",
                "1.5",
                "Inspect README.md",
            ]
        )

        self.assertEqual(args.model_timeout_s, 180.0)
        self.assertEqual(args.max_model_retries, 5)
        self.assertEqual(args.retry_base_delay_s, 1.5)


class PlainCliTests(unittest.TestCase):
    def test_expected_agent_failure_is_reported_without_traceback(self) -> None:
        class FailingAgent:
            def run(self, prompt: str, *, session_id: str | None = None) -> object:
                raise ModelRequestError(
                    "model authentication failed (HTTP 401)",
                    retryable=False,
                    status_code=401,
                )

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = _run_plain(  # type: ignore[arg-type]
                FailingAgent(), "Hello", None
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            stderr.getvalue(),
            "error: model authentication failed (HTTP 401)\n",
        )
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_success_prints_answer_and_session(self) -> None:
        class PassingAgent:
            def run(self, prompt: str, *, session_id: str | None = None) -> object:
                return SimpleNamespace(text="Done", session_id="session-1")

        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = _run_plain(  # type: ignore[arg-type]
                PassingAgent(), "Hello", None
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "Done\n")
        self.assertEqual(stderr.getvalue(), "session_id: session-1\n")


if __name__ == "__main__":
    unittest.main()
