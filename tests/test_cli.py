from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from coding_agent.cli import build_parser


class CliDefaultsTests(unittest.TestCase):
    def test_defaults_to_deepseek_v4_pro_configuration(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            args = build_parser().parse_args(["Inspect README.md"])

        self.assertEqual(args.base_url, "https://api.deepseek.com")
        self.assertEqual(args.reasoning_effort, "high")
        self.assertTrue(args.thinking)
        self.assertEqual(args.max_context_tokens, 24_000)
        self.assertEqual(args.context_summary_tokens, 2_000)
        self.assertEqual(args.history_search_limit, 5)
        self.assertEqual(args.approval_mode, "ask")
        self.assertEqual(args.sandbox, "required")
        self.assertEqual(args.sandbox_runtime, "auto")
        self.assertIsNone(args.sandbox_image)
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


if __name__ == "__main__":
    unittest.main()
