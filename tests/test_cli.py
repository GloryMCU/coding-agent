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


if __name__ == "__main__":
    unittest.main()
