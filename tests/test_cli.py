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


if __name__ == "__main__":
    unittest.main()
