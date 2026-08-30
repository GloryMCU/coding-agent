from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from coding_agent.quality import summarize_event_log, summarize_events


class QualitySummaryTests(unittest.TestCase):
    def test_summarizes_usage_latency_failures_and_termination(self) -> None:
        summary = summarize_events(
            [
                {
                    "type": "model_response",
                    "payload": {
                        "session_id": "session-1",
                        "duration_ms": 120,
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 4,
                        },
                    },
                },
                {
                    "type": "tool_result",
                    "payload": {
                        "session_id": "session-1",
                        "duration_ms": 30,
                        "ok": False,
                    },
                },
                {
                    "type": "model_request_error",
                    "payload": {"session_id": "session-1"},
                },
                {
                    "type": "agent_terminated",
                    "payload": {
                        "session_id": "session-1",
                        "reason": "final_response",
                    },
                },
            ]
        )

        self.assertEqual(summary["sessions_observed"], 1)
        self.assertEqual(summary["model_responses"], 1)
        self.assertEqual(summary["tool_failures"], 1)
        self.assertEqual(summary["model_request_errors"], 1)
        self.assertEqual(summary["model_duration_ms"], 120)
        self.assertEqual(summary["tool_duration_ms"], 30)
        self.assertEqual(summary["usage"]["prompt_tokens"], 10)
        self.assertEqual(summary["termination_reasons"]["final_response"], 1)

    def test_reads_jsonl_and_reports_the_bad_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            path.write_text(
                json.dumps({"type": "user_message", "payload": {}})
                + "\nnot-json\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, r"events\.jsonl:2"):
                summarize_event_log(path)


if __name__ == "__main__":
    unittest.main()
