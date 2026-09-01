from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from coding_agent.errors import ModelProtocolError, ModelRequestError
from coding_agent.model import DeepSeekV4ProClient, parse_openai_message


def sdk_message(
    *,
    content: str | None = None,
    arguments: str = "{}",
    reasoning_content: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=[
            SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(name="read_file", arguments=arguments),
            )
        ],
    )


class OpenAIMessageParserTests(unittest.TestCase):
    def test_parses_native_tool_call(self) -> None:
        response = parse_openai_message(
            sdk_message(arguments='{"path":"README.md","start_line":1}')
        )

        self.assertEqual(response.tool_calls[0].name, "read_file")
        self.assertEqual(response.tool_calls[0].arguments["path"], "README.md")

    def test_preserves_deepseek_reasoning_content(self) -> None:
        response = parse_openai_message(
            sdk_message(reasoning_content="Need to inspect the repository.")
        )

        self.assertEqual(
            response.assistant_message["reasoning_content"],
            "Need to inspect the repository.",
        )

    def test_rejects_invalid_json_arguments(self) -> None:
        with self.assertRaisesRegex(ModelProtocolError, "invalid JSON"):
            parse_openai_message(sdk_message(arguments="{not-json}"))

    def test_rejects_non_object_arguments(self) -> None:
        with self.assertRaisesRegex(ModelProtocolError, "JSON object"):
            parse_openai_message(sdk_message(arguments='["README.md"]'))

    def test_deepseek_client_sends_thinking_configuration(self) -> None:
        captured_client: dict[str, object] = {}
        captured_request: dict[str, object] = {}

        class FakeCompletions:
            def create(self, **request: object) -> SimpleNamespace:
                captured_request.update(request)
                return SimpleNamespace(
                    id="response-1",
                    usage=SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                    ),
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="Done",
                                reasoning_content="Reasoned",
                                tool_calls=None,
                            ),
                            finish_reason="stop",
                        )
                    ],
                )

        def fake_openai(**kwargs: object) -> SimpleNamespace:
            captured_client.update(kwargs)
            return SimpleNamespace(
                chat=SimpleNamespace(completions=FakeCompletions())
            )

        fake_module = SimpleNamespace(OpenAI=fake_openai)
        with patch.dict(sys.modules, {"openai": fake_module}):
            client = DeepSeekV4ProClient(api_key="test-key")
            response = client.generate([], [], timeout_s=12)

        self.assertEqual(captured_client["base_url"], "https://api.deepseek.com")
        self.assertEqual(captured_request["model"], "deepseek-v4-pro")
        self.assertEqual(captured_request["reasoning_effort"], "high")
        self.assertEqual(
            captured_request["extra_body"],
            {"thinking": {"type": "enabled"}},
        )
        self.assertEqual(response.assistant_message["reasoning_content"], "Reasoned")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.response_id, "response-1")
        self.assertEqual(response.usage["total_tokens"], 15)

    def test_client_rejects_truncated_response(self) -> None:
        class FakeCompletions:
            def create(self, **request: object) -> SimpleNamespace:
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="partial",
                                reasoning_content=None,
                                tool_calls=None,
                            ),
                            finish_reason="length",
                        )
                    ]
                )

        fake_module = SimpleNamespace(
            OpenAI=lambda **kwargs: SimpleNamespace(
                chat=SimpleNamespace(completions=FakeCompletions())
            )
        )
        with patch.dict(sys.modules, {"openai": fake_module}):
            client = DeepSeekV4ProClient(api_key="test-key")
            with self.assertRaisesRegex(ModelProtocolError, "did not complete normally"):
                client.generate([], [], timeout_s=12)

    def test_client_rejects_inconsistent_tool_finish_reason(self) -> None:
        class FakeCompletions:
            def create(self, **request: object) -> SimpleNamespace:
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="No call",
                                reasoning_content=None,
                                tool_calls=None,
                            ),
                            finish_reason="tool_calls",
                        )
                    ]
                )

        fake_module = SimpleNamespace(
            OpenAI=lambda **kwargs: SimpleNamespace(
                chat=SimpleNamespace(completions=FakeCompletions())
            )
        )
        with patch.dict(sys.modules, {"openai": fake_module}):
            client = DeepSeekV4ProClient(api_key="test-key")
            with self.assertRaisesRegex(ModelProtocolError, "without any tool call"):
                client.generate([], [], timeout_s=12)

    def test_client_classifies_http_failures_for_retry(self) -> None:
        cases = (
            (400, False, "request rejected"),
            (401, False, "authentication failed"),
            (403, False, "access denied"),
            (429, True, "rate limit"),
            (503, True, "service unavailable"),
        )
        for status_code, retryable, message in cases:
            with self.subTest(status_code=status_code):
                error = RuntimeError("vendor detail")
                error.status_code = status_code  # type: ignore[attr-defined]

                class FakeCompletions:
                    def create(self, **request: object) -> SimpleNamespace:
                        raise error

                fake_module = SimpleNamespace(
                    OpenAI=lambda **kwargs: SimpleNamespace(
                        chat=SimpleNamespace(completions=FakeCompletions())
                    )
                )
                with patch.dict(sys.modules, {"openai": fake_module}):
                    client = DeepSeekV4ProClient(api_key="test-key")
                    with self.assertRaises(ModelRequestError) as raised:
                        client.generate([], [], timeout_s=12)

                self.assertEqual(raised.exception.status_code, status_code)
                self.assertEqual(raised.exception.retryable, retryable)
                self.assertIn(message, str(raised.exception))

    def test_client_treats_transport_failure_as_retryable(self) -> None:
        class FakeCompletions:
            def create(self, **request: object) -> SimpleNamespace:
                raise ConnectionError("stream disconnected")

        fake_module = SimpleNamespace(
            OpenAI=lambda **kwargs: SimpleNamespace(
                chat=SimpleNamespace(completions=FakeCompletions())
            )
        )
        with patch.dict(sys.modules, {"openai": fake_module}):
            client = DeepSeekV4ProClient(api_key="test-key")
            with self.assertRaises(ModelRequestError) as raised:
                client.generate([], [], timeout_s=12)

        self.assertTrue(raised.exception.retryable)
        self.assertIsNone(raised.exception.status_code)
        self.assertIn("stream disconnected", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
