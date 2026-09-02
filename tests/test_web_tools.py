from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Mapping

from coding_agent.model import ToolCall
from coding_agent.errors import WebAccessError
from coding_agent.tools import create_workspace_registry
from coding_agent.web_tools import (
    HttpResponse,
    ResolvedHttpsTarget,
    WebAccessClient,
)


class FakeTransport:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.requests: list[
            tuple[ResolvedHttpsTarget, str, Mapping[str, str], bytes | None]
        ] = []

    def __call__(
        self,
        target: ResolvedHttpsTarget,
        method: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
        max_bytes: int,
    ) -> HttpResponse:
        self.requests.append((target, method, dict(headers), body))
        if not self.responses:
            raise AssertionError("unexpected HTTP request")
        return self.responses.pop(0)


def fake_resolver(hostname: str, port: int) -> list[str]:
    if hostname in {"localhost", "private.example"}:
        return ["127.0.0.1"]
    if hostname == "mixed.example":
        return ["93.184.216.34", "10.0.0.1"]
    return ["93.184.216.34"]


class WebSearchTests(unittest.TestCase):
    def test_search_calls_exa_mcp_and_returns_text_content(self) -> None:
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": "Official docs: https://docs.example/guide",
                        }
                    ]
                },
            }
        ).encode()
        transport = FakeTransport(
            HttpResponse(200, {"content-type": "application/json"}, body)
        )
        client = WebAccessClient(
            exa_api_key="exa test/secret",
            resolver=fake_resolver,
            transport=transport,
        )

        result = client.search(
            "current Python release",
            count=3,
            search_type="deep",
            livecrawl="preferred",
            context_max_characters=12_000,
        )

        self.assertEqual(
            result["content"], "Official docs: https://docs.example/guide"
        )
        self.assertEqual(result["provider"], "Exa MCP")
        target, method, headers, request_body = transport.requests[0]
        self.assertEqual(target.hostname, "mcp.exa.ai")
        self.assertEqual(target.request_target, "/mcp?exaApiKey=exa+test%2Fsecret")
        self.assertEqual(method, "POST")
        self.assertEqual(headers["Content-Type"], "application/json")
        payload = json.loads((request_body or b"").decode())
        self.assertEqual(payload["method"], "tools/call")
        self.assertEqual(payload["params"]["name"], "web_search_exa")
        self.assertEqual(
            payload["params"]["arguments"],
            {
                "query": "current Python release",
                "type": "deep",
                "numResults": 3,
                "livecrawl": "preferred",
                "contextMaxCharacters": 12_000,
            },
        )
        self.assertNotIn("exa test/secret", json.dumps(result))

    def test_search_allows_public_mcp_and_parses_sse(self) -> None:
        event = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": "SSE result"}]},
        }
        transport = FakeTransport(
            HttpResponse(
                200,
                {"content-type": "text/event-stream"},
                f"event: message\ndata: {json.dumps(event)}\n\n".encode(),
            )
        )
        client = WebAccessClient(resolver=fake_resolver, transport=transport)

        result = client.search("test")

        self.assertEqual(result["content"], "SSE result")
        target, _, _, _ = transport.requests[0]
        self.assertEqual(target.request_target, "/mcp")

    def test_search_refuses_redirects_and_surfaces_mcp_errors(self) -> None:

        redirected = WebAccessClient(
            exa_api_key="secret",
            resolver=fake_resolver,
            transport=FakeTransport(
                HttpResponse(302, {"location": "https://other.example/"}, b"")
            ),
        )
        with self.assertRaisesRegex(Exception, "redirects are not allowed"):
            redirected.search("test")

        failed = WebAccessClient(
            resolver=fake_resolver,
            transport=FakeTransport(
                HttpResponse(
                    200,
                    {"content-type": "application/json"},
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "result": {
                                "isError": True,
                                "content": [
                                    {"type": "text", "text": "rate limit exceeded"}
                                ],
                            },
                        }
                    ).encode(),
                )
            ),
        )
        with self.assertRaisesRegex(Exception, "rate limit exceeded"):
            failed.search("test")

    def test_dns_resolution_has_a_hard_total_timeout(self) -> None:
        released = threading.Event()

        def blocking_resolver(hostname: str, port: int) -> list[str]:
            del hostname, port
            released.wait(timeout=2)
            return ["93.184.216.34"]

        client = WebAccessClient(
            timeout_s=0.02,
            resolver=blocking_resolver,
            transport=FakeTransport(),
        )
        try:
            with self.assertRaisesRegex(WebAccessError, "DNS resolution"):
                client.search("test")
        finally:
            released.set()

    def test_custom_transport_cannot_exceed_total_timeout(self) -> None:
        released = threading.Event()

        def blocking_transport(*args, **kwargs) -> HttpResponse:
            del args, kwargs
            released.wait(timeout=2)
            return HttpResponse(200, {"content-type": "application/json"}, b"{}")

        client = WebAccessClient(
            timeout_s=0.02,
            resolver=fake_resolver,
            transport=blocking_transport,
        )
        try:
            with self.assertRaisesRegex(WebAccessError, "HTTPS request"):
                client.search("test")
        finally:
            released.set()


class FetchWebpageTests(unittest.TestCase):
    def test_extracts_visible_html_and_removes_active_content(self) -> None:
        html = b"""<!doctype html><html><head><title> Example page </title>
        <style>.hidden { display:none }</style><script>stealSecrets()</script></head>
        <body><h1>Hello</h1><p>Public &amp; useful text.</p></body></html>"""
        transport = FakeTransport(
            HttpResponse(200, {"Content-Type": "text/html; charset=utf-8"}, html)
        )
        client = WebAccessClient(resolver=fake_resolver, transport=transport)

        result = client.fetch_page("https://example.com/docs#section")

        self.assertEqual(result["url"], "https://example.com/docs")
        self.assertEqual(result["title"], "Example page")
        self.assertIn("Hello", result["text"])
        self.assertIn("Public & useful text.", result["text"])
        self.assertNotIn("stealSecrets", result["text"])
        self.assertNotIn("display:none", result["text"])
        self.assertTrue(result["untrusted"])

    def test_blocks_non_https_credentials_ip_literals_and_private_dns(self) -> None:
        client = WebAccessClient(resolver=fake_resolver, transport=FakeTransport())
        cases = {
            "http://example.com": "only HTTPS",
            "https://user:pass@example.com": "must not contain credentials",
            "https://127.0.0.1/secret": "not an IP literal",
            "https://private.example/secret": "local, private, reserved",
            "https://mixed.example/secret": "local, private, reserved",
            "https://example.com:444/": "port 443",
        }
        for url, message in cases.items():
            with self.subTest(url=url):
                with self.assertRaisesRegex(Exception, message):
                    client.fetch_page(url)

    def test_revalidates_redirects_before_second_request(self) -> None:
        transport = FakeTransport(
            HttpResponse(302, {"location": "https://private.example/secret"}, b"")
        )
        client = WebAccessClient(resolver=fake_resolver, transport=transport)

        with self.assertRaisesRegex(Exception, "local, private, reserved"):
            client.fetch_page("https://example.com/start")

        self.assertEqual(len(transport.requests), 1)

    def test_rejects_binary_and_marks_bounded_text_as_truncated(self) -> None:
        binary = WebAccessClient(
            resolver=fake_resolver,
            transport=FakeTransport(
                HttpResponse(200, {"content-type": "application/pdf"}, b"%PDF")
            ),
        )
        with self.assertRaisesRegex(Exception, "unsupported web page content type"):
            binary.fetch_page("https://example.com/file.pdf")

        text = WebAccessClient(
            resolver=fake_resolver,
            transport=FakeTransport(
                HttpResponse(
                    200,
                    {"content-type": "text/plain"},
                    b"x" * 2_000,
                    truncated=True,
                )
            ),
        )
        result = text.fetch_page("https://example.com/large", max_chars=1_000)
        self.assertTrue(result["truncated"])
        self.assertLessEqual(len(result["text"]), 1_001)


class WebToolRegistryTests(unittest.TestCase):
    def test_registry_exposes_and_executes_web_tools_when_client_is_supplied(self) -> None:
        transport = FakeTransport(
            HttpResponse(
                200,
                {"content-type": "application/json"},
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "result": {
                            "content": [{"type": "text", "text": "No results"}]
                        },
                    }
                ).encode(),
            )
        )
        client = WebAccessClient(
            exa_api_key="test-key",
            resolver=fake_resolver,
            transport=transport,
        )
        with tempfile.TemporaryDirectory() as directory:
            registry = create_workspace_registry(Path(directory), web_client=client)

            names = [schema["function"]["name"] for schema in registry.schemas()]
            self.assertIn("web_search", names)
            self.assertIn("fetch_webpage", names)
            result = registry.execute(
                ToolCall(
                    id="web-search-1",
                    name="web_search",
                    arguments={"query": "test", "count": 1},
                )
            )

        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.output["provider"], "Exa MCP")

    def test_registry_schema_rejects_oversized_url_before_network(self) -> None:
        client = WebAccessClient(resolver=fake_resolver, transport=FakeTransport())
        with tempfile.TemporaryDirectory() as directory:
            registry = create_workspace_registry(Path(directory), web_client=client)
            result = registry.execute(
                ToolCall(
                    id="fetch-1",
                    name="fetch_webpage",
                    arguments={"url": "https://example.com/" + "x" * 5_000},
                )
            )

        self.assertFalse(result.ok)
        self.assertIn("at most 4096 characters", result.error or "")


if __name__ == "__main__":
    unittest.main()
