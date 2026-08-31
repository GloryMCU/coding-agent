"""Restricted host-side HTTPS access for search and public web pages."""

from __future__ import annotations

import http.client
import ipaddress
import json
import re
import socket
import ssl
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable, Mapping
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit

from .errors import WebAccessError


_BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_ALLOWED_PAGE_TYPES = frozenset(
    {"text/html", "application/xhtml+xml", "text/plain", "application/json"}
)
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tr",
        "ul",
    }
)
_FRESHNESS = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}


@dataclass(frozen=True, slots=True)
class ResolvedHttpsTarget:
    """A validated URL pinned to one already checked public IP address."""

    url: str
    hostname: str
    port: int
    address: str
    request_target: str


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes
    truncated: bool = False


Resolver = Callable[[str, int], list[str]]
Transport = Callable[
    [ResolvedHttpsTarget, Mapping[str, str], float, int], HttpResponse
]


class WebAccessClient:
    """Search and fetch through a small, credential-isolated HTTPS surface."""

    def __init__(
        self,
        *,
        brave_api_key: str | None = None,
        timeout_s: float = 15.0,
        max_response_bytes: int = 512 * 1024,
        max_redirects: int = 3,
        resolver: Resolver | None = None,
        transport: Transport | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if max_response_bytes < 1024:
            raise ValueError("max_response_bytes must be at least 1024")
        if not 0 <= max_redirects <= 10:
            raise ValueError("max_redirects must be between 0 and 10")
        key = (brave_api_key or "").strip()
        if "\r" in key or "\n" in key:
            raise ValueError("Brave Search API key contains invalid characters")
        self._brave_api_key = key or None
        self._timeout_s = timeout_s
        self._max_response_bytes = max_response_bytes
        self._max_redirects = max_redirects
        self._resolver = resolver or _resolve_addresses
        self._transport = transport or _https_transport

    def search(
        self,
        query: str,
        *,
        count: int = 5,
        freshness: str | None = None,
    ) -> dict[str, object]:
        """Return a bounded projection of Brave Search web results."""

        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise WebAccessError("search query must not be empty")
        if len(normalized_query) > 400 or len(normalized_query.split()) > 50:
            raise WebAccessError("search query exceeds Brave's 400 character/50 word limit")
        if not 1 <= count <= 10:
            raise WebAccessError("search count must be between 1 and 10")
        if freshness is not None and freshness not in _FRESHNESS:
            raise WebAccessError("freshness must be day, week, month, or year")
        if self._brave_api_key is None:
            raise WebAccessError(
                "web_search requires the BRAVE_SEARCH_API_KEY environment variable"
            )

        parameters: dict[str, str | int] = {
            "q": normalized_query,
            "count": count,
            "safesearch": "moderate",
            "text_decorations": "false",
            "result_filter": "web",
        }
        if freshness is not None:
            parameters["freshness"] = _FRESHNESS[freshness]
        url = f"{_BRAVE_SEARCH_ENDPOINT}?{urlencode(parameters)}"
        response, _ = self._request(
            url,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self._brave_api_key,
            },
            follow_redirects=False,
        )
        if response.status != 200:
            raise WebAccessError(f"Brave Search returned HTTP {response.status}")
        if response.truncated:
            raise WebAccessError("Brave Search response exceeded the size limit")
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WebAccessError("Brave Search returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise WebAccessError("Brave Search returned an unexpected response")

        web = payload.get("web")
        raw_results = web.get("results", []) if isinstance(web, dict) else []
        results: list[dict[str, str]] = []
        if isinstance(raw_results, list):
            for item in raw_results[:count]:
                if not isinstance(item, dict):
                    continue
                result_url = _bounded_text(item.get("url"), 2048)
                if urlsplit(result_url).scheme.casefold() not in {"http", "https"}:
                    continue
                result = {
                    "title": _clean_fragment(item.get("title"), 500),
                    "url": result_url,
                    "description": _clean_fragment(item.get("description"), 2000),
                }
                age = _bounded_text(item.get("age"), 100)
                if age:
                    result["age"] = age
                results.append(result)
        return {
            "query": normalized_query,
            "result_count": len(results),
            "results": results,
            "provider": "Brave Search",
            "untrusted": True,
        }

    def fetch_page(self, url: str, *, max_chars: int = 50_000) -> dict[str, object]:
        """Fetch one public HTTPS page and return bounded visible text."""

        if not 1_000 <= max_chars <= 100_000:
            raise WebAccessError("max_chars must be between 1000 and 100000")
        response, final_url = self._request(
            url,
            headers={"Accept": "text/html,application/xhtml+xml,text/plain,application/json"},
            follow_redirects=True,
        )
        if not 200 <= response.status < 300:
            raise WebAccessError(f"web page returned HTTP {response.status}")
        content_type_header = response.headers.get("content-type", "")
        content_type = content_type_header.partition(";")[0].strip().casefold()
        if content_type not in _ALLOWED_PAGE_TYPES:
            rendered = content_type or "missing content type"
            raise WebAccessError(f"unsupported web page content type: {rendered}")
        content_encoding = response.headers.get("content-encoding", "identity").casefold()
        if content_encoding not in {"", "identity"}:
            raise WebAccessError(
                f"unsupported web page content encoding: {content_encoding}"
            )

        charset = _charset(content_type_header)
        try:
            decoded = response.body.decode(charset, errors="replace")
        except LookupError as exc:
            raise WebAccessError(f"unsupported web page charset: {charset}") from exc
        title = ""
        if content_type in {"text/html", "application/xhtml+xml"}:
            parser = _VisibleTextParser()
            parser.feed(decoded)
            parser.close()
            text = parser.text()
            title = parser.title()
        else:
            text = decoded
        text = _normalize_text(text)
        text_truncated = len(text) > max_chars
        if text_truncated:
            text = text[:max_chars].rstrip() + "…"
        return {
            "url": final_url,
            "title": title,
            "content_type": content_type,
            "text": text,
            "bytes_read": len(response.body),
            "truncated": response.truncated or text_truncated,
            "untrusted": True,
        }

    def _request(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        follow_redirects: bool,
    ) -> tuple[HttpResponse, str]:
        current_url = url
        for redirect_count in range(self._max_redirects + 1):
            target = _resolve_https_target(current_url, self._resolver)
            request_headers = {
                "Accept-Encoding": "identity",
                "Connection": "close",
                "User-Agent": "coding-agent/0.1 (+restricted-fetch)",
                **headers,
            }
            try:
                response = self._transport(
                    target,
                    request_headers,
                    self._timeout_s,
                    self._max_response_bytes,
                )
            except WebAccessError:
                raise
            except Exception as exc:
                raise WebAccessError(
                    f"HTTPS request failed: {type(exc).__name__}: {exc}"
                ) from exc
            normalized_headers = {
                str(key).casefold(): str(value)
                for key, value in response.headers.items()
            }
            response = HttpResponse(
                status=response.status,
                headers=normalized_headers,
                body=response.body,
                truncated=response.truncated,
            )
            if response.status not in _REDIRECT_STATUSES:
                return response, target.url
            if not follow_redirects:
                raise WebAccessError("redirects are not allowed for authenticated search")
            location = response.headers.get("location")
            if not location:
                raise WebAccessError("web page redirect is missing a Location header")
            if redirect_count >= self._max_redirects:
                raise WebAccessError("web page exceeded the redirect limit")
            current_url = urljoin(target.url, location)
        raise AssertionError("redirect loop exhausted unexpectedly")


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, target: ResolvedHttpsTarget, *, timeout: float) -> None:
        super().__init__(
            target.hostname,
            port=target.port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
        self._pinned_address = target.address

    def connect(self) -> None:
        raw_socket = socket.create_connection(
            (self._pinned_address, self.port),
            timeout=self.timeout,
        )
        try:
            self.sock = self._context.wrap_socket(
                raw_socket,
                server_hostname=self.host,
            )
        except Exception:
            raw_socket.close()
            raise


def _https_transport(
    target: ResolvedHttpsTarget,
    headers: Mapping[str, str],
    timeout_s: float,
    max_bytes: int,
) -> HttpResponse:
    connection = _PinnedHTTPSConnection(target, timeout=timeout_s)
    try:
        connection.request("GET", target.request_target, headers=dict(headers))
        response = connection.getresponse()
        body = response.read(max_bytes + 1)
        truncated = len(body) > max_bytes
        if truncated:
            body = body[:max_bytes]
        return HttpResponse(
            status=response.status,
            headers={key.casefold(): value for key, value in response.getheaders()},
            body=body,
            truncated=truncated,
        )
    finally:
        connection.close()


def _resolve_addresses(hostname: str, port: int) -> list[str]:
    try:
        records = socket.getaddrinfo(
            hostname,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise WebAccessError(f"could not resolve web host: {hostname}") from exc
    return [record[4][0] for record in records]


def _resolve_https_target(url: str, resolver: Resolver) -> ResolvedHttpsTarget:
    if not isinstance(url, str) or not url.strip():
        raise WebAccessError("web URL must be a non-empty string")
    candidate = url.strip()
    if len(candidate) > 4096:
        raise WebAccessError("web URL exceeds 4096 characters")
    if any(ord(character) < 32 or character in {" ", "\\"} for character in candidate):
        raise WebAccessError("web URL contains invalid characters")
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise WebAccessError("web URL is malformed") from exc
    if parsed.scheme.casefold() != "https":
        raise WebAccessError("only HTTPS web URLs are allowed")
    if parsed.username is not None or parsed.password is not None:
        raise WebAccessError("web URLs must not contain credentials")
    if not parsed.hostname:
        raise WebAccessError("web URL must include a hostname")
    if port not in {None, 443}:
        raise WebAccessError("web URLs may only use HTTPS port 443")
    try:
        hostname = parsed.hostname.rstrip(".").encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise WebAccessError("web URL hostname is invalid") from exc
    if not hostname:
        raise WebAccessError("web URL hostname is invalid")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise WebAccessError("web URLs must use a public DNS hostname, not an IP literal")
    addresses = resolver(hostname, 443)
    if not addresses:
        raise WebAccessError("web hostname resolved to no addresses")
    public_addresses: list[str] = []
    for address in addresses:
        try:
            parsed_address = ipaddress.ip_address(address)
        except ValueError as exc:
            raise WebAccessError("web hostname resolved to an invalid address") from exc
        if not parsed_address.is_global:
            raise WebAccessError(
                "web access to local, private, reserved, or special-use addresses is blocked"
            )
        public_addresses.append(str(parsed_address))
    path = parsed.path or "/"
    request_target = urlunsplit(("", "", path, parsed.query, ""))
    normalized_url = urlunsplit(("https", hostname, path, parsed.query, ""))
    return ResolvedHttpsTarget(
        url=normalized_url,
        hostname=hostname,
        port=443,
        address=public_addresses[0],
        request_target=request_target,
    )


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._title_parts: list[str] = []
        self._ignored_depth = 0
        self._title_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.casefold()
        if normalized in {"script", "style", "noscript", "svg", "template"}:
            self._ignored_depth += 1
        if normalized == "title" and self._ignored_depth == 0:
            self._title_depth += 1
        if normalized in _BLOCK_TAGS and self._ignored_depth == 0:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized == "title" and self._title_depth:
            self._title_depth -= 1
        if normalized in {"script", "style", "noscript", "svg", "template"}:
            if self._ignored_depth:
                self._ignored_depth -= 1
            return
        if normalized in _BLOCK_TAGS and self._ignored_depth == 0:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        self._parts.append(data)
        if self._title_depth:
            self._title_parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)

    def title(self) -> str:
        return _normalize_text("".join(self._title_parts))[:500]


def _charset(content_type: str) -> str:
    match = re.search(r"(?:^|;)\s*charset\s*=\s*['\"]?([^;'\"\s]+)", content_type, re.I)
    return match.group(1) if match else "utf-8"


def _bounded_text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    normalized = " ".join(value.split())
    return normalized[:limit]


def _clean_fragment(value: object, limit: int) -> str:
    raw = _bounded_text(value, limit * 2)
    if "<" not in raw and ">" not in raw:
        return raw[:limit]
    parser = _VisibleTextParser()
    parser.feed(raw)
    parser.close()
    return _normalize_text(parser.text())[:limit]


def _normalize_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[\t\f\v ]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()
