"""Thread-safe adapters between the synchronous agent and Textual."""

from __future__ import annotations

from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable

from ..permissions import ApprovalDecision, ApprovalPolicy, PermissionRequest

if TYPE_CHECKING:
    from .app import CodingAgentApp


class TuiEventSink:
    """Translate core events into messages on Textual's UI thread."""

    def __init__(self, app: CodingAgentApp) -> None:
        self._app = app

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self._app.call_from_thread(
            self._app.post_agent_event,
            event_type,
            payload,
        )


class TuiApprovalPolicy(ApprovalPolicy):
    """Block the agent worker while Textual presents an approval modal."""

    def __init__(
        self,
        app: CodingAgentApp,
        *,
        timeout_s: float = 300,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("approval timeout must be positive")
        self._app = app
        self._timeout_s = timeout_s
        self._pending: set[Future[ApprovalDecision]] = set()
        self._lock = Lock()

    def approve(self, request: PermissionRequest) -> ApprovalDecision:
        future: Future[ApprovalDecision] = Future()
        with self._lock:
            self._pending.add(future)
        self._app.call_from_thread(
            self._app.show_approval,
            request,
            future,
        )
        try:
            return future.result(timeout=self._timeout_s)
        except FutureTimeoutError:
            if not future.done():
                future.set_result(ApprovalDecision.DENY)
            self._app.call_from_thread(
                self._app.expire_approval,
                request,
            )
            return ApprovalDecision.DENY
        finally:
            with self._lock:
                self._pending.discard(future)

    def deny_pending(self) -> None:
        """Release worker threads when the app is shutting down."""

        with self._lock:
            pending = tuple(self._pending)
        for future in pending:
            if not future.done():
                future.set_result(ApprovalDecision.DENY)
