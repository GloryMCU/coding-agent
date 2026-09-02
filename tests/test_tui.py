from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from typing import Any

try:
    from textual.widgets import RichLog, Static

    from coding_agent.agent import AgentResult
    from coding_agent.conversation import ConversationState
    from coding_agent.errors import ModelRequestError
    from coding_agent.permissions import (
        ApprovalDecision,
        PermissionKind,
        PermissionRequest,
    )
    from coding_agent.tui import CodingAgentApp
    from coding_agent.tui.screens import ApprovalScreen
except ModuleNotFoundError as exc:
    if exc.name in {"textual", "rich"}:
        raise unittest.SkipTest("Textual optional dependency is not installed") from exc
    raise


class StubAgent:
    def __init__(self, events: Any) -> None:
        self.events = events
        self.prompts: list[str] = []

    def run(self, prompt: str, *, session_id: str | None = None) -> AgentResult:
        self.prompts.append(prompt)
        current_session = session_id or "session-example"
        self.events.emit(
            "user_message",
            {"session_id": current_session, "content": prompt},
        )
        self.events.emit(
            "model_response",
            {
                "session_id": current_session,
                "step": 1,
                "assistant_message": {"role": "assistant", "content": "Finished"},
                "text": "Finished",
                "tool_calls": [],
            },
        )
        self.events.emit(
            "agent_terminated",
            {"session_id": current_session, "step": 1, "reason": "final_response"},
        )
        return AgentResult(
            text="Finished",
            steps=1,
            termination_reason="final_response",
            conversation=ConversationState(),
            session_id=current_session,
        )


class ApprovalStubAgent(StubAgent):
    def __init__(self, events: Any, approval: Any) -> None:
        super().__init__(events)
        self.approval = approval
        self.approved: bool | None = None

    def run(self, prompt: str, *, session_id: str | None = None) -> AgentResult:
        self.approved = self.approval.approve(
            PermissionRequest(
                tool_name="run_command",
                kind=PermissionKind.EXECUTE,
                description="run argv=['python', '-m', 'unittest']",
            )
        )
        return super().run(prompt, session_id=session_id)


class FailingStubAgent(StubAgent):
    def run(self, prompt: str, *, session_id: str | None = None) -> AgentResult:
        raise ModelRequestError(
            "model authentication failed (HTTP 401)",
            retryable=False,
            status_code=401,
        )


class TuiAppTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.stub: StubAgent | None = None

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_app(self, *, initial_prompt: str | None = None) -> CodingAgentApp:
        def factory(events: Any, approval: Any) -> StubAgent:
            self.stub = StubAgent(events)
            return self.stub

        return CodingAgentApp(
            agent_factory=factory,  # type: ignore[arg-type]
            workspace=self.workspace,
            model_name="fake-model",
            approval_mode="ask",
            initial_prompt=initial_prompt,
        )

    async def test_mounts_with_workspace_and_ready_prompt(self) -> None:
        app = self.make_app()

        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause()
            context = str(app.query_one("#context-bar", Static).render())

            self.assertIn(str(app.workspace), context)
            self.assertIn("fake-model", context)
            self.assertFalse(app.busy)

    async def test_submits_prompt_on_worker_and_renders_response(self) -> None:
        app = self.make_app()

        async with app.run_test(size=(100, 32)) as pilot:
            app.submit_prompt("Inspect the project")
            for _ in range(100):
                await pilot.pause(0.01)
                if not app.busy:
                    break

            self.assertIsNotNone(self.stub)
            self.assertEqual(self.stub.prompts, ["Inspect the project"])
            self.assertEqual(app.session_id, "session-example")
            rendered = "\n".join(
                line.text for line in app.query_one("#conversation", RichLog).lines
            )
            self.assertIn("Inspect the project", rendered)
            self.assertIn("Finished", rendered)

    async def test_enter_submits_and_shift_enter_inserts_a_newline(self) -> None:
        app = self.make_app()

        async with app.run_test(size=(100, 32)) as pilot:
            prompt = app._prompt()
            prompt.focus()
            prompt.text = "First line"
            prompt.move_cursor((0, len(prompt.text)))
            await pilot.press("shift+enter")
            self.assertEqual(prompt.text, "First line\n")

            prompt.insert("Second line")
            await pilot.press("enter")
            for _ in range(100):
                await pilot.pause(0.01)
                if not app.busy:
                    break

            self.assertEqual(self.stub.prompts, ["First line\nSecond line"])

    async def test_tool_details_are_replaced_by_status_and_not_logged(self) -> None:
        app = self.make_app()

        async with app.run_test(size=(100, 32)) as pilot:
            app.post_agent_event(
                "model_response",
                {
                    "tool_calls": [
                        {
                            "name": "read_file",
                            "arguments": {"path": "private-details.py"},
                        }
                    ]
                },
            )
            activity = str(app.query_one("#activity", Static).render())
            self.assertIn("Reading file", activity)

            app.post_agent_event(
                "tool_result",
                {
                    "name": "read_file",
                    "ok": True,
                    "output": {"path": "private-details.py", "content": "hidden"},
                },
            )
            app.post_agent_event(
                "model_response",
                {"tool_calls": [], "text": "Final answer only"},
            )

            rendered = "\n".join(
                line.text for line in app.query_one("#conversation", RichLog).lines
            )
            self.assertIn("Final answer only", rendered)
            self.assertNotIn("read_file", rendered)
            self.assertNotIn("private-details.py", rendered)
            self.assertNotIn("hidden", rendered)

    async def test_activity_shows_live_model_and_tool_progress(self) -> None:
        app = self.make_app()

        async with app.run_test(size=(100, 32)):
            app.post_agent_event(
                "model_request_started",
                {"step": 2, "max_steps": 30, "finalizing": False},
            )
            activity = str(app.query_one("#activity", Static).render())
            self.assertIn("Model step 2/30", activity)
            self.assertIn("waiting for model", activity)

            assert app._activity_started_at is not None
            app._activity_started_at -= 65
            app._refresh_activity_elapsed()
            activity = str(app.query_one("#activity", Static).render())
            self.assertIn("1m 05s elapsed", activity)

            app.post_agent_event(
                "tool_started",
                {"step": 2, "name": "run_command"},
            )
            activity = str(app.query_one("#activity", Static).render())
            self.assertIn("Model step 2", activity)
            self.assertIn("Running command", activity)

    async def test_copies_latest_raw_agent_response(self) -> None:
        app = self.make_app()
        answer = "Use this code:\n\n```python\nprint('hello')\n```"

        async with app.run_test(size=(100, 32)) as pilot:
            app.post_agent_event(
                "model_response",
                {"tool_calls": [], "text": answer},
            )
            await pilot.press("ctrl+y")
            await pilot.pause()

            self.assertEqual(app.last_assistant_text, answer)
            self.assertEqual(app._clipboard, answer)

    async def test_copy_local_command_and_new_session_reset(self) -> None:
        app = self.make_app()

        async with app.run_test(size=(100, 32)) as pilot:
            app.post_agent_event(
                "model_response",
                {"tool_calls": [], "text": "Latest answer"},
            )
            app.submit_prompt("/copy")
            await pilot.pause()
            self.assertEqual(app._clipboard, "Latest answer")

            app.submit_prompt("/new")
            await pilot.pause()
            self.assertIsNone(app.last_assistant_text)

    async def test_provisional_final_is_hidden_while_verification_runs(self) -> None:
        app = self.make_app()

        async with app.run_test(size=(100, 32)):
            app.post_agent_event(
                "model_response",
                {
                    "tool_calls": [],
                    "text": "Unverified final answer",
                    "provisional": True,
                },
            )

            rendered = "\n".join(
                line.text for line in app.query_one("#conversation", RichLog).lines
            )
            activity = str(app.query_one("#activity", Static).render())
            self.assertNotIn("Unverified final answer", rendered)
            self.assertIn("Verifying project", activity)

    async def test_approval_modal_resolves_future(self) -> None:
        app = self.make_app()
        request = PermissionRequest(
            tool_name="run_command",
            kind=PermissionKind.EXECUTE,
            description="run argv=['python', '-m', 'unittest']",
        )
        future: Future[ApprovalDecision] = Future()

        async with app.run_test(size=(100, 32)) as pilot:
            app.show_approval(request, future)
            await pilot.pause()

            self.assertIsInstance(app.screen, ApprovalScreen)
            await pilot.click("#allow")
            await pilot.pause()
            self.assertEqual(
                future.result(timeout=1), ApprovalDecision.ALLOW_ONCE
            )

    async def test_approval_modal_can_allow_for_task(self) -> None:
        app = self.make_app()
        request = PermissionRequest(
            tool_name="delete_file",
            kind=PermissionKind.DELETE,
            description="delete old.py",
        )
        future: Future[ApprovalDecision] = Future()

        async with app.run_test(size=(100, 32)) as pilot:
            app.show_approval(request, future)
            await pilot.pause()
            await pilot.click("#allow-task")
            await pilot.pause()

            self.assertEqual(
                future.result(timeout=1), ApprovalDecision.ALLOW_TASK
            )

    async def test_agent_worker_waits_for_tui_approval(self) -> None:
        approval_stub: ApprovalStubAgent | None = None

        def factory(events: Any, approval: Any) -> ApprovalStubAgent:
            nonlocal approval_stub
            approval_stub = ApprovalStubAgent(events, approval)
            return approval_stub

        app = CodingAgentApp(
            agent_factory=factory,  # type: ignore[arg-type]
            workspace=self.workspace,
            model_name="fake-model",
            approval_mode="ask",
        )

        async with app.run_test(size=(100, 32)) as pilot:
            app.submit_prompt("Run tests")
            for _ in range(100):
                await pilot.pause(0.01)
                if isinstance(app.screen, ApprovalScreen):
                    break
            self.assertIsInstance(app.screen, ApprovalScreen)

            await pilot.pause()
            await pilot.click("#allow")
            await pilot.pause()
            for _ in range(100):
                await pilot.pause(0.01)
                if not app.busy:
                    break

            self.assertIsNotNone(approval_stub)
            self.assertTrue(approval_stub.approved)
            self.assertFalse(app.busy)

    async def test_agent_denies_approval_after_tui_timeout(self) -> None:
        approval_stub: ApprovalStubAgent | None = None

        def factory(events: Any, approval: Any) -> ApprovalStubAgent:
            nonlocal approval_stub
            approval_stub = ApprovalStubAgent(events, approval)
            return approval_stub

        app = CodingAgentApp(
            agent_factory=factory,  # type: ignore[arg-type]
            workspace=self.workspace,
            model_name="fake-model",
            approval_mode="ask",
            approval_timeout_s=0.02,
        )

        async with app.run_test(size=(100, 32)) as pilot:
            app.submit_prompt("Run tests")
            for _ in range(200):
                await pilot.pause(0.01)
                if not app.busy:
                    break

            self.assertIsNotNone(approval_stub)
            self.assertFalse(approval_stub.approved)
            self.assertFalse(app.busy)
            self.assertNotIsInstance(app.screen, ApprovalScreen)

    async def test_local_new_command_resets_session(self) -> None:
        app = self.make_app()
        app.session_id = "old-session"

        async with app.run_test(size=(100, 32)) as pilot:
            app.submit_prompt("/new")
            await pilot.pause()

            self.assertIsNone(app.session_id)
            self.assertEqual(self.stub.prompts, [])

    async def test_expected_failure_is_handled_without_worker_error(self) -> None:
        def factory(events: Any, approval: Any) -> FailingStubAgent:
            return FailingStubAgent(events)

        app = CodingAgentApp(
            agent_factory=factory,  # type: ignore[arg-type]
            workspace=self.workspace,
            model_name="fake-model",
            approval_mode="ask",
        )

        async with app.run_test(size=(100, 32)) as pilot:
            app.submit_prompt("Hello")
            for _ in range(100):
                await pilot.pause(0.01)
                if not app.busy:
                    break

            rendered = "\n".join(
                line.text for line in app.query_one("#conversation", RichLog).lines
            )
            self.assertFalse(app.busy)
            self.assertIn("Agent stopped: model authentication failed", rendered)
            self.assertNotIn("ModelRequestError", rendered)


if __name__ == "__main__":
    unittest.main()
