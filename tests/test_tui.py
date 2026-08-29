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
    from coding_agent.permissions import PermissionKind, PermissionRequest
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

            self.assertIn(str(self.workspace), context)
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

    async def test_approval_modal_resolves_future(self) -> None:
        app = self.make_app()
        request = PermissionRequest(
            tool_name="run_command",
            kind=PermissionKind.EXECUTE,
            description="run argv=['python', '-m', 'unittest']",
        )
        future: Future[bool] = Future()

        async with app.run_test(size=(100, 32)) as pilot:
            app.show_approval(request, future)
            await pilot.pause()

            self.assertIsInstance(app.screen, ApprovalScreen)
            await pilot.click("#allow")
            await pilot.pause()
            self.assertTrue(future.result(timeout=1))

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

    async def test_local_new_command_resets_session(self) -> None:
        app = self.make_app()
        app.session_id = "old-session"

        async with app.run_test(size=(100, 32)) as pilot:
            app.submit_prompt("/new")
            await pilot.pause()

            self.assertIsNone(app.session_id)
            self.assertEqual(self.stub.prompts, [])


if __name__ == "__main__":
    unittest.main()
