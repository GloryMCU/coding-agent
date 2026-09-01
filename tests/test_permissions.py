from __future__ import annotations

import io
import unittest

from coding_agent.permissions import (
    ApprovalDecision,
    InteractiveApprovalPolicy,
    PermissionKind,
    PermissionRequest,
    PermissionRule,
    PermissionRuleDecision,
    PermissionRuleEngine,
    create_approval_policy,
)


class RecordingReviewer:
    def __init__(self, *decisions: ApprovalDecision) -> None:
        self.decisions = list(decisions)
        self.requests: list[PermissionRequest] = []

    def approve(self, request: PermissionRequest) -> ApprovalDecision:
        self.requests.append(request)
        return self.decisions.pop(0)


def write_request(
    *, tool_name: str = "write_file", resource: str = "src/module.py"
) -> PermissionRequest:
    return PermissionRequest(
        tool_name=tool_name,
        kind=PermissionKind.WRITE,
        description=f"write {resource}",
        resource=resource,
        task_scope="workspace:write",
    )


class PermissionRuleEngineTests(unittest.TestCase):
    def test_most_restrictive_matching_rule_wins(self) -> None:
        request = write_request(resource="secrets.env")
        engine = PermissionRuleEngine(
            [
                PermissionRule(
                    PermissionRuleDecision.ALLOW,
                    kinds=frozenset({PermissionKind.WRITE}),
                ),
                PermissionRule(
                    PermissionRuleDecision.ASK,
                    resource_pattern="*.env",
                ),
                PermissionRule(
                    PermissionRuleDecision.DENY,
                    tool_pattern="write_*",
                    resource_pattern="secrets.*",
                ),
            ],
            default=PermissionRuleDecision.ALLOW,
        )

        self.assertEqual(engine.evaluate(request), PermissionRuleDecision.DENY)

    def test_ask_beats_allow_and_default_applies_only_without_matches(self) -> None:
        request = write_request(resource="config.toml")
        engine = PermissionRuleEngine(
            [
                PermissionRule(
                    PermissionRuleDecision.ALLOW,
                    kinds=frozenset({PermissionKind.WRITE}),
                ),
                PermissionRule(
                    PermissionRuleDecision.ASK,
                    resource_pattern="*.toml",
                ),
            ],
            default=PermissionRuleDecision.DENY,
        )

        self.assertEqual(engine.evaluate(request), PermissionRuleDecision.ASK)
        self.assertEqual(
            engine.evaluate(
                PermissionRequest(
                    tool_name="delete_file",
                    kind=PermissionKind.DELETE,
                    description="delete notes.txt",
                )
            ),
            PermissionRuleDecision.DENY,
        )

    def test_command_prefix_and_sandbox_must_both_match(self) -> None:
        rule = PermissionRule(
            PermissionRuleDecision.ALLOW,
            command_prefix=("python", "-m", "unittest"),
            sandboxed=True,
        )
        engine = PermissionRuleEngine([rule])

        sandboxed = PermissionRequest(
            tool_name="run_command",
            kind=PermissionKind.EXECUTE,
            description="run tests",
            command=("python", "-m", "unittest", "discover"),
            sandboxed=True,
        )
        unsandboxed = PermissionRequest(
            tool_name="run_command",
            kind=PermissionKind.EXECUTE,
            description="run tests",
            command=sandboxed.command,
            sandboxed=False,
        )

        self.assertEqual(engine.evaluate(sandboxed), PermissionRuleDecision.ALLOW)
        self.assertEqual(engine.evaluate(unsandboxed), PermissionRuleDecision.ASK)


class ScopedApprovalPolicyTests(unittest.TestCase):
    def test_workspace_mode_allows_writes_and_sandboxed_commands(self) -> None:
        reviewer = RecordingReviewer(ApprovalDecision.DENY)
        policy = create_approval_policy("workspace", reviewer=reviewer)

        self.assertTrue(policy.approve(write_request()))
        self.assertTrue(
            policy.approve(
                PermissionRequest(
                    tool_name="run_command",
                    kind=PermissionKind.EXECUTE,
                    description="run tests",
                    command=("python", "-m", "unittest"),
                    sandboxed=True,
                )
            )
        )
        self.assertEqual(reviewer.requests, [])

    def test_workspace_mode_asks_for_delete_and_unsandboxed_execution(self) -> None:
        reviewer = RecordingReviewer(
            ApprovalDecision.ALLOW_ONCE,
            ApprovalDecision.DENY,
        )
        policy = create_approval_policy("workspace", reviewer=reviewer)

        self.assertTrue(
            policy.approve(
                PermissionRequest(
                    tool_name="delete_file",
                    kind=PermissionKind.DELETE,
                    description="delete old.py",
                    task_scope="workspace:delete",
                )
            )
        )
        self.assertFalse(
            policy.approve(
                PermissionRequest(
                    tool_name="run_command",
                    kind=PermissionKind.EXECUTE,
                    description="run tests on host",
                    command=("python", "-m", "unittest"),
                    sandboxed=False,
                    task_scope="workspace:execute:python",
                )
            )
        )
        self.assertEqual(len(reviewer.requests), 2)

    def test_task_approval_is_reused_then_cleared_for_next_task(self) -> None:
        reviewer = RecordingReviewer(
            ApprovalDecision.ALLOW_TASK,
            ApprovalDecision.DENY,
        )
        policy = create_approval_policy("ask", reviewer=reviewer)

        self.assertTrue(policy.approve(write_request()))
        self.assertTrue(policy.approve(write_request(tool_name="apply_patch")))
        self.assertEqual(len(reviewer.requests), 1)

        policy.begin_task()

        self.assertFalse(policy.approve(write_request(tool_name="apply_patch")))
        self.assertEqual(len(reviewer.requests), 2)

    def test_deny_rule_overrides_task_grant(self) -> None:
        reviewer = RecordingReviewer(ApprovalDecision.ALLOW_TASK)
        policy = create_approval_policy(
            "ask",
            reviewer=reviewer,
            rules=[
                PermissionRule(
                    PermissionRuleDecision.DENY,
                    kinds=frozenset({PermissionKind.WRITE}),
                    resource_pattern="*.env",
                )
            ],
        )

        self.assertTrue(policy.approve(write_request(resource="src/module.py")))
        self.assertFalse(policy.approve(write_request(resource="secrets.env")))
        self.assertEqual(len(reviewer.requests), 1)

    def test_deny_mode_cannot_be_relaxed_by_an_allow_rule(self) -> None:
        policy = create_approval_policy(
            "deny",
            rules=[
                PermissionRule(
                    PermissionRuleDecision.ALLOW,
                    kinds=frozenset({PermissionKind.WRITE}),
                )
            ],
        )

        self.assertFalse(policy.approve(write_request()))

    def test_interactive_reviewer_supports_task_scope(self) -> None:
        reviewer = InteractiveApprovalPolicy(
            input_fn=lambda _: "t",
            output=io.StringIO(),
        )

        self.assertEqual(
            reviewer.approve(write_request()),
            ApprovalDecision.ALLOW_TASK,
        )


if __name__ == "__main__":
    unittest.main()
