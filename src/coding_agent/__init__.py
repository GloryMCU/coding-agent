"""Minimal, framework-free coding agent."""

from .agent import Agent, AgentConfig, AgentResult, AgentStatus
from .conversation import ConversationState
from .context import ContextBuilder, ContextConfig
from .events import CompositeEventSink
from .model import (
    DeepSeekV4ProClient,
    ModelClient,
    ModelResponse,
    OpenAIChatClient,
    ToolCall,
)
from .permissions import (
    ApprovalDecision,
    AllowAllApprovalPolicy,
    ApprovalPolicy,
    DenyApprovalPolicy,
    InteractiveApprovalPolicy,
    PermissionKind,
    PermissionRequest,
    PermissionRule,
    PermissionRuleDecision,
    PermissionRuleEngine,
    RuleBasedApprovalPolicy,
    create_approval_policy,
)
from .tools import ToolRegistry, create_read_only_registry, create_workspace_registry
from .storage import HistorySearchResult, SqliteConversationStore

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "AgentStatus",
    "ApprovalDecision",
    "AllowAllApprovalPolicy",
    "ApprovalPolicy",
    "CompositeEventSink",
    "ConversationState",
    "ContextBuilder",
    "ContextConfig",
    "DeepSeekV4ProClient",
    "DenyApprovalPolicy",
    "HistorySearchResult",
    "InteractiveApprovalPolicy",
    "ModelClient",
    "ModelResponse",
    "OpenAIChatClient",
    "PermissionKind",
    "PermissionRequest",
    "PermissionRule",
    "PermissionRuleDecision",
    "PermissionRuleEngine",
    "RuleBasedApprovalPolicy",
    "ToolCall",
    "ToolRegistry",
    "SqliteConversationStore",
    "create_read_only_registry",
    "create_approval_policy",
    "create_workspace_registry",
]
