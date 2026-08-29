"""Minimal, framework-free coding agent."""

from .agent import Agent, AgentConfig, AgentResult
from .conversation import ConversationState
from .context import ContextBuilder, ContextConfig
from .model import (
    DeepSeekV4ProClient,
    ModelClient,
    ModelResponse,
    OpenAIChatClient,
    ToolCall,
)
from .tools import ToolRegistry, create_read_only_registry, create_workspace_registry
from .storage import HistorySearchResult, SqliteConversationStore

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "ConversationState",
    "ContextBuilder",
    "ContextConfig",
    "DeepSeekV4ProClient",
    "HistorySearchResult",
    "ModelClient",
    "ModelResponse",
    "OpenAIChatClient",
    "ToolCall",
    "ToolRegistry",
    "SqliteConversationStore",
    "create_read_only_registry",
    "create_workspace_registry",
]
