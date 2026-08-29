"""Domain errors raised by the agent core."""


class CodingAgentError(Exception):
    """Base error for expected agent failures."""


class ModelRequestError(CodingAgentError):
    """The model request failed after the configured retries."""


class ModelProtocolError(CodingAgentError):
    """The model returned a response that cannot be interpreted safely."""


class MaxStepsExceeded(CodingAgentError):
    """The agent exhausted its step budget."""


class LoopDetected(CodingAgentError):
    """The agent repeated the same action too many times."""


class ToolError(CodingAgentError):
    """Base error for tool validation or execution failures."""


class ToolArgumentsError(ToolError):
    """Tool arguments do not match the declared schema."""


class WorkspaceAccessError(ToolError):
    """A tool attempted to access a path outside the workspace."""


class PermissionDenied(ToolError):
    """A local permission policy rejected a tool operation."""

