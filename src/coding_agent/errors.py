"""Domain errors raised by the agent core."""


class CodingAgentError(Exception):
    """Base error for expected agent failures."""


class ModelRequestError(CodingAgentError):
    """A model request failed, with enough metadata to decide on retries."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


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


class SandboxUnavailableError(ToolError):
    """A required operating-system isolation backend is unavailable."""


class WebAccessError(ToolError):
    """A restricted web request was invalid, unsafe, or unsuccessful."""


class VerificationRequiredError(CodingAgentError):
    """Workspace changes could not pass the mandatory verification gate."""

