"""Local tool definitions, validation, and execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .conversation import Message
from .errors import ToolArgumentsError
from .model import ToolCall
from .policy import WorkspacePolicy


JSONSchema = dict[str, Any]
ToolHandler = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: JSONSchema
    handler: ToolHandler

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    tool_call_id: str
    name: str
    ok: bool
    output: Any = None
    error: str | None = None

    def to_message(self) -> Message:
        payload: dict[str, Any] = {"ok": self.ok}
        if self.ok:
            payload["output"] = self.output
        else:
            payload["error"] = self.error
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": json.dumps(payload, ensure_ascii=False),
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ValueError(f"tool already registered: {definition.name}")
        self._tools[definition.name] = definition

    def schemas(self) -> list[dict[str, Any]]:
        return [definition.schema() for definition in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolExecutionResult:
        definition = self._tools.get(call.name)
        if definition is None:
            return ToolExecutionResult(
                tool_call_id=call.id,
                name=call.name,
                ok=False,
                error=f"unknown tool: {call.name}",
            )

        try:
            validate_json_value(definition.parameters, call.arguments, path="arguments")
            output = definition.handler(call.arguments)
        except Exception as exc:
            return ToolExecutionResult(
                tool_call_id=call.id,
                name=call.name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        return ToolExecutionResult(
            tool_call_id=call.id,
            name=call.name,
            ok=True,
            output=output,
        )


def validate_json_value(schema: JSONSchema, value: Any, *, path: str) -> None:
    """Validate the JSON Schema subset used by local tools."""

    expected = schema.get("type")
    type_ok = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(expected, True)
    if not type_ok:
        raise ToolArgumentsError(f"{path} must be {expected}")

    if "enum" in schema and value not in schema["enum"]:
        raise ToolArgumentsError(f"{path} must be one of {schema['enum']!r}")

    if expected == "object":
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                raise ToolArgumentsError(f"{path}.{required} is required")
        if schema.get("additionalProperties") is False:
            unknown = set(value) - set(properties)
            if unknown:
                raise ToolArgumentsError(
                    f"{path} contains unknown fields: {sorted(unknown)!r}"
                )
        for key, child in value.items():
            if key in properties:
                validate_json_value(properties[key], child, path=f"{path}.{key}")

    if expected == "array" and "items" in schema:
        for index, child in enumerate(value):
            validate_json_value(schema["items"], child, path=f"{path}[{index}]")

    if expected in {"integer", "number"}:
        if "minimum" in schema and value < schema["minimum"]:
            raise ToolArgumentsError(f"{path} must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ToolArgumentsError(f"{path} must be <= {schema['maximum']}")


def create_read_only_registry(
    workspace: str | Path,
    *,
    max_file_bytes: int = 64 * 1024,
) -> ToolRegistry:
    policy = WorkspacePolicy(Path(workspace))
    registry = ToolRegistry()

    def read_file(arguments: dict[str, Any]) -> dict[str, Any]:
        path = policy.resolve_read_path(arguments["path"])
        start_line = arguments.get("start_line", 1)
        end_line = arguments.get("end_line")
        if end_line is not None and end_line < start_line:
            raise ToolArgumentsError("end_line must be greater than or equal to start_line")

        with path.open("rb") as stream:
            raw = stream.read(max_file_bytes + 1)
        truncated = len(raw) > max_file_bytes
        text = raw[:max_file_bytes].decode("utf-8", errors="replace")
        lines = text.splitlines()
        selected = lines[start_line - 1 : end_line]

        return {
            "path": policy.display_path(path),
            "start_line": start_line,
            "end_line": start_line + max(len(selected) - 1, 0),
            "content": "\n".join(selected),
            "truncated": truncated,
        }

    registry.register(
        ToolDefinition(
            name="read_file",
            description=(
                "Read a UTF-8 text file inside the workspace. Paths must be "
                "relative to the workspace root."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative file path",
                    },
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "First one-based line to return",
                    },
                    "end_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Last one-based line to return, inclusive",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=read_file,
        )
    )
    return registry

