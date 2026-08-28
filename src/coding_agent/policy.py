"""Workspace access policy shared by local file tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import WorkspaceAccessError


@dataclass(frozen=True, slots=True)
class WorkspacePolicy:
    root: Path

    def __post_init__(self) -> None:
        resolved = self.root.expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(f"workspace is not a directory: {resolved}")
        object.__setattr__(self, "root", resolved)

    def resolve_read_path(self, raw_path: str) -> Path:
        requested = Path(raw_path)
        if requested.is_absolute():
            raise WorkspaceAccessError("absolute paths are not allowed")

        candidate = (self.root / requested).resolve(strict=False)
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceAccessError("path escapes the workspace") from exc

        if not candidate.exists():
            raise WorkspaceAccessError(f"path does not exist: {raw_path}")
        if not candidate.is_file():
            raise WorkspaceAccessError(f"path is not a regular file: {raw_path}")
        return candidate

    def display_path(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

