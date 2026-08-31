"""Workspace access policy shared by local file tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import WorkspaceAccessError


VERIFICATION_CONFIG_NAME = ".coding-agent-verification.toml"


@dataclass(frozen=True, slots=True)
class WorkspacePolicy:
    root: Path

    def __post_init__(self) -> None:
        resolved = self.root.expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(f"workspace is not a directory: {resolved}")
        object.__setattr__(self, "root", resolved)

    def resolve_read_path(self, raw_path: str) -> Path:
        candidate = self.resolve_existing_path(raw_path)
        if not candidate.is_file():
            raise WorkspaceAccessError(f"path is not a regular file: {raw_path}")
        return candidate

    def resolve_existing_path(self, raw_path: str) -> Path:
        """Resolve an existing workspace-relative file or directory."""

        candidate = self.resolve_workspace_path(raw_path)
        if not candidate.exists():
            raise WorkspaceAccessError(f"path does not exist: {raw_path}")
        return candidate

    def resolve_workspace_path(self, raw_path: str) -> Path:
        """Resolve a workspace-relative path without requiring it to exist."""

        requested = Path(raw_path)
        if requested.is_absolute():
            raise WorkspaceAccessError("absolute paths are not allowed")

        candidate = (self.root / requested).resolve(strict=False)
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceAccessError("path escapes the workspace") from exc

        return candidate

    def resolve_mutation_path(self, raw_path: str) -> Path:
        """Resolve a writable file path without following the final path component."""

        requested = Path(raw_path)
        if requested.is_absolute():
            raise WorkspaceAccessError("absolute paths are not allowed")
        if not requested.name or requested.name in {".", ".."}:
            raise WorkspaceAccessError("path must name a file inside the workspace")

        parent = (self.root / requested).parent.resolve(strict=False)
        try:
            parent.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceAccessError("path escapes the workspace") from exc

        candidate = parent / requested.name
        relative = candidate.relative_to(self.root)
        protected_root = relative.parts[0].casefold() if relative.parts else ""
        if protected_root in {".git", ".coding-agent"} or (
            len(relative.parts) == 1
            and protected_root == VERIFICATION_CONFIG_NAME.casefold()
        ):
            raise WorkspaceAccessError(
                "repository metadata, agent state, and verification policy paths "
                "cannot be modified"
            )
        if candidate.is_symlink():
            raise WorkspaceAccessError("symbolic links cannot be modified or deleted")
        return candidate

    def display_path(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

