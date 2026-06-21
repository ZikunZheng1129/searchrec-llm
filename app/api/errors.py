"""API error helpers for local demo artifact handling."""

from __future__ import annotations

from pathlib import Path


class ArtifactMissingError(FileNotFoundError):
    """Raised when a demo artifact is required but missing."""

    def __init__(self, path: str | Path, command_hint: str | None = None) -> None:
        self.path = str(path)
        self.command_hint = command_hint
        super().__init__(artifact_missing_message(path, command_hint))


class InvalidDemoRequestError(ValueError):
    """Raised when a demo request cannot be handled safely."""


def artifact_missing_message(path: str | Path, command_hint: str | None = None) -> str:
    """Build a user-facing missing artifact message."""
    message = f"Missing {path}."
    if command_hint:
        message += f" Run: {command_hint}"
    return message
