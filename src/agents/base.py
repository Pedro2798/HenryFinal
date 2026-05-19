"""Shared agent infrastructure."""

from __future__ import annotations


class AgentError(RuntimeError):
    """Raised when an agent's LLM call fails or returns an invalid object."""
