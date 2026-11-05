"""Single error type the CLI raises for invalid contract input."""

from __future__ import annotations


class ContractError(Exception):
    """Raised when CLI contract input is invalid or cannot be resolved."""
