"""MCP server components for WebShop+ green agent.

This package provides a FastMCP server for WebShop tool execution with
session-scoped state management.

Architecture:
- One global FastMCP instance (mcp) handles all requests
- Sessions are isolated via contextvar (current_session_id)
- SessionManager manages session lifecycle and state registration
"""

from .session_state import SessionState
from .session_manager import SessionManager
from .server import (
    mcp,
    current_session_id,
    register_session,
    unregister_session,
    get_session_state,
    is_session_registered,
    wait_for_completion,
    get_final_result,
    is_session_completed,
    signal_completion,
    get_mcp_app,
)

__all__ = [
    "SessionState",
    "SessionManager",
    "mcp",
    "current_session_id",
    "register_session",
    "unregister_session",
    "get_session_state",
    "is_session_registered",
    "wait_for_completion",
    "get_final_result",
    "is_session_completed",
    "signal_completion",
    "get_mcp_app",
]
