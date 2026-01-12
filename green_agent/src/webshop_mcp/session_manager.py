"""Session manager for MCP server sessions.

This module provides the SessionManager class that manages the lifecycle of
session state for the global MCP server. It handles session creation, retrieval,
expiration (TTL), and capacity limits (LRU eviction).

Note: This manager no longer creates per-session MCP servers. Instead, it
manages SessionState objects that are used by the single global FastMCP server
via contextvars for session isolation.
"""

import asyncio
import time
from typing import Any

import structlog

from .server import (
    get_final_result,
    is_session_completed,
    is_session_registered,
    register_session,
    unregister_session,
    wait_for_completion,
    get_session_state,
)
from .session_state import SessionState

logger = structlog.get_logger()


class SessionManager:
    """Manages session state for the global MCP server.

    This class provides lifecycle management for session-scoped state:
    - Create new sessions with unique IDs
    - Retrieve existing sessions by ID
    - Automatic cleanup of expired sessions (TTL-based)
    - Capacity limits with LRU eviction

    Example:
        manager = SessionManager(max_sessions=100, session_ttl=3600)

        # Create a new session
        state = await manager.create_session(
            session_id="abc123",
            goal="Find running shoes under $50",
            budget=50.0,
            constraints=["no synthetic"],
        )

        # Retrieve session later
        state = await manager.get_session("abc123")

        # Cleanup when done
        await manager.cleanup_session("abc123")

    Attributes:
        session_times: Dict mapping session IDs to creation timestamps.
        max_sessions: Maximum number of concurrent sessions allowed.
        session_ttl: Time-to-live in seconds before session expires.
    """

    def __init__(
        self,
        max_sessions: int = 100,
        session_ttl: int = 3600,
    ):
        """Initialize the session manager.

        Args:
            max_sessions: Maximum number of concurrent sessions. When exceeded,
                oldest sessions are evicted (LRU). Default: 100.
            session_ttl: Time-to-live in seconds for sessions. Sessions older
                than this are automatically cleaned up. Default: 3600 (1 hour).
        """
        self.session_times: dict[str, float] = {}
        self.max_sessions = max_sessions
        self.session_ttl = session_ttl
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        session_id: str,
        goal: str,
        budget: float,
        constraints: list[str] | None = None,
        max_turns: int = 30,
        webshop: Any | None = None,
    ) -> SessionState:
        """Create a new session with state registered to the global MCP server.

        Args:
            session_id: Unique identifier for this session.
            goal: The shopping task goal.
            budget: Maximum allowed spending amount.
            constraints: Optional list of constraints.
            max_turns: Maximum actions before session terminates. Default: 30.
            webshop: Optional WebShop interface (for testing).

        Returns:
            SessionState instance for this session.

        Raises:
            ValueError: If session_id already exists.
        """
        async with self._lock:
            # Check if session already exists
            if is_session_registered(session_id):
                raise ValueError(f"Session '{session_id}' already exists")

            # Cleanup expired and excess sessions
            await self._cleanup_if_needed()

            # Create session state
            state = SessionState(
                session_id=session_id,
                goal=goal,
                budget=budget,
                constraints=constraints or [],
                max_turns=max_turns,
            )

            # Register with global MCP server
            register_session(session_id, state, webshop)

            # Track session time for LRU
            self.session_times[session_id] = time.time()

            logger.info(
                "SessionManager: session created",
                session_id=session_id,
                manager_id=id(self),
                sessions_count=len(self.session_times),
            )

            return state

    async def get_session(self, session_id: str) -> SessionState | None:
        """Get an existing session's state by ID.

        Args:
            session_id: The session identifier to look up.

        Returns:
            SessionState if session exists, None otherwise.
        """
        async with self._lock:
            # Update access time for LRU tracking
            found = is_session_registered(session_id)
            logger.info(
                "SessionManager: get_session",
                session_id=session_id,
                found=found,
                manager_id=id(self),
                all_sessions=list(self.session_times.keys()),
            )
            if found:
                self.session_times[session_id] = time.time()
            return get_session_state(session_id)

    async def cleanup_session(self, session_id: str) -> bool:
        """Remove a session and release resources.

        Args:
            session_id: The session identifier to clean up.

        Returns:
            True if session was found and removed, False otherwise.
        """
        async with self._lock:
            if is_session_registered(session_id):
                unregister_session(session_id)
                self.session_times.pop(session_id, None)
                return True
            return False

    async def get_session_count(self) -> int:
        """Get the number of active sessions.

        Returns:
            Current number of active sessions.
        """
        return len(self.session_times)

    async def get_session_ids(self) -> list[str]:
        """Get list of all active session IDs.

        Returns:
            List of session ID strings.
        """
        return list(self.session_times.keys())

    async def is_session_active(self, session_id: str) -> bool:
        """Check if a session exists and is active.

        Args:
            session_id: The session identifier to check.

        Returns:
            True if session exists and not expired, False otherwise.
        """
        if not is_session_registered(session_id):
            return False

        # Check if expired
        age = time.time() - self.session_times.get(session_id, 0)
        return age <= self.session_ttl

    async def get_session_state_summary(self, session_id: str) -> dict[str, Any] | None:
        """Get the state summary of a session.

        Args:
            session_id: The session identifier.

        Returns:
            Session state summary dict, or None if session not found.
        """
        state = get_session_state(session_id)
        if state is None:
            return None
        return state.get_summary()

    async def wait_for_session_completion(
        self, session_id: str, timeout: float | None = None
    ) -> dict[str, Any]:
        """Wait for a session to complete.

        Args:
            session_id: Session to wait for.
            timeout: Maximum wait time in seconds.

        Returns:
            Final evaluation result.

        Raises:
            asyncio.TimeoutError: If timeout reached.
            ValueError: If session not found.
        """
        return await wait_for_completion(session_id, timeout)

    def is_session_completed(self, session_id: str) -> bool:
        """Check if session is completed.

        Args:
            session_id: Session identifier.

        Returns:
            True if session has completed.
        """
        return is_session_completed(session_id)

    def get_final_result(self, session_id: str) -> dict[str, Any] | None:
        """Get final result for a completed session.

        Args:
            session_id: Session identifier.

        Returns:
            Final result if completed, None otherwise.
        """
        return get_final_result(session_id)

    async def _cleanup_if_needed(self) -> None:
        """Remove expired or excess sessions.

        This method is called internally during create_session.
        It removes:
        1. Sessions older than session_ttl
        2. Oldest sessions if at max_sessions capacity (LRU eviction)
        """
        now = time.time()

        # Remove expired sessions
        expired = [
            sid for sid, t in self.session_times.items()
            if now - t > self.session_ttl
        ]
        for sid in expired:
            unregister_session(sid)
            del self.session_times[sid]

        # LRU eviction if at capacity
        while len(self.session_times) >= self.max_sessions:
            # Find oldest session
            if not self.session_times:
                break
            oldest = min(self.session_times, key=self.session_times.get)  # type: ignore
            unregister_session(oldest)
            del self.session_times[oldest]

    async def cleanup_all(self) -> int:
        """Remove all sessions.

        Returns:
            Number of sessions that were cleaned up.
        """
        async with self._lock:
            count = len(self.session_times)
            for sid in list(self.session_times.keys()):
                unregister_session(sid)
            self.session_times.clear()
            return count

    async def cleanup_completed(self) -> int:
        """Remove all completed sessions.

        Returns:
            Number of completed sessions that were cleaned up.
        """
        async with self._lock:
            completed = [
                sid for sid in self.session_times.keys()
                if is_session_completed(sid)
            ]
            for sid in completed:
                unregister_session(sid)
                del self.session_times[sid]
            return len(completed)
