"""Session manager for MCP server instances.

This module provides the SessionManager class that manages the lifecycle of
session-scoped MCP server instances. It handles session creation, retrieval,
expiration (TTL), and capacity limits (LRU eviction).
"""

import asyncio
import time
from typing import Any

from .server import WebShopMCPServer
from .session_state import SessionState


class SessionManager:
    """Manages MCP server instances for active sessions.

    This class provides lifecycle management for session-scoped MCP servers:
    - Create new sessions with unique IDs
    - Retrieve existing sessions by ID
    - Automatic cleanup of expired sessions (TTL-based)
    - Capacity limits with LRU eviction

    Example:
        manager = SessionManager(max_sessions=100, session_ttl=3600)

        # Create a new session
        server = await manager.create_session(
            session_id="abc123",
            goal="Find running shoes under $50",
            budget=50.0,
            constraints=["no synthetic"],
        )

        # Retrieve session later
        server = await manager.get_session("abc123")

        # Cleanup when done
        await manager.cleanup_session("abc123")

    Attributes:
        sessions: Dict mapping session IDs to WebShopMCPServer instances.
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
        self.sessions: dict[str, WebShopMCPServer] = {}
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
    ) -> WebShopMCPServer:
        """Create a new session with an MCP server.

        Args:
            session_id: Unique identifier for this session.
            goal: The shopping task goal.
            budget: Maximum allowed spending amount.
            constraints: Optional list of constraints.
            max_turns: Maximum actions before session terminates. Default: 30.
            webshop: Optional WebShop interface (for testing).

        Returns:
            WebShopMCPServer instance for this session.

        Raises:
            ValueError: If session_id already exists.
        """
        async with self._lock:
            # Check if session already exists
            if session_id in self.sessions:
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

            # Create MCP server for this session
            server = WebShopMCPServer(state, webshop=webshop)

            # Store session
            self.sessions[session_id] = server
            self.session_times[session_id] = time.time()

            return server

    async def get_session(self, session_id: str) -> WebShopMCPServer | None:
        """Get an existing session by ID.

        Args:
            session_id: The session identifier to look up.

        Returns:
            WebShopMCPServer if session exists, None otherwise.
        """
        async with self._lock:
            # Update access time for LRU tracking
            if session_id in self.sessions:
                self.session_times[session_id] = time.time()
            return self.sessions.get(session_id)

    async def cleanup_session(self, session_id: str) -> bool:
        """Remove a session and release resources.

        Args:
            session_id: The session identifier to clean up.

        Returns:
            True if session was found and removed, False otherwise.
        """
        async with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                del self.session_times[session_id]
                return True
            return False

    async def get_session_count(self) -> int:
        """Get the number of active sessions.

        Returns:
            Current number of active sessions.
        """
        return len(self.sessions)

    async def get_session_ids(self) -> list[str]:
        """Get list of all active session IDs.

        Returns:
            List of session ID strings.
        """
        return list(self.sessions.keys())

    async def is_session_active(self, session_id: str) -> bool:
        """Check if a session exists and is active.

        Args:
            session_id: The session identifier to check.

        Returns:
            True if session exists and not expired, False otherwise.
        """
        if session_id not in self.sessions:
            return False

        # Check if expired
        age = time.time() - self.session_times[session_id]
        return age <= self.session_ttl

    async def get_session_state(self, session_id: str) -> dict[str, Any] | None:
        """Get the state of a session.

        Args:
            session_id: The session identifier.

        Returns:
            Session state summary dict, or None if session not found.
        """
        server = await self.get_session(session_id)
        if server is None:
            return None
        return server.state.get_summary()

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
            del self.sessions[sid]
            del self.session_times[sid]

        # LRU eviction if at capacity
        while len(self.sessions) >= self.max_sessions:
            # Find oldest session
            if not self.session_times:
                break
            oldest = min(self.session_times, key=self.session_times.get)  # type: ignore
            del self.sessions[oldest]
            del self.session_times[oldest]

    async def cleanup_all(self) -> int:
        """Remove all sessions.

        Returns:
            Number of sessions that were cleaned up.
        """
        async with self._lock:
            count = len(self.sessions)
            self.sessions.clear()
            self.session_times.clear()
            return count

    async def cleanup_completed(self) -> int:
        """Remove all completed sessions.

        Returns:
            Number of completed sessions that were cleaned up.
        """
        async with self._lock:
            completed = [
                sid for sid, server in self.sessions.items()
                if server.state.completed
            ]
            for sid in completed:
                del self.sessions[sid]
                del self.session_times[sid]
            return len(completed)
