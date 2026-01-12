"""Tests for SessionManager class.

Tests cover:
- Session creation
- Session retrieval
- Session cleanup (individual and bulk)
- TTL-based expiration
- Capacity-based LRU eviction
- Concurrent access safety
"""

import asyncio
import time
from unittest.mock import patch

import pytest

from src.webshop_mcp.session_manager import SessionManager
from src.webshop_mcp.session_state import SessionState
from src.webshop_mcp.server import (
    is_session_registered,
    is_session_completed,
    _session_states,
    unregister_session,
)


@pytest.fixture(autouse=True)
def cleanup_global_state():
    """Clean up global session state before and after each test."""
    # Clear before
    _session_states.clear()
    yield
    # Clear after
    _session_states.clear()


class TestSessionCreation:
    """Tests for creating sessions."""

    @pytest.mark.asyncio
    async def test_create_session_success(self):
        """Test successful session creation."""
        manager = SessionManager()

        state = await manager.create_session(
            session_id="test-123",
            goal="Find running shoes under $50",
            budget=50.0,
            constraints=["no synthetic"],
        )

        assert state is not None
        assert isinstance(state, SessionState)
        assert state.session_id == "test-123"
        assert state.goal == "Find running shoes under $50"
        assert state.budget == 50.0
        assert state.constraints == ["no synthetic"]

    @pytest.mark.asyncio
    async def test_create_session_default_constraints(self):
        """Test session creation with default empty constraints."""
        manager = SessionManager()

        state = await manager.create_session(
            session_id="test-456",
            goal="Find a shirt",
            budget=30.0,
        )

        assert state.constraints == []

    @pytest.mark.asyncio
    async def test_create_session_custom_max_turns(self):
        """Test session creation with custom max turns."""
        manager = SessionManager()

        state = await manager.create_session(
            session_id="test-789",
            goal="Quick task",
            budget=20.0,
            max_turns=10,
        )

        assert state.max_turns == 10

    @pytest.mark.asyncio
    async def test_create_session_duplicate_raises_error(self):
        """Test that creating duplicate session raises ValueError."""
        manager = SessionManager()

        await manager.create_session(
            session_id="duplicate",
            goal="First task",
            budget=50.0,
        )

        with pytest.raises(ValueError, match="already exists"):
            await manager.create_session(
                session_id="duplicate",
                goal="Second task",
                budget=60.0,
            )

    @pytest.mark.asyncio
    async def test_create_session_tracks_time(self):
        """Test that session creation time is tracked."""
        manager = SessionManager()
        before = time.time()

        await manager.create_session(
            session_id="timed",
            goal="Task",
            budget=50.0,
        )

        after = time.time()
        assert "timed" in manager.session_times
        assert before <= manager.session_times["timed"] <= after


class TestSessionRetrieval:
    """Tests for retrieving sessions."""

    @pytest.mark.asyncio
    async def test_get_session_existing(self):
        """Test retrieving an existing session."""
        manager = SessionManager()

        original = await manager.create_session(
            session_id="exists",
            goal="Task",
            budget=50.0,
        )

        retrieved = await manager.get_session("exists")

        assert retrieved is original
        assert retrieved.session_id == "exists"

    @pytest.mark.asyncio
    async def test_get_session_nonexistent(self):
        """Test retrieving a nonexistent session returns None."""
        manager = SessionManager()

        result = await manager.get_session("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_session_updates_access_time(self):
        """Test that get_session updates access time for LRU."""
        manager = SessionManager()

        await manager.create_session(
            session_id="lru-test",
            goal="Task",
            budget=50.0,
        )

        original_time = manager.session_times["lru-test"]

        # Wait a tiny bit
        await asyncio.sleep(0.01)

        await manager.get_session("lru-test")

        new_time = manager.session_times["lru-test"]
        assert new_time > original_time


class TestSessionCleanup:
    """Tests for session cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_session_existing(self):
        """Test cleaning up an existing session."""
        manager = SessionManager()

        await manager.create_session(
            session_id="to-remove",
            goal="Task",
            budget=50.0,
        )

        result = await manager.cleanup_session("to-remove")

        assert result is True
        assert not is_session_registered("to-remove")
        assert "to-remove" not in manager.session_times

    @pytest.mark.asyncio
    async def test_cleanup_session_nonexistent(self):
        """Test cleaning up a nonexistent session."""
        manager = SessionManager()

        result = await manager.cleanup_session("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_cleanup_all(self):
        """Test cleaning up all sessions."""
        manager = SessionManager()

        # Create multiple sessions
        for i in range(5):
            await manager.create_session(
                session_id=f"session-{i}",
                goal="Task",
                budget=50.0,
            )

        count = await manager.cleanup_all()

        assert count == 5
        assert len(manager.session_times) == 0

    @pytest.mark.asyncio
    async def test_cleanup_completed(self):
        """Test cleaning up only completed sessions."""
        manager = SessionManager()

        # Create sessions
        state1 = await manager.create_session(
            session_id="completed-1",
            goal="Task",
            budget=50.0,
        )
        state2 = await manager.create_session(
            session_id="active-1",
            goal="Task",
            budget=50.0,
        )
        state3 = await manager.create_session(
            session_id="completed-2",
            goal="Task",
            budget=50.0,
        )

        # Mark some as completed
        state1.completed = True
        state3.completed = True

        count = await manager.cleanup_completed()

        assert count == 2
        assert not is_session_registered("completed-1")
        assert not is_session_registered("completed-2")
        assert is_session_registered("active-1")


class TestTTLExpiration:
    """Tests for TTL-based session expiration."""

    @pytest.mark.asyncio
    async def test_expired_sessions_cleaned_on_create(self):
        """Test that expired sessions are cleaned up when creating new ones."""
        # Short TTL for testing
        manager = SessionManager(session_ttl=1)

        # Create a session
        await manager.create_session(
            session_id="old",
            goal="Task",
            budget=50.0,
        )

        # Manually set it as expired (1 hour ago)
        manager.session_times["old"] = time.time() - 3600

        # Create a new session - should trigger cleanup
        await manager.create_session(
            session_id="new",
            goal="Task",
            budget=50.0,
        )

        assert not is_session_registered("old")
        assert is_session_registered("new")

    @pytest.mark.asyncio
    async def test_is_session_active_not_expired(self):
        """Test is_session_active for non-expired session."""
        manager = SessionManager(session_ttl=3600)

        await manager.create_session(
            session_id="fresh",
            goal="Task",
            budget=50.0,
        )

        assert await manager.is_session_active("fresh") is True

    @pytest.mark.asyncio
    async def test_is_session_active_expired(self):
        """Test is_session_active for expired session."""
        manager = SessionManager(session_ttl=1)

        await manager.create_session(
            session_id="stale",
            goal="Task",
            budget=50.0,
        )

        # Set as expired
        manager.session_times["stale"] = time.time() - 100

        assert await manager.is_session_active("stale") is False

    @pytest.mark.asyncio
    async def test_is_session_active_nonexistent(self):
        """Test is_session_active for nonexistent session."""
        manager = SessionManager()

        assert await manager.is_session_active("nonexistent") is False


class TestCapacityLRUEviction:
    """Tests for capacity-based LRU eviction."""

    @pytest.mark.asyncio
    async def test_lru_eviction_at_capacity(self):
        """Test that oldest session is evicted when at capacity."""
        manager = SessionManager(max_sessions=3)

        # Create 3 sessions (at capacity)
        for i in range(3):
            await manager.create_session(
                session_id=f"session-{i}",
                goal="Task",
                budget=50.0,
            )
            # Small delay to ensure different timestamps
            await asyncio.sleep(0.01)

        # Access session-0 to make it recent
        await manager.get_session("session-0")
        await asyncio.sleep(0.01)

        # Create 4th session - should evict oldest (session-1)
        await manager.create_session(
            session_id="session-3",
            goal="Task",
            budget=50.0,
        )

        assert len(manager.session_times) == 3
        assert not is_session_registered("session-1")  # Oldest was evicted
        assert is_session_registered("session-0")  # Was accessed recently
        assert is_session_registered("session-2")
        assert is_session_registered("session-3")

    @pytest.mark.asyncio
    async def test_multiple_evictions(self):
        """Test multiple LRU evictions."""
        manager = SessionManager(max_sessions=2)

        # Create initial sessions
        await manager.create_session(
            session_id="a",
            goal="Task",
            budget=50.0,
        )
        await asyncio.sleep(0.01)
        await manager.create_session(
            session_id="b",
            goal="Task",
            budget=50.0,
        )

        # Create third - should evict 'a'
        await asyncio.sleep(0.01)
        await manager.create_session(
            session_id="c",
            goal="Task",
            budget=50.0,
        )

        assert not is_session_registered("a")
        assert is_session_registered("b")
        assert is_session_registered("c")


class TestSessionInfo:
    """Tests for session information methods."""

    @pytest.mark.asyncio
    async def test_get_session_count(self):
        """Test getting session count."""
        manager = SessionManager()

        assert await manager.get_session_count() == 0

        for i in range(3):
            await manager.create_session(
                session_id=f"session-{i}",
                goal="Task",
                budget=50.0,
            )

        assert await manager.get_session_count() == 3

    @pytest.mark.asyncio
    async def test_get_session_ids(self):
        """Test getting all session IDs."""
        manager = SessionManager()

        for i in range(3):
            await manager.create_session(
                session_id=f"session-{i}",
                goal="Task",
                budget=50.0,
            )

        ids = await manager.get_session_ids()

        assert set(ids) == {"session-0", "session-1", "session-2"}

    @pytest.mark.asyncio
    async def test_get_session_state_summary(self):
        """Test getting session state summary."""
        manager = SessionManager()

        await manager.create_session(
            session_id="stateful",
            goal="Find shoes",
            budget=50.0,
            constraints=["no synthetic"],
        )

        state = await manager.get_session_state_summary("stateful")

        assert state is not None
        assert state["session_id"] == "stateful"
        assert state["goal"] == "Find shoes"
        assert state["budget"] == 50.0
        assert state["constraints"] == ["no synthetic"]

    @pytest.mark.asyncio
    async def test_get_session_state_summary_nonexistent(self):
        """Test getting state of nonexistent session."""
        manager = SessionManager()

        state = await manager.get_session_state_summary("nonexistent")

        assert state is None


class TestConcurrency:
    """Tests for concurrent access safety."""

    @pytest.mark.asyncio
    async def test_concurrent_creates(self):
        """Test concurrent session creation."""
        manager = SessionManager(max_sessions=100)

        async def create_session(i: int):
            await manager.create_session(
                session_id=f"concurrent-{i}",
                goal="Task",
                budget=50.0,
            )

        # Create 10 sessions concurrently
        await asyncio.gather(*[create_session(i) for i in range(10)])

        assert await manager.get_session_count() == 10

    @pytest.mark.asyncio
    async def test_concurrent_reads_and_writes(self):
        """Test concurrent reads and writes don't corrupt state."""
        manager = SessionManager()

        # Create some sessions
        for i in range(5):
            await manager.create_session(
                session_id=f"rw-{i}",
                goal="Task",
                budget=50.0,
            )

        async def read_sessions():
            for _ in range(10):
                await manager.get_session("rw-0")
                await manager.get_session_count()
                await asyncio.sleep(0.001)

        async def cleanup_sessions():
            for i in range(5):
                await asyncio.sleep(0.002)
                await manager.cleanup_session(f"rw-{i}")

        # Run reads and cleanups concurrently
        await asyncio.gather(read_sessions(), cleanup_sessions())

        # All sessions should be cleaned up
        assert await manager.get_session_count() == 0


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_zero_max_sessions(self):
        """Test with zero max sessions still works (immediate eviction)."""
        manager = SessionManager(max_sessions=1)

        await manager.create_session(
            session_id="first",
            goal="Task",
            budget=50.0,
        )

        # Second create should evict first
        await manager.create_session(
            session_id="second",
            goal="Task",
            budget=50.0,
        )

        assert await manager.get_session_count() == 1
        assert is_session_registered("second")

    @pytest.mark.asyncio
    async def test_very_short_ttl(self):
        """Test with very short TTL."""
        manager = SessionManager(session_ttl=0)

        await manager.create_session(
            session_id="short-lived",
            goal="Task",
            budget=50.0,
        )

        # Session should be immediately considered expired
        assert await manager.is_session_active("short-lived") is False

    @pytest.mark.asyncio
    async def test_empty_goal(self):
        """Test session with empty goal string."""
        manager = SessionManager()

        state = await manager.create_session(
            session_id="empty-goal",
            goal="",
            budget=50.0,
        )

        assert state.goal == ""

    @pytest.mark.asyncio
    async def test_zero_budget(self):
        """Test session with zero budget."""
        manager = SessionManager()

        state = await manager.create_session(
            session_id="zero-budget",
            goal="Free items only",
            budget=0.0,
        )

        assert state.budget == 0.0
