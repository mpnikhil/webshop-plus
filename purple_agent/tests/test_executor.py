"""
Tests for WebShop+ Purple Agent Executor.

This module contains tests for:
- Executor initialization
- Task execution via execute()
- TCK conformance test handling
- Cancel operation
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
import uuid

from a2a.types import (
    Message,
    Role,
    TaskState,
    TextPart,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_event_queue():
    """Create a mock event queue."""
    queue = AsyncMock()
    queue.enqueue_event = AsyncMock()
    return queue


def create_message(text: str, role: Role = Role.user, message_id: str | None = None) -> Message:
    """Helper to create a valid SDK Message with all required fields."""
    return Message(
        messageId=message_id or str(uuid.uuid4()),
        role=role,
        parts=[TextPart(text=text)],
    )


@pytest.fixture
def mock_request_context():
    """Create a mock request context with a simple message."""
    context = MagicMock()
    context.task_id = str(uuid.uuid4())
    context.context_id = str(uuid.uuid4())
    context.message = create_message("Hello world")
    context.metadata = None
    context.current_task = None
    return context


@pytest.fixture
def mock_request_context_no_message():
    """Create a mock request context without a message."""
    context = MagicMock()
    context.task_id = str(uuid.uuid4())
    context.context_id = str(uuid.uuid4())
    context.message = None
    context.metadata = None
    context.current_task = None
    return context


# =============================================================================
# Executor Initialization Tests
# =============================================================================


class TestExecutorInitialization:
    """Test Executor initialization."""

    def test_executor_creation(self):
        """Test Executor can be instantiated."""
        from src.executor import Executor

        executor = Executor()
        assert executor is not None

    def test_executor_has_shopping_agent(self):
        """Test Executor has a ShoppingAgent instance."""
        from src.executor import Executor
        from src.shopping_agent import ShoppingAgent

        executor = Executor()
        agent = executor.get_shopping_agent()
        assert agent is not None
        assert isinstance(agent, ShoppingAgent)

    def test_executor_has_empty_task_states(self):
        """Test Executor starts with empty simple task states."""
        from src.executor import Executor

        executor = Executor()
        assert executor._simple_task_states == {}

    def test_executor_clear_state(self):
        """Test clear_state method empties the task states."""
        from src.executor import Executor

        executor = Executor()
        executor._simple_task_states["test-task"] = {"message_count": 1}
        assert len(executor._simple_task_states) == 1

        executor.clear_state()
        assert len(executor._simple_task_states) == 0


# =============================================================================
# Execute Method Tests
# =============================================================================


class TestExecutorExecute:
    """Test Executor.execute() method."""

    @pytest.mark.asyncio
    async def test_execute_requires_message(self, mock_request_context_no_message, mock_event_queue):
        """Test execute raises ValueError when message is None."""
        from src.executor import Executor

        executor = Executor()

        with pytest.raises(ValueError, match="Message is required"):
            await executor.execute(mock_request_context_no_message, mock_event_queue)

    @pytest.mark.asyncio
    async def test_execute_simple_message_goes_to_tck_handler(self, mock_request_context, mock_event_queue):
        """Test simple messages (no MCP) go to TCK handler."""
        from src.executor import Executor

        executor = Executor()

        await executor.execute(mock_request_context, mock_event_queue)

        # Verify event was published (simple message handling)
        assert mock_event_queue.enqueue_event.called

    @pytest.mark.asyncio
    async def test_execute_tracks_simple_task(self, mock_request_context, mock_event_queue):
        """Test execute tracks simple tasks in _simple_task_states."""
        from src.executor import Executor

        executor = Executor()
        task_id = mock_request_context.task_id

        await executor.execute(mock_request_context, mock_event_queue)

        # Task should be tracked
        assert task_id in executor._simple_task_states


# =============================================================================
# TCK Conformance Tests
# =============================================================================


class TestTCKConformance:
    """Test TCK conformance handling."""

    @pytest.mark.asyncio
    async def test_tck_resubscribe_test_detection(self, mock_event_queue):
        """Test TCK resubscribe test is detected by messageId prefix."""
        from src.executor import Executor

        executor = Executor()

        context = MagicMock()
        context.task_id = str(uuid.uuid4())
        context.context_id = str(uuid.uuid4())
        context.message = create_message(
            "Test message",
            message_id="test-resubscribe-message-id-12345"
        )
        context.metadata = None
        context.current_task = None

        # This should trigger the TCK resubscribe handler (which takes time)
        # We'll just verify it doesn't raise and publishes events
        # Note: This test may take a few seconds due to TCK timeout
        import asyncio
        try:
            await asyncio.wait_for(
                executor.execute(context, mock_event_queue),
                timeout=1.0  # Short timeout to just verify detection
            )
        except asyncio.TimeoutError:
            pass  # Expected - TCK test takes time

        # Verify some events were published
        assert mock_event_queue.enqueue_event.called

    @pytest.mark.asyncio
    async def test_simple_message_completion(self, mock_event_queue):
        """Test simple message with 'done' completes the task."""
        from src.executor import Executor

        executor = Executor()

        context = MagicMock()
        context.task_id = "test-task-id"
        context.context_id = str(uuid.uuid4())
        context.message = create_message("done")
        context.metadata = None
        context.current_task = None

        await executor.execute(context, mock_event_queue)

        # Task should not be tracked (completed)
        assert "test-task-id" not in executor._simple_task_states


# =============================================================================
# Cancel Method Tests
# =============================================================================


class TestExecutorCancel:
    """Test Executor.cancel() method."""

    @pytest.mark.asyncio
    async def test_cancel_publishes_cancelled_status(self, mock_request_context, mock_event_queue):
        """Test cancel publishes cancelled status for TCK conformance."""
        from src.executor import Executor

        executor = Executor()

        # Cancel should not raise, it should publish cancelled status
        await executor.cancel(mock_request_context, mock_event_queue)

        # Verify cancelled status was published (via event queue)
        assert mock_event_queue.enqueue_event.called

    @pytest.mark.asyncio
    async def test_cancel_simple_task(self, mock_request_context, mock_event_queue):
        """Test cancelling a simple task that's being tracked."""
        from src.executor import Executor

        executor = Executor()
        task_id = mock_request_context.task_id

        # Track a simple task
        executor._simple_task_states[task_id] = {"message_count": 1}

        await executor.cancel(mock_request_context, mock_event_queue)

        # Task should be removed from tracking
        assert task_id not in executor._simple_task_states


# =============================================================================
# get_message_text Tests
# =============================================================================


class TestGetMessageText:
    """Test get_message_text utility function."""

    def test_get_message_text_single_part(self):
        """Test extracting text from single-part message."""
        from src.messenger import get_message_text

        message = create_message("Hello world")
        text = get_message_text(message)
        assert text == "Hello world"

    def test_get_message_text_multiple_parts(self):
        """Test extracting text from multi-part message."""
        from src.messenger import get_message_text

        message = Message(
            messageId=str(uuid.uuid4()),
            role=Role.user,
            parts=[
                TextPart(text="First line"),
                TextPart(text="Second line"),
            ],
        )
        text = get_message_text(message)
        assert "First line" in text
        assert "Second line" in text

    def test_get_message_text_empty_parts(self):
        """Test extracting text from message with no parts."""
        from src.messenger import get_message_text

        message = Message(
            messageId=str(uuid.uuid4()),
            role=Role.user,
            parts=[],
        )
        text = get_message_text(message)
        assert text == ""


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
