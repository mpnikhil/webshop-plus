"""
Tests for Executor integration with ShoppingAgent (AAA Stage 10).

This module tests the wiring between the A2A Executor and the ADK ShoppingAgent
for MCP-based shopping tasks.

Tests verify:
- Executor detects MCP-based tasks via resources in kickoff message
- Executor extracts MCP URI and task data correctly
- Executor delegates to ShoppingAgent.run() for MCP tasks
- Executor handles ShoppingAgent success/failure appropriately
- Executor handles missing MCP URI gracefully
- Executor handles ShoppingAgent exceptions
"""

import json
import uuid

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from a2a.types import Message, Role, TaskState, TextPart

from src.executor import Executor


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def executor():
    """Create a fresh Executor instance for each test."""
    return Executor()


@pytest.fixture
def mock_event_queue():
    """Create a mock event queue."""
    queue = AsyncMock()
    queue.enqueue_event = AsyncMock()
    return queue


def create_message(text: str, role: Role = Role.user) -> Message:
    """Helper to create a valid SDK Message with all required fields."""
    return Message(
        messageId=str(uuid.uuid4()),
        role=role,
        parts=[TextPart(text=text)],
    )


def create_mcp_kickoff_message(
    goal: str = "Find running shoes under $50",
    budget: float = 50.0,
    constraints: list[str] | None = None,
    mcp_uri: str = "http://localhost:8000/mcp/session-123",
) -> Message:
    """Create an MCP kickoff message with resources array."""
    payload = {
        "goal": goal,
        "budget": budget,
        "constraints": constraints or [],
        "resources": [
            {
                "type": "mcp",
                "uri": mcp_uri,
                "description": "WebShop MCP server",
            }
        ],
    }
    return create_message(json.dumps(payload))


def create_mock_context(message: Message, task_id: str | None = None, context_id: str | None = None):
    """Create a mock request context."""
    context = MagicMock()
    context.task_id = task_id or str(uuid.uuid4())
    context.context_id = context_id or str(uuid.uuid4())
    context.message = message
    context.metadata = None
    context.current_task = None
    return context


# =============================================================================
# Executor Initialization Tests
# =============================================================================


class TestExecutorShoppingAgentInit:
    """Test Executor initialization with ShoppingAgent."""

    def test_executor_has_shopping_agent(self, executor):
        """Executor initializes with a ShoppingAgent instance."""
        agent = executor.get_shopping_agent()
        assert agent is not None

    def test_executor_shopping_agent_is_shopping_agent_type(self, executor):
        """The shopping agent is of correct type."""
        from src.shopping_agent import ShoppingAgent

        agent = executor.get_shopping_agent()
        assert isinstance(agent, ShoppingAgent)


# =============================================================================
# MCP Task Detection Tests
# =============================================================================


class TestMcpTaskDetection:
    """Test Executor detection of MCP-based tasks."""

    @pytest.mark.asyncio
    async def test_executor_detects_mcp_task(self, executor, mock_event_queue):
        """Executor detects MCP URI in kickoff message and routes to ShoppingAgent."""
        message = create_mcp_kickoff_message()
        context = create_mock_context(message)

        with patch.object(executor, "_shopping_agent") as mock_shopping_agent:
            mock_shopping_agent.run = AsyncMock(return_value={
                "success": True,
                "final_message": "Shopping completed",
                "turns_used": 5,
            })

            await executor.execute(context, mock_event_queue)

            # Verify ShoppingAgent.run was called
            mock_shopping_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_executor_non_mcp_falls_through(self, executor, mock_event_queue):
        """Non-MCP messages fall through to simple message handler."""
        message = create_message("Hello world")
        context = create_mock_context(message)

        await executor.execute(context, mock_event_queue)

        # Should not call ShoppingAgent (would go to simple message handler)
        # Verify event was published (simple message handling)
        assert mock_event_queue.enqueue_event.called


# =============================================================================
# Executor Extracts and Runs Tests
# =============================================================================


class TestExecutorExtractsAndRuns:
    """Test Executor extracts MCP URI and task data, then runs ShoppingAgent."""

    @pytest.mark.asyncio
    async def test_executor_passes_mcp_uri_to_agent(self, executor, mock_event_queue):
        """Executor passes correct MCP URI to ShoppingAgent."""
        mcp_uri = "http://localhost:8000/mcp/session-xyz789"
        message = create_mcp_kickoff_message(mcp_uri=mcp_uri)
        context = create_mock_context(message)

        with patch.object(executor, "_shopping_agent") as mock_shopping_agent:
            mock_shopping_agent.run = AsyncMock(return_value={
                "success": True,
                "final_message": "Done",
                "turns_used": 3,
            })

            await executor.execute(context, mock_event_queue)

            # Verify MCP URI was passed correctly
            call_args = mock_shopping_agent.run.call_args
            assert call_args[0][0] == mcp_uri

    @pytest.mark.asyncio
    async def test_executor_passes_task_data_to_agent(self, executor, mock_event_queue):
        """Executor passes correct task data to ShoppingAgent."""
        goal = "Find waterproof hiking boots"
        budget = 150.0
        constraints = ["waterproof", "size 10"]

        message = create_mcp_kickoff_message(
            goal=goal,
            budget=budget,
            constraints=constraints,
        )
        context = create_mock_context(message)

        with patch.object(executor, "_shopping_agent") as mock_shopping_agent:
            mock_shopping_agent.run = AsyncMock(return_value={
                "success": True,
                "final_message": "Done",
                "turns_used": 3,
            })

            await executor.execute(context, mock_event_queue)

            # Verify task data was passed correctly
            call_args = mock_shopping_agent.run.call_args
            task_data = call_args[0][1]
            assert task_data["goal"] == goal
            assert task_data["budget"] == budget
            assert task_data["constraints"] == constraints

    @pytest.mark.asyncio
    async def test_executor_adds_session_id_to_task_data(self, executor, mock_event_queue):
        """Executor adds session_id (task_id) to task data."""
        message = create_mcp_kickoff_message()
        task_id = "test-task-id-12345"
        context = create_mock_context(message, task_id=task_id)

        with patch.object(executor, "_shopping_agent") as mock_shopping_agent:
            mock_shopping_agent.run = AsyncMock(return_value={
                "success": True,
                "final_message": "Done",
                "turns_used": 3,
            })

            await executor.execute(context, mock_event_queue)

            # Verify session_id was added
            call_args = mock_shopping_agent.run.call_args
            task_data = call_args[0][1]
            assert task_data["session_id"] == task_id


# =============================================================================
# Success Response Tests
# =============================================================================


class TestExecutorSuccessResponse:
    """Test Executor handles ShoppingAgent success response."""

    @pytest.mark.asyncio
    async def test_executor_completes_on_success(self, executor, mock_event_queue):
        """Executor calls updater.complete() on success."""
        message = create_mcp_kickoff_message()
        context = create_mock_context(message)

        with patch.object(executor, "_shopping_agent") as mock_shopping_agent:
            mock_shopping_agent.run = AsyncMock(return_value={
                "success": True,
                "final_message": "Successfully purchased running shoes",
                "turns_used": 7,
            })

            await executor.execute(context, mock_event_queue)

            # Verify completion event was published
            assert mock_event_queue.enqueue_event.called

    @pytest.mark.asyncio
    async def test_executor_includes_final_message_on_success(self, executor, mock_event_queue):
        """Executor includes final_message in completion."""
        message = create_mcp_kickoff_message()
        context = create_mock_context(message)
        final_message = "Found and purchased Nike Air Max for $45.99"

        with patch.object(executor, "_shopping_agent") as mock_shopping_agent:
            mock_shopping_agent.run = AsyncMock(return_value={
                "success": True,
                "final_message": final_message,
                "turns_used": 5,
            })

            await executor.execute(context, mock_event_queue)

            # Verify event queue received events with final message
            assert mock_event_queue.enqueue_event.called


# =============================================================================
# Failure Response Tests
# =============================================================================


class TestExecutorFailureResponse:
    """Test Executor handles ShoppingAgent failure response."""

    @pytest.mark.asyncio
    async def test_executor_fails_on_agent_failure(self, executor, mock_event_queue):
        """Executor calls updater.failed() when ShoppingAgent reports failure."""
        message = create_mcp_kickoff_message()
        context = create_mock_context(message)

        with patch.object(executor, "_shopping_agent") as mock_shopping_agent:
            mock_shopping_agent.run = AsyncMock(return_value={
                "success": False,
                "final_message": "Could not find products",
                "turns_used": 10,
                "error": "No products matched search",
            })

            await executor.execute(context, mock_event_queue)

            # Verify failure event was published
            assert mock_event_queue.enqueue_event.called


# =============================================================================
# Missing Data Tests
# =============================================================================


class TestExecutorHandlesMissingData:
    """Test Executor handles missing or invalid data gracefully."""

    @pytest.mark.asyncio
    async def test_executor_fails_on_missing_goal(self, executor, mock_event_queue):
        """Executor fails gracefully when goal is missing from kickoff."""
        # Create message with MCP URI but no goal
        payload = {
            "budget": 50.0,
            "resources": [{"type": "mcp", "uri": "http://localhost:8000/mcp/test"}],
        }
        message = create_message(json.dumps(payload))
        context = create_mock_context(message)

        await executor.execute(context, mock_event_queue)

        # Verify failure event was published (missing goal)
        assert mock_event_queue.enqueue_event.called

    @pytest.mark.asyncio
    async def test_executor_uses_default_budget(self, executor, mock_event_queue):
        """Executor uses default budget when not specified."""
        payload = {
            "goal": "Find shoes",
            "resources": [{"type": "mcp", "uri": "http://localhost:8000/mcp/test"}],
        }
        message = create_message(json.dumps(payload))
        context = create_mock_context(message)

        with patch.object(executor, "_shopping_agent") as mock_shopping_agent:
            mock_shopping_agent.run = AsyncMock(return_value={
                "success": True,
                "final_message": "Done",
                "turns_used": 3,
            })

            await executor.execute(context, mock_event_queue)

            # Verify default budget was used
            call_args = mock_shopping_agent.run.call_args
            task_data = call_args[0][1]
            assert task_data["budget"] == 100.0  # Default


# =============================================================================
# Exception Handling Tests
# =============================================================================


class TestExecutorExceptionHandling:
    """Test Executor handles ShoppingAgent exceptions gracefully."""

    @pytest.mark.asyncio
    async def test_executor_handles_agent_exception(self, executor, mock_event_queue):
        """Executor handles exceptions from ShoppingAgent.run()."""
        message = create_mcp_kickoff_message()
        context = create_mock_context(message)

        with patch.object(executor, "_shopping_agent") as mock_shopping_agent:
            mock_shopping_agent.run = AsyncMock(
                side_effect=Exception("Connection refused")
            )

            # Should not raise - exception is caught and reported via updater
            await executor.execute(context, mock_event_queue)

            # Verify failure event was published
            assert mock_event_queue.enqueue_event.called

    @pytest.mark.asyncio
    async def test_executor_handles_value_error(self, executor, mock_event_queue):
        """Executor handles ValueError from ShoppingAgent."""
        message = create_mcp_kickoff_message()
        context = create_mock_context(message)

        with patch.object(executor, "_shopping_agent") as mock_shopping_agent:
            mock_shopping_agent.run = AsyncMock(
                side_effect=ValueError("Invalid task data")
            )

            await executor.execute(context, mock_event_queue)

            # Verify failure event was published
            assert mock_event_queue.enqueue_event.called

    @pytest.mark.asyncio
    async def test_executor_handles_timeout_error(self, executor, mock_event_queue):
        """Executor handles timeout errors from ShoppingAgent."""
        message = create_mcp_kickoff_message()
        context = create_mock_context(message)

        with patch.object(executor, "_shopping_agent") as mock_shopping_agent:
            import asyncio
            mock_shopping_agent.run = AsyncMock(
                side_effect=asyncio.TimeoutError("MCP server timeout")
            )

            await executor.execute(context, mock_event_queue)

            # Verify failure event was published
            assert mock_event_queue.enqueue_event.called


# =============================================================================
# Start Work Status Tests
# =============================================================================


class TestExecutorStartWorkStatus:
    """Test Executor publishes start_work status for MCP tasks."""

    @pytest.mark.asyncio
    async def test_executor_publishes_start_work(self, executor, mock_event_queue):
        """Executor publishes start_work before running ShoppingAgent."""
        message = create_mcp_kickoff_message(goal="Find laptop under $500")
        context = create_mock_context(message)

        with patch.object(executor, "_shopping_agent") as mock_shopping_agent:
            mock_shopping_agent.run = AsyncMock(return_value={
                "success": True,
                "final_message": "Done",
                "turns_used": 3,
            })

            await executor.execute(context, mock_event_queue)

            # Verify start_work event was published (working state)
            assert mock_event_queue.enqueue_event.called


# =============================================================================
# Integration Tests
# =============================================================================


class TestExecutorMcpIntegration:
    """Integration tests for MCP task handling flow."""

    @pytest.mark.asyncio
    async def test_full_mcp_task_flow_success(self, executor, mock_event_queue):
        """Test complete MCP task flow from kickoff to completion."""
        goal = "Find wireless earbuds under $100"
        budget = 100.0
        constraints = ["bluetooth 5.0", "noise cancelling"]
        mcp_uri = "http://localhost:8000/mcp/session-integration-test"

        message = create_mcp_kickoff_message(
            goal=goal,
            budget=budget,
            constraints=constraints,
            mcp_uri=mcp_uri,
        )
        context = create_mock_context(message)

        with patch.object(executor, "_shopping_agent") as mock_shopping_agent:
            mock_shopping_agent.run = AsyncMock(return_value={
                "success": True,
                "final_message": "Purchased Sony WF-1000XM4 for $89.99",
                "turns_used": 8,
            })

            await executor.execute(context, mock_event_queue)

            # Verify full flow
            mock_shopping_agent.run.assert_called_once()
            call_args = mock_shopping_agent.run.call_args

            # Check MCP URI
            assert call_args[0][0] == mcp_uri

            # Check task data
            task_data = call_args[0][1]
            assert task_data["goal"] == goal
            assert task_data["budget"] == budget
            assert task_data["constraints"] == constraints
            assert "session_id" in task_data

    @pytest.mark.asyncio
    async def test_full_mcp_task_flow_failure(self, executor, mock_event_queue):
        """Test complete MCP task flow with failure."""
        message = create_mcp_kickoff_message()
        context = create_mock_context(message)

        with patch.object(executor, "_shopping_agent") as mock_shopping_agent:
            mock_shopping_agent.run = AsyncMock(return_value={
                "success": False,
                "final_message": "Budget exceeded",
                "turns_used": 5,
                "error": "Cart total $65.00 exceeds budget $50.00",
            })

            await executor.execute(context, mock_event_queue)

            # Verify ShoppingAgent was called
            mock_shopping_agent.run.assert_called_once()

            # Verify failure was reported
            assert mock_event_queue.enqueue_event.called


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
