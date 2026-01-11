"""
Tests for WebShop+ Purple Agent Executor.

This module contains comprehensive tests for:
- Executor initialization
- Task execution via execute()
- Error handling
- Agent lifecycle per context
- Cancel operation
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from a2a.types import (
    Message,
    Role,
    TaskState,
    TextPart,
    UnsupportedOperationError,
    Task,
    TaskStatus,
)
from a2a.utils.errors import ServerError


# =============================================================================
# Fixtures
# =============================================================================


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


@pytest.fixture
def mock_request_context():
    """Create a mock request context with a message."""
    context = MagicMock()
    context.task_id = str(uuid.uuid4())
    context.context_id = str(uuid.uuid4())
    context.message = create_message("Find running shoes under $100")
    context.current_task = None
    return context


@pytest.fixture
def mock_request_context_no_message():
    """Create a mock request context without a message."""
    context = MagicMock()
    context.task_id = str(uuid.uuid4())
    context.context_id = str(uuid.uuid4())
    context.message = None
    context.current_task = None
    return context


@pytest.fixture
def mock_request_context_with_task():
    """Create a mock request context with a current task in working state."""
    context = MagicMock()
    context.task_id = str(uuid.uuid4())
    context.context_id = str(uuid.uuid4())
    context.message = create_message("OBSERVATION: Search results: B07XYZ - Shoes - $50")
    # Create a mock task with working state
    context.current_task = MagicMock()
    context.current_task.status = MagicMock()
    context.current_task.status.state = TaskState.working
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

    def test_executor_has_empty_agents_dict(self):
        """Test Executor starts with empty agents dict."""
        from src.executor import Executor

        executor = Executor()
        assert executor._agents == {}

    def test_executor_clear_agents(self):
        """Test clear_agents method empties the dict."""
        from src.executor import Executor

        executor = Executor()
        executor._agents["test-context"] = MagicMock()
        assert len(executor._agents) == 1

        executor.clear_agents()
        assert len(executor._agents) == 0

    def test_executor_get_agent_not_found(self):
        """Test get_agent returns None for unknown context."""
        from src.executor import Executor

        executor = Executor()
        assert executor.get_agent("nonexistent") is None


# =============================================================================
# Execute Method Tests
# =============================================================================


class TestExecutorExecute:
    """Test Executor.execute() method."""

    @pytest.mark.asyncio
    async def test_execute_creates_new_agent(self, mock_request_context, mock_event_queue):
        """Test execute creates a new agent for new context_id."""
        from src.executor import Executor

        executor = Executor()

        # Mock ShopperAgent.run to be a no-op
        with patch("src.executor.ShopperAgent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock()
            MockAgent.return_value = mock_agent

            await executor.execute(mock_request_context, mock_event_queue)

            # Verify agent was created
            assert mock_request_context.context_id in executor._agents

    @pytest.mark.asyncio
    async def test_execute_reuses_existing_agent(self, mock_event_queue):
        """Test execute reuses existing agent for same context_id."""
        from src.executor import Executor

        executor = Executor()

        # Create two contexts with same context_id
        context1 = MagicMock()
        context1.task_id = str(uuid.uuid4())
        context1.context_id = "shared-context"
        context1.message = create_message("Task 1")
        context1.current_task = None

        context2 = MagicMock()
        context2.task_id = str(uuid.uuid4())
        context2.context_id = "shared-context"
        context2.message = create_message("Task 2")
        context2.current_task = None

        with patch("src.executor.ShopperAgent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock()
            MockAgent.return_value = mock_agent

            await executor.execute(context1, mock_event_queue)
            await executor.execute(context2, mock_event_queue)

            # Should only create one agent
            assert MockAgent.call_count == 1
            # Agent.run should be called twice
            assert mock_agent.run.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_calls_agent_run(self, mock_request_context, mock_event_queue):
        """Test execute calls agent.run with message and updater."""
        from src.executor import Executor

        executor = Executor()

        with patch("src.executor.ShopperAgent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock()
            MockAgent.return_value = mock_agent

            await executor.execute(mock_request_context, mock_event_queue)

            # Verify run was called with message
            mock_agent.run.assert_called_once()
            call_args = mock_agent.run.call_args
            assert call_args[0][0] == mock_request_context.message

    @pytest.mark.asyncio
    async def test_execute_requires_message(self, mock_request_context_no_message, mock_event_queue):
        """Test execute raises ValueError when message is None."""
        from src.executor import Executor

        executor = Executor()

        with pytest.raises(ValueError, match="Message is required"):
            await executor.execute(mock_request_context_no_message, mock_event_queue)

    @pytest.mark.asyncio
    async def test_execute_handles_exception(self, mock_request_context, mock_event_queue):
        """Test execute handles exceptions from agent.run."""
        from src.executor import Executor

        executor = Executor()

        with patch("src.executor.ShopperAgent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=Exception("Test error"))
            MockAgent.return_value = mock_agent

            # Should not raise - exception is caught and reported via updater
            await executor.execute(mock_request_context, mock_event_queue)

            # Verify the event queue received events (task failed)
            assert mock_event_queue.enqueue_event.called


# =============================================================================
# Cancel Method Tests
# =============================================================================


class TestExecutorCancel:
    """Test Executor.cancel() method."""

    @pytest.mark.asyncio
    async def test_cancel_raises_unsupported(self, mock_request_context, mock_event_queue):
        """Test cancel raises ServerError wrapping UnsupportedOperationError."""
        from src.executor import Executor

        executor = Executor()

        with pytest.raises(ServerError) as exc_info:
            await executor.cancel(mock_request_context, mock_event_queue)

        # Verify the error contains UnsupportedOperationError
        assert isinstance(exc_info.value.error, UnsupportedOperationError)
        assert "Cancel operation is not supported" in exc_info.value.error.message


# =============================================================================
# Terminal States Tests
# =============================================================================


class TestTerminalStates:
    """Test terminal state handling."""

    def test_terminal_states_defined(self):
        """Test TERMINAL_STATES constant is defined correctly."""
        from src.executor import TERMINAL_STATES

        assert TaskState.completed in TERMINAL_STATES
        assert TaskState.canceled in TERMINAL_STATES
        assert TaskState.failed in TERMINAL_STATES
        assert TaskState.rejected in TERMINAL_STATES
        assert TaskState.working not in TERMINAL_STATES
        assert TaskState.submitted not in TERMINAL_STATES


# =============================================================================
# Agent Lifecycle Tests
# =============================================================================


class TestAgentLifecycle:
    """Test agent lifecycle management."""

    def test_get_agent_returns_cached_agent(self):
        """Test get_agent returns the cached agent."""
        from src.executor import Executor
        from src.agent import ShopperAgent

        executor = Executor()
        agent = ShopperAgent()
        executor._agents["test-context"] = agent

        retrieved = executor.get_agent("test-context")
        assert retrieved is agent

    @pytest.mark.asyncio
    async def test_different_contexts_get_different_agents(self, mock_event_queue):
        """Test different context_ids get different agent instances."""
        from src.executor import Executor

        executor = Executor()

        context1 = MagicMock()
        context1.task_id = "task-1"
        context1.context_id = "context-1"
        context1.message = create_message("Task 1")
        context1.current_task = None

        context2 = MagicMock()
        context2.task_id = "task-2"
        context2.context_id = "context-2"
        context2.message = create_message("Task 2")
        context2.current_task = None

        with patch("src.executor.ShopperAgent") as MockAgent:
            mock_agent1 = MagicMock()
            mock_agent1.run = AsyncMock()
            mock_agent2 = MagicMock()
            mock_agent2.run = AsyncMock()
            MockAgent.side_effect = [mock_agent1, mock_agent2]

            await executor.execute(context1, mock_event_queue)
            await executor.execute(context2, mock_event_queue)

            # Should create two different agents
            assert MockAgent.call_count == 2
            assert executor._agents["context-1"] is mock_agent1
            assert executor._agents["context-2"] is mock_agent2


# =============================================================================
# Integration Tests
# =============================================================================


class TestExecutorIntegration:
    """Integration tests for Executor with real ShopperAgent (mocked LLM)."""

    @pytest.fixture
    def mock_llm_responses(self):
        """Set up mock LLM responses."""
        def mock_complete(messages, **kwargs):
            content = messages[-1]["content"] if messages else ""
            if "PRODUCT_TYPE:" in content or "Parse the following" in content:
                return """PRODUCT_TYPE: running shoes
BUDGET: 100
PREFERENCES: comfortable
CONSTRAINTS: none
COMPARISON_REQUIRED: no
SEARCH_QUERY: running shoes"""
            return "search[running shoes]"
        return mock_complete

    @pytest.mark.asyncio
    async def test_execute_with_real_agent(self, mock_event_queue, mock_llm_responses):
        """Test execute with real ShopperAgent (mocked LLM)."""
        from src.executor import Executor

        executor = Executor()

        context = MagicMock()
        context.task_id = str(uuid.uuid4())
        context.context_id = str(uuid.uuid4())
        context.message = create_message("Find running shoes under $100")
        context.current_task = None

        with patch("src.agent.LLMClient") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.complete = mock_llm_responses
            MockLLM.return_value = mock_llm

            await executor.execute(context, mock_event_queue)

            # Verify agent was created and is accessible
            agent = executor.get_agent(context.context_id)
            assert agent is not None
            # Verify task was processed (action history should have entries)
            assert len(agent.context.action_history) >= 1

    @pytest.mark.asyncio
    async def test_execute_multi_turn_conversation(self, mock_event_queue, mock_llm_responses):
        """Test multi-turn conversation within same context."""
        from src.executor import Executor

        executor = Executor()
        context_id = str(uuid.uuid4())

        # First turn: task instruction
        context1 = MagicMock()
        context1.task_id = str(uuid.uuid4())
        context1.context_id = context_id
        context1.message = create_message("Find running shoes under $100")
        context1.current_task = None

        # Second turn: observation
        context2 = MagicMock()
        context2.task_id = str(uuid.uuid4())
        context2.context_id = context_id
        context2.message = create_message("OBSERVATION: Search results: B07XYZ - Nike Shoes - $89.99")
        context2.current_task = None

        with patch("src.agent.LLMClient") as MockLLM:
            mock_llm = MagicMock()

            def mock_complete(messages, **kwargs):
                content = messages[-1]["content"] if messages else ""
                if "PRODUCT_TYPE:" in content or "Parse the following" in content:
                    return """PRODUCT_TYPE: running shoes
BUDGET: 100
PREFERENCES: comfortable
CONSTRAINTS: none
COMPARISON_REQUIRED: no
SEARCH_QUERY: running shoes"""
                elif "select the best" in content.lower():
                    return "B07XYZ"
                return "search[running shoes]"

            mock_llm.complete = mock_complete
            MockLLM.return_value = mock_llm

            await executor.execute(context1, mock_event_queue)
            await executor.execute(context2, mock_event_queue)

            # Verify same agent was used
            agent = executor.get_agent(context_id)
            assert agent is not None
            # Verify multiple actions were recorded
            assert len(agent.context.action_history) >= 2


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
# Agent run() Method Tests
# =============================================================================


class TestShopperAgentRun:
    """Test ShopperAgent.run() method."""

    @pytest.fixture
    def mock_updater(self):
        """Create a mock TaskUpdater."""
        updater = AsyncMock()
        updater.update_status = AsyncMock()
        updater.complete = AsyncMock()
        return updater

    @pytest.mark.asyncio
    async def test_run_calls_update_status(self, mock_updater):
        """Test run calls update_status with working state."""
        from src.agent import ShopperAgent

        agent = ShopperAgent()
        agent.llm_client = MagicMock()
        agent.llm_client.complete.return_value = """PRODUCT_TYPE: shoes
BUDGET: none
PREFERENCES: none
CONSTRAINTS: none
COMPARISON_REQUIRED: no
SEARCH_QUERY: shoes"""

        message = create_message("Find shoes")

        await agent.run(message, mock_updater)

        # Verify update_status was called with working state
        mock_updater.update_status.assert_called_once()
        call_kwargs = mock_updater.update_status.call_args[1]
        assert call_kwargs["state"] == TaskState.working

    @pytest.mark.asyncio
    async def test_run_calls_complete_with_action(self, mock_updater):
        """Test run calls complete with action message."""
        from src.agent import ShopperAgent

        agent = ShopperAgent()
        agent.llm_client = MagicMock()
        agent.llm_client.complete.return_value = """PRODUCT_TYPE: running shoes
BUDGET: 100
PREFERENCES: none
CONSTRAINTS: none
COMPARISON_REQUIRED: no
SEARCH_QUERY: running shoes"""

        message = create_message("Find running shoes under $100")

        await agent.run(message, mock_updater)

        # Verify complete was called
        mock_updater.complete.assert_called_once()
        call_kwargs = mock_updater.complete.call_args[1]
        # The message should contain the action
        result_message = call_kwargs["message"]
        assert result_message.role == Role.agent
        # Action should be in the text (SDK Part wraps the TextPart in .root)
        part = result_message.parts[0]
        actual_part = part.root if hasattr(part, "root") else part
        text = actual_part.text
        assert "search[" in text

    @pytest.mark.asyncio
    async def test_run_processes_task_instruction_when_no_requirements(self, mock_updater):
        """Test run processes as task instruction when requirements is None."""
        from src.agent import ShopperAgent

        agent = ShopperAgent()
        agent.llm_client = MagicMock()
        agent.llm_client.complete.return_value = """PRODUCT_TYPE: laptop
BUDGET: 500
PREFERENCES: none
CONSTRAINTS: none
COMPARISON_REQUIRED: no
SEARCH_QUERY: laptop"""

        assert agent.context.requirements is None

        message = create_message("Buy a laptop under $500")

        await agent.run(message, mock_updater)

        # Requirements should now be set
        assert agent.context.requirements is not None

    @pytest.mark.asyncio
    async def test_run_processes_observation_when_requirements_set(self, mock_updater):
        """Test run processes as observation when requirements are already set."""
        from src.agent import ShopperAgent, TaskRequirements, AgentState

        agent = ShopperAgent()
        agent.llm_client = MagicMock()
        agent.llm_client.complete.return_value = "B07XYZ123"

        # Set up existing requirements
        agent.context.requirements = TaskRequirements(
            product_type="laptop",
            budget=500.0,
            raw_instruction="Buy a laptop",
        )
        agent.context.state = AgentState.SEARCHING
        agent.context.action_history = ["search[laptop]"]

        message = create_message("Search results: B07XYZ123 - Laptop - $450")

        await agent.run(message, mock_updater)

        # Should have processed as observation (added to observation history)
        assert len(agent.context.observation_history) >= 1


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
