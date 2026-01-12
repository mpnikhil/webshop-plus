"""
Tests for the SDK-compatible AgentExecutor wrapper.

These tests verify that WebShopPlusExecutor correctly implements the a2a-sdk's
AgentExecutor interface and properly delegates to the underlying WebShopPlusAgent.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.a2a_executor import (
    WebShopPlusExecutor,
    _parse_config,
    _parse_participants,
    _parse_skill_from_message,
)
from src.models import AssessmentConfig


# =============================================================================
# Unit Tests for Helper Functions
# =============================================================================


class TestParseParticipants:
    """Tests for _parse_participants helper."""

    def test_valid_participants(self):
        """Should extract participants from metadata."""
        metadata = {
            "participants": {
                "shopper": "http://agent:8001/a2a",
                "advisor": "http://advisor:8002/a2a",
            }
        }
        result = _parse_participants(metadata)
        assert result == {
            "shopper": "http://agent:8001/a2a",
            "advisor": "http://advisor:8002/a2a",
        }

    def test_missing_participants(self):
        """Should raise ValueError when participants missing."""
        with pytest.raises(ValueError, match="No participants found"):
            _parse_participants({})

    def test_empty_participants(self):
        """Should raise ValueError when participants is empty."""
        with pytest.raises(ValueError, match="No participants found"):
            _parse_participants({"participants": {}})

    def test_invalid_url_type(self):
        """Should raise ValueError when URL is not a string."""
        metadata = {
            "participants": {
                "shopper": 12345,  # Not a string
            }
        }
        with pytest.raises(ValueError, match="must have a string URL"):
            _parse_participants(metadata)


class TestParseConfig:
    """Tests for _parse_config helper."""

    def test_default_config(self):
        """Should return default config when none provided."""
        result = _parse_config({})
        assert result.num_tasks == 80
        assert result.task_types == ["all"]

    def test_custom_num_tasks(self):
        """Should parse num_tasks from config."""
        metadata = {"config": {"num_tasks": 10}}
        result = _parse_config(metadata)
        assert result.num_tasks == 10

    def test_categories_mapped_to_task_types(self):
        """Should map 'categories' to 'task_types'."""
        metadata = {"config": {"categories": ["budget", "memory"]}}
        result = _parse_config(metadata)
        assert result.task_types == ["budget", "memory"]

    def test_timeout_per_task(self):
        """Should parse timeout_per_task."""
        metadata = {"config": {"timeout_per_task": 180}}
        result = _parse_config(metadata)
        assert result.timeout_per_task == 180


class TestParseSkillFromMessage:
    """Tests for _parse_skill_from_message helper."""

    def test_no_message(self):
        """Should return None for no message."""
        assert _parse_skill_from_message(None) is None

    def test_budget_skill(self):
        """Should detect budget assessment skill."""
        mock_message = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Run budget assessment"
        mock_message.parts = [mock_part]

        result = _parse_skill_from_message(mock_message)
        assert result == "budget-assessment"

    def test_memory_skill(self):
        """Should detect memory assessment skill."""
        mock_message = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Test memory capabilities"
        mock_message.parts = [mock_part]

        result = _parse_skill_from_message(mock_message)
        assert result == "memory-assessment"

    def test_constraint_skill(self):
        """Should detect constraint assessment skill."""
        mock_message = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Check constraint handling"
        mock_message.parts = [mock_part]

        result = _parse_skill_from_message(mock_message)
        assert result == "constraint-assessment"

    def test_reasoning_skill(self):
        """Should detect reasoning/comparative assessment skill."""
        mock_message = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Test comparative reasoning"
        mock_message.parts = [mock_part]

        result = _parse_skill_from_message(mock_message)
        assert result == "reasoning-assessment"

    def test_recovery_skill(self):
        """Should detect error recovery assessment skill."""
        mock_message = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Test error recovery"
        mock_message.parts = [mock_part]

        result = _parse_skill_from_message(mock_message)
        assert result == "recovery-assessment"

    def test_no_skill_match(self):
        """Should return None for unmatched message."""
        mock_message = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Just run all assessments"
        mock_message.parts = [mock_part]

        result = _parse_skill_from_message(mock_message)
        assert result is None


# =============================================================================
# Unit Tests for WebShopPlusExecutor
# =============================================================================


class TestWebShopPlusExecutor:
    """Tests for WebShopPlusExecutor class."""

    def test_init_default_config(self):
        """Should initialize with default config."""
        executor = WebShopPlusExecutor()
        assert executor._agent_config is not None
        assert executor._active_agents == {}

    def test_init_custom_config(self):
        """Should accept custom agent config."""
        from src.agent import AgentConfig

        config = AgentConfig(max_actions_per_task=50)
        executor = WebShopPlusExecutor(agent_config=config)
        assert executor._agent_config.max_actions_per_task == 50

    @pytest.mark.asyncio
    async def test_execute_handles_missing_participants_as_simple_message(self):
        """Should handle request without participants as simple echo message.

        This supports A2A TCK conformance testing where simple messages
        need to be processed without requiring assessment participants.
        """
        executor = WebShopPlusExecutor()

        # Create mock context with empty metadata
        mock_context = MagicMock()
        mock_context.task_id = "test-task-1"
        mock_context.context_id = "test-context-1"
        mock_context.metadata = {}
        mock_context.message = None

        # Create mock event queue
        mock_queue = AsyncMock()
        mock_queue.enqueue_event = AsyncMock()

        await executor.execute(mock_context, mock_queue)

        # Verify events were emitted (start_work + requires_input)
        assert mock_queue.enqueue_event.called
        events = [call.args[0] for call in mock_queue.enqueue_event.call_args_list]
        # Check that we got an input-required status (allowing continuation/cancellation)
        final_event = events[-1]
        assert final_event.status.state.value == "input-required"

    @pytest.mark.asyncio
    async def test_cancel_no_active_task(self):
        """Should handle cancel when no active task."""
        executor = WebShopPlusExecutor()

        mock_context = MagicMock()
        mock_context.task_id = "nonexistent-task"
        mock_context.context_id = "test-context"

        mock_queue = AsyncMock()
        mock_queue.enqueue_event = AsyncMock()

        await executor.cancel(mock_context, mock_queue)

        # Should still emit a canceled status
        assert mock_queue.enqueue_event.called

    @pytest.mark.asyncio
    async def test_cancel_active_task(self):
        """Should cancel an active task."""
        executor = WebShopPlusExecutor()

        # Add a mock active agent
        mock_agent = MagicMock()
        executor._active_agents["active-task"] = mock_agent

        mock_context = MagicMock()
        mock_context.task_id = "active-task"
        mock_context.context_id = "test-context"

        mock_queue = AsyncMock()
        mock_queue.enqueue_event = AsyncMock()

        await executor.cancel(mock_context, mock_queue)

        # Agent.cancel() should have been called
        mock_agent.cancel.assert_called_once()

        # Should emit canceled status
        assert mock_queue.enqueue_event.called

    def test_create_message(self):
        """Should create a Message with TextPart."""
        executor = WebShopPlusExecutor()
        msg = executor._create_message("Test message")

        assert msg.role.value == "agent"
        assert len(msg.parts) == 1
        # Parts are wrapped in a Part discriminated union
        part = msg.parts[0]
        # Access the inner TextPart via .root or directly check the kind
        assert part.root.text == "Test message"
        assert part.root.kind == "text"


# =============================================================================
# Integration-Style Tests (with mocked dependencies)
# =============================================================================


class TestExecutorIntegration:
    """Integration tests with mocked WebShopPlusAgent."""

    @pytest.mark.asyncio
    async def test_execute_successful_assessment(self):
        """Should run complete assessment and emit success events."""
        executor = WebShopPlusExecutor()

        # Create mock context
        mock_context = MagicMock()
        mock_context.task_id = "test-task"
        mock_context.context_id = "test-context"
        mock_context.metadata = {
            "participants": {"shopper": "http://agent:8001/a2a"},
            "config": {"num_tasks": 5},
        }
        mock_context.message = None

        mock_queue = AsyncMock()
        mock_queue.enqueue_event = AsyncMock()

        # Mock the WebShopPlusAgent
        with patch("src.a2a_executor.WebShopPlusAgent") as MockAgent:
            # Setup mock agent instance
            mock_agent_instance = AsyncMock()
            mock_agent_instance.__aenter__ = AsyncMock(return_value=mock_agent_instance)
            mock_agent_instance.__aexit__ = AsyncMock(return_value=None)

            # Setup mock results
            mock_results = MagicMock()
            mock_results.aggregate.successful_tasks = 5
            mock_results.aggregate.total_tasks = 5
            mock_results.aggregate.average_score = 0.85
            mock_results.model_dump.return_value = {
                "assessment_id": "test",
                "aggregate": {
                    "successful_tasks": 5,
                    "total_tasks": 5,
                    "average_score": 0.85,
                },
            }

            mock_agent_instance.run = AsyncMock(return_value=mock_results)
            MockAgent.return_value = mock_agent_instance

            await executor.execute(mock_context, mock_queue)

        # Verify events were emitted
        assert mock_queue.enqueue_event.called

        # Get all events
        events = [call.args[0] for call in mock_queue.enqueue_event.call_args_list]

        # Check for status update events
        status_events = [e for e in events if hasattr(e, "status")]
        assert len(status_events) >= 2  # At least working + completed

        # Verify final status is completed
        final_status = status_events[-1]
        assert final_status.status.state.value == "completed"

        # Check for artifact event
        artifact_events = [e for e in events if hasattr(e, "artifact")]
        assert len(artifact_events) == 1
        assert artifact_events[0].artifact.name == "assessment_results"


# =============================================================================
# MCP Integration Tests (AAA Stage 7)
# =============================================================================


class TestExecutorMCPSupport:
    """Tests for MCP session management in WebShopPlusExecutor."""

    def test_has_mcp_support_false_by_default(self):
        """Should return False when no SessionManager configured."""
        executor = WebShopPlusExecutor()
        assert executor.has_mcp_support() is False

    def test_has_mcp_support_true_with_session_manager(self):
        """Should return True when SessionManager is configured."""
        from src.webshop_mcp import SessionManager

        session_manager = SessionManager()
        executor = WebShopPlusExecutor(session_manager=session_manager)
        assert executor.has_mcp_support() is True

    def test_get_mcp_base_url(self):
        """Should return correct MCP base URL."""
        executor = WebShopPlusExecutor(mcp_host="myhost", mcp_port=9000)
        assert executor._get_mcp_base_url() == "http://myhost:9000/mcp"

    def test_build_mcp_kickoff(self):
        """Should build correct kickoff dict with MCP resource."""
        executor = WebShopPlusExecutor()

        kickoff = executor.build_mcp_kickoff(
            goal="Find running shoes under $50",
            budget=50.0,
            constraints=["no synthetic"],
            mcp_uri="http://localhost:8000/mcp/session123",
        )

        assert kickoff["goal"] == "Find running shoes under $50"
        assert kickoff["budget"] == 50.0
        assert kickoff["constraints"] == ["no synthetic"]
        assert len(kickoff["resources"]) == 1
        assert kickoff["resources"][0]["type"] == "mcp"
        assert kickoff["resources"][0]["uri"] == "http://localhost:8000/mcp/session123"

    @pytest.mark.asyncio
    async def test_create_mcp_session(self):
        """Should create MCP session and return session_id and URI."""
        from src.webshop_mcp import SessionManager

        session_manager = SessionManager()
        executor = WebShopPlusExecutor(
            session_manager=session_manager,
            mcp_host="localhost",
            mcp_port=8000,
        )

        session_id, mcp_uri = await executor.create_mcp_session(
            goal="Find shoes",
            budget=100.0,
            constraints=["waterproof"],
            max_turns=20,
        )

        assert session_id is not None
        assert mcp_uri.startswith("http://localhost:8000/mcp/")
        assert session_id in mcp_uri

        # Verify session was created
        session = await session_manager.get_session(session_id)
        assert session is not None
        assert session.state.goal == "Find shoes"
        assert session.state.budget == 100.0
        assert session.state.constraints == ["waterproof"]
        assert session.state.max_turns == 20

    @pytest.mark.asyncio
    async def test_create_mcp_session_without_manager_raises(self):
        """Should raise RuntimeError when no SessionManager configured."""
        executor = WebShopPlusExecutor()

        with pytest.raises(RuntimeError, match="SessionManager not configured"):
            await executor.create_mcp_session(
                goal="Find shoes",
                budget=100.0,
            )

    @pytest.mark.asyncio
    async def test_cleanup_mcp_session(self):
        """Should clean up MCP session."""
        from src.webshop_mcp import SessionManager

        session_manager = SessionManager()
        executor = WebShopPlusExecutor(session_manager=session_manager)

        # Create a session
        session_id, _ = await executor.create_mcp_session(
            goal="Find shoes",
            budget=100.0,
        )

        # Verify it exists
        assert await session_manager.get_session(session_id) is not None

        # Clean it up
        result = await executor.cleanup_mcp_session(session_id)
        assert result is True

        # Verify it's gone
        assert await session_manager.get_session(session_id) is None

    @pytest.mark.asyncio
    async def test_cleanup_mcp_session_nonexistent(self):
        """Should return False when cleaning up nonexistent session."""
        from src.webshop_mcp import SessionManager

        session_manager = SessionManager()
        executor = WebShopPlusExecutor(session_manager=session_manager)

        result = await executor.cleanup_mcp_session("nonexistent-session")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_mcp_session_result_not_completed(self):
        """Should return None for incomplete session."""
        from src.webshop_mcp import SessionManager

        session_manager = SessionManager()
        executor = WebShopPlusExecutor(session_manager=session_manager)

        session_id, _ = await executor.create_mcp_session(
            goal="Find shoes",
            budget=100.0,
        )

        # Session not completed yet
        result = await executor.get_mcp_session_result(session_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_mcp_session_result_completed(self):
        """Should return result for completed session."""
        from src.webshop_mcp import SessionManager

        session_manager = SessionManager()
        executor = WebShopPlusExecutor(session_manager=session_manager)

        session_id, _ = await executor.create_mcp_session(
            goal="Find shoes",
            budget=100.0,
        )

        # Complete the session manually
        session = await session_manager.get_session(session_id)
        session.state.mark_completed("checkout")
        final_result = {"terminated": True, "reason": "checkout", "score": 1.0}
        session.signal_completion(final_result)

        # Now get result
        result = await executor.get_mcp_session_result(session_id)
        assert result is not None
        assert result["terminated"] is True
        assert result["reason"] == "checkout"


class TestMCPServerWaitForCompletion:
    """Tests for WebShopMCPServer.wait_for_completion()."""

    @pytest.mark.asyncio
    async def test_wait_for_completion_already_completed(self):
        """Should return immediately if already completed."""
        from src.webshop_mcp import SessionState, WebShopMCPServer

        state = SessionState(
            session_id="test",
            goal="Find shoes",
            budget=100.0,
        )
        server = WebShopMCPServer(state)

        # Mark as completed
        state.mark_completed("checkout")
        server.signal_completion({"terminated": True, "score": 1.0})

        # Should return immediately
        result = await asyncio.wait_for(server.wait_for_completion(), timeout=1.0)
        assert result["terminated"] is True
        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_wait_for_completion_timeout(self):
        """Should raise TimeoutError if not completed in time."""
        from src.webshop_mcp import SessionState, WebShopMCPServer

        state = SessionState(
            session_id="test",
            goal="Find shoes",
            budget=100.0,
        )
        server = WebShopMCPServer(state)

        with pytest.raises(asyncio.TimeoutError):
            await server.wait_for_completion(timeout=0.1)

    @pytest.mark.asyncio
    async def test_wait_for_completion_signaled(self):
        """Should return when completion is signaled."""
        from src.webshop_mcp import SessionState, WebShopMCPServer

        state = SessionState(
            session_id="test",
            goal="Find shoes",
            budget=100.0,
        )
        server = WebShopMCPServer(state)

        # Signal completion in background
        async def signal_later():
            await asyncio.sleep(0.1)
            state.mark_completed("checkout")
            server.signal_completion({"terminated": True, "score": 1.0})

        asyncio.create_task(signal_later())

        # Wait for completion
        result = await server.wait_for_completion(timeout=1.0)
        assert result["terminated"] is True

    def test_is_completed(self):
        """Should return correct completion status."""
        from src.webshop_mcp import SessionState, WebShopMCPServer

        state = SessionState(
            session_id="test",
            goal="Find shoes",
            budget=100.0,
        )
        server = WebShopMCPServer(state)

        assert server.is_completed() is False

        state.mark_completed("checkout")
        assert server.is_completed() is True

    def test_get_final_result_before_completion(self):
        """Should return None before completion."""
        from src.webshop_mcp import SessionState, WebShopMCPServer

        state = SessionState(
            session_id="test",
            goal="Find shoes",
            budget=100.0,
        )
        server = WebShopMCPServer(state)

        assert server.get_final_result() is None

    def test_get_final_result_after_completion(self):
        """Should return result after completion."""
        from src.webshop_mcp import SessionState, WebShopMCPServer

        state = SessionState(
            session_id="test",
            goal="Find shoes",
            budget=100.0,
        )
        server = WebShopMCPServer(state)

        expected = {"terminated": True, "score": 0.8}
        server.signal_completion(expected)

        result = server.get_final_result()
        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
