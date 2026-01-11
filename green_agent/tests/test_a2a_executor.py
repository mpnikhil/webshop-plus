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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
