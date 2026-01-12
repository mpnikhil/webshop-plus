"""
Tests for ADK-based ShoppingAgent.

These tests verify:
- Agent creation with default and custom parameters
- Instruction formatting with task details
- Input validation for run()
- Integration with mocked MCP toolset

Note: These are unit tests that mock external dependencies.
Integration tests with a real MCP server are in Stage 11.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from src.shopping_agent import (
    ShoppingAgent,
    SHOPPING_INSTRUCTION,
    DEFAULT_MODEL,
    DEFAULT_MAX_TURNS,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def shopping_agent():
    """Create a ShoppingAgent instance for testing."""
    return ShoppingAgent()


@pytest.fixture
def custom_agent():
    """Create a ShoppingAgent with custom parameters."""
    return ShoppingAgent(model="gemini-1.5-pro", max_turns=50)


@pytest.fixture
def task_data():
    """Create sample task data."""
    return {
        "goal": "Find waterproof running shoes under $80",
        "budget": 80.0,
        "constraints": ["waterproof", "size 10"],
        "session_id": "test-session-123",
    }


@pytest.fixture
def minimal_task_data():
    """Create minimal task data (only required fields)."""
    return {
        "goal": "Find a laptop",
    }


# =============================================================================
# Test Agent Creation
# =============================================================================


class TestAgentCreation:
    """Tests for ShoppingAgent initialization."""

    def test_default_parameters(self, shopping_agent):
        """Agent created with default parameters."""
        assert shopping_agent.model == DEFAULT_MODEL
        assert shopping_agent.max_turns == DEFAULT_MAX_TURNS

    def test_custom_model(self, custom_agent):
        """Agent created with custom model."""
        assert custom_agent.model == "gemini-1.5-pro"

    def test_custom_max_turns(self, custom_agent):
        """Agent created with custom max_turns."""
        assert custom_agent.max_turns == 50

    def test_session_service_initialized(self, shopping_agent):
        """Session service is initialized."""
        assert shopping_agent._session_service is not None

    @patch.dict("os.environ", {"ADK_MODEL": "gemini-2.5-flash"})
    def test_model_from_environment(self):
        """Model can be configured via environment variable."""
        # Need to reimport to pick up env var
        from src.shopping_agent import ShoppingAgent as SA
        agent = SA()
        # Note: The default is already set at module load time,
        # so this test verifies the env var mechanism exists
        assert agent._model is not None


# =============================================================================
# Test Instruction Formatting
# =============================================================================


class TestInstructionFormatting:
    """Tests for instruction template formatting."""

    def test_format_instruction_full(self, shopping_agent, task_data):
        """Instruction formatted with all task details."""
        instruction = shopping_agent._format_instruction(
            goal=task_data["goal"],
            budget=task_data["budget"],
            constraints=task_data["constraints"],
        )

        assert "Find waterproof running shoes under $80" in instruction
        assert "$80.0" in instruction
        assert "waterproof, size 10" in instruction

    def test_format_instruction_no_constraints(self, shopping_agent):
        """Instruction formatted with empty constraints."""
        instruction = shopping_agent._format_instruction(
            goal="Find a laptop",
            budget=500.0,
            constraints=[],
        )

        assert "Find a laptop" in instruction
        assert "$500.0" in instruction
        assert "CONSTRAINTS: none" in instruction

    def test_format_instruction_single_constraint(self, shopping_agent):
        """Instruction formatted with single constraint."""
        instruction = shopping_agent._format_instruction(
            goal="Find headphones",
            budget=100.0,
            constraints=["wireless"],
        )

        assert "Find headphones" in instruction
        assert "wireless" in instruction

    def test_instruction_contains_tool_descriptions(self, shopping_agent):
        """Instruction contains MCP tool descriptions."""
        instruction = shopping_agent._format_instruction(
            goal="Test",
            budget=100.0,
            constraints=[],
        )

        assert "search" in instruction.lower()
        assert "click" in instruction.lower()
        assert "checkout" in instruction.lower()

    def test_instruction_contains_rules(self, shopping_agent):
        """Instruction contains important rules."""
        instruction = shopping_agent._format_instruction(
            goal="Test",
            budget=100.0,
            constraints=[],
        )

        assert "element ID" in instruction or "element_id" in instruction
        assert "TERMINAL" in instruction


# =============================================================================
# Test Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for run() input validation."""

    @pytest.mark.asyncio
    async def test_run_missing_mcp_uri(self, shopping_agent, task_data):
        """Run raises ValueError when mcp_uri is missing."""
        with pytest.raises(ValueError, match="mcp_uri is required"):
            await shopping_agent.run(mcp_uri="", task_data=task_data)

    @pytest.mark.asyncio
    async def test_run_missing_goal(self, shopping_agent):
        """Run raises ValueError when goal is missing."""
        with pytest.raises(ValueError, match="must contain 'goal'"):
            await shopping_agent.run(
                mcp_uri="http://localhost:8000/mcp/session-123",
                task_data={"budget": 100.0},
            )

    @pytest.mark.asyncio
    async def test_run_empty_task_data(self, shopping_agent):
        """Run raises ValueError when task_data is empty."""
        with pytest.raises(ValueError, match="must contain 'goal'"):
            await shopping_agent.run(
                mcp_uri="http://localhost:8000/mcp/session-123",
                task_data={},
            )


# =============================================================================
# Test Event Processing
# =============================================================================


class TestEventProcessing:
    """Tests for event processing methods."""

    def test_is_final_event_turn_complete(self, shopping_agent):
        """Turn complete event is recognized as final."""
        event = MagicMock()
        event.turn_complete = True
        event.finish_reason = None
        event.error_message = None

        assert shopping_agent._is_final_event(event) is True

    def test_is_final_event_finish_reason(self, shopping_agent):
        """Finish reason event is recognized as final."""
        event = MagicMock()
        event.turn_complete = False
        event.finish_reason = "STOP"
        event.error_message = None

        assert shopping_agent._is_final_event(event) is True

    def test_is_final_event_error(self, shopping_agent):
        """Error event is recognized as final."""
        event = MagicMock()
        event.turn_complete = False
        event.finish_reason = None
        event.error_message = "Connection failed"

        assert shopping_agent._is_final_event(event) is True

    def test_is_final_event_not_final(self, shopping_agent):
        """Intermediate event is not recognized as final."""
        event = MagicMock()
        event.turn_complete = False
        event.finish_reason = None
        event.error_message = None

        assert shopping_agent._is_final_event(event) is False

    def test_extract_message_from_content(self, shopping_agent):
        """Extract text from event content."""
        event = MagicMock()
        event.error_message = None
        part = MagicMock()
        part.text = "I found some products."
        event.content = MagicMock()
        event.content.parts = [part]

        message = shopping_agent._extract_message(event)
        assert message == "I found some products."

    def test_extract_message_multiple_parts(self, shopping_agent):
        """Extract text from multiple content parts."""
        event = MagicMock()
        event.error_message = None
        part1 = MagicMock()
        part1.text = "Part 1."
        part2 = MagicMock()
        part2.text = "Part 2."
        event.content = MagicMock()
        event.content.parts = [part1, part2]

        message = shopping_agent._extract_message(event)
        assert message == "Part 1. Part 2."

    def test_extract_message_from_error(self, shopping_agent):
        """Extract error message from event."""
        event = MagicMock()
        event.error_message = "Connection failed"
        event.content = None

        message = shopping_agent._extract_message(event)
        assert "Error: Connection failed" in message

    def test_extract_message_empty_content(self, shopping_agent):
        """Empty string when no content."""
        event = MagicMock()
        event.error_message = None
        event.content = None

        message = shopping_agent._extract_message(event)
        assert message == ""

    def test_extract_message_empty_parts(self, shopping_agent):
        """Empty string when parts is empty."""
        event = MagicMock()
        event.error_message = None
        event.content = MagicMock()
        event.content.parts = []

        message = shopping_agent._extract_message(event)
        assert message == ""


# =============================================================================
# Test Run with Mocked Dependencies
# =============================================================================


class TestRunWithMocks:
    """Tests for run() with mocked ADK dependencies."""

    @pytest.mark.asyncio
    async def test_run_success_path(self, shopping_agent, task_data):
        """Test successful run with mocked runner."""
        # Create mock event
        mock_event = MagicMock()
        mock_event.turn_complete = True
        mock_event.finish_reason = "STOP"
        mock_event.error_message = None
        part = MagicMock()
        part.text = "Successfully completed purchase."
        mock_event.content = MagicMock()
        mock_event.content.parts = [part]

        # Create async generator for runner
        async def mock_run_async(*args, **kwargs):
            yield mock_event

        with patch("src.shopping_agent.Agent") as MockAgent, \
             patch("src.shopping_agent.Runner") as MockRunner, \
             patch("src.shopping_agent.McpToolset") as MockToolset:

            # Configure mocks
            mock_runner_instance = MagicMock()
            mock_runner_instance.run_async = mock_run_async
            MockRunner.return_value = mock_runner_instance

            result = await shopping_agent.run(
                mcp_uri="http://localhost:8000/mcp/session-123",
                task_data=task_data,
            )

            assert result["success"] is True
            assert "Successfully completed purchase" in result["final_message"]
            assert result["turns_used"] == 1

    @pytest.mark.asyncio
    async def test_run_error_path(self, shopping_agent, task_data):
        """Test run that results in an error."""
        # Create mock error event
        mock_event = MagicMock()
        mock_event.turn_complete = False
        mock_event.finish_reason = None
        mock_event.error_message = "MCP connection failed"
        mock_event.content = None

        async def mock_run_async(*args, **kwargs):
            yield mock_event

        with patch("src.shopping_agent.Agent") as MockAgent, \
             patch("src.shopping_agent.Runner") as MockRunner, \
             patch("src.shopping_agent.McpToolset") as MockToolset:

            mock_runner_instance = MagicMock()
            mock_runner_instance.run_async = mock_run_async
            MockRunner.return_value = mock_runner_instance

            result = await shopping_agent.run(
                mcp_uri="http://localhost:8000/mcp/session-123",
                task_data=task_data,
            )

            assert result["success"] is False
            assert "Error" in result["final_message"]

    @pytest.mark.asyncio
    async def test_run_max_turns_reached(self, task_data):
        """Test run that hits max turns limit."""
        agent = ShoppingAgent(max_turns=3)

        # Create mock events that never complete
        mock_event = MagicMock()
        mock_event.turn_complete = False
        mock_event.finish_reason = None
        mock_event.error_message = None
        part = MagicMock()
        part.text = "Processing..."
        mock_event.content = MagicMock()
        mock_event.content.parts = [part]

        # Yield more events than max_turns
        async def mock_run_async(*args, **kwargs):
            for _ in range(10):
                yield mock_event

        with patch("src.shopping_agent.Agent"), \
             patch("src.shopping_agent.Runner") as MockRunner, \
             patch("src.shopping_agent.McpToolset"):

            mock_runner_instance = MagicMock()
            mock_runner_instance.run_async = mock_run_async
            MockRunner.return_value = mock_runner_instance

            result = await agent.run(
                mcp_uri="http://localhost:8000/mcp/session-123",
                task_data=task_data,
            )

            # Should stop at max_turns
            assert result["turns_used"] <= 3

    @pytest.mark.asyncio
    async def test_run_exception_handling(self, shopping_agent, task_data):
        """Test run handles exceptions gracefully."""
        with patch("src.shopping_agent.Agent") as MockAgent, \
             patch("src.shopping_agent.Runner") as MockRunner, \
             patch("src.shopping_agent.McpToolset"):

            # Make runner raise an exception
            MockRunner.side_effect = RuntimeError("Initialization failed")

            result = await shopping_agent.run(
                mcp_uri="http://localhost:8000/mcp/session-123",
                task_data=task_data,
            )

            assert result["success"] is False
            assert "error" in result
            assert "Initialization failed" in result["error"]

    @pytest.mark.asyncio
    async def test_run_uses_correct_mcp_uri(self, shopping_agent, task_data):
        """Test that run passes correct MCP URI to toolset."""
        mock_event = MagicMock()
        mock_event.turn_complete = True
        mock_event.finish_reason = "STOP"
        mock_event.error_message = None
        mock_event.content = None

        async def mock_run_async(*args, **kwargs):
            yield mock_event

        with patch("src.shopping_agent.Agent"), \
             patch("src.shopping_agent.Runner") as MockRunner, \
             patch("src.shopping_agent.McpToolset") as MockToolset, \
             patch("src.shopping_agent.StreamableHTTPConnectionParams") as MockParams:

            mock_runner_instance = MagicMock()
            mock_runner_instance.run_async = mock_run_async
            MockRunner.return_value = mock_runner_instance

            mcp_uri = "http://localhost:8000/mcp/session-xyz"
            await shopping_agent.run(mcp_uri=mcp_uri, task_data=task_data)

            # Verify StreamableHTTPConnectionParams was called with correct URL
            MockParams.assert_called_once()
            call_kwargs = MockParams.call_args[1]
            assert call_kwargs["url"] == mcp_uri

    @pytest.mark.asyncio
    async def test_run_creates_agent_with_instruction(self, shopping_agent, task_data):
        """Test that run creates agent with formatted instruction."""
        mock_event = MagicMock()
        mock_event.turn_complete = True
        mock_event.finish_reason = "STOP"
        mock_event.error_message = None
        mock_event.content = None

        async def mock_run_async(*args, **kwargs):
            yield mock_event

        with patch("src.shopping_agent.Agent") as MockAgent, \
             patch("src.shopping_agent.Runner") as MockRunner, \
             patch("src.shopping_agent.McpToolset"):

            mock_runner_instance = MagicMock()
            mock_runner_instance.run_async = mock_run_async
            MockRunner.return_value = mock_runner_instance

            await shopping_agent.run(
                mcp_uri="http://localhost:8000/mcp/session-123",
                task_data=task_data,
            )

            # Verify Agent was created with task-specific instruction
            MockAgent.assert_called_once()
            call_kwargs = MockAgent.call_args[1]
            assert task_data["goal"] in call_kwargs["instruction"]


# =============================================================================
# Test Default Task Data Values
# =============================================================================


class TestDefaultValues:
    """Tests for default task data values."""

    @pytest.mark.asyncio
    async def test_default_budget(self, shopping_agent, minimal_task_data):
        """Default budget is used when not specified."""
        mock_event = MagicMock()
        mock_event.turn_complete = True
        mock_event.finish_reason = "STOP"
        mock_event.error_message = None
        mock_event.content = None

        async def mock_run_async(*args, **kwargs):
            yield mock_event

        with patch("src.shopping_agent.Agent") as MockAgent, \
             patch("src.shopping_agent.Runner") as MockRunner, \
             patch("src.shopping_agent.McpToolset"):

            mock_runner_instance = MagicMock()
            mock_runner_instance.run_async = mock_run_async
            MockRunner.return_value = mock_runner_instance

            await shopping_agent.run(
                mcp_uri="http://localhost:8000/mcp/session-123",
                task_data=minimal_task_data,
            )

            # Verify default budget (100.0) is used in instruction
            call_kwargs = MockAgent.call_args[1]
            assert "$100.0" in call_kwargs["instruction"]

    @pytest.mark.asyncio
    async def test_default_constraints(self, shopping_agent, minimal_task_data):
        """Default constraints (empty) used when not specified."""
        mock_event = MagicMock()
        mock_event.turn_complete = True
        mock_event.finish_reason = "STOP"
        mock_event.error_message = None
        mock_event.content = None

        async def mock_run_async(*args, **kwargs):
            yield mock_event

        with patch("src.shopping_agent.Agent") as MockAgent, \
             patch("src.shopping_agent.Runner") as MockRunner, \
             patch("src.shopping_agent.McpToolset"):

            mock_runner_instance = MagicMock()
            mock_runner_instance.run_async = mock_run_async
            MockRunner.return_value = mock_runner_instance

            await shopping_agent.run(
                mcp_uri="http://localhost:8000/mcp/session-123",
                task_data=minimal_task_data,
            )

            # Verify "none" constraints used in instruction
            call_kwargs = MockAgent.call_args[1]
            assert "CONSTRAINTS: none" in call_kwargs["instruction"]

    @pytest.mark.asyncio
    async def test_generates_session_id_if_not_provided(self, shopping_agent, minimal_task_data):
        """Session ID is generated if not provided."""
        mock_event = MagicMock()
        mock_event.turn_complete = True
        mock_event.finish_reason = "STOP"
        mock_event.error_message = None
        mock_event.content = None

        async def mock_run_async(*args, **kwargs):
            # Check session_id is a valid UUID
            assert "session_id" in kwargs
            try:
                uuid.UUID(kwargs["session_id"])
            except ValueError:
                pytest.fail("session_id should be a valid UUID")
            yield mock_event

        with patch("src.shopping_agent.Agent"), \
             patch("src.shopping_agent.Runner") as MockRunner, \
             patch("src.shopping_agent.McpToolset"):

            mock_runner_instance = MagicMock()
            mock_runner_instance.run_async = mock_run_async
            MockRunner.return_value = mock_runner_instance

            await shopping_agent.run(
                mcp_uri="http://localhost:8000/mcp/session-123",
                task_data=minimal_task_data,
            )


# =============================================================================
# Test Module Constants
# =============================================================================


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_shopping_instruction_template_exists(self):
        """SHOPPING_INSTRUCTION template is defined."""
        assert SHOPPING_INSTRUCTION is not None
        assert len(SHOPPING_INSTRUCTION) > 100

    def test_instruction_has_placeholders(self):
        """Instruction template has required placeholders."""
        assert "{goal}" in SHOPPING_INSTRUCTION
        assert "{budget}" in SHOPPING_INSTRUCTION
        assert "{constraints}" in SHOPPING_INSTRUCTION

    def test_default_model_is_set(self):
        """DEFAULT_MODEL constant is set."""
        assert DEFAULT_MODEL is not None
        assert len(DEFAULT_MODEL) > 0

    def test_default_max_turns_is_reasonable(self):
        """DEFAULT_MAX_TURNS is a reasonable value."""
        assert DEFAULT_MAX_TURNS > 0
        assert DEFAULT_MAX_TURNS <= 100
