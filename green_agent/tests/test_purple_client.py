"""
Tests for PurpleAgentClient SDK-based A2A client.

Stage 7c of the AAA (A2A + MCP Agentification) implementation.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    Artifact,
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

from src.purple_client import (
    ConnectionError,
    PurpleAgentClient,
    PurpleAgentClientError,
    TaskError,
    TaskResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agent_card() -> AgentCard:
    """Create a mock agent card."""
    return AgentCard(
        name="Test Purple Agent",
        url="http://localhost:8001",
        version="1.0.0",
        description="A test purple agent for unit tests",
        skills=[],
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )


@pytest.fixture
def mock_task() -> Task:
    """Create a mock task."""
    return Task(
        id="task-123",
        context_id="ctx-456",
        status=TaskStatus(state=TaskState.completed),
    )


@pytest.fixture
def mock_client(mock_agent_card, mock_task) -> MagicMock:
    """Create a mock SDK Client."""
    client = MagicMock()
    client.get_card = AsyncMock(return_value=mock_agent_card)

    # Default: return completed task
    async def mock_send_message(*args, **kwargs):
        update = TaskStatusUpdateEvent(
            task_id=mock_task.id,
            context_id=mock_task.context_id,
            status=TaskStatus(state=TaskState.completed),
            final=True,
            kind="status-update",
        )
        yield (mock_task, update)

    client.send_message = mock_send_message
    return client


# =============================================================================
# Test: Connection
# =============================================================================


class TestConnection:
    """Tests for connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_client, mock_agent_card):
        """Test successful connection to purple agent."""
        with patch("src.purple_client.ClientFactory") as MockFactory:
            MockFactory.connect = AsyncMock(return_value=mock_client)

            client = PurpleAgentClient("http://localhost:8001")
            await client.connect()

            assert client.is_connected
            assert client.agent_card == mock_agent_card
            MockFactory.connect.assert_called_once_with("http://localhost:8001")

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure raises ConnectionError."""
        with patch("src.purple_client.ClientFactory") as MockFactory:
            MockFactory.connect = AsyncMock(side_effect=Exception("Connection refused"))

            client = PurpleAgentClient("http://localhost:8001")

            with pytest.raises(ConnectionError) as exc_info:
                await client.connect()

            assert "Connection refused" in str(exc_info.value)
            assert not client.is_connected

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_client, mock_agent_card):
        """Test async context manager connects and closes."""
        with patch("src.purple_client.ClientFactory") as MockFactory:
            MockFactory.connect = AsyncMock(return_value=mock_client)

            async with PurpleAgentClient("http://localhost:8001") as client:
                assert client.is_connected
                assert client.agent_card == mock_agent_card

            # After exit, should be disconnected
            assert not client.is_connected

    @pytest.mark.asyncio
    async def test_close_disconnects(self, mock_client):
        """Test close() disconnects the client."""
        with patch("src.purple_client.ClientFactory") as MockFactory:
            MockFactory.connect = AsyncMock(return_value=mock_client)

            client = PurpleAgentClient("http://localhost:8001")
            await client.connect()
            assert client.is_connected

            await client.close()
            assert not client.is_connected
            assert client.agent_card is None


# =============================================================================
# Test: Send Task
# =============================================================================


class TestSendTask:
    """Tests for send_task method."""

    @pytest.mark.asyncio
    async def test_send_task_success(self, mock_client, mock_task):
        """Test successful task execution."""
        with patch("src.purple_client.ClientFactory") as MockFactory:
            MockFactory.connect = AsyncMock(return_value=mock_client)

            async with PurpleAgentClient("http://localhost:8001") as client:
                result = await client.send_task(
                    goal="Find running shoes under $50",
                    budget=50.0,
                    constraints=["waterproof"],
                )

                assert result.success
                assert result.task_id == "task-123"
                assert result.context_id == "ctx-456"
                assert result.final_state == TaskState.completed

    @pytest.mark.asyncio
    async def test_send_task_with_mcp_uri(self, mock_client):
        """Test task with MCP URI included in kickoff."""
        messages_sent = []

        async def capture_send_message(message, *args, **kwargs):
            messages_sent.append(message)
            task = Task(
                id="task-mcp",
                context_id="ctx-mcp",
                status=TaskStatus(state=TaskState.completed),
            )
            update = TaskStatusUpdateEvent(
                task_id="task-mcp",
                context_id="ctx-mcp",
                status=TaskStatus(state=TaskState.completed),
                final=True,
                kind="status-update",
            )
            yield (task, update)

        mock_client.send_message = capture_send_message

        with patch("src.purple_client.ClientFactory") as MockFactory:
            MockFactory.connect = AsyncMock(return_value=mock_client)

            async with PurpleAgentClient("http://localhost:8001") as client:
                await client.send_task(
                    goal="Buy shoes",
                    budget=100.0,
                    constraints=[],
                    mcp_uri="http://localhost:8000/mcp/session-abc",
                )

                # Verify message content includes MCP resource
                assert len(messages_sent) == 1
                msg = messages_sent[0]
                assert len(msg.parts) == 1
                # SDK wraps TextPart in Part(root=TextPart(...))
                part = msg.parts[0]
                text = part.root.text if isinstance(part, Part) else part.text
                content = json.loads(text)
                assert content["goal"] == "Buy shoes"
                assert content["budget"] == 100.0
                assert "resources" in content
                assert content["resources"][0]["type"] == "mcp"
                assert content["resources"][0]["uri"] == "http://localhost:8000/mcp/session-abc"

    @pytest.mark.asyncio
    async def test_send_task_not_connected(self):
        """Test send_task raises error when not connected."""
        client = PurpleAgentClient("http://localhost:8001")

        with pytest.raises(ConnectionError) as exc_info:
            await client.send_task(goal="Test", budget=10.0)

        assert "Not connected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_send_task_failure(self, mock_client):
        """Test task that fails on the agent side."""
        # TaskStatus.message must be a Message object, not a string
        error_msg = Message(
            message_id="err-msg",
            role=Role.agent,
            parts=[Part(root=TextPart(text="Agent error"))],
        )

        async def failing_send_message(*args, **kwargs):
            task = Task(
                id="task-fail",
                context_id="ctx-fail",
                status=TaskStatus(state=TaskState.failed, message=error_msg),
            )
            update = TaskStatusUpdateEvent(
                task_id="task-fail",
                context_id="ctx-fail",
                status=TaskStatus(state=TaskState.failed, message=error_msg),
                final=True,
                kind="status-update",
            )
            yield (task, update)

        mock_client.send_message = failing_send_message

        with patch("src.purple_client.ClientFactory") as MockFactory:
            MockFactory.connect = AsyncMock(return_value=mock_client)

            async with PurpleAgentClient("http://localhost:8001") as client:
                result = await client.send_task(goal="Test", budget=10.0)

                assert not result.success
                assert result.final_state == TaskState.failed
                # Error should contain the message text
                assert result.error is not None
                assert "Agent error" in result.error


# =============================================================================
# Test: Result Extraction
# =============================================================================


class TestResultExtraction:
    """Tests for result extraction from various response formats."""

    @pytest.mark.asyncio
    async def test_extract_result_from_artifact(self, mock_client):
        """Test extraction of result from artifact update."""
        result_data = {"score": 0.95, "cart": [{"name": "Shoes", "price": 45.0}]}

        async def send_with_artifact(*args, **kwargs):
            task = Task(
                id="task-art",
                context_id="ctx-art",
                status=TaskStatus(state=TaskState.completed),
            )

            # First: working status
            yield (task, TaskStatusUpdateEvent(
                task_id="task-art",
                context_id="ctx-art",
                status=TaskStatus(state=TaskState.working),
                final=False,
                kind="status-update",
            ))

            # Second: artifact with result
            yield (task, TaskArtifactUpdateEvent(
                task_id="task-art",
                context_id="ctx-art",
                artifact=Artifact(
                    artifact_id="art-1",
                    parts=[Part(root=TextPart(text=json.dumps(result_data)))],
                ),
                kind="artifact-update",
            ))

            # Third: completed status
            yield (task, TaskStatusUpdateEvent(
                task_id="task-art",
                context_id="ctx-art",
                status=TaskStatus(state=TaskState.completed),
                final=True,
                kind="status-update",
            ))

        mock_client.send_message = send_with_artifact

        with patch("src.purple_client.ClientFactory") as MockFactory:
            MockFactory.connect = AsyncMock(return_value=mock_client)

            async with PurpleAgentClient("http://localhost:8001") as client:
                result = await client.send_task(goal="Test", budget=100.0)

                assert result.success
                assert result.result_data == result_data
                assert result.result_data["score"] == 0.95

    @pytest.mark.asyncio
    async def test_extract_result_from_direct_message(self, mock_client):
        """Test extraction of result from direct message response."""
        result_data = {"checkout": True, "total": 42.50}

        async def send_with_message(*args, **kwargs):
            # Yield a direct message response (not a task/update tuple)
            yield Message(
                message_id="msg-1",
                role=Role.agent,
                parts=[Part(root=TextPart(text=json.dumps(result_data)))],
            )

        mock_client.send_message = send_with_message

        with patch("src.purple_client.ClientFactory") as MockFactory:
            MockFactory.connect = AsyncMock(return_value=mock_client)

            async with PurpleAgentClient("http://localhost:8001") as client:
                result = await client.send_task(goal="Test", budget=100.0)

                # Direct message doesn't complete a task, so success depends on state
                assert result.result_data == result_data

    @pytest.mark.asyncio
    async def test_extract_result_from_history(self, mock_client):
        """Test extraction of result from task history."""
        result_data = {"found": True, "products": [1, 2, 3]}

        async def send_with_history(*args, **kwargs):
            task = Task(
                id="task-hist",
                context_id="ctx-hist",
                status=TaskStatus(state=TaskState.completed),
                history=[
                    Message(
                        message_id="m1",
                        role=Role.user,
                        parts=[Part(root=TextPart(text="Buy shoes"))],
                    ),
                    Message(
                        message_id="m2",
                        role=Role.agent,
                        parts=[Part(root=TextPart(text=json.dumps(result_data)))],
                    ),
                ],
            )
            yield (task, TaskStatusUpdateEvent(
                task_id="task-hist",
                context_id="ctx-hist",
                status=TaskStatus(state=TaskState.completed),
                final=True,
                kind="status-update",
            ))

        mock_client.send_message = send_with_history

        with patch("src.purple_client.ClientFactory") as MockFactory:
            MockFactory.connect = AsyncMock(return_value=mock_client)

            async with PurpleAgentClient("http://localhost:8001") as client:
                result = await client.send_task(goal="Test", budget=100.0)

                assert result.success
                assert result.result_data == result_data


# =============================================================================
# Test: Build Kickoff
# =============================================================================


class TestBuildKickoff:
    """Tests for kickoff message building."""

    def test_build_kickoff_basic(self):
        """Test basic kickoff without MCP."""
        client = PurpleAgentClient("http://localhost:8001")

        kickoff = client._build_kickoff(
            goal="Find running shoes under $50",
            budget=50.0,
            constraints=["waterproof", "size 10"],
            mcp_uri=None,
        )

        assert kickoff["goal"] == "Find running shoes under $50"
        assert kickoff["budget"] == 50.0
        assert kickoff["constraints"] == ["waterproof", "size 10"]
        assert "resources" not in kickoff

    def test_build_kickoff_with_mcp(self):
        """Test kickoff with MCP resource."""
        client = PurpleAgentClient("http://localhost:8001")

        kickoff = client._build_kickoff(
            goal="Buy shoes",
            budget=100.0,
            constraints=[],
            mcp_uri="http://localhost:8000/mcp/session-123",
        )

        assert kickoff["goal"] == "Buy shoes"
        assert kickoff["budget"] == 100.0
        assert kickoff["constraints"] == []
        assert "resources" in kickoff
        assert len(kickoff["resources"]) == 1
        assert kickoff["resources"][0]["type"] == "mcp"
        assert kickoff["resources"][0]["uri"] == "http://localhost:8000/mcp/session-123"
        assert "description" in kickoff["resources"][0]


# =============================================================================
# Test: Send Message
# =============================================================================


class TestSendMessage:
    """Tests for send_message lower-level API."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, mock_client, mock_task):
        """Test sending a raw message."""
        with patch("src.purple_client.ClientFactory") as MockFactory:
            MockFactory.connect = AsyncMock(return_value=mock_client)

            async with PurpleAgentClient("http://localhost:8001") as client:
                result = await client.send_message("Hello, agent!")

                assert result.success
                assert result.task_id == "task-123"

    @pytest.mark.asyncio
    async def test_send_message_with_role(self, mock_client):
        """Test sending message with custom role."""
        messages_sent = []

        async def capture_send(*args, **kwargs):
            messages_sent.append(args[0])
            task = Task(
                id="t1",
                context_id="c1",
                status=TaskStatus(state=TaskState.completed),
            )
            yield (task, TaskStatusUpdateEvent(
                task_id="t1",
                context_id="c1",
                status=TaskStatus(state=TaskState.completed),
                final=True,
                kind="status-update",
            ))

        mock_client.send_message = capture_send

        with patch("src.purple_client.ClientFactory") as MockFactory:
            MockFactory.connect = AsyncMock(return_value=mock_client)

            async with PurpleAgentClient("http://localhost:8001") as client:
                await client.send_message("System message", role=Role.user)

                assert len(messages_sent) == 1
                assert messages_sent[0].role == Role.user


# =============================================================================
# Test: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_task_execution_exception(self, mock_client):
        """Test handling of exceptions during task execution."""
        async def raise_error(*args, **kwargs):
            raise RuntimeError("Network error")
            yield  # Make it a generator

        mock_client.send_message = raise_error

        with patch("src.purple_client.ClientFactory") as MockFactory:
            MockFactory.connect = AsyncMock(return_value=mock_client)

            async with PurpleAgentClient("http://localhost:8001") as client:
                with pytest.raises(TaskError) as exc_info:
                    await client.send_task(goal="Test", budget=10.0)

                assert "Network error" in str(exc_info.value)

    def test_exception_inheritance(self):
        """Test that custom exceptions inherit from base class."""
        assert issubclass(ConnectionError, PurpleAgentClientError)
        assert issubclass(TaskError, PurpleAgentClientError)

    @pytest.mark.asyncio
    async def test_url_normalization(self, mock_client):
        """Test that trailing slashes are stripped from URL."""
        with patch("src.purple_client.ClientFactory") as MockFactory:
            MockFactory.connect = AsyncMock(return_value=mock_client)

            client = PurpleAgentClient("http://localhost:8001/")
            assert client.agent_url == "http://localhost:8001"


# =============================================================================
# Test: TaskResult
# =============================================================================


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_task_result_defaults(self):
        """Test TaskResult default values."""
        result = TaskResult(
            success=True,
            task_id="t1",
            context_id="c1",
            final_state=TaskState.completed,
        )

        assert result.success
        assert result.task_id == "t1"
        assert result.context_id == "c1"
        assert result.final_state == TaskState.completed
        assert result.result_data is None
        assert result.raw_task is None
        assert result.error is None

    def test_task_result_with_all_fields(self, mock_task):
        """Test TaskResult with all fields populated."""
        result_data = {"score": 0.9}

        result = TaskResult(
            success=True,
            task_id="t1",
            context_id="c1",
            final_state=TaskState.completed,
            result_data=result_data,
            raw_task=mock_task,
            error=None,
        )

        assert result.result_data == result_data
        assert result.raw_task == mock_task
