"""
Tests for WebShop+ A2A server and messenger modules.

This module tests:
- Agent card generation and serving
- A2A message formatting and parsing
- JSON-RPC request/response handling
- SSE streaming endpoints
- Assessment request handling
"""

import json
import pytest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from src.server_legacy import app, state
from src.messenger import (
    A2AClient,
    A2AMessage,
    A2ATask,
    AgentCard,
    Artifact,
    JSONRPCRequest,
    JSONRPCResponse,
    MessageRole,
    TaskState,
    TaskStatus,
    create_artifact_update_event,
    create_error_response,
    create_message_send_request,
    create_message_stream_request,
    create_status_update_event,
    create_task_response,
    create_text_message,
    create_webshop_plus_agent_card,
    extract_action_from_text,
    get_text_from_message,
    parse_action_from_response,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def client():
    """Create a test client."""
    # Set default card URL for tests
    state.card_url = "http://localhost:8000"
    state.agent_card = None
    state.active_assessments = {}
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create an async test client."""
    state.card_url = "http://localhost:8000"
    state.agent_card = None
    state.active_assessments = {}
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac


# =============================================================================
# Messenger Tests
# =============================================================================


class TestMessageCreation:
    """Test A2A message creation utilities."""

    def test_create_text_message(self):
        """Test creating a simple text message."""
        msg = create_text_message("Hello, agent!")

        assert msg.role == MessageRole.USER
        assert len(msg.parts) == 1
        assert msg.parts[0]["kind"] == "text"
        assert msg.parts[0]["text"] == "Hello, agent!"
        assert msg.messageId is not None

    def test_create_text_message_with_task_id(self):
        """Test creating a text message with task context."""
        msg = create_text_message(
            "Test message",
            role=MessageRole.AGENT,
            task_id="task-123",
            context_id="ctx-456",
        )

        assert msg.role == MessageRole.AGENT
        assert msg.taskId == "task-123"
        assert msg.contextId == "ctx-456"

    def test_create_message_send_request(self):
        """Test creating a message/send JSON-RPC request."""
        msg = create_text_message("Test")
        request = create_message_send_request(msg)

        assert request.jsonrpc == "2.0"
        assert request.method == "message/send"
        assert "message" in request.params
        assert request.id is not None

    def test_create_message_stream_request(self):
        """Test creating a message/stream JSON-RPC request."""
        msg = create_text_message("Test")
        request = create_message_stream_request(msg, metadata={"key": "value"})

        assert request.method == "message/stream"
        assert request.params["metadata"]["key"] == "value"


class TestTaskCreation:
    """Test A2A task creation utilities."""

    def test_create_task(self):
        """Test creating an A2A task."""
        task = A2ATask(
            status=TaskStatus(state=TaskState.WORKING, message="Processing"),
        )

        assert task.id is not None
        assert task.contextId is not None
        assert task.status.state == TaskState.WORKING
        assert task.kind == "task"

    def test_create_task_response(self):
        """Test creating a JSON-RPC task response."""
        task = A2ATask(status=TaskStatus(state=TaskState.COMPLETED))
        response = create_task_response(task, "req-123")

        assert response.jsonrpc == "2.0"
        assert response.id == "req-123"
        assert response.result is not None
        assert response.result["status"]["state"] == "completed"

    def test_create_error_response(self):
        """Test creating a JSON-RPC error response."""
        response = create_error_response(
            code=-32600,
            message="Invalid request",
            request_id="req-123",
            data={"detail": "Missing field"},
        )

        assert response.error is not None
        assert response.error["code"] == -32600
        assert response.error["message"] == "Invalid request"
        assert response.error["data"]["detail"] == "Missing field"


class TestStatusEvents:
    """Test SSE event creation."""

    def test_create_status_update_event(self):
        """Test creating a status update event."""
        event = create_status_update_event(
            task_id="task-123",
            context_id="ctx-456",
            state=TaskState.WORKING,
            message="Processing task",
            final=False,
            request_id="req-789",
        )

        assert event["jsonrpc"] == "2.0"
        assert event["id"] == "req-789"
        assert event["result"]["taskId"] == "task-123"
        assert event["result"]["status"]["state"] == "working"
        assert event["result"]["kind"] == "status-update"

    def test_create_artifact_update_event(self):
        """Test creating an artifact update event."""
        artifact = Artifact(
            name="result",
            parts=[{"kind": "text", "text": "Result data"}],
        )
        event = create_artifact_update_event(
            task_id="task-123",
            context_id="ctx-456",
            artifact=artifact,
            append=False,
            last_chunk=True,
            request_id="req-789",
        )

        assert event["result"]["artifact"]["name"] == "result"
        assert event["result"]["lastChunk"] is True
        assert event["result"]["kind"] == "artifact-update"


class TestActionParsing:
    """Test action parsing from agent responses."""

    def test_extract_search_action(self):
        """Test extracting search action from text."""
        text = "I'll search for tents. search[2-person camping tent]"
        action = extract_action_from_text(text)

        assert action == "search[2-person camping tent]"

    def test_extract_click_action(self):
        """Test extracting click action from text."""
        text = "Let me click on that product. click[Buy Now]"
        action = extract_action_from_text(text)

        assert action == "click[Buy Now]"

    def test_extract_action_case_insensitive(self):
        """Test that action extraction is case insensitive."""
        text = "SEARCH[camping gear]"
        action = extract_action_from_text(text)

        assert action == "search[camping gear]"

    def test_extract_action_no_match(self):
        """Test extraction when no action is present."""
        text = "I'm thinking about what to do next."
        action = extract_action_from_text(text)

        assert action is None

    def test_get_text_from_message(self):
        """Test extracting text from message."""
        message = {
            "parts": [
                {"kind": "text", "text": "First part. "},
                {"kind": "text", "text": "Second part."},
                {"kind": "file", "file": {"name": "test.txt"}},
            ]
        }
        text = get_text_from_message(message)

        assert "First part." in text
        assert "Second part." in text

    def test_parse_action_from_response(self):
        """Test parsing action from JSON-RPC response."""
        response = JSONRPCResponse(
            result={
                "history": [
                    {
                        "role": "agent",
                        "parts": [
                            {"kind": "text", "text": "I'll search for tents. search[camping tent]"}
                        ],
                    }
                ]
            },
            id="req-123",
        )

        action = parse_action_from_response(response)
        assert action == "search[camping tent]"

    def test_parse_action_from_error_response(self):
        """Test parsing returns None for error responses."""
        response = JSONRPCResponse(
            error={"code": -32600, "message": "Invalid"},
            id="req-123",
        )

        action = parse_action_from_response(response)
        assert action is None


class TestAgentCard:
    """Test agent card generation."""

    def test_create_webshop_plus_agent_card(self):
        """Test creating the WebShop+ agent card."""
        card = create_webshop_plus_agent_card("http://localhost:8000")

        assert card.name == "WebShop+ Benchmark"
        assert card.protocolVersion == "0.3.0"
        assert card.url == "http://localhost:8000/a2a"
        assert card.capabilities.streaming is True
        assert len(card.skills) > 0

    def test_agent_card_skills(self):
        """Test that agent card has expected skills."""
        card = create_webshop_plus_agent_card("http://localhost:8000")

        skill_ids = [s.id for s in card.skills]
        assert "assessment" in skill_ids
        assert "budget-assessment" in skill_ids
        assert "memory-assessment" in skill_ids


# =============================================================================
# Server Tests
# =============================================================================


class TestAgentCardEndpoint:
    """Test the agent card endpoint."""

    def test_get_agent_card(self, client):
        """Test fetching the agent card."""
        response = client.get("/.well-known/agent-card.json")

        assert response.status_code == 200
        data = response.json()

        assert data["name"] == "WebShop+ Benchmark"
        assert data["protocolVersion"] == "0.3.0"
        assert "skills" in data
        assert "capabilities" in data

    def test_agent_card_content_type(self, client):
        """Test that agent card has correct content type."""
        response = client.get("/.well-known/agent-card.json")

        assert "application/json" in response.headers["content-type"]


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_check(self, client):
        """Test the health check returns healthy."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "webshop-plus-green"


class TestA2AEndpoint:
    """Test the A2A message handling endpoint."""

    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/a2a",
            content="not json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["error"]["code"] == -32700  # Parse error

    def test_invalid_jsonrpc_request(self, client):
        """Test handling of invalid JSON-RPC request."""
        response = client.post("/a2a", json={"not": "valid"})

        assert response.status_code == 400
        data = response.json()
        assert data["error"]["code"] == -32600  # Invalid request

    def test_unknown_method(self, client):
        """Test handling of unknown method."""
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "unknown/method",
                "params": {},
                "id": "req-123",
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["error"]["code"] == -32601  # Method not found

    def test_message_send_simple(self, client):
        """Test simple message/send."""
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "Hello!"}],
                    }
                },
                "id": "req-123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "req-123"
        assert "result" in data
        assert data["result"]["status"]["state"] == "completed"

    def test_message_send_assessment_request(self, client):
        """Test message/send with assessment request."""
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "Run assessment"}],
                    },
                    "metadata": {
                        "participants": {"shopper": "http://agent:8001/a2a"},
                        "config": {"task_types": ["budget_constrained"], "num_tasks": 5},
                    },
                },
                "id": "req-123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["result"]["status"]["state"] == "submitted"

    def test_tasks_get_not_found(self, client):
        """Test tasks/get with non-existent task."""
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "tasks/get",
                "params": {"id": "nonexistent-task"},
                "id": "req-123",
            },
        )

        assert response.status_code == 404
        data = response.json()
        assert data["error"]["code"] == -32001

    def test_tasks_get_missing_id(self, client):
        """Test tasks/get without task ID."""
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "tasks/get",
                "params": {},
                "id": "req-123",
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["error"]["code"] == -32602  # Invalid params

    def test_tasks_cancel_not_found(self, client):
        """Test tasks/cancel with non-existent task."""
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "tasks/cancel",
                "params": {"id": "nonexistent-task"},
                "id": "req-123",
            },
        )

        assert response.status_code == 404

    def test_get_authenticated_extended_card(self, client):
        """Test agent/getAuthenticatedExtendedCard method."""
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "agent/getAuthenticatedExtendedCard",
                "params": {},
                "id": "req-123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["result"]["name"] == "WebShop+ Benchmark"


class TestMessageStream:
    """Test SSE streaming endpoint."""

    @pytest.mark.asyncio
    async def test_message_stream_simple(self, async_client):
        """Test message/stream returns SSE events."""
        response = await async_client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "message/stream",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "Hello!"}],
                    }
                },
                "id": "req-123",
            },
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Parse SSE events
        content = response.content.decode()
        events = [line for line in content.split("\n\n") if line.startswith("data: ")]
        assert len(events) > 0

        # Check first event is a task
        first_event = json.loads(events[0].replace("data: ", ""))
        assert "result" in first_event

    @pytest.mark.asyncio
    async def test_message_stream_assessment(self, async_client):
        """Test message/stream with assessment request."""
        response = await async_client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "message/stream",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "Run assessment"}],
                    },
                    "metadata": {
                        "participants": {"shopper": "http://agent:8001/a2a"},
                        "config": {"num_tasks": 3},
                    },
                },
                "id": "req-123",
            },
        )

        assert response.status_code == 200

        # Parse events
        content = response.content.decode()
        events = []
        for line in content.split("\n\n"):
            if line.startswith("data: "):
                events.append(json.loads(line.replace("data: ", "")))

        # Should have multiple status updates
        status_updates = [
            e for e in events if e.get("result", {}).get("kind") == "status-update"
        ]
        assert len(status_updates) > 0

        # Should end with completed or have artifact
        final_events = [e for e in events if e.get("result", {}).get("final") is True]
        assert len(final_events) > 0

    @pytest.mark.asyncio
    async def test_message_stream_no_participants(self, async_client):
        """Test message/stream fails gracefully without participants."""
        response = await async_client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "message/stream",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "Run assessment"}],
                    },
                    "metadata": {
                        "config": {"num_tasks": 3},
                    },
                },
                "id": "req-123",
            },
        )

        assert response.status_code == 200

        content = response.content.decode()
        events = []
        for line in content.split("\n\n"):
            if line.startswith("data: "):
                events.append(json.loads(line.replace("data: ", "")))

        # Should have a failed status
        final_events = [
            e for e in events
            if e.get("result", {}).get("status", {}).get("state") == "failed"
        ]
        assert len(final_events) > 0


class TestAssessmentTracking:
    """Test assessment state tracking."""

    def test_assessment_creates_state(self, client):
        """Test that assessment requests create state."""
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "Start assessment"}],
                    },
                    "metadata": {
                        "participants": {"shopper": "http://agent:8001/a2a"},
                    },
                },
                "id": "req-123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        task_id = data["result"]["id"]

        # Task should be tracked
        assert task_id in state.active_assessments

    def test_tasks_get_active_assessment(self, client):
        """Test getting status of active assessment."""
        # First create an assessment
        response1 = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "Start assessment"}],
                    },
                    "metadata": {
                        "participants": {"shopper": "http://agent:8001/a2a"},
                    },
                },
                "id": "req-1",
            },
        )
        task_id = response1.json()["result"]["id"]

        # Now get task status
        response2 = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "tasks/get",
                "params": {"id": task_id},
                "id": "req-2",
            },
        )

        assert response2.status_code == 200
        data = response2.json()
        assert data["result"]["id"] == task_id

    def test_tasks_cancel_active_assessment(self, client):
        """Test canceling an active assessment."""
        # Create an assessment
        response1 = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "Start assessment"}],
                    },
                    "metadata": {
                        "participants": {"shopper": "http://agent:8001/a2a"},
                    },
                },
                "id": "req-1",
            },
        )
        task_id = response1.json()["result"]["id"]

        # Cancel it
        response2 = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "tasks/cancel",
                "params": {"id": task_id},
                "id": "req-2",
            },
        )

        assert response2.status_code == 200
        data = response2.json()
        assert data["result"]["status"]["state"] == "canceled"
        assert state.active_assessments[task_id]["canceled"] is True


# =============================================================================
# A2A Client Tests
# =============================================================================


class TestA2AClient:
    """Test the A2A HTTP client."""

    @pytest.mark.asyncio
    async def test_client_context_manager(self):
        """Test client works as context manager."""
        async with A2AClient() as client:
            assert client._client is not None
        assert client._client is None

    @pytest.mark.asyncio
    async def test_client_not_initialized_error(self):
        """Test client raises error when not initialized."""
        client = A2AClient()
        with pytest.raises(RuntimeError, match="Client not initialized"):
            await client.send_message("http://test", create_text_message("test"))

    @pytest.mark.asyncio
    async def test_get_agent_card_from_server(self, async_client):
        """Test fetching agent card via client."""
        async with A2AClient() as a2a_client:
            # We need to mock or use the actual test server
            # For now, test the server directly
            response = await async_client.get("/.well-known/agent-card.json")
            assert response.status_code == 200
            data = response.json()
            card = AgentCard(**data)
            assert card.name == "WebShop+ Benchmark"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full A2A flow."""

    def test_full_a2a_discovery_flow(self, client):
        """Test full A2A discovery flow."""
        # 1. Fetch agent card
        card_response = client.get("/.well-known/agent-card.json")
        assert card_response.status_code == 200
        card = card_response.json()

        # 2. Verify endpoint URL
        assert "/a2a" in card["url"]

        # 3. Send a message to the A2A endpoint
        msg_response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "What can you do?"}],
                    }
                },
                "id": "discovery-test",
            },
        )

        assert msg_response.status_code == 200
        data = msg_response.json()
        assert data["result"]["status"]["state"] == "completed"

    def test_assessment_request_flow(self, client):
        """Test assessment request flow."""
        # 1. Send assessment request
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "Run full assessment"}],
                    },
                    "metadata": {
                        "participants": {
                            "shopper": "http://purple-agent:8001/a2a"
                        },
                        "config": {
                            "task_types": ["all"],
                            "num_tasks": 80,
                            "timeout_per_task": 300,
                        },
                    },
                },
                "id": "assessment-test",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should be submitted (actual processing happens via stream)
        assert data["result"]["status"]["state"] == "submitted"
        assert data["result"]["id"] is not None

        # 2. Check task status
        task_id = data["result"]["id"]
        status_response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "tasks/get",
                "params": {"id": task_id},
                "id": "status-check",
            },
        )

        assert status_response.status_code == 200
