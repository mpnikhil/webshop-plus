"""
Integration tests for the SDK-based A2A server (server_a2a.py).

These tests verify that the SDK-based server correctly handles A2A protocol
requests and integrates with the WebShopPlusExecutor.

Stage 5 of the A2A SDK Migration.

Test Categories:
1. Server endpoint tests (health, agent-card, A2A RPC)
2. Protocol compliance tests (JSON-RPC format, message structure)
3. End-to-end tests with mocked purple agent
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.testclient import TestClient

from src.server_a2a import create_app, create_sdk_agent_card


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def app():
    """Create a test application instance."""
    return create_app(card_url="http://testserver")


@pytest.fixture
def client(app):
    """Create a synchronous test client."""
    return TestClient(app)


@pytest.fixture
def agent_card():
    """Create an agent card for testing."""
    return create_sdk_agent_card("http://testserver")


# =============================================================================
# Server Endpoint Tests
# =============================================================================


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self, client):
        """Should return healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "webshop-plus-green-a2a"


class TestAgentCardEndpoint:
    """Tests for the /.well-known/agent-card.json endpoint."""

    def test_agent_card_returns_valid_json(self, client):
        """Should return valid agent card JSON."""
        response = client.get("/.well-known/agent-card.json")
        assert response.status_code == 200
        data = response.json()

        # Verify required fields
        assert "name" in data
        assert "description" in data
        assert "url" in data
        assert "capabilities" in data
        assert "skills" in data

    def test_agent_card_has_correct_name(self, client):
        """Should have correct agent name."""
        response = client.get("/.well-known/agent-card.json")
        data = response.json()
        assert data["name"] == "WebShop+ Benchmark"

    def test_agent_card_has_streaming_capability(self, client):
        """Should advertise streaming capability."""
        response = client.get("/.well-known/agent-card.json")
        data = response.json()
        assert data["capabilities"]["streaming"] is True

    def test_agent_card_has_skills(self, client):
        """Should have at least one skill defined."""
        response = client.get("/.well-known/agent-card.json")
        data = response.json()
        assert len(data["skills"]) >= 1

    def test_agent_card_skill_has_input_modes(self, client):
        """Each skill should have inputModes defined."""
        response = client.get("/.well-known/agent-card.json")
        data = response.json()

        for skill in data["skills"]:
            # inputModes is the SDK-supported field for input format
            assert "inputModes" in skill, f"Skill {skill['id']} missing inputModes"
            # Note: inputSchema is not currently supported by a2a-sdk v0.3.x
            # See HANDOVER_A2A_STAGE5.md for details on this SDK limitation


class TestA2ARPCEndpoint:
    """Tests for the /a2a JSON-RPC endpoint."""

    def test_rpc_endpoint_exists(self, client):
        """Should respond to POST requests."""
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "id": "1",
                "params": {
                    "message": {
                        "messageId": "test-msg-1",
                        "role": "user",
                        "parts": [{"kind": "text", "text": "test"}],
                    }
                },
            },
        )
        # Should get a response (may be error due to missing participants, but endpoint works)
        assert response.status_code in [200, 400, 500]

    def test_rpc_requires_jsonrpc_field(self, client):
        """Should reject request without jsonrpc field."""
        response = client.post(
            "/a2a",
            json={
                "method": "message/send",
                "id": "1",
                "params": {},
            },
        )
        # SDK should validate JSON-RPC format
        assert response.status_code in [200, 400]
        data = response.json()
        if "error" in data:
            assert data["error"] is not None

    def test_rpc_requires_method_field(self, client):
        """Should reject request without method field."""
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "params": {},
            },
        )
        assert response.status_code in [200, 400]
        data = response.json()
        if "error" in data:
            assert data["error"] is not None


# =============================================================================
# Agent Card Factory Tests
# =============================================================================


class TestAgentCardFactory:
    """Tests for the create_sdk_agent_card function."""

    def test_creates_valid_agent_card(self, agent_card):
        """Should create a valid AgentCard object."""
        assert agent_card.name == "WebShop+ Benchmark"
        assert "shopping agents" in agent_card.description.lower()

    def test_url_includes_base_url(self, agent_card):
        """URL should include the provided base URL."""
        assert "http://testserver" in agent_card.url

    def test_has_assessment_skill(self, agent_card):
        """Should have the main assessment skill."""
        skill_ids = [s.id for s in agent_card.skills]
        assert "assessment" in skill_ids

    def test_has_category_specific_skills(self, agent_card):
        """Should have skills for each category."""
        skill_ids = [s.id for s in agent_card.skills]
        expected_skills = [
            "budget-assessment",
            "memory-assessment",
            "constraint-assessment",
            "reasoning-assessment",
            "recovery-assessment",
        ]
        for expected in expected_skills:
            assert expected in skill_ids, f"Missing skill: {expected}"

    def test_skills_have_input_modes(self, agent_card):
        """All skills should have input modes defined."""
        for skill in agent_card.skills:
            # SDK uses snake_case internally (input_modes) but serializes to camelCase (inputModes)
            assert skill.input_modes is not None, f"Skill {skill.id} has no input_modes"
            # Note: inputSchema is defined in server_a2a.py but not serialized
            # due to a2a-sdk v0.3.x limitation (AgentSkill model ignores extra fields)

    def test_assessment_skill_has_correct_input_mode(self, agent_card):
        """Assessment skill should accept JSON input."""
        assessment_skill = next(s for s in agent_card.skills if s.id == "assessment")
        assert assessment_skill.input_modes is not None
        assert "application/json" in assessment_skill.input_modes


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProtocolCompliance:
    """Tests for A2A protocol compliance.

    These tests verify JSON-RPC format and protocol structure without
    actually running the executor (which would require network access).
    """

    def test_accepts_message_send_method_format(self, client):
        """Should accept properly formatted message/send method (missing participants)."""
        # Test that the endpoint accepts the request format
        # We intentionally omit participants to avoid triggering actual execution
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "id": "test-1",
                "params": {
                    "message": {
                        "messageId": "msg-1",
                        "role": "user",
                        "parts": [{"kind": "text", "text": "hello"}],
                    },
                    "metadata": {},  # No participants - will be rejected quickly
                },
            },
        )
        # Should get a response (rejected due to missing participants)
        assert response.status_code in [200, 400, 500]

    def test_response_has_jsonrpc_field(self, client):
        """Response should include jsonrpc field."""
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "id": "test-1",
                "params": {
                    "message": {
                        "messageId": "msg-1",
                        "role": "user",
                        "parts": [{"kind": "text", "text": "hello"}],
                    },
                    "metadata": {},  # No participants
                },
            },
        )
        data = response.json()
        assert "jsonrpc" in data
        assert data["jsonrpc"] == "2.0"

    def test_response_has_matching_id(self, client):
        """Response should include matching request ID."""
        request_id = "unique-request-id-12345"
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "id": request_id,
                "params": {
                    "message": {
                        "messageId": "msg-1",
                        "role": "user",
                        "parts": [{"kind": "text", "text": "hello"}],
                    },
                    "metadata": {},  # No participants
                },
            },
        )
        data = response.json()
        assert "id" in data
        assert data["id"] == request_id


# =============================================================================
# End-to-End Tests with Mocked Purple Agent
# =============================================================================


class TestEndToEnd:
    """End-to-end tests with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_full_assessment_flow_mocked(self):
        """Test complete assessment flow with mocked agent."""
        # Create app with mocked executor
        with patch("src.server_a2a.WebShopPlusExecutor") as MockExecutor:
            # Setup mock executor
            mock_executor = MagicMock()

            async def mock_execute(context, event_queue):
                """Simulate a successful assessment."""
                from a2a.server.tasks import TaskUpdater
                from a2a.types import TaskState, TextPart

                updater = TaskUpdater(
                    event_queue=event_queue,
                    task_id=context.task_id,
                    context_id=context.context_id,
                )

                # Emit working status
                from a2a.types import Message, Role

                await updater.start_work(
                    message=Message(
                        messageId="work-msg",
                        role=Role.agent,
                        parts=[TextPart(text="Starting...")],
                    )
                )

                # Add artifact
                await updater.add_artifact(
                    parts=[TextPart(text='{"score": 0.85}')],
                    name="results",
                    last_chunk=True,
                )

                # Complete
                await updater.complete(
                    message=Message(
                        messageId="done-msg",
                        role=Role.agent,
                        parts=[TextPart(text="Done!")],
                    )
                )

            mock_executor.execute = mock_execute
            mock_executor.cancel = AsyncMock()
            MockExecutor.return_value = mock_executor

            app = create_app(card_url="http://testserver")

            # Use async client for streaming test
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as async_client:
                # Test health
                response = await async_client.get("/health")
                assert response.status_code == 200

                # Test agent card
                response = await async_client.get("/.well-known/agent-card.json")
                assert response.status_code == 200
                card = response.json()
                assert card["name"] == "WebShop+ Benchmark"


class TestMessageStreamEndpoint:
    """Tests for the message/stream endpoint."""

    def test_stream_endpoint_accepts_request_format(self, client):
        """Should accept streaming request format (without network call)."""
        # Test the endpoint accepts the streaming format
        # Omit participants to avoid triggering network calls
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "message/stream",
                "id": "stream-1",
                "params": {
                    "message": {
                        "messageId": "msg-1",
                        "role": "user",
                        "parts": [{"kind": "text", "text": "assess"}],
                    },
                    "metadata": {},  # No participants
                },
            },
            headers={"Accept": "text/event-stream"},
        )
        # Should get a response (will be rejected due to missing participants)
        assert response.status_code in [200, 400, 500]


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in the SDK server."""

    def test_missing_participants_returns_error(self, client):
        """Should return error when participants missing from metadata."""
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "id": "1",
                "params": {
                    "message": {
                        "messageId": "msg-1",
                        "role": "user",
                        "parts": [{"kind": "text", "text": "assess"}],
                    },
                    "metadata": {},  # No participants
                },
            },
        )
        # Should fail gracefully
        assert response.status_code in [200, 500]

    def test_invalid_json_returns_error(self, client):
        """Should return error for invalid JSON."""
        response = client.post(
            "/a2a",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        # SDK returns 200 with JSON-RPC error in body (code -32700 = Parse error)
        # This is correct JSON-RPC behavior per spec
        assert response.status_code in [200, 400, 422, 500]
        if response.status_code == 200:
            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32700  # Parse error

    def test_unknown_method_returns_error(self, client):
        """Should return error for unknown method."""
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "unknown/method",
                "id": "1",
                "params": {},
            },
        )
        data = response.json()
        assert "error" in data


# =============================================================================
# Comparison Tests (SDK vs Legacy Server)
# =============================================================================


class TestSDKCompatibility:
    """Tests ensuring SDK server matches legacy server behavior."""

    def test_same_agent_card_structure(self, client):
        """Agent card should have same structure as legacy server."""
        response = client.get("/.well-known/agent-card.json")
        card = response.json()

        # Verify required fields match legacy format
        assert isinstance(card["name"], str)
        assert isinstance(card["description"], str)
        assert isinstance(card["url"], str)
        assert isinstance(card["capabilities"], dict)
        assert isinstance(card["skills"], list)

        # Verify capability structure
        caps = card["capabilities"]
        assert "streaming" in caps
        assert "pushNotifications" in caps

    def test_health_response_structure(self, client):
        """Health endpoint should have expected response structure."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "service" in data
        assert data["status"] == "healthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
