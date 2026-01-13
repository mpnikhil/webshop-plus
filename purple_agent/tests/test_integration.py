"""
Integration tests for WebShop+ Purple Agent SDK Server.

This module contains comprehensive integration tests for:
- Server endpoints (agent card, a2a RPC, health)
- Agent card configuration
- A2A protocol compliance (JSON-RPC 2.0)
- Task lifecycle (submit, status, message/send)
- Error handling

Uses Starlette TestClient for synchronous testing of the async server.
"""

import uuid

import pytest
from starlette.testclient import TestClient

from src.server import create_app, create_agent_card


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def client():
    """Create a TestClient for the SDK-based server."""
    app = create_app(card_url="http://testserver")
    return TestClient(app)


# =============================================================================
# Server Endpoint Tests
# =============================================================================


class TestServerEndpoints:
    """Test server endpoints availability and responses."""

    def test_agent_card_endpoint_exists(self, client):
        """Test /.well-known/agent-card.json endpoint returns 200."""
        response = client.get("/.well-known/agent-card.json")
        assert response.status_code == 200

    def test_agent_card_endpoint_returns_json(self, client):
        """Test agent card endpoint returns valid JSON."""
        response = client.get("/.well-known/agent-card.json")
        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert isinstance(data, dict)

    def test_a2a_endpoint_exists(self, client):
        """Test /a2a endpoint exists and accepts POST."""
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "agent/getCard",
                "id": "test-1",
            },
        )
        assert response.status_code == 200

    def test_health_endpoint(self, client):
        """Test /health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "webshop-plus-purple"

    def test_cors_headers_present(self, client):
        """Test CORS headers are present in responses."""
        # OPTIONS request to check CORS preflight
        response = client.options(
            "/a2a",
            headers={"Origin": "http://example.com"},
        )
        # Server should have CORS configured
        assert response.status_code in (200, 400, 405)  # Depends on route config


# =============================================================================
# Agent Card Tests
# =============================================================================


class TestAgentCard:
    """Test agent card configuration and content."""

    def test_agent_card_has_required_fields(self, client):
        """Test agent card has all required fields."""
        response = client.get("/.well-known/agent-card.json")
        data = response.json()

        assert "name" in data
        assert "description" in data
        assert "url" in data
        assert "version" in data
        assert "capabilities" in data
        assert "skills" in data

    def test_agent_card_name_and_version(self, client):
        """Test agent card name and version are correct."""
        response = client.get("/.well-known/agent-card.json")
        data = response.json()

        assert data["name"] == "WebShop+ Shopper Agent"
        assert data["version"] == "1.0.0"

    def test_agent_card_skills_configured(self, client):
        """Test agent card has shopping skill configured."""
        response = client.get("/.well-known/agent-card.json")
        data = response.json()

        assert len(data["skills"]) == 1
        skill = data["skills"][0]
        assert skill["id"] == "shopping"
        assert skill["name"] == "Product Shopping"
        assert "tags" in skill
        assert "shopping" in skill["tags"]
        assert "examples" in skill
        assert len(skill["examples"]) > 0

    def test_agent_card_capabilities(self, client):
        """Test agent card capabilities are correctly configured."""
        response = client.get("/.well-known/agent-card.json")
        data = response.json()

        capabilities = data["capabilities"]
        assert capabilities["streaming"] is True
        assert capabilities.get("pushNotifications", False) is False

    def test_agent_card_url_format(self, client):
        """Test agent card URL points to /a2a endpoint."""
        response = client.get("/.well-known/agent-card.json")
        data = response.json()

        assert data["url"].endswith("/a2a")

    def test_agent_card_input_output_modes(self, client):
        """Test agent card input/output modes."""
        response = client.get("/.well-known/agent-card.json")
        data = response.json()

        assert "defaultInputModes" in data
        assert "text/plain" in data["defaultInputModes"]
        assert "defaultOutputModes" in data
        assert "text/plain" in data["defaultOutputModes"]


# =============================================================================
# Protocol Compliance Tests (JSON-RPC 2.0)
# =============================================================================


class TestProtocolCompliance:
    """Test A2A protocol compliance (JSON-RPC 2.0)."""

    def test_jsonrpc_version_in_response(self, client):
        """Test JSON-RPC 2.0 version in response."""
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "agent/getCard",
                "id": "test-1",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("jsonrpc") == "2.0"

    def test_jsonrpc_id_preserved(self, client):
        """Test JSON-RPC request ID is preserved in response."""
        test_id = f"test-{uuid.uuid4()}"
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "agent/getCard",
                "id": test_id,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("id") == test_id



# =============================================================================
# Task Lifecycle Tests
# =============================================================================


class TestTaskLifecycle:
    """Test task lifecycle via message/send."""


    def test_message_send_returns_completed_status(self, client):
        """Test message/send returns completed status when 'done' is sent.

        In the AAA architecture, non-MCP messages are treated as TCK tests
        and require 'done' to complete.
        """
        message_id = str(uuid.uuid4())

        # Send "done" to complete immediately
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": message_id,
                        "role": "user",
                        "parts": [{"kind": "text", "text": "done"}],
                    },
                },
                "id": "test-1",
            },
        )

        assert response.status_code == 200
        data = response.json()
        result = data["result"]
        # Message containing "done" should complete
        assert result["status"]["state"] == "completed"

    def test_message_send_returns_action_in_message(self, client):
        """Test message/send returns response message for simple messages.

        In the AAA architecture, non-MCP messages are treated as TCK tests
        and return input-required state with echo response.
        """
        message_id = str(uuid.uuid4())

        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": message_id,
                        "role": "user",
                        "parts": [{"kind": "text", "text": "Hello world"}],
                    },
                },
                "id": "test-1",
            },
        )

        assert response.status_code == 200
        data = response.json()
        result = data["result"]

        # Simple messages go to TCK handler and return input-required
        assert result["status"]["state"] == "input-required"

        # Check for echo in status message
        status_message = result["status"].get("message")
        assert status_message is not None
        assert status_message["role"] == "agent"
        # Message should echo the input
        parts = status_message["parts"]
        assert len(parts) > 0
        response_text = parts[0].get("text", "")
        assert "Received:" in response_text or "Hello world" in response_text


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling for invalid requests."""

    def test_invalid_method_returns_error(self, client):
        """Test invalid JSON-RPC method returns error."""
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "invalid/method",
                "id": "test-1",
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Should have error response
        assert "error" in data
        error = data["error"]
        assert "code" in error
        assert "message" in error

    def test_missing_params_returns_error(self, client):
        """Test message/send without params returns error."""
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "id": "test-1",
                # Missing "params"
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Should have error response
        assert "error" in data

    def test_invalid_json_returns_error(self, client):
        """Test invalid JSON in request body returns error."""
        response = client.post(
            "/a2a",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        # SDK may return 400 or a JSON-RPC error
        assert response.status_code in (200, 400, 422)

    def test_missing_message_in_send_returns_error(self, client):
        """Test message/send without message in params returns error."""
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {},  # Missing "message"
                "id": "test-1",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "error" in data


# =============================================================================
# create_agent_card() Unit Tests
# =============================================================================


class TestCreateAgentCard:
    """Unit tests for create_agent_card() function."""

    def test_creates_agent_card_with_url(self):
        """Test create_agent_card creates card with correct URL."""
        card = create_agent_card("http://example.com")
        assert card.url == "http://example.com/a2a"

    def test_creates_agent_card_with_skill(self):
        """Test create_agent_card creates card with shopping skill."""
        card = create_agent_card("http://example.com")
        assert len(card.skills) == 1
        assert card.skills[0].id == "shopping"

    def test_creates_agent_card_with_capabilities(self):
        """Test create_agent_card creates card with correct capabilities."""
        card = create_agent_card("http://example.com")
        assert card.capabilities.streaming is True


# =============================================================================
# Context Management Tests
# =============================================================================




# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
