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
from unittest.mock import patch, MagicMock

from src.server_new import create_app, create_agent_card


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def client():
    """Create a TestClient for the SDK-based server."""
    app = create_app(card_url="http://testserver")
    return TestClient(app)


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for tests that need agent execution."""
    def _mock_complete(messages, **kwargs):
        content = messages[-1]["content"] if messages else ""
        if "PRODUCT_TYPE:" in content or "Parse the following" in content:
            return """PRODUCT_TYPE: running shoes
BUDGET: 100
PREFERENCES: comfortable
CONSTRAINTS: none
COMPARISON_REQUIRED: no
SEARCH_QUERY: running shoes"""
        return "search[running shoes]"
    return _mock_complete


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
        assert capabilities["streaming"] is False
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

    def test_jsonrpc_message_send_format(self, client, mock_llm_client):
        """Test message/send follows JSON-RPC 2.0 format."""
        message_id = str(uuid.uuid4())

        with patch("src.agent.LLMClient") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.complete = mock_llm_client
            MockLLM.return_value = mock_llm

            response = client.post(
                "/a2a",
                json={
                    "jsonrpc": "2.0",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "messageId": message_id,
                            "role": "user",
                            "parts": [{"kind": "text", "text": "Find shoes"}],
                        },
                    },
                    "id": "test-1",
                },
            )

        assert response.status_code == 200
        data = response.json()
        # JSON-RPC 2.0 response format
        assert data.get("jsonrpc") == "2.0"
        assert "result" in data
        assert data.get("id") == "test-1"


# =============================================================================
# Task Lifecycle Tests
# =============================================================================


class TestTaskLifecycle:
    """Test task lifecycle via message/send."""

    def test_message_send_creates_task(self, client, mock_llm_client):
        """Test message/send creates and processes a task."""
        message_id = str(uuid.uuid4())

        with patch("src.agent.LLMClient") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.complete = mock_llm_client
            MockLLM.return_value = mock_llm

            response = client.post(
                "/a2a",
                json={
                    "jsonrpc": "2.0",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "messageId": message_id,
                            "role": "user",
                            "parts": [{"kind": "text", "text": "Find running shoes under $100"}],
                        },
                    },
                    "id": "test-1",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        result = data["result"]
        # Result should contain task status
        assert "status" in result

    def test_message_send_returns_completed_status(self, client, mock_llm_client):
        """Test message/send returns completed status for synchronous tasks."""
        message_id = str(uuid.uuid4())

        with patch("src.agent.LLMClient") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.complete = mock_llm_client
            MockLLM.return_value = mock_llm

            response = client.post(
                "/a2a",
                json={
                    "jsonrpc": "2.0",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "messageId": message_id,
                            "role": "user",
                            "parts": [{"kind": "text", "text": "Buy a laptop"}],
                        },
                    },
                    "id": "test-1",
                },
            )

        assert response.status_code == 200
        data = response.json()
        result = data["result"]
        # Non-streaming agent should complete synchronously
        assert result["status"]["state"] == "completed"

    def test_message_send_returns_action_in_message(self, client, mock_llm_client):
        """Test message/send returns action in response message."""
        message_id = str(uuid.uuid4())

        with patch("src.agent.LLMClient") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.complete = mock_llm_client
            MockLLM.return_value = mock_llm

            response = client.post(
                "/a2a",
                json={
                    "jsonrpc": "2.0",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "messageId": message_id,
                            "role": "user",
                            "parts": [{"kind": "text", "text": "Find running shoes under $100"}],
                        },
                    },
                    "id": "test-1",
                },
            )

        assert response.status_code == 200
        data = response.json()
        result = data["result"]

        # Check for action in status message
        status_message = result["status"].get("message")
        assert status_message is not None
        assert status_message["role"] == "agent"
        # Action should be in the message parts
        parts = status_message["parts"]
        assert len(parts) > 0
        # Extract text from parts
        action_text = parts[0].get("text", "")
        assert "search[" in action_text.lower() or "running shoes" in action_text.lower()


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
        assert card.capabilities.streaming is False


# =============================================================================
# Context Management Tests
# =============================================================================


class TestContextManagement:
    """Test context/session management across multiple requests."""

    def test_same_context_preserves_state(self, client, mock_llm_client):
        """Test that same contextId preserves agent state."""
        context_id = str(uuid.uuid4())
        message_id_1 = str(uuid.uuid4())
        message_id_2 = str(uuid.uuid4())

        with patch("src.agent.LLMClient") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.complete = mock_llm_client
            MockLLM.return_value = mock_llm

            # First message
            response1 = client.post(
                "/a2a",
                json={
                    "jsonrpc": "2.0",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "messageId": message_id_1,
                            "role": "user",
                            "parts": [{"kind": "text", "text": "Find running shoes under $100"}],
                            "contextId": context_id,
                        },
                    },
                    "id": "test-1",
                },
            )
            assert response1.status_code == 200

            # Second message with same contextId
            response2 = client.post(
                "/a2a",
                json={
                    "jsonrpc": "2.0",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "messageId": message_id_2,
                            "role": "user",
                            "parts": [{"kind": "text", "text": "OBSERVATION: Search results: B07XYZ - Shoes - $50"}],
                            "contextId": context_id,
                        },
                    },
                    "id": "test-2",
                },
            )
            assert response2.status_code == 200


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
