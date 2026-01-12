"""Tests for the MCP route handler in server.py.

These tests verify that the MCPRouteHandler correctly routes requests
to the appropriate session's MCP application.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.webshop_mcp import SessionManager, SessionState, WebShopMCPServer


class MockASGIApp:
    """Mock ASGI app for testing."""

    def __init__(self):
        self.called = False
        self.scope = None
        self.receive = None
        self.send = None

    async def __call__(self, scope, receive, send):
        self.called = True
        self.scope = scope
        self.receive = receive
        self.send = send


class MockSend:
    """Mock send callable for capturing ASGI responses."""

    def __init__(self):
        self.responses = []

    async def __call__(self, message):
        self.responses.append(message)


class TestMCPRouteHandler:
    """Tests for MCPRouteHandler class."""

    @pytest.fixture
    def session_manager(self):
        """Create a session manager for testing."""
        return SessionManager()

    @pytest.fixture
    def handler(self, session_manager):
        """Create an MCP route handler for testing."""
        # Import here to avoid circular issues
        from src.server import MCPRouteHandler

        return MCPRouteHandler(session_manager)

    @pytest.mark.asyncio
    async def test_non_http_scope_ignored(self, handler):
        """Should ignore non-HTTP requests."""
        scope = {"type": "websocket", "path": "/mcp/session123"}
        receive = AsyncMock()
        send = MockSend()

        await handler(scope, receive, send)

        # No response should be sent
        assert len(send.responses) == 0

    @pytest.mark.asyncio
    async def test_invalid_path_returns_error(self, handler):
        """Should return 400 for invalid MCP path."""
        scope = {"type": "http", "path": "/invalid/path"}
        receive = AsyncMock()
        send = MockSend()

        await handler(scope, receive, send)

        # Should have response start and body
        assert len(send.responses) == 2
        assert send.responses[0]["status"] == 400

        body = json.loads(send.responses[1]["body"])
        assert "error" in body
        assert "Invalid MCP path" in body["error"]

    @pytest.mark.asyncio
    async def test_missing_session_id_returns_error(self, handler):
        """Should return 400 for missing session ID."""
        scope = {"type": "http", "path": "/mcp/"}
        receive = AsyncMock()
        send = MockSend()

        await handler(scope, receive, send)

        assert len(send.responses) == 2
        assert send.responses[0]["status"] == 400

        body = json.loads(send.responses[1]["body"])
        assert "Missing session_id" in body["error"]

    @pytest.mark.asyncio
    async def test_nonexistent_session_returns_404(self, handler):
        """Should return 404 for nonexistent session."""
        scope = {"type": "http", "path": "/mcp/nonexistent-session"}
        receive = AsyncMock()
        send = MockSend()

        await handler(scope, receive, send)

        assert len(send.responses) == 2
        assert send.responses[0]["status"] == 404

        body = json.loads(send.responses[1]["body"])
        assert "not found" in body["error"]

    @pytest.mark.asyncio
    async def test_valid_session_routes_to_mcp_app(self, handler, session_manager):
        """Should route to session's MCP app for valid session."""
        # Create a session
        await session_manager.create_session(
            session_id="test-session",
            goal="Find shoes",
            budget=100.0,
        )

        # Get the session and replace its MCP app with a mock
        session = await session_manager.get_session("test-session")
        mock_app = MockASGIApp()
        session.get_app = MagicMock(return_value=mock_app)

        # Make request
        scope = {"type": "http", "path": "/mcp/test-session"}
        receive = AsyncMock()
        send = MockSend()

        await handler(scope, receive, send)

        # Should have called the mock app
        assert mock_app.called
        assert mock_app.scope["path"] == "/"

    @pytest.mark.asyncio
    async def test_path_after_session_id_preserved(self, handler, session_manager):
        """Should preserve path after session ID for MCP app."""
        await session_manager.create_session(
            session_id="test-session",
            goal="Find shoes",
            budget=100.0,
        )

        session = await session_manager.get_session("test-session")
        mock_app = MockASGIApp()
        session.get_app = MagicMock(return_value=mock_app)

        # Make request with path after session ID
        scope = {"type": "http", "path": "/mcp/test-session/tools/search"}
        receive = AsyncMock()
        send = MockSend()

        await handler(scope, receive, send)

        assert mock_app.called
        assert mock_app.scope["path"] == "/tools/search"


class TestMCPRouteHandlerEdgeCases:
    """Edge case tests for MCPRouteHandler."""

    @pytest.fixture
    def session_manager(self):
        return SessionManager()

    @pytest.fixture
    def handler(self, session_manager):
        from src.server import MCPRouteHandler

        return MCPRouteHandler(session_manager)

    @pytest.mark.asyncio
    async def test_session_id_with_special_characters(self, handler, session_manager):
        """Should handle session IDs with hyphens (UUID format)."""
        session_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        await session_manager.create_session(
            session_id=session_id,
            goal="Find shoes",
            budget=100.0,
        )

        session = await session_manager.get_session(session_id)
        mock_app = MockASGIApp()
        session.get_app = MagicMock(return_value=mock_app)

        scope = {"type": "http", "path": f"/mcp/{session_id}"}
        receive = AsyncMock()
        send = MockSend()

        await handler(scope, receive, send)

        assert mock_app.called

    @pytest.mark.asyncio
    async def test_empty_path_after_mcp_prefix(self, handler):
        """Should return 400 for /mcp with no session ID."""
        # Note: This depends on how the path is split
        scope = {"type": "http", "path": "/mcp"}
        receive = AsyncMock()
        send = MockSend()

        await handler(scope, receive, send)

        # Should return an error (either 400 or similar)
        assert len(send.responses) == 2
        # The first response should be an error status
        assert send.responses[0]["status"] in [400, 404]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
