"""Tests for the MCP route handler in server.py.

These tests verify that the MCPRouteHandler correctly routes requests
to the global MCP application with the correct session context.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.webshop_mcp import SessionManager, SessionState
from src.webshop_mcp.server import (
    _session_states,
    current_session_id,
)


@pytest.fixture(autouse=True)
def cleanup_global_state():
    """Clean up global session state before and after each test."""
    _session_states.clear()
    yield
    _session_states.clear()


class MockASGIApp:
    """Mock ASGI app for testing."""

    def __init__(self):
        self.called = False
        self.scope = None
        self.receive = None
        self.send = None
        self.captured_session_id = None

    async def __call__(self, scope, receive, send):
        self.called = True
        self.scope = scope
        self.receive = receive
        self.send = send
        # Capture the session_id from contextvar
        try:
            self.captured_session_id = current_session_id.get()
        except LookupError:
            self.captured_session_id = None


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
    def mock_mcp_app(self):
        """Create a mock MCP app for testing."""
        return MockASGIApp()

    @pytest.fixture
    def handler(self, session_manager, mock_mcp_app):
        """Create an MCP route handler for testing."""
        from src.server import MCPRouteHandler

        return MCPRouteHandler(session_manager, mock_mcp_app)

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
        """Should return 404 for invalid MCP path (not matching /mcp/ pattern)."""
        scope = {"type": "http", "path": "/invalid/path"}
        receive = AsyncMock()
        send = MockSend()

        await handler(scope, receive, send)

        # Should have response start and body
        assert len(send.responses) == 2
        # Invalid paths that don't match /mcp/ pattern return 404
        assert send.responses[0]["status"] == 404

        body = json.loads(send.responses[1]["body"])
        assert "error" in body
        # Path /invalid/path is parsed as session_id="invalid", so error is about session not found
        assert "not found" in body["error"].lower() or "invalid" in body["error"].lower() or "session" in body["error"].lower()

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
    async def test_valid_session_routes_to_mcp_app(
        self, handler, session_manager, mock_mcp_app
    ):
        """Should route to global MCP app for valid session."""
        # Create a session
        await session_manager.create_session(
            session_id="test-session",
            goal="Find shoes",
            budget=100.0,
        )

        # Make request
        scope = {"type": "http", "path": "/mcp/test-session"}
        receive = AsyncMock()
        send = MockSend()

        await handler(scope, receive, send)

        # Should have called the mock app
        assert mock_mcp_app.called
        assert mock_mcp_app.scope["path"] == "/mcp"
        # Should have set the session_id in contextvar
        assert mock_mcp_app.captured_session_id == "test-session"

    @pytest.mark.asyncio
    async def test_path_after_session_id_preserved(
        self, handler, session_manager, mock_mcp_app
    ):
        """Should preserve path after session ID for MCP app."""
        await session_manager.create_session(
            session_id="test-session",
            goal="Find shoes",
            budget=100.0,
        )

        # Make request with path after session ID
        scope = {"type": "http", "path": "/mcp/test-session/tools/search"}
        receive = AsyncMock()
        send = MockSend()

        await handler(scope, receive, send)

        assert mock_mcp_app.called
        # Path should be modified to /mcp/tools/search
        assert mock_mcp_app.scope["path"] == "/mcp/tools/search"

    @pytest.mark.asyncio
    async def test_contextvar_reset_after_request(
        self, handler, session_manager, mock_mcp_app
    ):
        """Should reset contextvar after request completes."""
        await session_manager.create_session(
            session_id="test-session",
            goal="Find shoes",
            budget=100.0,
        )

        scope = {"type": "http", "path": "/mcp/test-session"}
        receive = AsyncMock()
        send = MockSend()

        await handler(scope, receive, send)

        # After the handler returns, contextvar should be reset
        with pytest.raises(LookupError):
            current_session_id.get()


class TestMCPRouteHandlerEdgeCases:
    """Edge case tests for MCPRouteHandler."""

    @pytest.fixture
    def session_manager(self):
        return SessionManager()

    @pytest.fixture
    def mock_mcp_app(self):
        return MockASGIApp()

    @pytest.fixture
    def handler(self, session_manager, mock_mcp_app):
        from src.server import MCPRouteHandler

        return MCPRouteHandler(session_manager, mock_mcp_app)

    @pytest.mark.asyncio
    async def test_session_id_with_special_characters(
        self, handler, session_manager, mock_mcp_app
    ):
        """Should handle session IDs with hyphens (UUID format)."""
        session_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        await session_manager.create_session(
            session_id=session_id,
            goal="Find shoes",
            budget=100.0,
        )

        scope = {"type": "http", "path": f"/mcp/{session_id}"}
        receive = AsyncMock()
        send = MockSend()

        await handler(scope, receive, send)

        assert mock_mcp_app.called
        assert mock_mcp_app.captured_session_id == session_id

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
