"""FastMCP server for WebShop+ tool execution.

This module provides the WebShopMCPServer class that exposes shopping tools
(search, click, checkout) via the MCP protocol for use by shopping agents.

Tools return structured responses with element IDs that agents use for
subsequent interactions. The server is session-scoped - each assessment
session gets its own server instance with isolated state.
"""

from typing import Any

from mcp.server.fastmcp import FastMCP

from .session_state import SessionState


class WebShopMCPServer:
    """Session-scoped MCP server for WebShop tools.

    This server provides three tools for shopping:
    - search(query): Search the product catalog
    - click(element_id): Click on an element from previous observation
    - checkout(): Complete purchase (terminal action)

    Each instance is bound to a SessionState that tracks cart, budget,
    turn count, and visible elements.

    Example:
        state = SessionState(
            session_id="abc123",
            goal="Find running shoes under $50",
            budget=50.0,
            constraints=["no synthetic"],
        )
        server = WebShopMCPServer(state)
        app = server.get_app()  # Mount in Starlette/FastAPI
    """

    def __init__(self, state: SessionState):
        """Initialize MCP server with session state.

        Args:
            state: SessionState instance for tracking session data.
        """
        self.state = state
        self.mcp = FastMCP(f"WebShop-{state.session_id}")
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all shopping tools with the MCP server."""
        # Store reference to self for use in closures
        server = self

        @self.mcp.tool()
        def search(query: str) -> dict[str, Any]:
            """Search the store catalog for products.

            Args:
                query: Search terms (e.g., "running shoes", "blue cotton shirt")

            Returns:
                Structured results with clickable element IDs, or error dict
                if not implemented yet.
            """
            # Placeholder - will be implemented in Stage 3
            return {
                "error": "Not implemented",
                "message": "search() will be implemented in Stage 3",
                "query": query,
                "session_id": server.state.session_id,
            }

        @self.mcp.tool()
        def click(element_id: str) -> dict[str, Any]:
            """Click on an element by its ID from previous observation.

            Args:
                element_id: ID from previous observation (e.g., "p1", "size_10")

            Returns:
                New page state with available element IDs, or error dict
                if element not found or not implemented.
            """
            # Placeholder - will be implemented in Stage 4
            return {
                "error": "Not implemented",
                "message": "click() will be implemented in Stage 4",
                "element_id": element_id,
                "session_id": server.state.session_id,
            }

        @self.mcp.tool()
        def checkout() -> dict[str, Any]:
            """Complete purchase and end session. TERMINAL action.

            This is a terminal action - calling it ends the session and
            returns the final evaluation with success/failure and score.

            Returns:
                Evaluation dict with terminated=True, cart contents,
                total, budget comparison, and score.
            """
            # Placeholder - will be implemented in Stage 5
            return {
                "error": "Not implemented",
                "message": "checkout() will be implemented in Stage 5",
                "session_id": server.state.session_id,
            }

    def get_app(self) -> Any:
        """Return Starlette app for mounting.

        The returned app handles MCP protocol via streamable HTTP transport.
        Mount this at a path like "/mcp/{session_id}" in your main server.

        Returns:
            ASGI application that handles MCP requests.
        """
        return self.mcp.streamable_http_app()
