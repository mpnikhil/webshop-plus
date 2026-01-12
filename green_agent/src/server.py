"""
A2A server for WebShop+ green agent using the official a2a-sdk.

This module implements the A2A protocol server with:
- Agent card endpoint at /.well-known/agent-card.json
- A2A message handling endpoint at /a2a
- MCP tool endpoint at /mcp/{session_id} for session-scoped tool execution
- SSE streaming for real-time task updates
- SDK-managed request handling and task lifecycle

Usage:
    uv run python src/server.py --host 0.0.0.0 --port 8000 --card-url http://localhost:8000
"""

import argparse
from contextlib import asynccontextmanager

import structlog
import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentProvider,
    AgentSkill,
)

from src.a2a_executor import WebShopPlusExecutor
from src.agent import AgentConfig
from src.webshop_mcp import SessionManager
from src.webshop_mcp.server import (
    current_session_id,
    is_session_registered,
    mcp,
    get_mcp_app,
)

logger = structlog.get_logger()


# =============================================================================
# MCP Route Handler
# =============================================================================


class MCPRouteHandler:
    """ASGI handler that routes MCP requests to the global MCP server.

    This handler extracts the session_id from the URL path, sets the
    contextvar for session isolation, and delegates to the global MCP app.

    Example:
        handler = MCPRouteHandler(session_manager)
        # Requests to /mcp/{session_id}/* are routed to the global MCP app
        # with session_id set in contextvar
    """

    def __init__(self, session_manager: SessionManager, mcp_app):
        """Initialize the MCP route handler.

        Args:
            session_manager: SessionManager instance for looking up sessions.
            mcp_app: The global MCP Starlette app.
        """
        self.session_manager = session_manager
        self.mcp_app = mcp_app

    async def __call__(self, scope, receive, send) -> None:
        """ASGI entrypoint for MCP requests.

        Args:
            scope: ASGI scope dict.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        if scope["type"] != "http":
            return

        # Extract session_id from path
        # NOTE: Starlette Mount strips the mount prefix, so:
        # - If mounted at "/mcp", request to "/mcp/session123" arrives as "/session123"
        # - If NOT mounted (direct route), request arrives as "/mcp/session123"
        path = scope.get("path", "")

        # Handle both mounted (path starts with /) and non-mounted cases
        if path.startswith("/mcp/"):
            # Non-mounted case: /mcp/{session_id}/...
            path_after_prefix = path[5:]  # Remove "/mcp/"
        elif path.startswith("/"):
            # Mounted case: /{session_id}/...
            path_after_prefix = path[1:]  # Remove leading "/"
        else:
            await self._send_error(send, 400, f"Invalid MCP path: {path}")
            return

        # Split to get session_id and remaining path
        path_parts = path_after_prefix.split("/", 1)
        session_id = path_parts[0] if path_parts else ""

        # Remaining path after session_id (if any)
        remaining_after_session = path_parts[1] if len(path_parts) > 1 else ""

        if not session_id:
            await self._send_error(send, 400, "Missing session_id in path")
            return

        # Verify session exists
        if not is_session_registered(session_id):
            logger.warning("MCP session not found", session_id=session_id)
            await self._send_error(send, 404, f"Session '{session_id}' not found")
            return

        # Set the contextvar for this request
        # This allows tools to access the correct session state
        token = current_session_id.set(session_id)

        try:
            # Adjust the path for the MCP app
            # FastMCP streamable_http_app() creates routes at /mcp
            modified_path = "/mcp"
            if remaining_after_session:
                modified_path += "/" + remaining_after_session
            modified_scope = dict(scope)
            modified_scope["path"] = modified_path
            # Also reset root_path since we're handling the full path ourselves
            modified_scope["root_path"] = ""

            logger.info(
                "Routing MCP request",
                session_id=session_id,
                original_path=path,
                modified_path=modified_path,
                method=scope.get("method"),
                root_path=scope.get("root_path", ""),
            )

            # Log available routes for debugging
            if hasattr(self.mcp_app, "routes"):
                route_info = [
                    f"{r.path} ({getattr(r, 'methods', 'any')})"
                    for r in self.mcp_app.routes
                ]
                logger.debug("FastMCP routes", routes=route_info)

            await self.mcp_app(modified_scope, receive, send)
        finally:
            # Reset the contextvar
            current_session_id.reset(token)

    async def _send_error(self, send, status: int, message: str) -> None:
        """Send an HTTP error response.

        Args:
            send: ASGI send callable.
            status: HTTP status code.
            message: Error message.
        """
        import json

        body = json.dumps({"error": message}).encode()

        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(body)).encode()],
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body,
        })


# =============================================================================
# SDK Agent Card Factory
# =============================================================================


def create_sdk_agent_card(base_url: str) -> AgentCard:
    """Create the WebShop+ agent card using SDK types.

    Args:
        base_url: The base URL where the agent is hosted (e.g., http://localhost:8001).

    Returns:
        An SDK AgentCard for the WebShop+ benchmark.
    """
    # Common schema for participants - required for all skills
    participants_schema = {
        "type": "object",
        "properties": {
            "shopper": {
                "type": "string",
                "format": "uri",
                "description": "The A2A endpoint URL of the shopping agent to evaluate.",
            }
        },
        "required": ["shopper"],
    }

    # Full assessment config schema
    full_config_schema = {
        "type": "object",
        "properties": {
            "num_tasks": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 80,
                "description": "Total number of tasks to run across all categories.",
            },
            "categories": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "budget_constrained",
                        "preference_memory",
                        "negative_constraint",
                        "comparative_reasoning",
                        "error_recovery",
                    ],
                },
                "description": "Task types to include (uses exact TaskType enum values). Default: all.",
            },
            "timeout_per_task": {
                "type": "integer",
                "minimum": 30,
                "maximum": 600,
                "default": 120,
                "description": "Timeout in seconds for each task.",
            },
            "max_steps_per_task": {
                "type": "integer",
                "minimum": 5,
                "maximum": 50,
                "default": 15,
                "description": "Maximum interaction steps per task.",
            },
        },
    }

    # Category-specific config schema
    category_config_schema = {
        "type": "object",
        "properties": {
            "num_tasks": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 16,
                "description": "Number of tasks to run for this category.",
            },
            "timeout_per_task": {
                "type": "integer",
                "minimum": 30,
                "maximum": 600,
                "default": 120,
                "description": "Timeout in seconds for each task.",
            },
            "max_steps_per_task": {
                "type": "integer",
                "minimum": 5,
                "maximum": 50,
                "default": 15,
                "description": "Maximum interaction steps per task.",
            },
        },
    }

    # Full assessment input schema
    full_assessment_schema = {
        "type": "object",
        "properties": {
            "participants": participants_schema,
            "config": full_config_schema,
        },
        "required": ["participants"],
    }

    # Category assessment input schema
    category_assessment_schema = {
        "type": "object",
        "properties": {
            "participants": participants_schema,
            "config": category_config_schema,
        },
        "required": ["participants"],
    }

    return AgentCard(
        name="WebShop+ Benchmark",
        description="Evaluates shopping agents on budget management, preference memory, "
        "constraint satisfaction, comparative reasoning, and error recovery.",
        version="1.0.0",
        url=f"{base_url}/a2a",
        provider=AgentProvider(
            organization="WebShop+ Team",
            url="https://github.com/mpnikhil/webshop-plus",
        ),
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False,
            stateTransitionHistory=False,
        ),
        defaultInputModes=["application/json"],
        defaultOutputModes=["application/json"],
        skills=[
            AgentSkill(
                id="assessment",
                name="Shopping Agent Assessment",
                description="Run a comprehensive assessment of a shopping agent across "
                "80 tasks covering budget management, preference memory, constraint "
                "satisfaction, comparative reasoning, and error recovery.",
                tags=["assessment", "benchmark", "shopping", "evaluation"],
                examples=[
                    "Assess the shopping agent at http://agent:8001/a2a",
                    "Run budget constraint tasks only",
                    "Evaluate with 20 tasks per category",
                ],
                inputModes=["application/json"],
                outputModes=["application/json"],
                inputSchema=full_assessment_schema,
            ),
            AgentSkill(
                id="budget-assessment",
                name="Budget Constraint Assessment",
                description="Evaluate agent on budget management tasks.",
                tags=["budget", "shopping", "constraints"],
                examples=["Test budget constraint handling"],
                inputModes=["application/json"],
                outputModes=["application/json"],
                inputSchema=category_assessment_schema,
            ),
            AgentSkill(
                id="memory-assessment",
                name="Preference Memory Assessment",
                description="Evaluate agent on preference recall across sessions.",
                tags=["memory", "preferences", "recall"],
                examples=["Test preference memory"],
                inputModes=["application/json"],
                outputModes=["application/json"],
                inputSchema=category_assessment_schema,
            ),
            AgentSkill(
                id="constraint-assessment",
                name="Negative Constraint Assessment",
                description="Evaluate agent on avoiding forbidden attributes.",
                tags=["constraints", "avoidance", "shopping"],
                examples=["Test negative constraint handling"],
                inputModes=["application/json"],
                outputModes=["application/json"],
                inputSchema=category_assessment_schema,
            ),
            AgentSkill(
                id="reasoning-assessment",
                name="Comparative Reasoning Assessment",
                description="Evaluate agent on product comparison and justification.",
                tags=["reasoning", "comparison", "shopping"],
                examples=["Test comparative reasoning"],
                inputModes=["application/json"],
                outputModes=["application/json"],
                inputSchema=category_assessment_schema,
            ),
            AgentSkill(
                id="recovery-assessment",
                name="Error Recovery Assessment",
                description="Evaluate agent on identifying and fixing cart errors.",
                tags=["recovery", "errors", "cart"],
                examples=["Test error recovery"],
                inputModes=["application/json"],
                outputModes=["application/json"],
                inputSchema=category_assessment_schema,
            ),
        ],
    )


# =============================================================================
# Health Check Endpoint
# =============================================================================


async def health_check(request):
    """Health check endpoint."""
    return JSONResponse({"status": "healthy", "service": "webshop-plus-green-a2a"})


# =============================================================================
# Server Factory
# =============================================================================


def create_app(
    card_url: str = "http://localhost:8001",
    host: str = "localhost",
    port: int = 8000,
    advertise_host: str | None = None,
) -> Starlette:
    """Create the SDK-based A2A Starlette application.

    Args:
        card_url: Base URL for the agent card.
        host: Host to bind the server to (e.g., "0.0.0.0" for all interfaces).
        port: Port for MCP URI generation.
        advertise_host: Hostname for MCP URIs that clients will use to connect.
                       If None, defaults to 'localhost' when host is '0.0.0.0',
                       otherwise uses host. In Docker, set to service name (e.g., 'green').

    Returns:
        A Starlette application with A2A and MCP routes.
    """
    # Create session manager for MCP sessions
    session_manager = SessionManager(max_sessions=100, session_ttl=3600)

    # Determine MCP host for URI generation
    # Use advertise_host if provided, otherwise default to localhost when binding to all interfaces
    if advertise_host:
        mcp_host = advertise_host
    else:
        mcp_host = "localhost" if host == "0.0.0.0" else host

    # Create SDK components
    agent_card = create_sdk_agent_card(card_url)
    task_store = InMemoryTaskStore()
    executor = WebShopPlusExecutor(
        agent_config=AgentConfig(mcp_host=mcp_host, mcp_port=port),
        session_manager=session_manager,
        mcp_host=mcp_host,
        mcp_port=port,
    )

    # Create the SDK request handler
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )

    # Get the global MCP app
    mcp_app = get_mcp_app()

    # Create MCP route handler with the global MCP app
    mcp_handler = MCPRouteHandler(session_manager, mcp_app)

    # Create the A2A Starlette application builder
    a2a_builder = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    # Create the main Starlette app with CORS middleware
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        ),
    ]

    # Create routes including health check and MCP mount
    routes = [
        Route("/health", health_check, methods=["GET"]),
        Mount("/mcp", app=mcp_handler),
    ]

    # Create lifespan context manager that runs the MCP session manager
    @asynccontextmanager
    async def lifespan(app):
        logger.info(
            "Starting WebShop+ A2A server (SDK)",
            card_url=card_url,
            mcp_base=f"http://{mcp_host}:{port}/mcp",
        )
        # Run the MCP session manager - this is REQUIRED for FastMCP to work
        async with mcp.session_manager.run():
            logger.info("MCP session manager started")
            # Log FastMCP app routes for debugging
            if hasattr(mcp_app, "routes"):
                for route in mcp_app.routes:
                    logger.info(
                        "FastMCP route registered",
                        path=getattr(route, "path", "unknown"),
                        methods=getattr(route, "methods", "any"),
                        name=getattr(route, "name", "unnamed"),
                    )
            yield
        # Cleanup session manager on shutdown
        await session_manager.cleanup_all()
        logger.info("Shutting down WebShop+ A2A server (SDK)")

    app = Starlette(
        routes=routes,
        middleware=middleware,
        lifespan=lifespan,
    )

    # Add A2A routes to our app
    # This adds /.well-known/agent-card.json and /a2a (the RPC endpoint)
    a2a_builder.add_routes_to_app(
        app=app,
        agent_card_url="/.well-known/agent-card.json",
        rpc_url="/a2a",
    )

    return app


# =============================================================================
# CLI Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="WebShop+ Green Agent Server (SDK-based)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,  # Default to 8001 for parallel testing
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--card-url",
        type=str,
        default=None,
        help="Base URL for the agent card (defaults to http://localhost:PORT)",
    )
    parser.add_argument(
        "--advertise-host",
        type=str,
        default=None,
        help="Hostname for MCP URIs (defaults to 'localhost' if --host is 0.0.0.0, otherwise same as --host). In Docker, set this to the service name (e.g., 'green').",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the server."""
    args = parse_args()

    # Set up structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set card URL
    card_url = args.card_url or f"http://localhost:{args.port}"

    # Determine advertise host for MCP URIs
    # If not specified, default to localhost when binding to 0.0.0.0, otherwise use bind host
    advertise_host = args.advertise_host
    if advertise_host is None:
        advertise_host = "localhost" if args.host == "0.0.0.0" else args.host

    logger.info(
        "Starting WebShop+ green agent (SDK-based)",
        host=args.host,
        port=args.port,
        card_url=card_url,
        advertise_host=advertise_host,
    )

    # Create and run the app
    app = create_app(card_url=card_url, host=args.host, port=args.port, advertise_host=advertise_host)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
