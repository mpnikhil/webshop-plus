"""
A2A SDK-based server for WebShop+ purple (shopping) agent.

This module implements the A2A protocol server using the official a2a-sdk:
- A2AStarletteApplication for server framework
- DefaultRequestHandler for request routing
- InMemoryTaskStore for task state management

Usage:
    uv run python src/server.py --host 0.0.0.0 --port 8001 --card-url http://localhost:8001

Based on the RDI Foundation agent-template pattern:
https://github.com/RDI-Foundation/agent-template
"""

import argparse
from contextlib import asynccontextmanager

import structlog
import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from src.executor import Executor

logger = structlog.get_logger()


def create_agent_card(card_url: str) -> AgentCard:
    """Create the WebShop+ purple (shopper) agent card.

    Args:
        card_url: The base URL where the agent is hosted.

    Returns:
        An AgentCard for the shopping agent.
    """
    skill = AgentSkill(
        id="shopping",
        name="Product Shopping",
        description=(
            "Navigate WebShop to find and purchase products based on "
            "requirements, constraints, and preferences. Supports search, "
            "product comparison, and purchase decisions."
        ),
        tags=["shopping", "search", "purchase", "comparison"],
        examples=[
            "Find running shoes under $100",
            "Buy a laptop with at least 16GB RAM",
            "Find the cheapest wireless headphones",
        ],
    )

    return AgentCard(
        name="WebShop+ Shopper Agent",
        description=(
            "Baseline shopping agent that navigates WebShop to find and purchase "
            "products based on given instructions. Supports search, product comparison, "
            "and purchase decisions."
        ),
        url=f"{card_url}/a2a",
        version="1.0.0",
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        capabilities=AgentCapabilities(
            streaming=True,  # Required for task status updates
            pushNotifications=False,
            stateTransitionHistory=False,
        ),
        skills=[skill],
    )


# =============================================================================
# Health Check Endpoint
# =============================================================================


async def health_check(request):
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "service": "webshop-plus-purple",
    })


# =============================================================================
# Server Factory
# =============================================================================


def create_app(card_url: str = "http://localhost:8001") -> Starlette:
    """Create the SDK-based A2A Starlette application.

    Args:
        card_url: Base URL for the agent card.

    Returns:
        A Starlette application with A2A routes.
    """
    # Create SDK components
    agent_card = create_agent_card(card_url)
    task_store = InMemoryTaskStore()
    executor = Executor()

    # Create the SDK request handler
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )

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

    # Create routes including health check
    routes = [
        Route("/health", health_check, methods=["GET"]),
    ]

    # Create lifespan context manager for startup/shutdown logging
    @asynccontextmanager
    async def lifespan(app):
        logger.info("Starting WebShop+ purple agent (SDK)", card_url=card_url)
        yield
        logger.info("Shutting down WebShop+ purple agent (SDK)")

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
        description="WebShop+ Purple (Shopper) Agent Server (SDK-based)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--card-url",
        type=str,
        default=None,
        help="Base URL for the agent card (defaults to http://HOST:PORT)",
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
    """Main entry point for the SDK-based server."""
    args = parse_args()

    # Configure structured logging
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

    # Build card URL
    card_url = args.card_url or f"http://{args.host}:{args.port}"

    logger.info(
        "Starting WebShop+ purple (shopper) agent (SDK-based)",
        host=args.host,
        port=args.port,
        card_url=card_url,
    )

    # Create and run the app
    app = create_app(card_url=card_url)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        access_log=False,
    )


if __name__ == "__main__":
    main()
