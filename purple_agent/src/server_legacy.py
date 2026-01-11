"""
FastAPI server for WebShop+ purple (shopping) agent.

This module implements the A2A protocol server with:
- Agent card endpoint at /.well-known/agent-card.json
- A2A message handling endpoint at /a2a
- Session management for shopping conversations

Usage:
    uv run python src/server.py --host 0.0.0.0 --port 8001 --card-url http://localhost:8001
"""

import argparse
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.agent import ShopperAgent
from src.messenger import (
    A2AMessage,
    A2ATask,
    AgentCard,
    JSONRPCRequest,
    MessageRole,
    TaskState,
    TaskStatus,
    create_error_response,
    create_shopper_agent_card,
    create_task_response,
    extract_observation,
    extract_task_instruction,
    get_text_from_message,
)

logger = structlog.get_logger()


# =============================================================================
# Global State
# =============================================================================


class ServerState:
    """Global server state container."""

    def __init__(self):
        self.card_url: str = "http://localhost:8001"
        self.agent_card: Optional[AgentCard] = None
        self.sessions: dict[str, ShopperAgent] = {}


state = ServerState()


# =============================================================================
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan."""
    logger.info("Starting WebShop+ purple (shopper) agent server", card_url=state.card_url)
    state.agent_card = create_shopper_agent_card(state.card_url)
    yield
    logger.info("Shutting down WebShop+ purple agent server")
    # Clean up sessions
    state.sessions.clear()


# =============================================================================
# FastAPI Application
# =============================================================================


app = FastAPI(
    title="WebShop+ Purple Agent",
    description="Baseline shopping agent for the WebShop+ benchmark. "
    "Navigates WebShop to find and purchase products based on instructions.",
    version="1.0.0",
    lifespan=lifespan,
)


# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Agent Card Endpoint
# =============================================================================


@app.get("/.well-known/agent-card.json")
async def get_agent_card() -> JSONResponse:
    """Serve the agent card for A2A discovery.

    Returns:
        JSON response containing the agent card.
    """
    if not state.agent_card:
        state.agent_card = create_shopper_agent_card(state.card_url)

    return JSONResponse(
        content=state.agent_card.model_dump(mode="json", exclude_none=True),
        media_type="application/json",
    )


# =============================================================================
# A2A Endpoint
# =============================================================================


@app.post("/a2a", response_model=None)
async def handle_a2a(request: Request):
    """Handle A2A protocol messages.

    Supported methods:
    - message/send: Process a message and return an action

    Returns:
        JSON-RPC response with the agent's action.
    """
    try:
        body = await request.json()
        logger.debug("Received A2A request", body=body)
    except Exception as e:
        logger.error("Failed to parse request body", error=str(e))
        return JSONResponse(
            content=create_error_response(
                code=-32700,
                message="Parse error",
                request_id="",
                data={"detail": str(e)},
            ).model_dump(mode="json", exclude_none=True),
            status_code=400,
        )

    # Parse as JSON-RPC request
    try:
        rpc_request = JSONRPCRequest(**body)
    except Exception as e:
        logger.error("Invalid JSON-RPC request", error=str(e))
        return JSONResponse(
            content=create_error_response(
                code=-32600,
                message="Invalid request",
                request_id=body.get("id", ""),
                data={"detail": str(e)},
            ).model_dump(mode="json", exclude_none=True),
            status_code=400,
        )

    # Route by method
    method = rpc_request.method
    params = rpc_request.params
    request_id = rpc_request.id

    logger.info("Processing A2A method", method=method, request_id=request_id)

    if method == "message/send":
        return await handle_message_send(params, request_id)
    elif method == "agent/getAuthenticatedExtendedCard":
        return await handle_get_extended_card(request_id)
    else:
        logger.warning("Unknown method", method=method)
        return JSONResponse(
            content=create_error_response(
                code=-32601,
                message=f"Method not found: {method}",
                request_id=request_id,
            ).model_dump(mode="json", exclude_none=True),
            status_code=400,
        )


# =============================================================================
# Message Handlers
# =============================================================================


async def handle_message_send(params: dict[str, Any], request_id: str) -> JSONResponse:
    """Handle message/send method.

    Processes incoming messages (task instructions or observations) and
    returns the agent's next action.

    Args:
        params: Request parameters containing the message.
        request_id: The JSON-RPC request ID.

    Returns:
        JSON response with task containing the agent's action.
    """
    message_data = params.get("message", {})
    metadata = params.get("metadata", {})

    # Get or create session
    context_id = message_data.get("contextId") or metadata.get("context_id") or str(uuid.uuid4())
    task_id = message_data.get("taskId") or metadata.get("task_id") or str(uuid.uuid4())

    agent = _get_or_create_agent(context_id, task_id)

    # Extract text content from message
    text_content = get_text_from_message(message_data)
    logger.info("Received message", text=text_content[:200] if text_content else "", context_id=context_id)

    # Determine message type and process
    action = _process_message(agent, text_content, metadata)

    # Create response task with the action
    task = A2ATask(
        id=task_id,
        contextId=context_id,
        status=TaskStatus(state=TaskState.COMPLETED),
        history=[
            A2AMessage(
                role=MessageRole.USER,
                parts=message_data.get("parts", []),
                messageId=message_data.get("messageId", str(uuid.uuid4())),
                taskId=task_id,
                contextId=context_id,
            ),
            A2AMessage(
                role=MessageRole.AGENT,
                parts=[{"kind": "text", "text": action}],
                taskId=task_id,
                contextId=context_id,
            ),
        ],
    )

    return JSONResponse(
        content=create_task_response(task, request_id).model_dump(mode="json", exclude_none=True)
    )


async def handle_get_extended_card(request_id: str) -> JSONResponse:
    """Handle agent/getAuthenticatedExtendedCard method.

    Args:
        request_id: The JSON-RPC request ID.

    Returns:
        JSON response with agent card.
    """
    if not state.agent_card:
        state.agent_card = create_shopper_agent_card(state.card_url)

    from src.messenger import JSONRPCResponse

    return JSONResponse(
        content=JSONRPCResponse(
            result=state.agent_card.model_dump(mode="json", exclude_none=True),
            id=request_id,
        ).model_dump(mode="json", exclude_none=True)
    )


# =============================================================================
# Session Management
# =============================================================================


def _get_or_create_agent(context_id: str, task_id: str) -> ShopperAgent:
    """Get or create a shopping agent for the session.

    Args:
        context_id: The context/session ID.
        task_id: The task ID.

    Returns:
        A ShopperAgent instance for the session.
    """
    if context_id not in state.sessions:
        logger.info("Creating new shopping session", context_id=context_id, task_id=task_id)
        agent = ShopperAgent()
        agent.reset(task_id=task_id)
        state.sessions[context_id] = agent

    return state.sessions[context_id]


def _process_message(agent: ShopperAgent, text: str, metadata: dict[str, Any]) -> str:
    """Process a message and return the agent's action.

    Args:
        agent: The ShopperAgent instance.
        text: The message text.
        metadata: Message metadata.

    Returns:
        The action string (e.g., "search[running shoes]" or "click[B07XYZ]").
    """
    # Check metadata for message type hints
    message_type = metadata.get("type", "").lower()

    # Detect message type from content if not specified
    if not message_type:
        text_lower = text.lower()
        if text_lower.startswith("task:") or "find " in text_lower or "buy " in text_lower or "purchase " in text_lower:
            message_type = "task_instruction"
        elif text_lower.startswith("observation:") or "products found" in text_lower or "search results" in text_lower:
            message_type = "observation"
        elif text_lower.startswith("error:") or "error" in text_lower:
            message_type = "error"
        else:
            # Default to observation for subsequent messages
            message_type = "observation" if agent.context.action_history else "task_instruction"

    logger.debug("Processing message", message_type=message_type, text_preview=text[:100])

    # Process based on type
    if message_type == "task_instruction":
        # Reset agent for new task
        agent.reset(task_id=agent.context.task_id)
        instruction = extract_task_instruction({"parts": [{"kind": "text", "text": text}]}) or text
        return agent.process_task_instruction(instruction)
    elif message_type == "error":
        return agent.process_error(text)
    else:
        # Observation
        observation = extract_observation({"parts": [{"kind": "text", "text": text}]}) or text
        return agent.process_observation(observation)


# =============================================================================
# Health Check
# =============================================================================


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "webshop-plus-purple",
        "active_sessions": str(len(state.sessions)),
    }


# =============================================================================
# CLI Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="WebShop+ Purple (Shopper) Agent Server",
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
        default=8001,
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--card-url",
        type=str,
        default=None,
        help="Base URL for the agent card (defaults to http://localhost:PORT)",
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
    state.card_url = args.card_url or f"http://localhost:{args.port}"

    logger.info(
        "Starting WebShop+ purple (shopper) agent",
        host=args.host,
        port=args.port,
        card_url=state.card_url,
    )

    # Run the server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
