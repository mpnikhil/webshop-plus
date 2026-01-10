"""
FastAPI server for WebShop+ green agent.

This module implements the A2A protocol server with:
- Agent card endpoint at /.well-known/agent-card.json
- A2A message handling endpoint at /a2a
- SSE streaming for real-time task updates
- CLI argument parsing for host, port, and card-url

Usage:
    uv run python src/server.py --host 0.0.0.0 --port 8000 --card-url http://localhost:8000
"""

import argparse
import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Optional

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

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
    create_status_update_event,
    create_task_response,
    create_text_message,
    create_webshop_plus_agent_card,
    extract_action_from_text,
    get_text_from_message,
)
from src.agent import AgentConfig, WebShopPlusAgent
from src.models import (
    AssessmentConfig,
    AssessmentRequest,
    AssessmentResults,
    EvaluationResult,
    TaskType,
    TaskUpdate,
)

logger = structlog.get_logger()


# =============================================================================
# Global State
# =============================================================================


class ServerState:
    """Global server state container."""

    def __init__(self):
        self.card_url: str = "http://localhost:8000"
        self.agent_card: Optional[AgentCard] = None
        self.active_assessments: dict[str, dict[str, Any]] = {}


state = ServerState()


# =============================================================================
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan."""
    logger.info("Starting WebShop+ green agent server", card_url=state.card_url)
    state.agent_card = create_webshop_plus_agent_card(state.card_url)
    yield
    logger.info("Shutting down WebShop+ green agent server")


# =============================================================================
# FastAPI Application
# =============================================================================


app = FastAPI(
    title="WebShop+ Green Agent",
    description="Evaluates shopping agents on budget management, preference memory, "
    "constraint satisfaction, comparative reasoning, and error recovery.",
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
        state.agent_card = create_webshop_plus_agent_card(state.card_url)

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
    - message/send: Send a message and receive synchronous response
    - message/stream: Send a message and receive SSE stream
    - tasks/get: Get task status by ID
    - tasks/cancel: Cancel a running task

    Returns:
        JSON-RPC response or SSE stream depending on method.
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
    elif method == "message/stream":
        return await handle_message_stream(params, request_id)
    elif method == "tasks/get":
        return await handle_tasks_get(params, request_id)
    elif method == "tasks/cancel":
        return await handle_tasks_cancel(params, request_id)
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

    This method creates a new task or continues an existing one.
    For WebShop+, this initiates an assessment.

    Args:
        params: Request parameters containing the message.
        request_id: The JSON-RPC request ID.

    Returns:
        JSON response with task result.
    """
    message_data = params.get("message", {})
    metadata = params.get("metadata", {})

    # Extract text content from message
    text_content = get_text_from_message(message_data)
    logger.info("Received message", text=text_content[:200] if text_content else "")

    # Check if this is an assessment request
    if "participants" in metadata or _is_assessment_request(text_content):
        return await handle_assessment_request(params, request_id)

    # For simple messages, create a quick response task
    task_id = str(uuid.uuid4())
    context_id = message_data.get("contextId", str(uuid.uuid4()))

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
                parts=[
                    {
                        "kind": "text",
                        "text": "Welcome to WebShop+ Benchmark! To start an assessment, "
                        "please send an assessment request with participant endpoints in the metadata.",
                    }
                ],
                taskId=task_id,
                contextId=context_id,
            ),
        ],
    )

    return JSONResponse(
        content=create_task_response(task, request_id).model_dump(mode="json", exclude_none=True)
    )


async def handle_message_stream(
    params: dict[str, Any], request_id: str
) -> StreamingResponse:
    """Handle message/stream method with SSE.

    This method streams task updates as Server-Sent Events.

    Args:
        params: Request parameters containing the message.
        request_id: The JSON-RPC request ID.

    Returns:
        SSE stream response.
    """
    message_data = params.get("message", {})
    metadata = params.get("metadata", {})

    # Create task
    task_id = str(uuid.uuid4())
    context_id = message_data.get("contextId", str(uuid.uuid4()))

    async def generate_events() -> AsyncGenerator[str, None]:
        """Generate SSE events."""
        # Send initial task event
        initial_task = A2ATask(
            id=task_id,
            contextId=context_id,
            status=TaskStatus(state=TaskState.SUBMITTED),
            history=[
                A2AMessage(
                    role=MessageRole.USER,
                    parts=message_data.get("parts", []),
                    messageId=message_data.get("messageId", str(uuid.uuid4())),
                    taskId=task_id,
                    contextId=context_id,
                )
            ],
        )

        task_data = create_task_response(initial_task, request_id).model_dump(mode="json", exclude_none=True)
        yield f"data: {json.dumps(task_data)}\n\n"

        # Check if this is an assessment request
        if "participants" in metadata or _is_assessment_request(
            get_text_from_message(message_data)
        ):
            # Stream assessment updates
            async for event in stream_assessment(
                params, task_id, context_id, request_id
            ):
                yield f"data: {json.dumps(event)}\n\n"
        else:
            # Simple response - just mark as completed
            status_event = create_status_update_event(task_id, context_id, TaskState.WORKING, "Processing message...", False, request_id)
            yield f"data: {json.dumps(status_event)}\n\n"
            await asyncio.sleep(0.1)

            # Create response artifact
            artifact = Artifact(
                name="response",
                parts=[
                    {
                        "kind": "text",
                        "text": "Welcome to WebShop+ Benchmark! To start an assessment, "
                        "provide participant endpoints in the request metadata.",
                    }
                ],
            )
            artifact_event = create_artifact_update_event(task_id, context_id, artifact, False, True, request_id)
            yield f"data: {json.dumps(artifact_event)}\n\n"

            # Final status
            final_event = create_status_update_event(task_id, context_id, TaskState.COMPLETED, "Message processed", True, request_id)
            yield f"data: {json.dumps(final_event)}\n\n"

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def handle_tasks_get(params: dict[str, Any], request_id: str) -> JSONResponse:
    """Handle tasks/get method.

    Args:
        params: Request parameters containing task_id.
        request_id: The JSON-RPC request ID.

    Returns:
        JSON response with task details.
    """
    task_id = params.get("id") or params.get("task_id")

    if not task_id:
        return JSONResponse(
            content=create_error_response(
                code=-32602,
                message="Missing required parameter: id",
                request_id=request_id,
            ).model_dump(mode="json", exclude_none=True),
            status_code=400,
        )

    # Check active assessments
    if task_id in state.active_assessments:
        assessment = state.active_assessments[task_id]
        task = A2ATask(
            id=task_id,
            contextId=assessment.get("context_id", ""),
            status=TaskStatus(
                state=TaskState(assessment.get("status", "working")),
                message=assessment.get("message"),
            ),
        )
        return JSONResponse(
            content=create_task_response(task, request_id).model_dump(mode="json", exclude_none=True)
        )

    return JSONResponse(
        content=create_error_response(
            code=-32001,
            message=f"Task not found: {task_id}",
            request_id=request_id,
        ).model_dump(mode="json", exclude_none=True),
        status_code=404,
    )


async def handle_tasks_cancel(params: dict[str, Any], request_id: str) -> JSONResponse:
    """Handle tasks/cancel method.

    Args:
        params: Request parameters containing task_id.
        request_id: The JSON-RPC request ID.

    Returns:
        JSON response confirming cancellation.
    """
    task_id = params.get("id") or params.get("task_id")

    if not task_id:
        return JSONResponse(
            content=create_error_response(
                code=-32602,
                message="Missing required parameter: id",
                request_id=request_id,
            ).model_dump(mode="json", exclude_none=True),
            status_code=400,
        )

    if task_id in state.active_assessments:
        state.active_assessments[task_id]["status"] = "canceled"
        state.active_assessments[task_id]["canceled"] = True
        logger.info("Task canceled", task_id=task_id)

        task = A2ATask(
            id=task_id,
            contextId=state.active_assessments[task_id].get("context_id", ""),
            status=TaskStatus(state=TaskState.CANCELED, message="Task canceled by user"),
        )
        return JSONResponse(
            content=create_task_response(task, request_id).model_dump(mode="json", exclude_none=True)
        )

    return JSONResponse(
        content=create_error_response(
            code=-32001,
            message=f"Task not found: {task_id}",
            request_id=request_id,
        ).model_dump(mode="json", exclude_none=True),
        status_code=404,
    )


async def handle_get_extended_card(request_id: str) -> JSONResponse:
    """Handle agent/getAuthenticatedExtendedCard method.

    Returns the same card as the public endpoint for now.

    Args:
        request_id: The JSON-RPC request ID.

    Returns:
        JSON response with extended agent card.
    """
    if not state.agent_card:
        state.agent_card = create_webshop_plus_agent_card(state.card_url)

    return JSONResponse(
        content=JSONRPCResponse(
            result=state.agent_card.model_dump(mode="json", exclude_none=True),
            id=request_id,
        ).model_dump(mode="json", exclude_none=True)
    )


# =============================================================================
# Assessment Handling
# =============================================================================


def _is_assessment_request(text: str) -> bool:
    """Check if a message text indicates an assessment request."""
    keywords = ["assess", "evaluate", "benchmark", "test", "run assessment", "start assessment"]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)


async def handle_assessment_request(
    params: dict[str, Any], request_id: str
) -> JSONResponse:
    """Handle an assessment request synchronously.

    This creates a task and returns immediately. The actual assessment
    runs in the background or via streaming.

    Args:
        params: Request parameters with participants and config.
        request_id: The JSON-RPC request ID.

    Returns:
        JSON response with task created.
    """
    metadata = params.get("metadata", {})
    message_data = params.get("message", {})

    # Extract participants from metadata or message
    participants = metadata.get("participants", {})
    if not participants:
        # Try to parse from message text
        text = get_text_from_message(message_data)
        # For now, require explicit participants
        if not participants:
            logger.warning("No participants provided in assessment request")

    # Extract config
    config_data = metadata.get("config", {})
    config = AssessmentConfig(**config_data) if config_data else AssessmentConfig()

    # Create task
    task_id = str(uuid.uuid4())
    context_id = message_data.get("contextId", str(uuid.uuid4()))

    # Store assessment state
    state.active_assessments[task_id] = {
        "context_id": context_id,
        "status": "submitted",
        "participants": participants,
        "config": config.model_dump(),
        "started_at": datetime.utcnow().isoformat(),
        "canceled": False,
    }

    task = A2ATask(
        id=task_id,
        contextId=context_id,
        status=TaskStatus(
            state=TaskState.SUBMITTED,
            message="Assessment request received. Use message/stream for real-time updates.",
        ),
        history=[
            A2AMessage(
                role=MessageRole.USER,
                parts=message_data.get("parts", []),
                taskId=task_id,
                contextId=context_id,
            )
        ],
    )

    logger.info(
        "Assessment request created",
        task_id=task_id,
        participants=participants,
        config=config.model_dump(),
    )

    return JSONResponse(
        content=create_task_response(task, request_id).model_dump(mode="json", exclude_none=True)
    )


async def stream_assessment(
    params: dict[str, Any],
    task_id: str,
    context_id: str,
    request_id: str,
) -> AsyncGenerator[dict[str, Any], None]:
    """Stream assessment progress via SSE.

    Uses the WebShopPlusAgent to orchestrate the actual assessment.

    Args:
        params: Request parameters.
        task_id: The task ID.
        context_id: The context ID.
        request_id: The request ID.

    Yields:
        SSE event data dictionaries.
    """
    metadata = params.get("metadata", {})
    participants = metadata.get("participants", {})
    config_data = metadata.get("config", {})

    # Update assessment state
    if task_id in state.active_assessments:
        state.active_assessments[task_id]["status"] = "working"

    # Send working status
    yield create_status_update_event(
        task_id,
        context_id,
        TaskState.WORKING,
        "Assessment starting...",
        False,
        request_id,
    )

    if not participants:
        # No participants - send error and complete
        yield create_status_update_event(
            task_id,
            context_id,
            TaskState.FAILED,
            "No participants provided. Please include 'participants' in metadata with agent endpoints.",
            True,
            request_id,
        )
        if task_id in state.active_assessments:
            state.active_assessments[task_id]["status"] = "failed"
        return

    # Parse assessment config
    config = AssessmentConfig(**config_data) if config_data else AssessmentConfig()

    # Create agent config
    agent_config = AgentConfig(
        task_timeout_seconds=config.timeout_per_task,
    )

    try:
        # Create and run the orchestration agent
        async with WebShopPlusAgent(config=agent_config) as agent:
            # Check for cancellation
            def check_canceled():
                return state.active_assessments.get(task_id, {}).get("canceled", False)

            # Stream results from the agent
            async for event in agent.run_streaming(
                participants=participants,
                config=config,
                task_id=task_id,
                context_id=context_id,
                request_id=request_id,
            ):
                # Check for cancellation
                if check_canceled():
                    agent.cancel()
                    yield create_status_update_event(
                        task_id,
                        context_id,
                        TaskState.CANCELED,
                        "Assessment canceled by user",
                        True,
                        request_id,
                    )
                    if task_id in state.active_assessments:
                        state.active_assessments[task_id]["status"] = "canceled"
                    return

                yield event

                # Check if this is a final event
                result = event.get("result", {})
                if result.get("final"):
                    event_state = result.get("status", {}).get("state")
                    if event_state in ("completed", "failed", "canceled"):
                        if task_id in state.active_assessments:
                            state.active_assessments[task_id]["status"] = event_state
                        return

    except Exception as e:
        logger.error("Assessment error", task_id=task_id, error=str(e))
        yield create_status_update_event(
            task_id,
            context_id,
            TaskState.FAILED,
            f"Assessment failed: {str(e)}",
            True,
            request_id,
        )
        if task_id in state.active_assessments:
            state.active_assessments[task_id]["status"] = "failed"


# =============================================================================
# Health Check
# =============================================================================


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "webshop-plus-green"}


# =============================================================================
# CLI Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="WebShop+ Green Agent Server",
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
        default=8000,
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
        "Starting WebShop+ green agent",
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
