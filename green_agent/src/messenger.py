"""
A2A Protocol messenger utilities for WebShop+ green agent.

This module provides utilities for A2A (Agent-to-Agent) protocol communication:
- JSON-RPC message formatting and parsing
- HTTP client for sending messages to purple agents
- SSE (Server-Sent Events) event creation utilities
- Response parsing utilities

Note: Agent card types and server-side request handling are now in server.py
using the official a2a-sdk types.

Based on A2A Protocol v0.3.0 specification.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Optional

import httpx
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


# =============================================================================
# Enums
# =============================================================================


class TaskState(str, Enum):
    """A2A task states."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    AUTH_REQUIRED = "auth-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    REJECTED = "rejected"


class MessageRole(str, Enum):
    """Message role in A2A conversation."""

    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


# =============================================================================
# A2A Message Models
# =============================================================================


class TextPart(BaseModel):
    """A text part of a message."""

    kind: str = "text"
    text: str


class FilePart(BaseModel):
    """A file part of a message."""

    kind: str = "file"
    file: dict[str, Any]  # Contains mimeType, data (base64), name, etc.


# Union type for message parts
MessagePart = TextPart | FilePart


class A2AMessage(BaseModel):
    """An A2A protocol message."""

    role: MessageRole
    parts: list[dict[str, Any]]  # List of TextPart or FilePart dicts
    messageId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    taskId: Optional[str] = None
    contextId: Optional[str] = None


class TaskStatus(BaseModel):
    """Status of an A2A task."""

    state: TaskState
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Artifact(BaseModel):
    """An artifact produced by a task."""

    artifactId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    description: Optional[str] = None
    parts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class A2ATask(BaseModel):
    """An A2A task representation."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    contextId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus
    artifacts: list[Artifact] = Field(default_factory=list)
    history: list[A2AMessage] = Field(default_factory=list)
    kind: str = "task"
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# JSON-RPC Models
# =============================================================================


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request."""

    jsonrpc: str = "2.0"
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response."""

    jsonrpc: str = "2.0"
    result: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None
    id: str


class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error."""

    code: int
    message: str
    data: Optional[dict[str, Any]] = None


# =============================================================================
# Message Factory Functions
# =============================================================================


def create_text_message(
    text: str,
    role: MessageRole = MessageRole.USER,
    task_id: Optional[str] = None,
    context_id: Optional[str] = None,
) -> A2AMessage:
    """Create a simple text message.

    Args:
        text: The text content of the message.
        role: The role of the sender (user/agent/system).
        task_id: Optional task ID to associate with the message.
        context_id: Optional context ID for conversation tracking.

    Returns:
        An A2AMessage instance.
    """
    return A2AMessage(
        role=role,
        parts=[{"kind": "text", "text": text}],
        taskId=task_id,
        contextId=context_id,
    )


def create_message_send_request(
    message: A2AMessage,
    metadata: Optional[dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> JSONRPCRequest:
    """Create a message/send JSON-RPC request.

    Args:
        message: The A2A message to send.
        metadata: Optional metadata for the request.
        request_id: Optional request ID (auto-generated if not provided).

    Returns:
        A JSONRPCRequest for message/send.
    """
    return JSONRPCRequest(
        method="message/send",
        params={
            "message": message.model_dump(exclude_none=True),
            "metadata": metadata or {},
        },
        id=request_id or str(uuid.uuid4()),
    )


def create_message_stream_request(
    message: A2AMessage,
    metadata: Optional[dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> JSONRPCRequest:
    """Create a message/stream JSON-RPC request.

    Args:
        message: The A2A message to send.
        metadata: Optional metadata for the request.
        request_id: Optional request ID (auto-generated if not provided).

    Returns:
        A JSONRPCRequest for message/stream.
    """
    return JSONRPCRequest(
        method="message/stream",
        params={
            "message": message.model_dump(exclude_none=True),
            "metadata": metadata or {},
        },
        id=request_id or str(uuid.uuid4()),
    )


def create_status_update_event(
    task_id: str,
    context_id: str,
    state: TaskState,
    message: Optional[str] = None,
    final: bool = False,
    request_id: str = "",
) -> dict[str, Any]:
    """Create a status update SSE event.

    Args:
        task_id: The task ID.
        context_id: The context ID.
        state: The new task state.
        message: Optional status message.
        final: Whether this is the final update.
        request_id: The request ID.

    Returns:
        A dictionary for the SSE event data.
    """
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "taskId": task_id,
            "contextId": context_id,
            "status": {
                "state": state.value,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
            },
            "final": final,
            "kind": "status-update",
        },
    }


def create_artifact_update_event(
    task_id: str,
    context_id: str,
    artifact: Artifact,
    append: bool = False,
    last_chunk: bool = False,
    request_id: str = "",
) -> dict[str, Any]:
    """Create an artifact update SSE event.

    Args:
        task_id: The task ID.
        context_id: The context ID.
        artifact: The artifact to include.
        append: Whether to append to existing artifact.
        last_chunk: Whether this is the last chunk.
        request_id: The request ID.

    Returns:
        A dictionary for the SSE event data.
    """
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "taskId": task_id,
            "contextId": context_id,
            "artifact": artifact.model_dump(exclude_none=True),
            "append": append,
            "lastChunk": last_chunk,
            "kind": "artifact-update",
        },
    }


# =============================================================================
# A2A HTTP Client
# =============================================================================


class A2AClient:
    """HTTP client for A2A protocol communication.

    This client handles sending messages to purple agents and receiving responses.
    It supports both synchronous (message/send) and streaming (message/stream) modes.
    """

    def __init__(
        self,
        timeout: float = 300.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize the A2A client.

        Args:
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
            retry_delay: Base delay between retries in seconds.
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "A2AClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def send_message(
        self,
        endpoint: str,
        message: A2AMessage,
        metadata: Optional[dict[str, Any]] = None,
    ) -> JSONRPCResponse:
        """Send a message to an agent and wait for response.

        Args:
            endpoint: The agent's A2A endpoint URL.
            message: The message to send.
            metadata: Optional metadata for the request.

        Returns:
            The JSON-RPC response from the agent.

        Raises:
            httpx.HTTPError: If the request fails after retries.
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context.")

        request = create_message_send_request(message, metadata)
        request_data = request.model_dump(exclude_none=True)

        for attempt in range(self.max_retries):
            try:
                response = await self._client.post(
                    endpoint,
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
                return JSONRPCResponse(**data)
            except httpx.HTTPError as e:
                logger.warning(
                    "A2A request failed",
                    endpoint=endpoint,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < self.max_retries - 1:
                    import asyncio

                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

        # Should not reach here
        raise RuntimeError("Unexpected state in send_message")

    async def stream_message(
        self,
        endpoint: str,
        message: A2AMessage,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a message and stream responses via SSE.

        Args:
            endpoint: The agent's A2A endpoint URL.
            message: The message to send.
            metadata: Optional metadata for the request.

        Yields:
            Parsed SSE event data dictionaries.

        Raises:
            httpx.HTTPError: If the request fails.
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context.")

        request = create_message_stream_request(message, metadata)
        request_data = request.model_dump(exclude_none=True)

        async with self._client.stream(
            "POST",
            endpoint,
            json=request_data,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    import json

                    data_str = line[6:].strip()
                    if data_str:
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse SSE data", data=data_str)

    async def get_agent_card(self, base_url: str) -> Optional[dict[str, Any]]:
        """Fetch an agent's card from the well-known endpoint.

        Args:
            base_url: The agent's base URL.

        Returns:
            The agent card as a dictionary, or None if not found.
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context.")

        try:
            # Try the well-known endpoint
            url = f"{base_url.rstrip('/')}/.well-known/agent-card.json"
            response = await self._client.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning("Failed to fetch agent card", url=base_url, error=str(e))
            return None


# =============================================================================
# Response Parsing Utilities
# =============================================================================


def parse_action_from_response(response: JSONRPCResponse) -> Optional[str]:
    """Extract an action string from an agent's response.

    The action should be in the format: search[query] or click[element]
    Actions are expected in agent messages (history or direct message), not artifacts.

    Args:
        response: The JSON-RPC response from the purple agent.

    Returns:
        The extracted action string, or None if not found.
    """
    if response.error:
        logger.warning("Response contains error", error=response.error)
        return None

    if not response.result:
        return None

    # Try to extract from task result
    result = response.result

    # Check if it's a task response with history
    if "history" in result:
        history = result.get("history", [])
        if history:
            # Get the last agent message
            for msg in reversed(history):
                if msg.get("role") == "agent":
                    parts = msg.get("parts", [])
                    for part in parts:
                        if part.get("kind") == "text":
                            text = part.get("text", "")
                            return extract_action_from_text(text)

    # Check for direct message in result
    if "message" in result:
        message = result["message"]
        parts = message.get("parts", [])
        for part in parts:
            if part.get("kind") == "text":
                text = part.get("text", "")
                return extract_action_from_text(text)

    return None


def extract_action_from_text(text: str) -> Optional[str]:
    """Extract a WebShop action from text.

    Actions are in the format:
    - search[query text]
    - click[element name or button]

    Args:
        text: The text to search for actions.

    Returns:
        The extracted action string, or None if not found.
    """
    import re

    # Pattern for search[...] or click[...]
    pattern = r"(search|click)\[([^\]]+)\]"
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        action_type = match.group(1).lower()
        action_arg = match.group(2)
        return f"{action_type}[{action_arg}]"

    return None


def get_text_from_message(message: dict[str, Any]) -> str:
    """Extract all text content from an A2A message.

    Args:
        message: An A2A message dictionary.

    Returns:
        Concatenated text content from all text parts.
    """
    parts = message.get("parts", [])
    texts = []
    for part in parts:
        if part.get("kind") == "text":
            texts.append(part.get("text", ""))
    return "\n".join(texts)
