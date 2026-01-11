"""
A2A Protocol messenger utilities for WebShop+ purple agent.

This module provides utilities for A2A (Agent-to-Agent) protocol communication:
- JSON-RPC message parsing and response formatting
- Agent card generation for the shopping agent
- Message parsing utilities

Based on A2A Protocol v0.3.0 specification.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

import structlog
from a2a.types import Message as SDKMessage
from a2a.types import TextPart as SDKTextPart
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
    file: dict[str, Any]


MessagePart = TextPart | FilePart


class A2AMessage(BaseModel):
    """An A2A protocol message."""

    role: MessageRole
    parts: list[dict[str, Any]]
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
# Agent Card Models
# =============================================================================


class AgentSkill(BaseModel):
    """A skill that an agent can perform."""

    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    inputModes: list[str] = Field(default_factory=lambda: ["text/plain"])
    outputModes: list[str] = Field(default_factory=lambda: ["text/plain"])


class AgentCapabilities(BaseModel):
    """Capabilities of an agent."""

    streaming: bool = False  # Purple agent doesn't need streaming
    pushNotifications: bool = False
    stateTransitionHistory: bool = False


class AgentProvider(BaseModel):
    """Provider information for an agent."""

    organization: str
    url: Optional[str] = None


class AgentCard(BaseModel):
    """Agent card describing an agent's capabilities and endpoints."""

    protocolVersion: str = "0.3.0"
    name: str
    description: str
    version: str = "1.0.0"
    url: str
    preferredTransport: str = "JSONRPC"
    provider: Optional[AgentProvider] = None
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    defaultInputModes: list[str] = Field(default_factory=lambda: ["text/plain"])
    defaultOutputModes: list[str] = Field(default_factory=lambda: ["text/plain"])
    skills: list[AgentSkill] = Field(default_factory=list)


# =============================================================================
# Message Factory Functions
# =============================================================================


def create_text_message(
    text: str,
    role: MessageRole = MessageRole.AGENT,
    task_id: Optional[str] = None,
    context_id: Optional[str] = None,
) -> A2AMessage:
    """Create a simple text message.

    Args:
        text: The text content of the message.
        role: The role of the sender (default: agent for purple agent).
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


def create_task_response(
    task: A2ATask,
    request_id: str,
) -> JSONRPCResponse:
    """Create a JSON-RPC response containing a task.

    Args:
        task: The A2A task to include in the response.
        request_id: The request ID to echo back.

    Returns:
        A JSONRPCResponse with the task as result.
    """
    return JSONRPCResponse(
        result=task.model_dump(exclude_none=True),
        id=request_id,
    )


def create_error_response(
    code: int,
    message: str,
    request_id: str,
    data: Optional[dict[str, Any]] = None,
) -> JSONRPCResponse:
    """Create a JSON-RPC error response.

    Args:
        code: Error code (negative for server errors).
        message: Error message.
        request_id: The request ID to echo back.
        data: Optional additional error data.

    Returns:
        A JSONRPCResponse with the error.
    """
    return JSONRPCResponse(
        error={
            "code": code,
            "message": message,
            "data": data,
        },
        id=request_id,
    )


# =============================================================================
# Agent Card Factory
# =============================================================================


def create_shopper_agent_card(base_url: str) -> AgentCard:
    """Create the WebShop+ purple (shopper) agent card.

    Args:
        base_url: The base URL where the agent is hosted (e.g., http://localhost:8001).

    Returns:
        An AgentCard for the shopping agent.
    """
    return AgentCard(
        name="WebShop+ Shopper Agent",
        description="Baseline shopping agent that navigates WebShop to find and purchase "
        "products based on given instructions. Supports search, product comparison, "
        "and purchase decisions.",
        version="1.0.0",
        url=f"{base_url}/a2a",
        provider=AgentProvider(
            organization="WebShop+ Team",
            url="https://github.com/mpnikhil/webshop-plus",
        ),
        capabilities=AgentCapabilities(
            streaming=False,
            pushNotifications=False,
            stateTransitionHistory=False,
        ),
        skills=[
            AgentSkill(
                id="shopping",
                name="Product Shopping",
                description="Navigate WebShop to find and purchase products based on "
                "requirements, constraints, and preferences.",
                tags=["shopping", "search", "purchase", "comparison"],
                examples=[
                    "Find running shoes under $100",
                    "Buy a laptop with at least 16GB RAM",
                    "Find the cheapest wireless headphones",
                ],
                inputModes=["text/plain"],
                outputModes=["text/plain"],
            ),
        ],
    )


# =============================================================================
# Message Parsing Utilities
# =============================================================================


def get_message_text(message: SDKMessage) -> str:
    """Extract all text content from an A2A SDK Message.

    Args:
        message: An A2A SDK Message object.

    Returns:
        Concatenated text content from all text parts.
    """
    texts = []
    for part in message.parts:
        # SDK Part wraps the actual part type in a 'root' attribute
        actual_part = part.root if hasattr(part, "root") else part
        # Check if the actual part has a 'text' attribute (TextPart)
        if hasattr(actual_part, "text"):
            texts.append(actual_part.text)
    return "\n".join(texts)


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


def extract_task_instruction(message: dict[str, Any]) -> Optional[str]:
    """Extract task instruction from a message.

    Looks for TASK: prefix or returns the full text content.

    Args:
        message: An A2A message dictionary.

    Returns:
        The task instruction text, or None if not found.
    """
    text = get_text_from_message(message)
    if not text:
        return None

    # Look for explicit TASK: prefix
    lines = text.strip().split("\n")
    for line in lines:
        if line.upper().startswith("TASK:"):
            return line.split(":", 1)[1].strip()

    # Otherwise return the full text as the instruction
    return text.strip()


def extract_observation(message: dict[str, Any]) -> Optional[str]:
    """Extract observation from a message.

    Looks for OBSERVATION: prefix or returns the full text content.

    Args:
        message: An A2A message dictionary.

    Returns:
        The observation text, or None if not found.
    """
    text = get_text_from_message(message)
    if not text:
        return None

    # Look for explicit OBSERVATION: prefix
    if "OBSERVATION:" in text.upper():
        idx = text.upper().index("OBSERVATION:")
        return text[idx + len("OBSERVATION:") :].strip()

    # Check if it looks like WebShop output
    if any(
        keyword in text.lower()
        for keyword in ["products found", "product page", "search results", "price:", "options:"]
    ):
        return text.strip()

    return text.strip()


def format_action_response(action: str) -> str:
    """Format an action for the response.

    Args:
        action: The action string (e.g., "search[running shoes]").

    Returns:
        The formatted action string.
    """
    # Ensure the action is in the correct format
    action = action.strip()
    if not (action.startswith("search[") or action.startswith("click[")):
        # Try to detect intent
        action_lower = action.lower()
        if "search" in action_lower:
            # Extract search query
            if "search for" in action_lower:
                query = action_lower.split("search for", 1)[1].strip()
                return f"search[{query}]"
            elif "search" in action_lower:
                query = action_lower.split("search", 1)[1].strip()
                return f"search[{query}]"
        elif "click" in action_lower or "buy" in action_lower or "select" in action_lower:
            # Try to extract click target
            if "buy now" in action_lower:
                return "click[buy now]"
            elif "add to cart" in action_lower:
                return "click[add to cart]"
    return action
