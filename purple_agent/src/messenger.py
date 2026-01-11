"""
A2A Protocol messenger utilities for WebShop+ purple agent.

This module provides utilities for A2A (Agent-to-Agent) protocol communication:
- Message creation and text extraction utilities
- Messenger class for agent-to-agent communication
- Parsing utilities for task instructions and observations

Based on RDI Foundation agent-template pattern:
https://github.com/RDI-Foundation/agent-template
"""

import uuid
from typing import Any, Optional

import httpx
import structlog
from a2a.types import Message, Part, Role, TextPart

logger = structlog.get_logger()


# =============================================================================
# Message Utilities
# =============================================================================


def create_message(
    role: Role,
    text: str,
    context_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> Message:
    """Create an A2A Message with text content.

    Args:
        role: The role of the message sender (user, agent).
        text: The text content of the message.
        context_id: Optional context ID for conversation tracking.
        message_id: Optional message ID (generated if not provided).

    Returns:
        An A2A Message instance.

    Example:
        >>> msg = create_message(Role.agent, "search[running shoes]")
        >>> print(msg.parts[0].text)
        search[running shoes]
    """
    return Message(
        message_id=message_id or str(uuid.uuid4()),
        role=role,
        parts=[TextPart(text=text)],
        context_id=context_id,
    )


def get_message_text(message: Message) -> str:
    """Extract all text content from an A2A SDK Message.

    Args:
        message: An A2A SDK Message object.

    Returns:
        Concatenated text content from all text parts.

    Example:
        >>> text = get_message_text(message)
        >>> print(text)
        Find running shoes under $100
    """
    texts = []
    for part in message.parts:
        # SDK Part wraps the actual part type in a 'root' attribute
        actual_part = part.root if hasattr(part, "root") else part
        # Check if the actual part has a 'text' attribute (TextPart)
        if hasattr(actual_part, "text"):
            texts.append(actual_part.text)
    return "\n".join(texts)


def merge_parts(parts: list[Part]) -> str:
    """Merge message parts into a single string.

    Args:
        parts: List of A2A Part objects.

    Returns:
        Concatenated text content from all text parts.

    Example:
        >>> text = merge_parts(message.parts)
        >>> print(text)
        Combined text from all parts
    """
    texts = []
    for part in parts:
        actual_part = part.root if hasattr(part, "root") else part
        if hasattr(actual_part, "text"):
            texts.append(actual_part.text)
    return "\n".join(texts)


# =============================================================================
# A2A Communication
# =============================================================================


async def send_message(
    url: str,
    message: Message,
    context_id: Optional[str] = None,
) -> dict[str, Any]:
    """Send a message to another agent via A2A protocol.

    Args:
        url: The A2A endpoint URL (e.g., "http://localhost:8001/a2a").
        message: The message to send.
        context_id: Optional context ID for session tracking.

    Returns:
        The JSON response from the agent.

    Raises:
        httpx.HTTPError: If the HTTP request fails.

    Example:
        >>> response = await send_message("http://localhost:8001/a2a", message)
        >>> print(response["result"]["status"]["state"])
        completed
    """
    # Convert message to dict for JSON serialization
    message_dict: dict[str, Any] = {
        "messageId": message.message_id,
        "role": message.role.value if hasattr(message.role, "value") else str(message.role),
        "parts": [],
    }

    # Extract text from parts
    for p in message.parts:
        actual_part = p.root if hasattr(p, "root") else p
        if hasattr(actual_part, "text"):
            message_dict["parts"].append({"kind": "text", "text": actual_part.text})

    if context_id:
        message_dict["contextId"] = context_id

    payload = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {"message": message_dict},
        "id": str(uuid.uuid4()),
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()


# =============================================================================
# Messenger Class
# =============================================================================


class Messenger:
    """Helper class for agent-to-agent communication.

    Manages context IDs across multiple message exchanges with agents.

    Example:
        >>> messenger = Messenger()
        >>> response = await messenger.talk_to_agent("Find shoes", "http://localhost:8001/a2a")
        >>> print(response)
        search[shoes]
        >>> messenger.reset()
    """

    def __init__(self) -> None:
        """Initialize the messenger with empty context tracking."""
        self._context_ids: dict[str, str] = {}

    async def talk_to_agent(self, message: str, url: str) -> str:
        """Send a message to another agent and get the response.

        Maintains context ID per agent URL for multi-turn conversations.

        Args:
            message: The text message to send.
            url: The agent's A2A endpoint URL.

        Returns:
            The text response from the agent.

        Example:
            >>> response = await messenger.talk_to_agent("Find running shoes", "http://localhost:8001/a2a")
            >>> print(response)
            search[running shoes]
        """
        # Get or create context ID for this agent
        context_id = self._context_ids.get(url, str(uuid.uuid4()))
        self._context_ids[url] = context_id

        # Create and send message
        msg = create_message(Role.user, message, context_id=context_id)
        response = await send_message(url, msg, context_id)

        # Extract text from response
        result = response.get("result", {})

        # Try to get text from status message
        status = result.get("status", {})
        status_message = status.get("message", {})
        if status_message:
            parts = status_message.get("parts", [])
            for part in parts:
                if part.get("kind") == "text" or "text" in part:
                    return part.get("text", "")

        # Fallback: check history for agent messages
        history = result.get("history", [])
        for msg_entry in reversed(history):
            if msg_entry.get("role") == "agent":
                parts = msg_entry.get("parts", [])
                for part in parts:
                    if part.get("kind") == "text" or "text" in part:
                        return part.get("text", "")

        return ""

    def reset(self) -> None:
        """Clear stored context IDs.

        Use this to start fresh conversations with agents.
        """
        self._context_ids.clear()

    def get_context_id(self, url: str) -> Optional[str]:
        """Get the context ID for a given agent URL.

        Args:
            url: The agent's A2A endpoint URL.

        Returns:
            The context ID or None if not set.
        """
        return self._context_ids.get(url)


# =============================================================================
# Parsing Utilities (preserved from legacy for ShopperAgent)
# =============================================================================


def get_text_from_dict_message(message: dict[str, Any]) -> str:
    """Extract all text content from an A2A message dictionary.

    Args:
        message: An A2A message dictionary.

    Returns:
        Concatenated text content from all text parts.

    Example:
        >>> message = {"parts": [{"kind": "text", "text": "Hello"}]}
        >>> text = get_text_from_dict_message(message)
        >>> print(text)
        Hello
    """
    parts = message.get("parts", [])
    texts = []
    for part in parts:
        if part.get("kind") == "text":
            texts.append(part.get("text", ""))
    return "\n".join(texts)


def extract_task_instruction(text: str) -> Optional[str]:
    """Extract task instruction from text.

    Looks for TASK: prefix or returns the full text content.

    Args:
        text: The text to extract instruction from.

    Returns:
        The task instruction text, or None if not found.

    Example:
        >>> instruction = extract_task_instruction("TASK: Find running shoes under $100")
        >>> print(instruction)
        Find running shoes under $100
    """
    if not text:
        return None

    # Look for explicit TASK: prefix
    lines = text.strip().split("\n")
    for line in lines:
        if line.upper().startswith("TASK:"):
            return line.split(":", 1)[1].strip()

    # Otherwise return the full text as the instruction
    return text.strip()


def extract_observation(text: str) -> Optional[str]:
    """Extract observation from text.

    Looks for OBSERVATION: prefix or recognizes WebShop output patterns.

    Args:
        text: The text to extract observation from.

    Returns:
        The observation text, or None if not found.

    Example:
        >>> obs = extract_observation("OBSERVATION: Found 10 products")
        >>> print(obs)
        Found 10 products
    """
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

    Attempts to normalize action strings into the WebShop format
    (search[query] or click[element]).

    Args:
        action: The action string.

    Returns:
        The formatted action string.

    Example:
        >>> format_action_response("search for running shoes")
        'search[running shoes]'
        >>> format_action_response("search[running shoes]")
        'search[running shoes]'
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
