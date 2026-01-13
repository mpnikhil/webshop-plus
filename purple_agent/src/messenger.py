"""
A2A Protocol messenger utilities for WebShop+ purple agent.

This module provides minimal utilities for A2A message text extraction.

Based on RDI Foundation agent-template pattern:
https://github.com/RDI-Foundation/agent-template
"""

from a2a.types import Message


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
