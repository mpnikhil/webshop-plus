"""
A2A Protocol messenger utilities for WebShop+ green agent.

This module provides utilities for A2A (Agent-to-Agent) protocol communication:
- SSE (Server-Sent Events) event creation utilities
- Task state and artifact models

Note: The A2AClient class and legacy executor support were removed in Stage 7e.
Use PurpleAgentClient from purple_client.py for A2A communication.

Based on A2A Protocol v0.3.0 specification.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


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


# =============================================================================
# A2A Models
# =============================================================================


class Artifact(BaseModel):
    """An artifact produced by a task."""

    artifactId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    description: Optional[str] = None
    parts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# SSE Event Factory Functions
# =============================================================================


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
