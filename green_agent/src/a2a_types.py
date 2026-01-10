"""
A2A SDK Type Compatibility Layer.

This module provides a bridge between the a2a-sdk types and the existing
messenger.py types. It re-exports SDK types for use in the new SDK-based
server implementation while allowing the legacy code to continue working.

Stage 1 of the A2A SDK Migration - establishes SDK dependency and type mappings.
"""

# =============================================================================
# SDK Server Components
# =============================================================================
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue

# =============================================================================
# SDK Type Imports
# =============================================================================
from a2a.types import (
    # Agent Card types
    AgentCard as SDKAgentCard,
    AgentCapabilities as SDKAgentCapabilities,
    AgentSkill as SDKAgentSkill,
    AgentProvider as SDKAgentProvider,

    # Message types
    Message as SDKMessage,
    Part as SDKPart,
    TextPart as SDKTextPart,
    DataPart as SDKDataPart,
    FilePart as SDKFilePart,
    FileWithBytes as SDKFileWithBytes,
    FileWithUri as SDKFileWithUri,

    # Task types
    Task as SDKTask,
    TaskStatus as SDKTaskStatus,
    TaskState as SDKTaskState,

    # Artifact types
    Artifact as SDKArtifact,

    # JSON-RPC types
    JSONRPCRequest as SDKJSONRPCRequest,
    JSONRPCResponse as SDKJSONRPCResponse,
    JSONRPCError as SDKJSONRPCError,

    # Other useful types
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
    PushNotificationConfig,
)

# =============================================================================
# Type Aliases for Migration
# =============================================================================

# These aliases allow gradual migration from custom types to SDK types.
# In Stage 7 (cleanup), these will be removed and imports will use SDK types directly.

# Server components (new - no legacy equivalent)
StarlettteApp = A2AStarletteApplication
RequestHandler = DefaultRequestHandler
TaskStore = InMemoryTaskStore

# Task state enum - SDK uses lowercase, will need mapping in usage
TaskState = SDKTaskState

# Agent card types - functionally equivalent
AgentCard = SDKAgentCard
AgentCapabilities = SDKAgentCapabilities
AgentSkill = SDKAgentSkill
AgentProvider = SDKAgentProvider

# Message types
Message = SDKMessage
TextPart = SDKTextPart
FilePart = SDKFilePart

# Task types
Task = SDKTask
TaskStatus = SDKTaskStatus

# Artifact
Artifact = SDKArtifact

# JSON-RPC types
JSONRPCRequest = SDKJSONRPCRequest
JSONRPCResponse = SDKJSONRPCResponse
JSONRPCError = SDKJSONRPCError

# =============================================================================
# SDK TaskState Value Mapping
# =============================================================================

# The SDK uses lowercase enum values (e.g., TaskState.submitted)
# while legacy code uses UPPERCASE (e.g., TaskState.SUBMITTED)
# This mapping helps during transition:

LEGACY_TO_SDK_STATE = {
    "submitted": SDKTaskState.submitted,
    "working": SDKTaskState.working,
    "input_required": SDKTaskState.input_required,
    "input-required": SDKTaskState.input_required,  # legacy format
    "auth_required": SDKTaskState.auth_required,
    "auth-required": SDKTaskState.auth_required,  # legacy format
    "completed": SDKTaskState.completed,
    "failed": SDKTaskState.failed,
    "canceled": SDKTaskState.canceled,
    "rejected": SDKTaskState.rejected,
}

def to_sdk_state(state_str: str) -> SDKTaskState:
    """Convert a state string to SDK TaskState enum.

    Args:
        state_str: State as string (e.g., "working", "completed")

    Returns:
        The corresponding SDKTaskState enum value.

    Raises:
        KeyError: If the state string is not recognized.
    """
    return LEGACY_TO_SDK_STATE[state_str.lower().replace("_", "-")]


__all__ = [
    # Server components
    "A2AStarletteApplication",
    "DefaultRequestHandler",
    "InMemoryTaskStore",
    "AgentExecutor",
    "EventQueue",
    "RequestContext",
    "TaskUpdater",

    # Aliases
    "StarlettteApp",
    "RequestHandler",
    "TaskStore",

    # Types
    "AgentCard",
    "AgentCapabilities",
    "AgentSkill",
    "AgentProvider",
    "Message",
    "TextPart",
    "FilePart",
    "Task",
    "TaskStatus",
    "TaskState",
    "Artifact",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "JSONRPCError",

    # SDK-prefixed types (explicit SDK types)
    "SDKAgentCard",
    "SDKAgentCapabilities",
    "SDKAgentSkill",
    "SDKAgentProvider",
    "SDKMessage",
    "SDKPart",
    "SDKTextPart",
    "SDKDataPart",
    "SDKFilePart",
    "SDKFileWithBytes",
    "SDKFileWithUri",
    "SDKTask",
    "SDKTaskStatus",
    "SDKTaskState",
    "SDKArtifact",
    "SDKJSONRPCRequest",
    "SDKJSONRPCResponse",
    "SDKJSONRPCError",

    # Request types
    "MessageSendParams",
    "SendMessageRequest",
    "SendStreamingMessageRequest",
    "PushNotificationConfig",

    # Utilities
    "to_sdk_state",
    "LEGACY_TO_SDK_STATE",
]
