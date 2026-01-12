"""
SDK-based A2A client for communicating with purple agents.

This module provides a clean interface for sending tasks to purple agents
using the official a2a-sdk Client instead of custom HTTP code.

Stage 7c of the AAA (A2A + MCP Agentification) implementation.
"""

import json
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import structlog
from a2a.client import Client
from a2a.client.client import ClientConfig
from a2a.client.client_factory import ClientFactory
from a2a.client.helpers import create_text_message_object
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

logger = structlog.get_logger()


class PurpleAgentClientError(Exception):
    """Base exception for PurpleAgentClient errors."""
    pass


class ConnectionError(PurpleAgentClientError):
    """Failed to connect to the purple agent."""
    pass


class TaskError(PurpleAgentClientError):
    """Task execution failed on the purple agent."""
    pass


class TimeoutError(PurpleAgentClientError):
    """Task execution timed out."""
    pass


@dataclass
class TaskResult:
    """Result from a purple agent task execution.

    Attributes:
        success: Whether the task completed successfully.
        task_id: The A2A task ID.
        context_id: The A2A context ID.
        final_state: The final TaskState.
        result_data: Parsed result data if available.
        raw_task: The raw Task object from the SDK.
        error: Error message if the task failed.
    """

    success: bool
    task_id: str
    context_id: str
    final_state: TaskState
    result_data: Optional[dict[str, Any]] = None
    raw_task: Optional[Task] = None
    error: Optional[str] = None


class PurpleAgentClient:
    """Client for sending tasks to purple agents via A2A protocol.

    Uses the official a2a-sdk Client instead of custom HTTP code.
    Supports both MCP-enabled tasks (with resources) and legacy tasks.

    Example:
        async with PurpleAgentClient("http://localhost:8001") as client:
            result = await client.send_task(
                goal="Find running shoes under $50",
                budget=50.0,
                constraints=["waterproof"],
                mcp_uri="http://localhost:8000/mcp/session-123",
            )
            if result.success:
                print(f"Task completed: {result.result_data}")
    """

    def __init__(
        self,
        agent_url: str,
        timeout: float = 120.0,
    ):
        """Initialize the PurpleAgentClient.

        Args:
            agent_url: The purple agent's URL (e.g., "http://localhost:8001" or
                      "http://localhost:8001/a2a"). The /a2a suffix is stripped
                      since the SDK expects the base URL for agent card fetching.
            timeout: Request timeout in seconds.
        """
        # Strip /a2a suffix if present - SDK expects base URL for agent card
        url = agent_url.rstrip("/")
        if url.endswith("/a2a"):
            url = url[:-4]
        self.agent_url = url
        self.timeout = timeout
        self._client: Optional[Client] = None
        self._agent_card: Optional[AgentCard] = None

    async def __aenter__(self) -> "PurpleAgentClient":
        """Async context manager entry - connects to the agent."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - closes the connection."""
        await self.close()

    async def connect(self) -> None:
        """Connect to the purple agent.

        Fetches the agent card and creates the SDK client.
        Uses a custom httpx client with the configured timeout.

        Raises:
            ConnectionError: If connection fails.
        """
        try:
            logger.info("Connecting to purple agent", url=self.agent_url, timeout=self.timeout)

            # Create custom httpx client with configured timeout
            httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
            client_config = ClientConfig(httpx_client=httpx_client)

            self._client = await ClientFactory.connect(
                self.agent_url,
                client_config=client_config,
            )
            self._agent_card = await self._client.get_card()
            logger.info(
                "Connected to purple agent",
                name=self._agent_card.name,
                url=self.agent_url,
            )
        except Exception as e:
            logger.error("Failed to connect to purple agent", url=self.agent_url, error=str(e))
            raise ConnectionError(f"Failed to connect to {self.agent_url}: {e}") from e

    async def close(self) -> None:
        """Close the client connection."""
        self._client = None
        self._agent_card = None

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected."""
        return self._client is not None

    @property
    def agent_card(self) -> Optional[AgentCard]:
        """Get the connected agent's card."""
        return self._agent_card

    async def send_task(
        self,
        goal: str,
        budget: float,
        constraints: Optional[list[str]] = None,
        mcp_uri: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TaskResult:
        """Send a task to the purple agent.

        Creates a kickoff message with the goal, budget, constraints,
        and optional MCP resource URI. Waits for task completion.

        Args:
            goal: The shopping task goal.
            budget: Maximum allowed spending.
            constraints: Optional list of constraints.
            mcp_uri: Optional MCP server URI for tool execution.
            metadata: Optional additional metadata.

        Returns:
            TaskResult with success status and result data.

        Raises:
            ConnectionError: If not connected.
            TaskError: If task execution fails.
        """
        if not self._client:
            raise ConnectionError("Not connected. Call connect() first.")

        # Build kickoff payload
        kickoff = self._build_kickoff(goal, budget, constraints or [], mcp_uri)

        # Create message with kickoff as JSON text
        message = create_text_message_object(
            role=Role.user,
            content=json.dumps(kickoff),
        )

        # Merge additional metadata
        request_metadata = metadata or {}

        logger.info(
            "Sending task to purple agent",
            goal=goal[:50],
            budget=budget,
            has_mcp=mcp_uri is not None,
        )

        # Send message and collect events
        try:
            return await self._execute_task(message, request_metadata)
        except Exception as e:
            logger.error("Task execution failed", error=str(e))
            raise TaskError(f"Task execution failed: {e}") from e

    async def send_message(
        self,
        content: str,
        role: Role = Role.user,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TaskResult:
        """Send a raw message to the purple agent.

        Lower-level API for sending arbitrary messages.

        Args:
            content: The message text content.
            role: The message role (default: user).
            metadata: Optional request metadata.

        Returns:
            TaskResult with completion status.

        Raises:
            ConnectionError: If not connected.
            TaskError: If execution fails.
        """
        if not self._client:
            raise ConnectionError("Not connected. Call connect() first.")

        message = create_text_message_object(role=role, content=content)
        request_metadata = metadata or {}

        try:
            return await self._execute_task(message, request_metadata)
        except Exception as e:
            logger.error("Message send failed", error=str(e))
            raise TaskError(f"Message send failed: {e}") from e

    async def _execute_task(
        self,
        message: Message,
        request_metadata: dict[str, Any],
    ) -> TaskResult:
        """Execute a task by sending a message and collecting events.

        Args:
            message: The message to send.
            request_metadata: Request metadata dict.

        Returns:
            TaskResult with final state.
        """
        assert self._client is not None

        final_task: Optional[Task] = None
        final_state = TaskState.submitted
        error_message: Optional[str] = None
        result_data: Optional[dict[str, Any]] = None

        async for event in self._client.send_message(message, request_metadata=request_metadata):
            # Event is either tuple[Task, Update] or Message
            if isinstance(event, tuple):
                task, update = event
                final_task = task

                if update is not None:
                    if isinstance(update, TaskStatusUpdateEvent):
                        final_state = update.status.state
                        logger.debug(
                            "Task status update",
                            task_id=task.id,
                            state=final_state.value,
                        )

                        # Check for error in status message
                        if final_state == TaskState.failed:
                            error_message = self._extract_error_from_status(update.status)

                    elif isinstance(update, TaskArtifactUpdateEvent):
                        # Extract result from artifact
                        result_data = self._extract_result_from_artifact(update)

            elif isinstance(event, Message):
                # Direct message response (usually for simple agents)
                logger.debug("Received direct message response")
                result_data = self._extract_result_from_message(event)

        # Build final result
        task_id = final_task.id if final_task else ""
        context_id = final_task.context_id if final_task else ""

        success = final_state == TaskState.completed

        # If no result from artifacts, try to extract from task history
        if result_data is None and final_task and final_task.history:
            result_data = self._extract_result_from_history(final_task.history)

        return TaskResult(
            success=success,
            task_id=task_id,
            context_id=context_id,
            final_state=final_state,
            result_data=result_data,
            raw_task=final_task,
            error=error_message,
        )

    def _build_kickoff(
        self,
        goal: str,
        budget: float,
        constraints: list[str],
        mcp_uri: Optional[str],
    ) -> dict[str, Any]:
        """Build a kickoff message payload.

        Args:
            goal: The shopping task goal.
            budget: Maximum allowed spending.
            constraints: List of constraints.
            mcp_uri: Optional MCP server URI.

        Returns:
            Kickoff dict ready to be serialized to JSON.
        """
        kickoff: dict[str, Any] = {
            "goal": goal,
            "budget": budget,
            "constraints": constraints,
        }

        if mcp_uri:
            kickoff["resources"] = [
                {
                    "type": "mcp",
                    "uri": mcp_uri,
                    "description": "WebShop MCP server for search, click, and checkout tools",
                }
            ]

        return kickoff

    def _extract_result_from_artifact(
        self,
        update: TaskArtifactUpdateEvent,
    ) -> Optional[dict[str, Any]]:
        """Extract result data from an artifact update.

        Args:
            update: The artifact update event.

        Returns:
            Parsed result dict or None.
        """
        if not update.artifact or not update.artifact.parts:
            return None

        for part in update.artifact.parts:
            text = self._get_text_from_part(part)
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    # Not JSON, skip
                    pass

        return None

    def _extract_result_from_message(self, message: Message) -> Optional[dict[str, Any]]:
        """Extract result data from a direct message response.

        Args:
            message: The response message.

        Returns:
            Parsed result dict or None.
        """
        if not message.parts:
            return None

        for part in message.parts:
            text = self._get_text_from_part(part)
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    # Not JSON, skip
                    pass

        return None

    def _get_text_from_part(self, part: Any) -> Optional[str]:
        """Extract text from a Part or TextPart.

        The SDK uses Part(root=TextPart(...)) wrappers, while
        direct construction may use TextPart directly.

        Args:
            part: Either a Part wrapper or TextPart directly.

        Returns:
            The text content, or None if not a text part.
        """
        # Handle Part wrapper (SDK style)
        if isinstance(part, Part):
            inner = part.root
            if isinstance(inner, TextPart) and inner.text:
                return inner.text
        # Handle TextPart directly (test code style)
        elif isinstance(part, TextPart) and part.text:
            return part.text
        return None

    def _extract_error_from_status(self, status: TaskStatus) -> Optional[str]:
        """Extract error message from a TaskStatus.

        TaskStatus.message is a Message object, not a string.
        This method extracts the text content from it.

        Args:
            status: The TaskStatus with a potential error message.

        Returns:
            The error message text, or None if not available.
        """
        if not status.message:
            return None

        # status.message is a Message object
        if isinstance(status.message, Message):
            result = self._extract_result_from_message(status.message)
            if result:
                # If it parsed as JSON, stringify it
                return json.dumps(result)
            # Otherwise try to get raw text
            if status.message.parts:
                texts = []
                for part in status.message.parts:
                    text = self._get_text_from_part(part)
                    if text:
                        texts.append(text)
                if texts:
                    return " ".join(texts)

        return None

    def _extract_result_from_history(
        self,
        history: list[Message],
    ) -> Optional[dict[str, Any]]:
        """Extract result from task history (last agent message).

        Args:
            history: List of messages from task history.

        Returns:
            Parsed result dict or None.
        """
        # Look for the last agent message
        for message in reversed(history):
            if message.role == Role.agent:
                result = self._extract_result_from_message(message)
                if result:
                    return result

        return None
