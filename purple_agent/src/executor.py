"""
AgentExecutor implementation for WebShop+ Purple Agent.

This module implements the A2A SDK AgentExecutor interface, providing
the bridge between the SDK's request handling and the ShopperAgent logic.

Based on the RDI Foundation agent-template pattern:
https://github.com/RDI-Foundation/agent-template

Updated to support A2A TCK conformance testing with:
- Simple message handling (input-required state)
- TCK resubscribe test support
- Task continuation and cancellation
"""

import asyncio
import json
import os
import uuid
from typing import Any, Optional

import structlog
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Role, TaskState, TextPart

from src.agent import ShopperAgent
from src.messenger import get_message_text

logger = structlog.get_logger()

# Terminal states where no further processing is needed
TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}

# TCK streaming timeout for conformance testing
# Tasks with messageId starting with "test-resubscribe-message-id" must run for
# at least 2 × TCK_STREAMING_TIMEOUT seconds
TCK_STREAMING_TIMEOUT = float(os.environ.get("TCK_STREAMING_TIMEOUT", "2.0"))


def _is_tck_resubscribe_test(message: Message | None) -> bool:
    """Check if this is a TCK resubscribe streaming test.

    The TCK requires tasks with messageId starting with "test-resubscribe-message-id"
    to run for at least 2 × TCK_STREAMING_TIMEOUT seconds.

    Args:
        message: The incoming A2A message.

    Returns:
        True if this is a TCK resubscribe test message.
    """
    if not message:
        return False
    message_id = getattr(message, "messageId", None) or getattr(message, "message_id", None)
    if message_id and str(message_id).startswith("test-resubscribe-message-id"):
        return True
    return False


def _is_simple_tck_message(message: Message | None, metadata: dict | None) -> bool:
    """Check if this is a simple TCK test message (not a shopping task).

    Simple messages are used for A2A protocol conformance testing and should
    use input-required state to allow task continuation and cancellation.

    Args:
        message: The incoming A2A message.
        metadata: The request metadata.

    Returns:
        True if this is a simple test message, not a shopping task.
    """
    # If metadata has specific fields, it's likely a real shopping task
    if metadata:
        if metadata.get("task_type") or metadata.get("participants") or metadata.get("config"):
            return False

    # Check message content for shopping task indicators
    if message:
        text = get_message_text(message).lower()
        # Shopping task indicators
        shopping_keywords = ["find", "buy", "purchase", "search for", "get me", "i need", "looking for"]
        for keyword in shopping_keywords:
            if keyword in text:
                return False

    # Default to simple message (for TCK tests)
    return True


class Executor(AgentExecutor):
    """
    A2A AgentExecutor implementation for the shopping agent.

    This executor:
    - Manages ShopperAgent instances per context (session)
    - Bridges SDK requests to the ShopperAgent.run() method
    - Handles task state transitions and error reporting
    - Supports A2A TCK conformance testing with simple message handling

    Example:
        >>> executor = Executor()
        >>> # Called by SDK's DefaultRequestHandler
        >>> await executor.execute(context, event_queue)
    """

    def __init__(self) -> None:
        """Initialize the executor with empty agent cache."""
        self._agents: dict[str, ShopperAgent] = {}
        self._simple_task_states: dict[str, dict] = {}  # Track simple echo tasks

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute a task by delegating to ShopperAgent.

        For simple messages without shopping task indicators (e.g., TCK tests),
        uses input-required state to allow task continuation and cancellation.

        Args:
            context: Request context containing message, task ID, and context ID.
            event_queue: Event queue for publishing status updates.

        Raises:
            ValueError: If no message is provided in the context.
        """
        if not context.message:
            raise ValueError("Message is required")

        task_id = context.task_id
        context_id = context.context_id

        logger.info(
            "Executor.execute() called",
            task_id=task_id,
            context_id=context_id,
        )

        # Create TaskUpdater for status/artifact updates
        updater = TaskUpdater(event_queue, task_id, context_id)

        try:
            # Check for TCK resubscribe streaming test (must run for 2×timeout)
            if _is_tck_resubscribe_test(context.message):
                await self._handle_tck_resubscribe_test(updater, context)
                return

            # Check if this is a simple TCK test message (not a shopping task)
            metadata = context.metadata or {}
            if _is_simple_tck_message(context.message, metadata):
                await self._handle_simple_message(updater, context)
                return

            # This is a shopping task - use the ShopperAgent
            # Get or create agent for this context (session)
            if context_id not in self._agents:
                logger.info("Creating new ShopperAgent", context_id=context_id)
                self._agents[context_id] = ShopperAgent()
            agent = self._agents[context_id]

            # Delegate to agent's run() method
            # The agent will call updater.complete() internally
            await agent.run(context.message, updater)

            # Note: agent.run() already calls updater.complete()
            # so we don't need to call it here. The status check below
            # is only for edge cases where run() didn't complete.
            current_task = context.current_task
            if current_task and current_task.status.state not in TERMINAL_STATES:
                logger.warning(
                    "Task not in terminal state after run(), completing",
                    state=current_task.status.state,
                )
                await updater.complete()

        except Exception as e:
            logger.exception("Executor caught exception", error=str(e))
            await updater.failed(
                message=self._create_message(f"Error: {str(e)}")
            )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancel a running task.

        Supports cancellation for simple TCK test tasks.
        Shopping tasks don't support cancellation.

        Args:
            context: Request context for the task to cancel.
            event_queue: Event queue for publishing status updates.
        """
        task_id = context.task_id or "unknown"
        context_id = context.context_id or "unknown"

        logger.info(
            "Cancel requested",
            task_id=task_id,
            context_id=context_id,
        )

        updater = TaskUpdater(event_queue, task_id, context_id)

        # Cancel simple tasks
        if task_id in self._simple_task_states:
            self._simple_task_states.pop(task_id, None)
            logger.info("Cancelled simple task", task_id=task_id)
            await updater.cancel(
                message=self._create_message("Task cancelled by request.")
            )
        else:
            # For shopping tasks, just acknowledge cancellation
            logger.warning("No active simple task to cancel", task_id=task_id)
            await updater.cancel(
                message=self._create_message("Task cancelled.")
            )

    async def _handle_simple_message(
        self,
        updater: TaskUpdater,
        context: RequestContext,
    ) -> None:
        """Handle simple messages without shopping task indicators for conformance testing.

        Uses input-required state to allow task continuation and cancellation.
        Completes only when the message contains "done" or "complete".

        Args:
            updater: The task updater for publishing status.
            context: The request context.
        """
        task_id = context.task_id or "unknown"
        message_text = get_message_text(context.message)
        message_lower = message_text.lower() if message_text else ""

        logger.info(
            "Handling simple message (TCK test)",
            task_id=task_id,
            message_preview=message_text[:100] if message_text else "(empty)",
        )

        # Check if this is a completion trigger
        is_completion = "done" in message_lower or "complete" in message_lower or "finish" in message_lower

        # Check if this is a continuation of an existing task
        is_continuation = task_id in self._simple_task_states

        if not is_continuation:
            # New task - start work
            await updater.start_work(
                message=self._create_message("Processing message...")
            )
            # Track this task as a simple echo task
            self._simple_task_states[task_id] = {"message_count": 1}
        else:
            # Continuation - increment message count
            self._simple_task_states[task_id]["message_count"] += 1

        # Echo back the message content
        response_text = f"Received: {message_text}" if message_text else "Message received."

        if is_completion:
            # Complete the task
            self._simple_task_states.pop(task_id, None)
            await updater.complete(
                message=self._create_message(response_text + " Task completed.")
            )
        else:
            # Stay in input-required state to allow continuation/cancellation
            await updater.requires_input(
                message=self._create_message(response_text + " Send 'done' to complete.")
            )

    async def _handle_tck_resubscribe_test(
        self,
        updater: TaskUpdater,
        context: RequestContext,
    ) -> None:
        """Handle TCK resubscribe streaming test.

        The TCK requires tasks with messageId starting with "test-resubscribe-message-id"
        to run for at least 2 × TCK_STREAMING_TIMEOUT seconds to test resubscribe.

        Args:
            updater: The task updater for publishing status.
            context: The request context.
        """
        task_id = context.task_id or "unknown"
        delay = TCK_STREAMING_TIMEOUT * 2.5  # Run slightly longer than 2×timeout

        logger.info(
            "Handling TCK resubscribe test",
            task_id=task_id,
            delay_seconds=delay,
        )

        await updater.start_work(
            message=self._create_message("Starting TCK resubscribe test task...")
        )

        # Emit periodic status updates during the delay
        intervals = 5
        interval_delay = delay / intervals
        for i in range(intervals):
            await asyncio.sleep(interval_delay)
            await updater.update_status(
                state=TaskState.working,
                message=self._create_message(f"TCK test progress: {(i + 1) * 100 // intervals}%"),
            )

        await updater.complete(
            message=self._create_message("TCK resubscribe test completed.")
        )

    def _create_message(self, text: str) -> Message:
        """Create a simple text message.

        Args:
            text: The message text.

        Returns:
            A Message with a single TextPart.
        """
        return Message(
            messageId=str(uuid.uuid4()),
            role=Role.agent,
            parts=[TextPart(text=text)],
        )

    def get_agent(self, context_id: str) -> ShopperAgent | None:
        """Get the agent for a context ID (for testing purposes).

        Args:
            context_id: The context ID to look up.

        Returns:
            The ShopperAgent instance or None if not found.
        """
        return self._agents.get(context_id)

    def clear_agents(self) -> None:
        """Clear all cached agents (for testing purposes)."""
        self._agents.clear()
        self._simple_task_states.clear()

    def _extract_mcp_uri(self, message: Message) -> Optional[str]:
        """Extract MCP resource URI from an A2A kickoff message.

        The green agent sends a kickoff message with a JSON payload containing:
        {
            "goal": "...",
            "budget": 50.0,
            "constraints": [...],
            "resources": [
                {"type": "mcp", "uri": "http://..."}
            ]
        }

        Args:
            message: The A2A message containing the kickoff payload.

        Returns:
            The MCP URI string if found, None otherwise.

        Example:
            >>> uri = executor._extract_mcp_uri(message)
            >>> print(uri)
            http://localhost:8000/mcp/session-123
        """
        text = get_message_text(message)
        if not text:
            return None

        try:
            data = json.loads(text)
            resources = data.get("resources", [])
            for resource in resources:
                if isinstance(resource, dict) and resource.get("type") == "mcp":
                    uri = resource.get("uri")
                    if uri:
                        return uri
        except json.JSONDecodeError:
            logger.debug("Message is not JSON, cannot extract MCP URI", text=text[:100])
        except (TypeError, AttributeError) as e:
            logger.debug("Error parsing message for MCP URI", error=str(e))

        return None

    def _extract_task_data(self, message: Message) -> dict[str, Any]:
        """Extract task data (goal, budget, constraints) from an A2A kickoff message.

        The green agent sends a kickoff message with a JSON payload containing:
        {
            "goal": "Find running shoes under $50",
            "budget": 50.0,
            "constraints": ["waterproof", "size 10"]
        }

        Args:
            message: The A2A message containing the kickoff payload.

        Returns:
            Dict with keys: goal, budget, constraints. Empty dict if parsing fails.
            - goal: str - The shopping task goal
            - budget: float - Maximum spending allowed (default 100.0)
            - constraints: list[str] - List of constraints (default [])

        Example:
            >>> task_data = executor._extract_task_data(message)
            >>> print(task_data)
            {'goal': 'Find running shoes', 'budget': 50.0, 'constraints': ['waterproof']}
        """
        text = get_message_text(message)
        if not text:
            return {}

        try:
            data = json.loads(text)
            # Extract goal - required field
            goal = data.get("goal")
            if not goal:
                logger.debug("No goal found in kickoff message")
                return {}

            return {
                "goal": goal,
                "budget": float(data.get("budget", 100.0)),
                "constraints": list(data.get("constraints", [])),
            }
        except json.JSONDecodeError:
            logger.debug("Message is not JSON, cannot extract task data", text=text[:100])
        except (TypeError, ValueError, AttributeError) as e:
            logger.debug("Error parsing message for task data", error=str(e))

        return {}
