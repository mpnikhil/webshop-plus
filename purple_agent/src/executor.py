"""
AgentExecutor implementation for WebShop+ Purple Agent.

This module implements the A2A SDK AgentExecutor interface for:
1. MCP-based shopping tasks (via ADK ShoppingAgent)
2. TCK conformance testing (simple message echo, resubscribe tests)

Based on the RDI Foundation agent-template pattern:
https://github.com/RDI-Foundation/agent-template

AAA Architecture:
- Shopping tasks require MCP resource URI in kickoff message
- ShoppingAgent uses ADK + McpToolset for ReAct loop execution
- No legacy regex-based action parsing
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

from src.messenger import get_message_text
from src.shopping_agent import ShoppingAgent

logger = structlog.get_logger()

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


class Executor(AgentExecutor):
    """
    A2A AgentExecutor implementation for the shopping agent.

    This executor handles:
    1. MCP-based shopping tasks - Routes to ADK ShoppingAgent
    2. TCK conformance tests - Simple message echo and resubscribe tests

    Shopping tasks MUST include an MCP resource URI in the kickoff message.
    Messages without MCP resources are treated as TCK conformance tests.

    Example:
        >>> executor = Executor()
        >>> # Called by SDK's DefaultRequestHandler
        >>> await executor.execute(context, event_queue)
    """

    def __init__(self) -> None:
        """Initialize the executor."""
        self._simple_task_states: dict[str, dict] = {}  # Track TCK echo tasks
        self._shopping_agent = ShoppingAgent()  # ADK-based agent for MCP tasks

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute a task.

        Routes to:
        1. TCK resubscribe test handler (special messageId prefix)
        2. MCP shopping task handler (has MCP resource in message)
        3. Simple TCK message handler (fallback for conformance tests)

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

            # Check if this is an MCP-based shopping task
            # MCP-based tasks have a resources array with type="mcp"
            mcp_uri = self._extract_mcp_uri(context.message)
            if mcp_uri:
                await self._handle_mcp_task(updater, context, mcp_uri)
                return

            # No MCP resource - treat as simple TCK test message
            await self._handle_simple_message(updater, context)

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

    async def _handle_mcp_task(
        self,
        updater: TaskUpdater,
        context: RequestContext,
        mcp_uri: str,
    ) -> None:
        """Handle MCP-based shopping tasks using ADK ShoppingAgent.

        This method is called when the kickoff message contains an MCP resource URI.
        It extracts task data and delegates to the ShoppingAgent which uses ADK's
        ReAct loop with McpToolset to execute the shopping task.

        Args:
            updater: The task updater for publishing status.
            context: The request context containing the kickoff message.
            mcp_uri: The MCP server URI extracted from the kickoff message.

        AAA Stage 10: Wire executor to ShoppingAgent.
        """
        task_id = context.task_id or "unknown"

        logger.info(
            "Handling MCP-based shopping task",
            task_id=task_id,
            mcp_uri=mcp_uri,
        )

        # Extract task data from the kickoff message
        task_data = self._extract_task_data(context.message)
        if not task_data:
            logger.error("Failed to extract task data from MCP kickoff message")
            await updater.failed(
                message=self._create_message("Error: Invalid kickoff message - missing goal")
            )
            return

        # Add session ID for tracking
        task_data["session_id"] = task_id

        logger.info(
            "Extracted task data for MCP shopping",
            goal=task_data.get("goal"),
            budget=task_data.get("budget"),
            constraints=task_data.get("constraints"),
        )

        # Start work status
        await updater.start_work(
            message=self._create_message(f"Starting shopping task: {task_data.get('goal')}")
        )

        # Run the ADK ShoppingAgent
        result = None
        shopping_error = None

        try:
            result = await self._shopping_agent.run(mcp_uri, task_data)

            logger.info(
                "ShoppingAgent completed",
                success=result.get("success"),
                turns_used=result.get("turns_used"),
            )

        except Exception as e:
            logger.exception("ShoppingAgent raised exception", error=str(e))
            shopping_error = str(e)

        # Send result even if there was a cleanup error
        # (MCP client cleanup sometimes raises but task may have completed successfully)
        if result is not None:
            # Send result data as JSON for green agent to parse
            result_json = json.dumps({
                "success": result.get("success", False),
                "turns_used": result.get("turns_used", 0),
                "final_message": result.get("final_message", ""),
                "error": result.get("error"),
            })

            # Send the JSON result as the message content
            if result.get("success"):
                await updater.complete(
                    message=self._create_message(result_json)
                )
            else:
                await updater.failed(
                    message=self._create_message(result_json)
                )
        else:
            # No result due to error before completion
            await updater.failed(
                message=self._create_message(f"Shopping agent error: {shopping_error}")
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

    def clear_state(self) -> None:
        """Clear all cached state (for testing purposes)."""
        self._simple_task_states.clear()

    def get_shopping_agent(self) -> ShoppingAgent:
        """Get the ADK ShoppingAgent instance (for testing purposes).

        Returns:
            The ShoppingAgent instance used for MCP-based tasks.
        """
        return self._shopping_agent

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
