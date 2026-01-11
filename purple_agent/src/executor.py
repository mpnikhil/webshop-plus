"""
AgentExecutor implementation for WebShop+ Purple Agent.

This module implements the A2A SDK AgentExecutor interface, providing
the bridge between the SDK's request handling and the ShopperAgent logic.

Based on the RDI Foundation agent-template pattern:
https://github.com/RDI-Foundation/agent-template
"""

import uuid

import structlog
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Role, TaskState, TextPart, UnsupportedOperationError
from a2a.utils.errors import ServerError

from src.agent import ShopperAgent

logger = structlog.get_logger()

# Terminal states where no further processing is needed
TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class Executor(AgentExecutor):
    """
    A2A AgentExecutor implementation for the shopping agent.

    This executor:
    - Manages ShopperAgent instances per context (session)
    - Bridges SDK requests to the ShopperAgent.run() method
    - Handles task state transitions and error reporting

    Example:
        >>> executor = Executor()
        >>> # Called by SDK's DefaultRequestHandler
        >>> await executor.execute(context, event_queue)
    """

    def __init__(self) -> None:
        """Initialize the executor with empty agent cache."""
        self._agents: dict[str, ShopperAgent] = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute a task by delegating to ShopperAgent.

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

        # Get or create agent for this context (session)
        # Each context_id maps to a separate ShopperAgent instance
        if context_id not in self._agents:
            logger.info("Creating new ShopperAgent", context_id=context_id)
            self._agents[context_id] = ShopperAgent()
        agent = self._agents[context_id]

        # Create TaskUpdater for status/artifact updates
        updater = TaskUpdater(event_queue, task_id, context_id)

        try:
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
                message=Message(
                    messageId=str(uuid.uuid4()),
                    role=Role.agent,
                    parts=[TextPart(text=f"Error: {str(e)}")],
                )
            )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancel a running task.

        Currently not supported by the shopping agent.

        Args:
            context: Request context for the task to cancel.
            event_queue: Event queue for publishing status updates.

        Raises:
            ServerError: Always, wrapping UnsupportedOperationError.
        """
        logger.info(
            "Cancel requested (not supported)",
            task_id=context.task_id,
        )
        raise ServerError(error=UnsupportedOperationError(message="Cancel operation is not supported"))

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
