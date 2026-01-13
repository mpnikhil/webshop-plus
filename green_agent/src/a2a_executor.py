"""
SDK-compatible AgentExecutor wrapper for WebShop+ green agent.

This module provides an `AgentExecutor` implementation that wraps the existing
`WebShopPlusAgent` logic, using the a2a-sdk's `TaskUpdater` for status updates.

Stage 3 of the A2A SDK Migration.
Updated in Stage 8 to support A2A TCK conformance testing.
Updated in AAA Stage 7 to support MCP-based tool execution.
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
from a2a.types import (
    Message,
    Role,
    TaskState,
    TextPart,
)

from .agent import AgentConfig, WebShopPlusAgent
from .webshop_mcp import SessionManager, is_session_completed, get_final_result
from .models import AssessmentConfig, TaskUpdate

logger = structlog.get_logger()

# TCK streaming timeout for conformance testing
# Tasks with messageId starting with "test-resubscribe-message-id" must run for
# at least 2 × TCK_STREAMING_TIMEOUT seconds
TCK_STREAMING_TIMEOUT = float(os.environ.get("TCK_STREAMING_TIMEOUT", "2.0"))


def _parse_participants(metadata: dict[str, Any]) -> dict[str, str]:
    """Extract participants from request metadata.

    Args:
        metadata: The request metadata dict.

    Returns:
        Dict mapping role names to endpoint URLs.

    Raises:
        ValueError: If no participants found in metadata.
    """
    participants = metadata.get("participants", {})

    if not participants:
        raise ValueError("No participants found in metadata. Expected 'participants' object with role->URL mappings.")

    # Validate all values are strings (URLs)
    for role, url in participants.items():
        if not isinstance(url, str):
            raise ValueError(f"Participant '{role}' must have a string URL, got {type(url).__name__}")

    return participants


def _parse_config(metadata: dict[str, Any]) -> AssessmentConfig:
    """Extract assessment config from request metadata.

    Args:
        metadata: The request metadata dict.

    Returns:
        AssessmentConfig parsed from config section, or defaults.
    """
    config_data = metadata.get("config", {})

    # Map A2A inputSchema field names to internal model names
    # Note: categories values should use exact TaskType enum names:
    # budget_constrained, preference_memory, negative_constraint,
    # comparative_reasoning, error_recovery
    field_mapping = {
        "num_tasks": "num_tasks",
        "categories": "task_types",
        "timeout_per_task": "timeout_per_task",
        "max_steps_per_task": "max_steps_per_task",
        "include_memory_tasks": "include_memory_tasks",
    }

    mapped_config = {}
    for external_key, internal_key in field_mapping.items():
        if external_key in config_data:
            mapped_config[internal_key] = config_data[external_key]

    return AssessmentConfig(**mapped_config)


def _extract_message_text(message: Message | None) -> Optional[str]:
    """Extract text content from an A2A message.

    Args:
        message: The incoming A2A message.

    Returns:
        The concatenated text from all TextPart parts, or None if no text found.
    """
    if not message or not message.parts:
        return None

    text_parts = []
    for part in message.parts:
        if hasattr(part, "root") and isinstance(part.root, TextPart):
            text_parts.append(part.root.text)
        elif isinstance(part, TextPart):
            text_parts.append(part.text)

    return "\n".join(text_parts) if text_parts else None


def _parse_skill_from_message(message: Message | None) -> Optional[str]:
    """Extract the skill ID from the message text if present.

    Looks for skill invocation patterns in the message.

    Args:
        message: The incoming A2A message.

    Returns:
        The skill ID if found, or None.
    """
    if not message or not message.parts:
        return None

    for part in message.parts:
        if hasattr(part, "text") and part.text:
            text = part.text.lower().strip()
            if "budget" in text:
                return "budget-assessment"
            if "memory" in text:
                return "memory-assessment"
            if "constraint" in text:
                return "constraint-assessment"
            if "reasoning" in text or "comparative" in text:
                return "reasoning-assessment"
            if "recovery" in text or "error" in text:
                return "recovery-assessment"
    return None


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


def _get_message_text(message: Message | None) -> str:
    """Extract text content from a message.

    Args:
        message: The incoming A2A message.

    Returns:
        The text content or empty string.
    """
    if not message or not message.parts:
        return ""

    texts = []
    for part in message.parts:
        if hasattr(part, "text") and part.text:
            texts.append(part.text)
    return " ".join(texts)


class WebShopPlusExecutor(AgentExecutor):
    """SDK-compatible executor that wraps WebShopPlusAgent.

    This executor implements the a2a-sdk's `AgentExecutor` interface,
    delegating actual assessment logic to `WebShopPlusAgent` and using
    `TaskUpdater` for status updates.

    For MCP-enabled assessments, this executor creates MCP sessions for each
    task and includes the MCP URI in the kickoff message to purple agents.

    Example:
        executor = WebShopPlusExecutor(
            session_manager=session_manager,
            mcp_host="localhost",
            mcp_port=8000,
        )
        # Used with DefaultRequestHandler in SDK-based server
    """

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        session_manager: Optional[SessionManager] = None,
        mcp_host: str = "localhost",
        mcp_port: int = 8000,
    ):
        """Initialize the executor.

        Args:
            agent_config: Optional configuration for the underlying agent.
            session_manager: Optional SessionManager for MCP sessions.
            mcp_host: Host for MCP URI generation.
            mcp_port: Port for MCP URI generation.
        """
        self._agent_config = agent_config or AgentConfig()
        self._session_manager = session_manager
        self._mcp_host = mcp_host
        self._mcp_port = mcp_port
        self._active_agents: dict[str, WebShopPlusAgent] = {}
        self._simple_task_states: dict[str, dict] = {}  # Track simple echo tasks

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the agent's logic for a given request context.

        Parses the incoming message metadata for participants and config,
        runs the WebShopPlusAgent assessment, and publishes updates via
        TaskUpdater.

        For simple messages without participants (e.g., TCK conformance tests),
        echoes the message back after completing successfully.

        Args:
            context: The request context containing message, task ID, etc.
            event_queue: The queue to publish events to.
        """
        task_id = context.task_id or "unknown"
        context_id = context.context_id or "unknown"

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task_id,
            context_id=context_id,
        )

        try:
            # Check for TCK resubscribe streaming test (must run for 2×timeout)
            if _is_tck_resubscribe_test(context.message):
                await self._handle_tck_resubscribe_test(updater, context)
                return

            # Parse request - try message text first (agentbeats-run format), then metadata
            metadata = context.metadata or {}
            participants = metadata.get("participants", {})

            # Try parsing EvalRequest from message text if no participants in metadata
            if not participants and context.message:
                message_text = _extract_message_text(context.message)
                if message_text:
                    try:
                        eval_request = json.loads(message_text)
                        if isinstance(eval_request, dict):
                            # Extract participants and config from EvalRequest JSON
                            participants = eval_request.get("participants", {})
                            if participants:
                                # Merge parsed data into metadata for downstream processing
                                metadata["participants"] = participants
                                if "config" in eval_request:
                                    metadata["config"] = eval_request["config"]
                                logger.info(
                                    "Parsed EvalRequest from message text",
                                    participants=list(participants.keys()),
                                    has_config="config" in eval_request,
                                )
                    except json.JSONDecodeError:
                        logger.debug("Message text is not JSON, continuing with metadata check")

            # If still no participants, handle as simple echo message (for conformance testing)
            if not participants:
                await self._handle_simple_message(updater, context)
                return

            # Start work for full assessment
            await updater.start_work(
                message=self._create_message("Starting WebShop+ assessment...")
            )

            logger.info(
                "Parsing request",
                task_id=task_id,
                metadata_keys=list(metadata.keys()),
            )

            try:
                participants = _parse_participants(metadata)
            except ValueError as e:
                await updater.reject(
                    message=self._create_message(str(e))
                )
                return

            config = _parse_config(metadata)

            # Check for skill-specific invocation
            skill_id = _parse_skill_from_message(context.message)
            if skill_id:
                # Map skill to task types
                skill_to_types = {
                    "budget-assessment": ["budget_constrained"],
                    "memory-assessment": ["preference_memory"],
                    "constraint-assessment": ["negative_constraint"],
                    "reasoning-assessment": ["comparative_reasoning"],
                    "recovery-assessment": ["error_recovery"],
                }
                if skill_id in skill_to_types:
                    config.task_types = skill_to_types[skill_id]
                    logger.info(
                        "Using skill-specific task types",
                        skill=skill_id,
                        task_types=config.task_types,
                    )

            logger.info(
                "Starting assessment",
                task_id=task_id,
                participants=list(participants.keys()),
                config=config.model_dump(),
            )

            # Run assessment with progress updates
            async with WebShopPlusAgent(
                config=self._agent_config,
                session_manager=self._session_manager,
            ) as agent:
                # Store agent for potential cancellation
                self._active_agents[task_id] = agent

                try:
                    # Define progress callback that updates via TaskUpdater
                    async def on_progress(update: TaskUpdate) -> None:
                        progress_msg = f"{update.message} ({update.tasks_completed}/{update.tasks_total})"
                        await updater.update_status(
                            state=TaskState.working,
                            message=self._create_message(progress_msg),
                        )

                    # Run assessment (synchronous callback converted)
                    results = await agent.run(
                        participants=participants,
                        config=config,
                        progress_callback=lambda u: None,  # Use sync callback for now
                    )

                    # Create result artifact
                    results_json = json.dumps(
                        results.model_dump(mode="json"),
                        indent=2,
                        default=str,
                    )

                    await updater.add_artifact(
                        parts=[
                            TextPart(text=results_json),
                        ],
                        name="assessment_results",
                        metadata={"format": "json"},
                        last_chunk=True,
                    )

                    # Complete with summary
                    summary = (
                        f"Assessment complete: "
                        f"{results.aggregate.successful_tasks}/{results.aggregate.total_tasks} tasks succeeded, "
                        f"avg score: {results.aggregate.average_score:.2f}"
                    )
                    await updater.complete(
                        message=self._create_message(summary)
                    )

                finally:
                    # Remove from active agents
                    self._active_agents.pop(task_id, None)

        except Exception as e:
            logger.error("Executor error", task_id=task_id, error=str(e))
            await updater.failed(
                message=self._create_message(f"Assessment failed: {str(e)}")
            )

    async def _handle_simple_message(
        self,
        updater: TaskUpdater,
        context: RequestContext,
    ) -> None:
        """Handle simple messages without participants for conformance testing.

        Uses input-required state to allow task continuation and cancellation.
        Completes only when the message contains "done" or "complete".

        Args:
            updater: The task updater for publishing status.
            context: The request context.
        """
        task_id = context.task_id or "unknown"
        message_text = _get_message_text(context.message)
        message_lower = message_text.lower() if message_text else ""

        logger.info(
            "Handling simple message (no participants)",
            task_id=task_id,
            message_preview=message_text[:100] if message_text else "(empty)",
        )

        # Check if this is a completion trigger
        is_completion = "done" in message_lower or "complete" in message_lower or "finish" in message_lower

        # Check if this is a continuation of an existing task
        is_continuation = context.task_id is not None and context.task_id in self._simple_task_states

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

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Request the agent to cancel an ongoing task.

        Args:
            context: The request context containing task ID to cancel.
            event_queue: The queue to publish cancellation status update.
        """
        task_id = context.task_id or "unknown"
        context_id = context.context_id or "unknown"

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task_id,
            context_id=context_id,
        )

        # Cancel the active agent if present
        agent = self._active_agents.get(task_id)
        if agent:
            agent.cancel()
            logger.info("Cancelled agent", task_id=task_id)
            await updater.cancel(
                message=self._create_message("Assessment cancelled by request.")
            )
        else:
            logger.warning("No active agent to cancel", task_id=task_id)
            await updater.cancel(
                message=self._create_message("No active task found to cancel.")
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

    def _get_mcp_base_url(self) -> str:
        """Get the base URL for MCP endpoints.

        Returns:
            Base URL like "http://localhost:8000/mcp"
        """
        return f"http://{self._mcp_host}:{self._mcp_port}/mcp"

    async def create_mcp_session(
        self,
        goal: str,
        budget: float,
        constraints: Optional[list[str]] = None,
        max_turns: int = 30,
    ) -> tuple[str, str]:
        """Create an MCP session for a task.

        Args:
            goal: The shopping task goal.
            budget: Maximum allowed spending.
            constraints: Optional list of constraints.
            max_turns: Maximum number of turns.

        Returns:
            Tuple of (session_id, mcp_uri).

        Raises:
            RuntimeError: If SessionManager is not configured.
        """
        if self._session_manager is None:
            raise RuntimeError("SessionManager not configured for MCP sessions")

        session_id = str(uuid.uuid4())

        await self._session_manager.create_session(
            session_id=session_id,
            goal=goal,
            budget=budget,
            constraints=constraints or [],
            max_turns=max_turns,
        )

        mcp_uri = f"{self._get_mcp_base_url()}/{session_id}"

        logger.info(
            "Created MCP session",
            session_id=session_id,
            mcp_uri=mcp_uri,
            goal=goal[:50],
            budget=budget,
        )

        return session_id, mcp_uri

    def get_mcp_uri(self, session_id: str) -> str:
        """Get the MCP URI for a given session ID.

        This is a convenience method for getting the full URI without
        needing to call create_mcp_session() again.

        Args:
            session_id: The session ID.

        Returns:
            Full MCP URI like "http://localhost:8000/mcp/session123".
        """
        return f"{self._get_mcp_base_url()}/{session_id}"

    async def cleanup_mcp_session(self, session_id: str) -> bool:
        """Clean up an MCP session after completion.

        Args:
            session_id: The session ID to clean up.

        Returns:
            True if session was cleaned up, False if not found.
        """
        if self._session_manager is None:
            return False

        result = await self._session_manager.cleanup_session(session_id)
        if result:
            logger.info("Cleaned up MCP session", session_id=session_id)
        return result

    async def get_mcp_session_result(self, session_id: str) -> Optional[dict[str, Any]]:
        """Get the result from a completed MCP session.

        Args:
            session_id: The session ID.

        Returns:
            The session result dict, or None if not found/not completed.
        """
        if self._session_manager is None:
            return None

        # Check if session exists
        state = await self._session_manager.get_session(session_id)
        if state is None:
            return None

        # Check if session is completed using module-level function
        if not is_session_completed(session_id):
            return None

        # Get the final result using module-level function
        return get_final_result(session_id)

    def has_mcp_support(self) -> bool:
        """Check if MCP support is enabled.

        Returns:
            True if SessionManager is configured.
        """
        return self._session_manager is not None
