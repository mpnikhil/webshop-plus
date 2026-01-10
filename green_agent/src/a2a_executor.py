"""
SDK-compatible AgentExecutor wrapper for WebShop+ green agent.

This module provides an `AgentExecutor` implementation that wraps the existing
`WebShopPlusAgent` logic, using the a2a-sdk's `TaskUpdater` for status updates.

Stage 3 of the A2A SDK Migration.
"""

import json
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
from .models import AssessmentConfig, TaskUpdate

logger = structlog.get_logger()


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
    field_mapping = {
        "num_tasks": "num_tasks",
        "categories": "task_types",
        "timeout_per_task": "timeout_per_task",
        "max_steps_per_task": "max_steps_per_task",
    }

    mapped_config = {}
    for external_key, internal_key in field_mapping.items():
        if external_key in config_data:
            mapped_config[internal_key] = config_data[external_key]

    return AssessmentConfig(**mapped_config)


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


class WebShopPlusExecutor(AgentExecutor):
    """SDK-compatible executor that wraps WebShopPlusAgent.

    This executor implements the a2a-sdk's `AgentExecutor` interface,
    delegating actual assessment logic to `WebShopPlusAgent` and using
    `TaskUpdater` for status updates.

    Example:
        executor = WebShopPlusExecutor()
        # Used with DefaultRequestHandler in SDK-based server
    """

    def __init__(self, agent_config: Optional[AgentConfig] = None):
        """Initialize the executor.

        Args:
            agent_config: Optional configuration for the underlying agent.
        """
        self._agent_config = agent_config or AgentConfig()
        self._active_agents: dict[str, WebShopPlusAgent] = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the agent's logic for a given request context.

        Parses the incoming message metadata for participants and config,
        runs the WebShopPlusAgent assessment, and publishes updates via
        TaskUpdater.

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
            # Start work
            await updater.start_work(
                message=self._create_message("Starting WebShop+ assessment...")
            )

            # Parse request
            metadata = context.metadata
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
            async with WebShopPlusAgent(config=self._agent_config) as agent:
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
