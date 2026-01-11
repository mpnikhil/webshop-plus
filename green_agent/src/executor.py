"""
A2A Request Executor for WebShop+ green agent.

This module handles routing and executing A2A requests to purple agents.
It provides:
- Request construction for task instructions and observations
- Response parsing to extract actions
- Error handling and retry logic
- Streaming support for real-time interaction
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Optional

import structlog

from .messenger import (
    A2AClient,
    A2AMessage,
    JSONRPCResponse,
    MessageRole,
    create_text_message,
    extract_action_from_text,
    parse_action_from_response,
)

logger = structlog.get_logger()


class ExecutorError(Exception):
    """Base exception for executor errors."""
    pass


class AgentTimeoutError(ExecutorError):
    """Agent did not respond in time."""
    pass


class AgentCommunicationError(ExecutorError):
    """Failed to communicate with agent."""
    pass


class InvalidActionError(ExecutorError):
    """Agent returned an invalid action."""
    pass


class RequestType(str, Enum):
    """Types of requests to send to purple agents."""

    TASK_INSTRUCTION = "task_instruction"
    OBSERVATION = "observation"
    ERROR_NOTICE = "error_notice"
    SESSION_END = "session_end"


@dataclass
class ExecutorConfig:
    """Configuration for the executor."""

    timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 1.0
    action_timeout: float = 60.0


@dataclass
class ExecutorResult:
    """Result from executing a request."""

    action: Optional[str] = None
    raw_response: Optional[JSONRPCResponse] = None
    full_text: str = ""
    error: Optional[str] = None
    timed_out: bool = False


class Executor:
    """
    Executor for A2A communication with purple agents.

    The Executor handles:
    - Sending task instructions to purple agents
    - Forwarding WebShop observations
    - Parsing agent actions from responses
    - Error recovery and retries
    """

    def __init__(
        self,
        config: Optional[ExecutorConfig] = None,
        client: Optional[A2AClient] = None,
    ):
        """
        Initialize the Executor.

        Args:
            config: Executor configuration.
            client: Optional A2A client to use (creates new one if not provided).
        """
        self.config = config or ExecutorConfig()
        self._client = client
        self._owned_client = False

    async def __aenter__(self) -> "Executor":
        """Async context manager entry."""
        if self._client is None:
            self._client = A2AClient(
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay,
            )
            self._owned_client = True
            await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._owned_client and self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
            self._client = None

    async def send_task_instruction(
        self,
        endpoint: str,
        instruction: str,
        task_id: str,
        context_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ExecutorResult:
        """
        Send a task instruction to a purple agent.

        Args:
            endpoint: The agent's A2A endpoint.
            instruction: The task instruction text.
            task_id: The task ID.
            context_id: The context ID for the conversation.
            metadata: Optional metadata to include.

        Returns:
            ExecutorResult with the agent's action or error.
        """
        message = create_text_message(
            text=f"TASK: {instruction}\n\nRespond with an action in the format: search[query] or click[element]",
            role=MessageRole.USER,
            task_id=task_id,
            context_id=context_id,
        )

        request_metadata = {
            "type": RequestType.TASK_INSTRUCTION.value,
            "task_id": task_id,
            **(metadata or {}),
        }

        return await self._execute_request(endpoint, message, request_metadata)

    async def send_observation(
        self,
        endpoint: str,
        observation: str,
        task_id: str,
        context_id: str,
        available_actions: Optional[dict[str, Any]] = None,
        reward: float = 0.0,
        done: bool = False,
    ) -> ExecutorResult:
        """
        Send an observation to a purple agent.

        Args:
            endpoint: The agent's A2A endpoint.
            observation: The observation from WebShop.
            task_id: The task ID.
            context_id: The context ID.
            available_actions: Optional dict of available actions.
            reward: The reward received.
            done: Whether the task is complete.

        Returns:
            ExecutorResult with the agent's next action or error.
        """
        # Build observation message
        parts = [f"OBSERVATION:\n{observation[:4000]}"]  # Truncate long observations

        if available_actions:
            clickables = available_actions.get("clickables", [])
            if clickables:
                parts.append(f"\nAvailable clickables: {clickables[:20]}")
            if available_actions.get("has_search_bar"):
                parts.append("Search bar is available.")

        if done:
            parts.append(f"\n[TASK COMPLETE - Reward: {reward:.2f}]")
        else:
            parts.append("\nRespond with your next action: search[query] or click[element]")

        text = "\n".join(parts)
        message = create_text_message(
            text=text,
            role=MessageRole.USER,
            task_id=task_id,
            context_id=context_id,
        )

        request_metadata = {
            "type": RequestType.OBSERVATION.value,
            "task_id": task_id,
            "reward": reward,
            "done": done,
        }

        return await self._execute_request(endpoint, message, request_metadata)

    async def send_error_notice(
        self,
        endpoint: str,
        error_message: str,
        task_id: str,
        context_id: str,
    ) -> ExecutorResult:
        """
        Notify the agent of an error.

        Args:
            endpoint: The agent's A2A endpoint.
            error_message: Description of the error.
            task_id: The task ID.
            context_id: The context ID.

        Returns:
            ExecutorResult (may not contain action).
        """
        message = create_text_message(
            text=f"ERROR: {error_message}\n\nPlease try a different action.",
            role=MessageRole.SYSTEM,
            task_id=task_id,
            context_id=context_id,
        )

        request_metadata = {
            "type": RequestType.ERROR_NOTICE.value,
            "task_id": task_id,
        }

        return await self._execute_request(endpoint, message, request_metadata)

    async def _execute_request(
        self,
        endpoint: str,
        message: A2AMessage,
        metadata: dict[str, Any],
    ) -> ExecutorResult:
        """
        Execute an A2A request and parse the response.

        Args:
            endpoint: The agent's A2A endpoint.
            message: The message to send.
            metadata: Request metadata.

        Returns:
            ExecutorResult with parsed action or error.
        """
        if not self._client:
            raise RuntimeError("Executor not initialized. Use 'async with' context.")

        result = ExecutorResult()

        try:
            # Send message with timeout
            response = await asyncio.wait_for(
                self._client.send_message(endpoint, message, metadata),
                timeout=self.config.action_timeout,
            )

            result.raw_response = response

            if response.error:
                result.error = f"Agent error: {response.error.get('message', 'Unknown error')}"
                logger.warning("Agent returned error", error=response.error)
                return result

            # Parse action from response
            action = parse_action_from_response(response)
            if action:
                result.action = action
                logger.debug("Parsed action from response", action=action)
            else:
                # Try to extract text for logging
                if response.result:
                    result.full_text = self._extract_response_text(response.result)
                    # Try one more time to find action in full text
                    action = extract_action_from_text(result.full_text)
                    if action:
                        result.action = action
                        logger.debug("Extracted action from full text", action=action)

            return result

        except asyncio.TimeoutError:
            result.timed_out = True
            result.error = f"Agent did not respond within {self.config.action_timeout}s"
            logger.warning("Agent timeout", endpoint=endpoint, timeout=self.config.action_timeout)
            return result

        except Exception as e:
            result.error = f"Communication error: {str(e)}"
            logger.error("Executor error", error=str(e), endpoint=endpoint)
            return result

    def _extract_response_text(self, result: dict[str, Any]) -> str:
        """Extract all text content from a response result."""
        texts = []

        # Check history
        for msg in result.get("history", []):
            if msg.get("role") == "agent":
                for part in msg.get("parts", []):
                    if part.get("kind") == "text":
                        texts.append(part.get("text", ""))

        # Check artifacts
        for artifact in result.get("artifacts", []):
            for part in artifact.get("parts", []):
                if part.get("kind") == "text":
                    texts.append(part.get("text", ""))

        # Check direct message
        if "message" in result:
            for part in result["message"].get("parts", []):
                if part.get("kind") == "text":
                    texts.append(part.get("text", ""))

        return "\n".join(texts)

    async def stream_interaction(
        self,
        endpoint: str,
        message: A2AMessage,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream an interaction with a purple agent.

        This is useful for long-running tasks where you want to see
        intermediate updates from the agent.

        Args:
            endpoint: The agent's A2A endpoint.
            message: The message to send.
            metadata: Optional metadata.

        Yields:
            SSE event data from the agent.
        """
        if not self._client:
            raise RuntimeError("Executor not initialized. Use 'async with' context.")

        try:
            async for event in self._client.stream_message(
                endpoint, message, metadata
            ):
                yield event
        except Exception as e:
            logger.error("Stream error", error=str(e), endpoint=endpoint)
            yield {"error": str(e)}


class MultiAgentExecutor:
    """
    Executor for managing multiple purple agents.

    This is used when an assessment involves multiple agents
    or when comparing agent performance.
    """

    def __init__(self, config: Optional[ExecutorConfig] = None):
        """
        Initialize the multi-agent executor.

        Args:
            config: Executor configuration.
        """
        self.config = config or ExecutorConfig()
        self._executors: dict[str, Executor] = {}
        self._client: Optional[A2AClient] = None

    async def __aenter__(self) -> "MultiAgentExecutor":
        """Async context manager entry."""
        self._client = A2AClient(
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
        )
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
            self._client = None

    def get_executor(self, agent_id: str) -> Executor:
        """
        Get or create an executor for an agent.

        Args:
            agent_id: The agent identifier.

        Returns:
            An Executor instance for the agent.
        """
        if agent_id not in self._executors:
            self._executors[agent_id] = Executor(
                config=self.config,
                client=self._client,
            )
        return self._executors[agent_id]

    async def send_to_all(
        self,
        endpoints: dict[str, str],
        instruction: str,
        task_id: str,
        context_id: str,
    ) -> dict[str, ExecutorResult]:
        """
        Send a task instruction to all agents.

        Args:
            endpoints: Dict mapping agent_id to endpoint URL.
            instruction: The task instruction.
            task_id: The task ID.
            context_id: The context ID.

        Returns:
            Dict mapping agent_id to ExecutorResult.
        """
        tasks = []
        agent_ids = []

        for agent_id, endpoint in endpoints.items():
            executor = self.get_executor(agent_id)
            tasks.append(
                executor.send_task_instruction(
                    endpoint, instruction, task_id, context_id
                )
            )
            agent_ids.append(agent_id)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            agent_id: (
                result
                if isinstance(result, ExecutorResult)
                else ExecutorResult(error=str(result))
            )
            for agent_id, result in zip(agent_ids, results)
        }
