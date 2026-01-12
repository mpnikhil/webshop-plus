"""
ADK-based Shopping Agent for WebShop+ Purple Agent.

This module implements an internal ADK agent that uses McpToolset to connect
to the green agent's MCP server for tool execution (search, click, checkout).

The agent uses a ReAct loop automatically managed by ADK, eliminating the need
for manual action parsing and observation handling.

Architecture:
    A2A Executor (executor.py)
           │
           ▼
    ShoppingAgent (this file)
           │
           ├── ADK Agent (LLM-based decisions)
           │
           └── McpToolset (connects to Green MCP server)
                   │
                   ▼
            Green MCP Server (search/click/checkout tools)
"""

import os
import uuid
from typing import Any, Optional

import structlog
from google.adk import Agent, Runner
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool.mcp_toolset import StreamableHTTPConnectionParams
from google.genai.types import Content, Part

logger = structlog.get_logger()


# =============================================================================
# Constants
# =============================================================================

# Default model for the ADK agent
# Can be overridden via environment variable
DEFAULT_MODEL = os.environ.get("ADK_MODEL", "gemini-2.0-flash")

# Maximum number of turns the agent can take before timing out
DEFAULT_MAX_TURNS = 30

# Connection timeout for MCP server
MCP_TIMEOUT = 30.0

# SSE read timeout for long-running tool calls
MCP_SSE_READ_TIMEOUT = 120.0


# =============================================================================
# Instruction Template
# =============================================================================

SHOPPING_INSTRUCTION = """You are a shopping assistant navigating an online store via MCP tools.

TASK: {goal}
BUDGET: ${budget}
CONSTRAINTS: {constraints}

AVAILABLE TOOLS:
- search(query: str): Search the store catalog. Returns products with element IDs.
- click(element_id: str): Click on an element by its ID from the previous observation.
- checkout(): Complete purchase. TERMINAL - this ends the session.

IMPORTANT RULES:
1. ONLY use element IDs from the most recent observation.
2. Do NOT guess or make up element IDs.
3. If a product looks good, click on it to see details.
4. If the product page has options (size, color), click to select them before adding to cart.
5. Call checkout() when you're ready to complete the purchase.
6. Stay within the budget constraint.
7. Follow any specific constraints provided.

STRATEGY:
1. Start by searching for products matching the task goal.
2. Review search results and click on promising products.
3. On product pages, evaluate if the product meets requirements.
4. Select any required options (size, color, etc.).
5. Add to cart if the product is suitable.
6. Call checkout() to complete the purchase.

Begin by searching for products that match your task."""


# =============================================================================
# ShoppingAgent Class
# =============================================================================


class ShoppingAgent:
    """
    Internal ADK agent for executing shopping tasks via MCP tools.

    This agent:
    - Connects to a green agent's MCP server dynamically
    - Uses ADK's built-in ReAct loop for decision-making
    - Executes shopping tasks using search/click/checkout tools

    Example:
        >>> agent = ShoppingAgent()
        >>> result = await agent.run(
        ...     mcp_uri="http://localhost:8000/mcp/session-123",
        ...     task_data={"goal": "Find running shoes under $50", "budget": 50.0, "constraints": []}
        ... )
    """

    def __init__(self, model: Optional[str] = None, max_turns: int = DEFAULT_MAX_TURNS):
        """Initialize the ShoppingAgent.

        Args:
            model: LLM model to use for decisions. Defaults to DEFAULT_MODEL.
            max_turns: Maximum number of turns before timeout. Defaults to DEFAULT_MAX_TURNS.
        """
        self._model = model or DEFAULT_MODEL
        self._max_turns = max_turns
        self._session_service = InMemorySessionService()
        logger.info(
            "ShoppingAgent initialized",
            model=self._model,
            max_turns=self._max_turns,
        )

    async def run(self, mcp_uri: str, task_data: dict[str, Any]) -> dict[str, Any]:
        """Run a shopping task using MCP tools.

        Args:
            mcp_uri: URI of the green agent's MCP server.
            task_data: Task information containing:
                - goal: str - The shopping task goal
                - budget: float - Maximum spending allowed
                - constraints: list[str] - List of constraints

        Returns:
            dict with keys:
                - success: bool - Whether the task completed successfully
                - final_message: str - Final response from the agent
                - turns_used: int - Number of turns taken
                - error: str (optional) - Error message if failed

        Raises:
            ValueError: If mcp_uri or goal is not provided.
        """
        if not mcp_uri:
            raise ValueError("mcp_uri is required")

        goal = task_data.get("goal")
        if not goal:
            raise ValueError("task_data must contain 'goal'")

        budget = task_data.get("budget", 100.0)
        constraints = task_data.get("constraints", [])
        session_id = task_data.get("session_id", str(uuid.uuid4()))

        logger.info(
            "ShoppingAgent.run() starting",
            mcp_uri=mcp_uri,
            goal=goal,
            budget=budget,
            constraints=constraints,
            session_id=session_id,
        )

        try:
            # Format the instruction with task details
            instruction = self._format_instruction(goal, budget, constraints)

            # Create MCP toolset connection
            mcp_toolset = McpToolset(
                connection_params=StreamableHTTPConnectionParams(
                    url=mcp_uri,
                    timeout=MCP_TIMEOUT,
                    sse_read_timeout=MCP_SSE_READ_TIMEOUT,
                )
            )

            # Create the ADK agent
            agent = Agent(
                name="shopping_assistant",
                model=self._model,
                instruction=instruction,
                tools=[mcp_toolset],
            )

            # Create runner with session service
            runner = Runner(
                app_name="webshop_assessment",
                agent=agent,
                session_service=self._session_service,
            )

            # Run the agent
            result = await self._execute_runner(
                runner=runner,
                session_id=session_id,
                goal=goal,
            )
            logger.info("ShoppingAgent.run() completed", result=result)
            return result

        except Exception as e:
            logger.exception("ShoppingAgent.run() failed", error=str(e))
            return {
                "success": False,
                "final_message": "",
                "turns_used": 0,
                "error": str(e),
            }

    def _format_instruction(
        self, goal: str, budget: float, constraints: list[str]
    ) -> str:
        """Format the agent instruction with task details.

        Args:
            goal: The shopping task goal.
            budget: Maximum spending allowed.
            constraints: List of constraints.

        Returns:
            Formatted instruction string.
        """
        constraint_text = ", ".join(constraints) if constraints else "none"
        return SHOPPING_INSTRUCTION.format(
            goal=goal,
            budget=budget,
            constraints=constraint_text,
        )

    async def _execute_runner(
        self,
        runner: Runner,
        session_id: str,
        goal: str,
    ) -> dict[str, Any]:
        """Execute the runner and collect results.

        Args:
            runner: The ADK Runner instance.
            session_id: Session ID for the run.
            goal: The task goal (used as initial message).

        Returns:
            Result dict with success, final_message, turns_used.
        """
        # Create initial message content
        initial_message = Content(
            parts=[Part(text=f"Please complete this shopping task: {goal}")],
            role="user",
        )

        final_message = ""
        turns_used = 0
        success = False

        # Run the agent and process events
        async for event in runner.run_async(
            user_id="assessment",
            session_id=session_id,
            new_message=initial_message,
        ):
            turns_used += 1

            # Check for max turns
            if turns_used >= self._max_turns:
                logger.warning(
                    "Max turns reached",
                    turns_used=turns_used,
                    max_turns=self._max_turns,
                )
                break

            # Process event
            if self._is_final_event(event):
                final_message = self._extract_message(event)
                success = not event.error_message
                logger.info(
                    "Final event received",
                    success=success,
                    final_message=final_message[:100] if final_message else "",
                )
                break

            # Log intermediate events
            if event.content:
                logger.debug(
                    "Intermediate event",
                    turn=turns_used,
                    content_preview=self._extract_message(event)[:100],
                )

        return {
            "success": success,
            "final_message": final_message,
            "turns_used": turns_used,
        }

    def _is_final_event(self, event: Event) -> bool:
        """Check if an event is the final response.

        Args:
            event: The ADK event.

        Returns:
            True if this is a final event.
        """
        # Check for turn completion
        if event.turn_complete:
            return True

        # Check for finish reason
        if event.finish_reason:
            return True

        # Check for error
        if event.error_message:
            return True

        return False

    def _extract_message(self, event: Event) -> str:
        """Extract text message from an event.

        Args:
            event: The ADK event.

        Returns:
            Extracted text message or empty string.
        """
        if event.error_message:
            return f"Error: {event.error_message}"

        if event.content and event.content.parts:
            texts = []
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    texts.append(part.text)
            return " ".join(texts)

        return ""

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model

    @property
    def max_turns(self) -> int:
        """Get the max turns limit."""
        return self._max_turns
