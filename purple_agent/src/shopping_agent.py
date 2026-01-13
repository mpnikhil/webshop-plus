"""
ADK-based Shopping Agent for WebShop+ Purple Agent.

This module implements an internal ADK agent that uses McpToolset to connect
to the green agent's MCP server for tool execution (search, click, checkout).

The agent uses a ReAct loop automatically managed by ADK, eliminating the need
for manual action parsing and observation handling.

LLM Configuration:
    Uses LiteLLM via ADK's LiteLlm wrapper for provider-agnostic LLM access.
    - Local: openai/qwen3-coder-30b-a3b-instruct-mlx via LM Studio (localhost:1234)
    - Production: Can use any LiteLLM-supported model

Architecture:
    A2A Executor (executor.py)
           │
           ▼
    ShoppingAgent (this file)
           │
           ├── ADK Agent (LiteLLM-based decisions)
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
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool.mcp_toolset import StreamableHTTPConnectionParams
from google.genai.types import Content, Part

logger = structlog.get_logger()


# =============================================================================
# Constants
# =============================================================================

# Default model for the ADK agent using LiteLLM
# Uses openai/ prefix for LM Studio's OpenAI-compatible API
# Can be overridden via environment variable
# Use LLM_MODEL (from Docker) or ADK_MODEL (from local) or default model
DEFAULT_MODEL = os.environ.get("LLM_MODEL") or os.environ.get("ADK_MODEL", "openai/qwen3-coder-30b-a3b-instruct-mlx")

# LM Studio API base URL (for local development)
# LM Studio provides an OpenAI-compatible API
# Use LLM_API_BASE (from Docker) or OPENAI_API_BASE (from local) or default to localhost
OPENAI_API_BASE = os.environ.get("LLM_API_BASE") or os.environ.get("OPENAI_API_BASE", "http://localhost:1234/v1")
OPENAI_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "lm-studio")

# Maximum number of turns the agent can take before timing out
DEFAULT_MAX_TURNS = 30

# Connection timeout for MCP server
MCP_TIMEOUT = 30.0

# SSE read timeout for long-running tool calls
MCP_SSE_READ_TIMEOUT = 120.0


# =============================================================================
# Instruction Template
# =============================================================================

SHOPPING_INSTRUCTION = """You are a shopping assistant. Buy products using these tools:

TASK: {goal}
BUDGET: ${budget}

TOOLS:
- search(query): Returns products p1, p2, p3... with prices
- click(element_id): Click "p1" to view, "add_to_cart" to add
- checkout(): Complete purchase (MUST call this to finish)

REQUIRED STEPS (do exactly this):
1. search("relevant terms")
2. click("p1") - pick ANY product within budget
3. click("add_to_cart")
4. checkout()

IMPORTANT: The catalog may not have exact matches. Buy the best available product within budget. Do NOT keep searching - pick from what's shown and complete the purchase.

Start now:"""


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
                   For LM Studio, use 'openai/model_name' format.
                   For Ollama, use 'ollama_chat/model_name' format.
            max_turns: Maximum number of turns before timeout. Defaults to DEFAULT_MAX_TURNS.
        """
        self._model = model or DEFAULT_MODEL
        self._max_turns = max_turns
        self._session_service = InMemorySessionService()

        # Set API configuration based on model provider
        if self._model.startswith("openai/"):
            # LM Studio or OpenAI-compatible API
            os.environ.setdefault("OPENAI_API_BASE", OPENAI_API_BASE)
            os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)
            api_base = os.environ.get("OPENAI_API_BASE")
        elif self._model.startswith("ollama"):
            # Ollama API
            os.environ.setdefault("OLLAMA_API_BASE", "http://localhost:11434")
            api_base = os.environ.get("OLLAMA_API_BASE")
        else:
            api_base = None

        logger.info(
            "ShoppingAgent initialized",
            model=self._model,
            max_turns=self._max_turns,
            api_base=api_base,
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

        print(f"[DEBUG] ShoppingAgent.run() called with MCP URI: {mcp_uri}")
        print(f"[DEBUG] Goal: {goal}, Budget: {budget}")
        logger.info(
            "ShoppingAgent.run() starting",
            mcp_uri=mcp_uri,
            goal=goal,
            budget=budget,
            constraints=constraints,
            session_id=session_id,
        )
        print(f"[DEBUG] After logger.info")

        try:
            print(f"[DEBUG] In try block")
            # Format the instruction with task details
            instruction = self._format_instruction(goal, budget, constraints)
            print(f"[DEBUG] Instruction formatted")

            logger.info("Creating MCP toolset", mcp_uri=mcp_uri)
            print(f"[DEBUG] About to create McpToolset")

            # Create MCP toolset connection
            try:
                mcp_toolset = McpToolset(
                    connection_params=StreamableHTTPConnectionParams(
                        url=mcp_uri,
                        timeout=MCP_TIMEOUT,
                        sse_read_timeout=MCP_SSE_READ_TIMEOUT,
                    )
                )
                print(f"[DEBUG] McpToolset created")
            except Exception as e:
                print(f"[DEBUG] McpToolset creation failed: {e}")
                raise

            logger.info("MCP toolset created successfully")

            # Create the ADK agent with LiteLLM wrapper
            # This allows using Ollama/Qwen instead of Gemini
            logger.info("Creating LiteLLM model", model=self._model)
            print(f"[DEBUG] Creating LiteLlm model: {self._model}")
            try:
                llm_model = LiteLlm(model=self._model)
                print(f"[DEBUG] LiteLlm model created")
            except Exception as e:
                print(f"[DEBUG] LiteLlm creation failed: {e}")
                raise

            logger.info("Creating ADK Agent with MCP toolset")
            print(f"[DEBUG] Creating ADK Agent")
            try:
                agent = Agent(
                    name="shopping_assistant",
                    model=llm_model,
                    instruction=instruction,
                    tools=[mcp_toolset],
                )
                print(f"[DEBUG] ADK Agent created")
            except Exception as e:
                print(f"[DEBUG] Agent creation failed: {e}")
                raise

            # Create runner with session service
            # Note: app_name must be "agents" to match ADK's internal module structure
            # when using LiteLlm wrapper (Agent loaded from google.adk.agents)
            logger.info("Creating ADK Runner")
            print(f"[DEBUG] Creating ADK Runner")
            app_name = "agents"
            try:
                runner = Runner(
                    app_name=app_name,
                    agent=agent,
                    session_service=self._session_service,
                )
                print(f"[DEBUG] ADK Runner created")
            except Exception as e:
                print(f"[DEBUG] Runner creation failed: {e}")
                raise
            logger.info("ADK Runner created successfully")

            # Create session BEFORE running (required by ADK's InMemorySessionService)
            print(f"[DEBUG] Creating ADK session: {session_id}")
            try:
                await self._session_service.create_session(
                    app_name=app_name,
                    user_id="assessment",
                    session_id=session_id,
                )
                print(f"[DEBUG] ADK session created")
            except Exception as e:
                print(f"[DEBUG] Session creation failed: {e}")
                raise

            # Run the agent
            print(f"[DEBUG] Calling _execute_runner")
            result = await self._execute_runner(
                runner=runner,
                session_id=session_id,
                goal=goal,
            )
            print(f"[DEBUG] _execute_runner returned: {result}")
            logger.info("ShoppingAgent.run() completed", result=result)
            logger.info("Returning result to executor")
            return result

        except Exception as e:
            logger.exception("ShoppingAgent.run() failed", error=str(e))
            return {
                "success": False,
                "final_message": "",
                "turns_used": 0,
                "error": str(e),
            }
        finally:
            logger.info("ShoppingAgent.run() exiting (cleanup phase)")

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
        print(f"[DEBUG] _execute_runner called with goal: {goal[:50]}")
        # Create initial message content
        initial_message = Content(
            parts=[Part(text=f"Please complete this shopping task: {goal}")],
            role="user",
        )
        print(f"[DEBUG] Initial message created")

        final_message = ""
        turns_used = 0
        success = False

        # Run the agent and process events
        # ADK handles the entire ReAct loop automatically - we just collect events
        logger.info("Starting runner.run_async()", session_id=session_id)
        print(f"[DEBUG] About to start runner.run_async()")
        async for event in runner.run_async(
            user_id="assessment",
            session_id=session_id,
            new_message=initial_message,
        ):
            turns_used += 1

            # Log event details for debugging
            function_calls = event.get_function_calls()
            function_responses = event.get_function_responses()

            logger.info(
                "Event received",
                turn=turns_used,
                has_function_calls=bool(function_calls),
                num_function_calls=len(function_calls) if function_calls else 0,
                has_function_responses=bool(function_responses),
                num_function_responses=len(function_responses) if function_responses else 0,
                has_error=bool(event.error_message),
                is_final=event.is_final_response(),
            )

            # Check for max turns
            if turns_used >= self._max_turns:
                logger.warning(
                    "Max turns reached",
                    turns_used=turns_used,
                    max_turns=self._max_turns,
                )
                break

            # Check for errors
            if event.error_message:
                logger.error(
                    "Error event received",
                    error=event.error_message,
                )
                break

            # Use ADK's built-in method to identify final response
            # This returns True only when there are NO function calls/responses left
            if event.is_final_response():
                final_message = self._extract_message(event)
                success = True
                logger.info(
                    "Final response received",
                    final_message=final_message[:200] if final_message else "",
                    turns_used=turns_used,
                )
                break

        result = {
            "success": success,
            "final_message": final_message,
            "turns_used": turns_used,
        }

        logger.info(
            "Event loop completed",
            success=success,
            turns_used=turns_used,
            final_message_len=len(final_message) if final_message else 0,
        )

        return result

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
