"""
WebShop+ Assessment Orchestration Agent.

This module provides the main orchestration logic for running assessments:
- WebShopPlusAgent: Main agent class that orchestrates task execution
- Task dispatch: single kickoff with PurpleAgentClient + MCP tools
- Result aggregation and reporting
"""

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Optional, TYPE_CHECKING

import structlog

from .evaluator import Evaluator
from .llm_client import LLMClient
from .messenger import (
    Artifact,
    TaskState,
    create_artifact_update_event,
    create_status_update_event,
)
from .models import (
    AgentMemory,
    AssessmentConfig,
    AssessmentResults,
    BudgetConstrainedTask,
    ErrorRecoveryTask,
    EvaluationResult,
    NegativeConstraintTask,
    PreferenceMemoryTask,
    Task,
    TaskType,
    TaskUpdate,
)
from .purple_client import PurpleAgentClient, TaskResult, PurpleAgentClientError
from .task_generator import TaskGenerator
from .webshop_mcp.session_state import SessionState as MCPSessionState
from .webshop_wrapper import WebShopWrapper

if TYPE_CHECKING:
    from .webshop_mcp import SessionManager

logger = structlog.get_logger()


@dataclass
class AgentConfig:
    """Configuration for the WebShop+ agent."""

    max_actions_per_task: int = 30
    task_timeout_seconds: float = 300.0
    action_timeout_seconds: float = 60.0
    max_retries_per_action: int = 3
    use_llm_evaluation: bool = True
    webshop_mode: str = "preview"
    # MCP configuration
    mcp_host: str = "localhost"
    mcp_port: int = 8000
    default_budget: float = 100.0  # Default budget when not specified in task


@dataclass
class TaskExecutionResult:
    """Result from executing a single task."""

    task_id: str
    session_id: str
    completed: bool = False
    total_reward: float = 0.0
    actions_taken: int = 0
    evaluation: Optional[EvaluationResult] = None
    error: Optional[str] = None
    timed_out: bool = False


class WebShopPlusAgent:
    """
    Main orchestration agent for WebShop+ assessments.

    This agent coordinates:
    - Task selection and loading from TaskGenerator
    - Session management via SessionManager
    - WebShop environment interaction via WebShopWrapper
    - A2A communication with purple agents via PurpleAgentClient
    - MCP-based tool execution via SessionManager
    - Scoring via Evaluator

    Example:
        async with WebShopPlusAgent() as agent:
            results = await agent.run(
                participants={"shopper": "http://agent:8001/a2a"},
                config=AssessmentConfig(num_tasks=10),
            )
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        task_generator: Optional[TaskGenerator] = None,
        webshop: Optional[WebShopWrapper] = None,
        evaluator: Optional[Evaluator] = None,
        llm_client: Optional[LLMClient] = None,
        session_manager: Optional["SessionManager"] = None,
    ):
        """
        Initialize the WebShopPlusAgent.

        Args:
            config: Agent configuration.
            task_generator: Task generator instance (creates default if None).
            webshop: WebShop wrapper instance (creates default if None).
            evaluator: Evaluator instance (creates default if None).
            llm_client: LLM client for evaluation (creates default if None).
            session_manager: MCP session manager for tool execution.
        """
        self.config = config or AgentConfig()

        # Components - lazy initialization
        self._task_generator = task_generator
        self._webshop = webshop
        self._evaluator = evaluator
        self._llm_client = llm_client
        self._session_manager = session_manager

        # Runtime state
        self._initialized = False
        self._canceled = False

    @property
    def task_generator(self) -> TaskGenerator:
        if self._task_generator is None:
            self._task_generator = TaskGenerator()
        return self._task_generator

    @property
    def webshop(self) -> WebShopWrapper:
        if self._webshop is None:
            self._webshop = WebShopWrapper(mode=self.config.webshop_mode)
        return self._webshop

    @property
    def evaluator(self) -> Evaluator:
        if self._evaluator is None:
            llm = self._llm_client if self.config.use_llm_evaluation else None
            self._evaluator = Evaluator(llm_client=llm)
        return self._evaluator

    async def __aenter__(self) -> "WebShopPlusAgent":
        """Async context manager entry."""
        self._initialized = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self._initialized = False

    def cancel(self) -> None:
        """Cancel the current assessment."""
        self._canceled = True

    async def run(
        self,
        participants: dict[str, str],
        config: Optional[AssessmentConfig] = None,
        progress_callback: Optional[Callable[[TaskUpdate], None]] = None,
    ) -> AssessmentResults:
        """
        Run a complete assessment.

        Args:
            participants: Dict mapping role to endpoint URL (e.g., {"shopper": "http://..."}).
            config: Assessment configuration.
            progress_callback: Optional callback for progress updates.

        Returns:
            AssessmentResults with all evaluation results.
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Use 'async with' context.")

        self._canceled = False
        config = config or AssessmentConfig()
        assessment_id = str(uuid.uuid4())

        logger.info(
            "Starting assessment",
            assessment_id=assessment_id,
            participants=participants,
            config=config.model_dump(),
        )

        # Get shopper endpoint (main participant)
        shopper_endpoint = participants.get("shopper") or participants.get("agent")
        if not shopper_endpoint:
            # Try to get first endpoint
            shopper_endpoint = next(iter(participants.values()), None)

        if not shopper_endpoint:
            return AssessmentResults(
                assessment_id=assessment_id,
                participants=participants,
                config=config,
                results=[],
            )

        # Select tasks based on config
        tasks = self._select_tasks(config)
        total_tasks = len(tasks)

        logger.info(
            "Selected tasks for assessment",
            total_tasks=total_tasks,
            task_types=[t.task_type.value for t in tasks[:5]],  # Log first 5
        )

        results: list[EvaluationResult] = []

        for i, task in enumerate(tasks):
            if self._canceled:
                logger.info("Assessment canceled", completed=i, total=total_tasks)
                break

            # Send progress update
            if progress_callback:
                progress_callback(
                    TaskUpdate(
                        task_id=assessment_id,
                        status="working",
                        current_task=task.task_id,
                        tasks_completed=i,
                        tasks_total=total_tasks,
                        message=f"Processing task {i + 1}/{total_tasks}: {task.task_type.value}",
                        progress=i / total_tasks,
                    )
                )

            # Execute the task
            execution_result = await self._execute_task(
                task=task,
                shopper_endpoint=shopper_endpoint,
                agent_id=list(participants.keys())[0] if participants else "unknown",
            )

            if execution_result.evaluation:
                results.append(execution_result.evaluation)

            logger.info(
                "Task completed",
                task_id=task.task_id,
                task_type=task.task_type.value,
                success=execution_result.evaluation.success if execution_result.evaluation else False,
                score=execution_result.evaluation.overall_score if execution_result.evaluation else 0,
                actions=execution_result.actions_taken,
            )

        # Build final results
        assessment_results = AssessmentResults(
            assessment_id=assessment_id,
            participants=participants,
            config=config,
            results=results,
        )
        assessment_results.calculate_aggregate()

        # Final progress update
        if progress_callback:
            progress_callback(
                TaskUpdate(
                    task_id=assessment_id,
                    status="completed",
                    tasks_completed=len(results),
                    tasks_total=total_tasks,
                    message="Assessment complete",
                    progress=1.0,
                )
            )

        logger.info(
            "Assessment completed",
            assessment_id=assessment_id,
            total_tasks=assessment_results.aggregate.total_tasks,
            successful=assessment_results.aggregate.successful_tasks,
            avg_score=assessment_results.aggregate.average_score,
        )

        return assessment_results

    async def run_streaming(
        self,
        participants: dict[str, str],
        config: Optional[AssessmentConfig] = None,
        task_id: str = "",
        context_id: str = "",
        request_id: str = "",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Run assessment with streaming updates.

        Args:
            participants: Dict mapping role to endpoint URL.
            config: Assessment configuration.
            task_id: Task ID for SSE events.
            context_id: Context ID for SSE events.
            request_id: Request ID for SSE events.

        Yields:
            SSE event dictionaries.
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Use 'async with' context.")

        self._canceled = False
        config = config or AssessmentConfig()
        assessment_id = task_id or str(uuid.uuid4())
        context_id = context_id or str(uuid.uuid4())

        # Get shopper endpoint
        shopper_endpoint = participants.get("shopper") or participants.get("agent")
        if not shopper_endpoint:
            shopper_endpoint = next(iter(participants.values()), None)

        if not shopper_endpoint:
            yield create_status_update_event(
                task_id=assessment_id,
                context_id=context_id,
                state=TaskState.FAILED,
                message="No participant endpoints provided",
                final=True,
                request_id=request_id,
            )
            return

        # Select tasks
        tasks = self._select_tasks(config)
        total_tasks = len(tasks)

        yield create_status_update_event(
            task_id=assessment_id,
            context_id=context_id,
            state=TaskState.WORKING,
            message=f"Starting assessment with {total_tasks} tasks",
            final=False,
            request_id=request_id,
        )

        results: list[EvaluationResult] = []

        for i, task in enumerate(tasks):
            if self._canceled:
                yield create_status_update_event(
                    task_id=assessment_id,
                    context_id=context_id,
                    state=TaskState.CANCELED,
                    message=f"Assessment canceled after {i} tasks",
                    final=True,
                    request_id=request_id,
                )
                return

            # Progress update
            yield create_status_update_event(
                task_id=assessment_id,
                context_id=context_id,
                state=TaskState.WORKING,
                message=f"Task {i + 1}/{total_tasks}: {task.task_type.value} ({task.task_id})",
                final=False,
                request_id=request_id,
            )

            # Execute task
            execution_result = await self._execute_task(
                task=task,
                shopper_endpoint=shopper_endpoint,
                agent_id=list(participants.keys())[0] if participants else "unknown",
            )

            if execution_result.evaluation:
                results.append(execution_result.evaluation)

        # Build and yield final results
        assessment_results = AssessmentResults(
            assessment_id=assessment_id,
            participants=participants,
            config=config,
            results=results,
        )
        assessment_results.calculate_aggregate()

        # Create results artifact
        import json
        artifact = Artifact(
            name="assessment_results",
            description="WebShop+ assessment results",
            parts=[
                {
                    "kind": "text",
                    "text": json.dumps(
                        assessment_results.model_dump(mode="json"),
                        indent=2,
                        default=str,
                    ),
                }
            ],
            metadata={"format": "json"},
        )

        yield create_artifact_update_event(
            task_id=assessment_id,
            context_id=context_id,
            artifact=artifact,
            append=False,
            last_chunk=True,
            request_id=request_id,
        )

        yield create_status_update_event(
            task_id=assessment_id,
            context_id=context_id,
            state=TaskState.COMPLETED,
            message=f"Assessment complete: {assessment_results.aggregate.successful_tasks}/{total_tasks} tasks succeeded, avg score: {assessment_results.aggregate.average_score:.2f}",
            final=True,
            request_id=request_id,
        )

    def _select_tasks(self, config: AssessmentConfig) -> list[Task]:
        """Select tasks based on configuration.

        Ensures balanced distribution across task types when using "all".
        """
        task_types = config.task_types
        num_tasks = config.num_tasks

        if "all" in task_types:
            # Get all available tasks
            all_tasks = self.task_generator.get_all_tasks()

            # Filter out memory tasks if disabled
            if not config.include_memory_tasks:
                all_tasks = [
                    t for t in all_tasks
                    if t.task_type != TaskType.PREFERENCE_MEMORY
                ]

            # If requesting all tasks (or more than available), return all
            if num_tasks >= len(all_tasks):
                return all_tasks

            # Otherwise, distribute evenly across types
            all_types = list(TaskType)
            if not config.include_memory_tasks:
                all_types = [t for t in all_types if t != TaskType.PREFERENCE_MEMORY]

            # Calculate tasks per type (round-robin)
            tasks_per_type = num_tasks // len(all_types)
            remainder = num_tasks % len(all_types)

            selected_tasks = []
            for i, task_type in enumerate(all_types):
                type_tasks = self.task_generator.get_tasks_by_type(task_type)
                # Distribute remainder across first types
                count = tasks_per_type + (1 if i < remainder else 0)
                selected_tasks.extend(type_tasks[:count])

            # Return exactly num_tasks (in case some types have fewer tasks)
            return selected_tasks[:num_tasks]
        else:
            # Specific task types requested
            all_tasks = []
            for task_type in task_types:
                try:
                    type_tasks = self.task_generator.get_tasks_by_type(task_type)
                    all_tasks.extend(type_tasks)
                except (ValueError, KeyError):
                    logger.warning(f"Unknown task type: {task_type}")

            # Filter out memory tasks if disabled
            if not config.include_memory_tasks:
                all_tasks = [
                    t for t in all_tasks
                    if t.task_type != TaskType.PREFERENCE_MEMORY
                ]

            # Limit to requested number
            return all_tasks[:num_tasks]

    def _extract_task_kickoff_data(self, task: Task) -> tuple[str, float, list[str], str]:
        """Extract goal, budget, constraints, and user history from a task.

        Args:
            task: The task to extract data from.

        Returns:
            Tuple of (goal, budget, constraints, user_history).
        """
        goal = task.instruction

        # Extract budget from task constraints if available
        budget = self.config.default_budget
        constraints: list[str] = []
        user_history: str = ""

        if isinstance(task, BudgetConstrainedTask):
            budget = task.constraints.budget
            # Convert required items to constraint strings
            for item in task.constraints.required_items:
                if item.category:
                    constraints.append(f"category: {item.category}")
                if item.attributes:
                    for key, value in item.attributes.items():
                        constraints.append(f"{key}: {value}")
            if task.constraints.optimization_goal:
                constraints.append(f"optimization: {task.constraints.optimization_goal.value}")

        elif isinstance(task, NegativeConstraintTask):
            budget = task.constraints.budget if task.constraints.budget else self.config.default_budget
            # Add forbidden attributes as constraints (list of strings)
            if task.constraints.forbidden_attributes:
                for attr in task.constraints.forbidden_attributes:
                    constraints.append(f"NOT: {attr}")
            # Add forbidden terms
            if task.constraints.forbidden_terms:
                for term in task.constraints.forbidden_terms:
                    constraints.append(f"FORBIDDEN: {term}")
            # Add required attributes (list of strings)
            if task.constraints.required_attributes:
                for attr in task.constraints.required_attributes:
                    constraints.append(f"REQUIRE: {attr}")

        elif isinstance(task, PreferenceMemoryTask):
            # Compile session sequence into a history string
            history_lines = []
            for i, session in enumerate(task.session_sequence):
                history_lines.append(f"Session {i+1}:")
                history_lines.append(f"  Request: {session.instruction}")
                if session.establishes:
                    preferences = ", ".join(f"{k}={v}" for k, v in session.establishes.items())
                    history_lines.append(f"  Outcome: User established preference for [{preferences}]")
            user_history = "\n".join(history_lines)

        return goal, budget, constraints, user_history

    def _get_mcp_uri(self, session_id: str) -> str:
        """Build the MCP URI for a session.

        Args:
            session_id: The MCP session ID.

        Returns:
            Full MCP URI for the session.
        """
        return f"http://{self.config.mcp_host}:{self.config.mcp_port}/mcp/{session_id}"

    async def _dispatch_task_to_purple(
        self,
        task: Task,
        shopper_endpoint: str,
        agent_id: str,
    ) -> TaskExecutionResult:
        """Execute a task using MCP-based flow.

        This method uses PurpleAgentClient to send a single kickoff message
        with an MCP URI. The purple agent then handles all shopping steps
        via MCP tool calls.

        Args:
            task: The task to execute.
            shopper_endpoint: The purple agent's endpoint.
            agent_id: The agent's ID for memory tracking.

        Returns:
            TaskExecutionResult with evaluation.
        """
        result = TaskExecutionResult(
            task_id=task.task_id,
            session_id="",
        )

        # Extract task data for kickoff
        goal, budget, constraints, user_history = self._extract_task_kickoff_data(task)

        # Create MCP session
        mcp_session_id: Optional[str] = None
        mcp_uri: Optional[str] = None
        mcp_state: Optional[MCPSessionState] = None

        if self._session_manager is not None:
            mcp_session_id = str(uuid.uuid4())
            result.session_id = mcp_session_id  # Use MCP session ID as result session ID

            # Extract initial cart for error_recovery tasks
            initial_cart = None
            if isinstance(task, ErrorRecoveryTask) and task.setup.cart_contents:
                initial_cart = [
                    {
                        "product_id": item.product_id,
                        "name": item.product_name,
                        "price": item.price,
                        "quantity": item.quantity,
                        "options": item.attributes,
                    }
                    for item in task.setup.cart_contents
                ]
                logger.info(
                    "Pre-populating cart for error_recovery task",
                    task_id=task.task_id,
                    cart_items=len(initial_cart),
                )

            await self._session_manager.create_session(
                session_id=mcp_session_id,
                goal=goal,
                budget=budget,
                constraints=constraints,
                max_turns=self.config.max_actions_per_task,
                initial_cart=initial_cart,
            )
            mcp_uri = self._get_mcp_uri(mcp_session_id)

            logger.info(
                "Created MCP session for task",
                task_id=task.task_id,
                mcp_session_id=mcp_session_id,
                mcp_uri=mcp_uri,
            )

        try:
            # Send task to purple agent via PurpleAgentClient
            async with PurpleAgentClient(shopper_endpoint) as client:
                task_result = await client.send_task(
                    goal=goal,
                    budget=budget,
                    constraints=constraints,
                    user_history=user_history,
                    mcp_uri=mcp_uri,
                )

                if task_result.success:
                    result.completed = True
                    result.total_reward = 1.0  # MCP tasks report completion as success

                    # Extract result data if available
                    if task_result.result_data:
                        # Record result in session for evaluation
                        if "actions" in task_result.result_data:
                            result.actions_taken = len(task_result.result_data["actions"])
                        if "turns_used" in task_result.result_data:
                            logger.info(
                                "Setting actions_taken from purple agent result_data",
                                turns_used=task_result.result_data["turns_used"],
                                task_id=task.task_id,
                            )
                            result.actions_taken = task_result.result_data["turns_used"]

                    logger.info(
                        "MCP task completed successfully",
                        task_id=task.task_id,
                        result_data=task_result.result_data,
                        actions_taken=result.actions_taken,
                        mcp_session_id=mcp_session_id,
                    )
                else:
                    result.error = task_result.error or "Task failed"
                    logger.warning(
                        "MCP task failed",
                        task_id=task.task_id,
                        error=task_result.error,
                    )

            # Get final MCP session state for evaluation
            mcp_state: Optional[MCPSessionState] = None
            if mcp_session_id and self._session_manager:
                mcp_state = await self._session_manager.get_session(mcp_session_id)

                if mcp_state:
                    logger.info(
                        "Retrieved MCP state for evaluation",
                        mcp_session_id=mcp_session_id,
                        history_len=len(mcp_state.history),
                        cart_len=len(mcp_state.cart),
                        turns_used=mcp_state.turn_count,
                        task_id=task.task_id,
                    )
                    # Update result with MCP metrics
                    result.actions_taken = mcp_state.turn_count

                    # Attach reasoning from purple agent for evaluation
                    if task_result.result_data and "reasoning_summary" in task_result.result_data:
                        mcp_state.reasoning_summary = task_result.result_data["reasoning_summary"]
                        logger.info(
                            "Attached reasoning_summary to MCP state",
                            reasoning_len=len(mcp_state.reasoning_summary),
                            task_id=task.task_id,
                        )

                # Clean up MCP session
                await self._session_manager.cleanup_session(mcp_session_id)

        except PurpleAgentClientError as e:
            result.error = f"Purple agent error: {str(e)}"
            logger.error("PurpleAgentClient error", task_id=task.task_id, error=str(e))
        except asyncio.TimeoutError:
            result.timed_out = True
            result.error = "Task execution timed out"
        except Exception as e:
            result.error = f"Task execution error: {str(e)}"
            logger.error("MCP task execution error", task_id=task.task_id, error=str(e))

        return self._finalize_task(result, task, mcp_state)

    async def _execute_task(
        self,
        task: Task,
        shopper_endpoint: str,
        agent_id: str,
    ) -> TaskExecutionResult:
        """
        Execute a single task using MCP-based execution.

        Args:
            task: The task to execute.
            shopper_endpoint: The purple agent's endpoint.
            agent_id: The agent's ID for memory tracking.

        Returns:
            TaskExecutionResult with evaluation.
        """
        return await self._dispatch_task_to_purple(task, shopper_endpoint, agent_id)

    def _finalize_task(
        self,
        result: TaskExecutionResult,
        task: Task,
        mcp_state: Optional[MCPSessionState],
    ) -> TaskExecutionResult:
        """Finalize task execution and evaluate using MCP state."""
        # Evaluate using MCP state directly
        if mcp_state:
            evaluation = self.evaluator.evaluate(mcp_state, task)
            result.evaluation = evaluation
        else:
            logger.warning(
                "No MCP state available for evaluation",
                task_id=task.task_id,
            )
            result.error = result.error or "No MCP state available for evaluation"

        return result

    async def execute_single_task(
        self,
        task: Task,
        shopper_endpoint: str,
        agent_id: str = "test",
    ) -> TaskExecutionResult:
        """
        Execute a single task (useful for testing).

        Args:
            task: The task to execute.
            shopper_endpoint: The purple agent's endpoint.
            agent_id: The agent's ID.

        Returns:
            TaskExecutionResult with evaluation.
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Use 'async with' context.")

        return await self._execute_task(task, shopper_endpoint, agent_id)


class MockPurpleAgent:
    """
    Mock purple agent for testing.

    This can be used to test the orchestration without a real purple agent.
    """

    def __init__(self, action_sequence: Optional[list[str]] = None):
        """
        Initialize mock agent.

        Args:
            action_sequence: Predefined sequence of actions to return.
        """
        self.action_sequence = action_sequence or [
            "search[shoes]",
            "click[B07XYZ123]",
            "click[buy now]",
        ]
        self._action_index = 0

    def get_next_action(self, observation: str) -> str:
        """Get the next action in the sequence."""
        if self._action_index < len(self.action_sequence):
            action = self.action_sequence[self._action_index]
            self._action_index += 1
            return action
        return "click[buy now]"  # Default fallback

    def reset(self) -> None:
        """Reset action index."""
        self._action_index = 0
