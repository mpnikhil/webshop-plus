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
    EvaluationResult,
    NegativeConstraintTask,
    PreferenceMemoryTask,
    SessionState,
    Task,
    TaskType,
    TaskUpdate,
)
from .purple_client import PurpleAgentClient, TaskResult, PurpleAgentClientError
from .state_manager import StateManager
from .task_generator import TaskGenerator
from .webshop_wrapper import WebShopWrapper

if TYPE_CHECKING:
    from .webshop_mcp import SessionManager

# Runtime imports for MCP session functions
from .webshop_mcp import is_session_completed, get_final_result

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
    - Session management via StateManager
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
        state_manager: Optional[StateManager] = None,
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
            state_manager: State manager instance (creates default if None).
            webshop: WebShop wrapper instance (creates default if None).
            evaluator: Evaluator instance (creates default if None).
            llm_client: LLM client for evaluation (creates default if None).
            session_manager: MCP session manager for tool execution.
        """
        self.config = config or AgentConfig()

        # Components - lazy initialization
        self._task_generator = task_generator
        self._state_manager = state_manager
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
    def state_manager(self) -> StateManager:
        if self._state_manager is None:
            self._state_manager = StateManager()
        return self._state_manager

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
        """Select tasks based on configuration."""
        task_types = config.task_types
        num_tasks = config.num_tasks

        if "all" in task_types:
            all_tasks = self.task_generator.get_all_tasks()
        else:
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

    def _extract_task_kickoff_data(self, task: Task) -> tuple[str, float, list[str]]:
        """Extract goal, budget, and constraints from a task.

        Args:
            task: The task to extract data from.

        Returns:
            Tuple of (goal, budget, constraints).
        """
        goal = task.instruction

        # Extract budget from task constraints if available
        budget = self.config.default_budget
        constraints: list[str] = []

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

        return goal, budget, constraints

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

        # Create session for tracking
        session = self.state_manager.create_session(task.task_id, agent_id)
        result.session_id = session.session_id

        # Get agent memory for preference tasks
        memory: Optional[AgentMemory] = None
        if isinstance(task, PreferenceMemoryTask):
            memory = self.state_manager.get_agent_memory(agent_id)

        try:
            # Extract task data for kickoff
            goal, budget, constraints = self._extract_task_kickoff_data(task)

            # Create MCP session if session manager is available
            mcp_session_id: Optional[str] = None
            mcp_uri: Optional[str] = None

            if self._session_manager is not None:
                mcp_session_id = str(uuid.uuid4())
                await self._session_manager.create_session(
                    session_id=mcp_session_id,
                    goal=goal,
                    budget=budget,
                    constraints=constraints,
                    max_turns=self.config.max_actions_per_task,
                )
                mcp_uri = self._get_mcp_uri(mcp_session_id)

                logger.info(
                    "Created MCP session for task",
                    task_id=task.task_id,
                    mcp_session_id=mcp_session_id,
                    mcp_uri=mcp_uri,
                )

            # Send task to purple agent via PurpleAgentClient
            async with PurpleAgentClient(shopper_endpoint) as client:
                task_result = await client.send_task(
                    goal=goal,
                    budget=budget,
                    constraints=constraints,
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

            # Get final result from MCP session if available
            if mcp_session_id and self._session_manager:
                session_completed = is_session_completed(mcp_session_id)
                logger.info(
                    "Checking MCP session completion",
                    mcp_session_id=mcp_session_id,
                    session_completed=session_completed,
                    task_id=task.task_id,
                )
                if session_completed:
                    mcp_result = get_final_result(mcp_session_id)
                    logger.info(
                        "Retrieved MCP final result",
                        mcp_session_id=mcp_session_id,
                        has_result=mcp_result is not None,
                        result_keys=list(mcp_result.keys()) if mcp_result else [],
                        task_id=task.task_id,
                    )
                    if mcp_result:
                        # Merge MCP result into task execution result
                        if "turns_used" in mcp_result:
                            logger.info(
                                "Setting actions_taken from MCP result",
                                turns_used=mcp_result["turns_used"],
                                task_id=task.task_id,
                            )
                            result.actions_taken = mcp_result["turns_used"]
                        if "success" in mcp_result:
                            result.completed = mcp_result["success"]
                        if "reward" in mcp_result:
                            result.total_reward = mcp_result["reward"]
                else:
                    logger.warning(
                        "MCP session not marked as completed",
                        mcp_session_id=mcp_session_id,
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

        return self._finalize_task(result, session, task, memory)

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
        session: SessionState,
        task: Task,
        memory: Optional[AgentMemory],
    ) -> TaskExecutionResult:
        """Finalize task execution and evaluate."""
        # Update session with actions_taken from result
        # (result.actions_taken was set from MCP final result)
        session.set_action_count(result.actions_taken)

        # Complete session
        session_summary = self.state_manager.complete_session(
            session.session_id,
            task_type=task.task_type.value,
        )

        # Evaluate
        evaluation = self.evaluator.evaluate(session, task, memory)
        result.evaluation = evaluation

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
