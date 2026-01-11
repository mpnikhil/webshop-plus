"""
WebShop+ Assessment Orchestration Agent.

This module provides the main orchestration logic for running assessments:
- WebShopPlusAgent: Main agent class that orchestrates task execution
- Task dispatch loop: send task -> receive action -> step -> send observation -> evaluate
- Result aggregation and reporting
"""

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Optional

import structlog

from .evaluator import Evaluator
from .executor import Executor, ExecutorConfig
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
    EvaluationResult,
    ErrorRecoveryTask,
    PreferenceMemoryTask,
    SessionState,
    Task,
    TaskType,
    TaskUpdate,
)
from .state_manager import StateManager
from .task_generator import TaskGenerator
from .webshop_wrapper import WebShopWrapper

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
    - A2A communication with purple agents via Executor
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
        """
        self.config = config or AgentConfig()

        # Components - lazy initialization
        self._task_generator = task_generator
        self._state_manager = state_manager
        self._webshop = webshop
        self._evaluator = evaluator
        self._llm_client = llm_client
        self._executor: Optional[Executor] = None

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
        executor_config = ExecutorConfig(
            timeout=self.config.task_timeout_seconds,
            action_timeout=self.config.action_timeout_seconds,
            max_retries=self.config.max_retries_per_action,
        )
        self._executor = Executor(config=executor_config)
        await self._executor.__aenter__()
        self._initialized = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._executor:
            await self._executor.__aexit__(exc_type, exc_val, exc_tb)
            self._executor = None
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

    async def _execute_task(
        self,
        task: Task,
        shopper_endpoint: str,
        agent_id: str,
    ) -> TaskExecutionResult:
        """
        Execute a single task.

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

        # Create session
        session = self.state_manager.create_session(task.task_id, agent_id)
        result.session_id = session.session_id

        # Handle special setup for error recovery tasks
        if isinstance(task, ErrorRecoveryTask):
            self.state_manager.inject_cart_from_setup(task.setup.cart_contents)

        # Get agent memory for preference tasks
        memory: Optional[AgentMemory] = None
        if isinstance(task, PreferenceMemoryTask):
            memory = self.state_manager.get_agent_memory(agent_id)

        try:
            # Reset WebShop environment
            initial_observation = self.webshop.reset()
            context_id = str(uuid.uuid4())

            # Send task instruction to agent
            exec_result = await self._executor.send_task_instruction(
                endpoint=shopper_endpoint,
                instruction=task.instruction,
                task_id=task.task_id,
                context_id=context_id,
            )

            if exec_result.error and not exec_result.action:
                result.error = exec_result.error
                session.error = exec_result.error
                return self._finalize_task(result, session, task, memory)

            # Get first action
            action = exec_result.action

            # Task loop
            done = False
            total_reward = 0.0

            while (
                not done
                and result.actions_taken < self.config.max_actions_per_task
                and not self._canceled
            ):
                if not action:
                    # No action received, try once more
                    exec_result = await self._executor.send_error_notice(
                        endpoint=shopper_endpoint,
                        error_message="No valid action received. Please respond with search[query] or click[element].",
                        task_id=task.task_id,
                        context_id=context_id,
                    )
                    action = exec_result.action
                    if not action:
                        result.error = "Agent failed to provide valid actions"
                        break

                # Execute action in WebShop
                step_result = self.webshop.step(action)
                result.actions_taken += 1

                # Record action in session
                self.state_manager.record_action(
                    session.session_id,
                    action,
                    step_result.observation,
                    step_result.reward,
                )

                total_reward += step_result.reward
                done = step_result.done

                if done:
                    break

                # Get available actions
                available_actions = self.webshop.get_available_actions()

                # Send observation to agent
                exec_result = await self._executor.send_observation(
                    endpoint=shopper_endpoint,
                    observation=step_result.observation,
                    task_id=task.task_id,
                    context_id=context_id,
                    available_actions=available_actions,
                    reward=step_result.reward,
                    done=done,
                )

                if exec_result.timed_out:
                    result.timed_out = True
                    result.error = "Agent timed out"
                    break

                action = exec_result.action

            result.total_reward = total_reward
            result.completed = done

        except asyncio.TimeoutError:
            result.timed_out = True
            result.error = "Task execution timed out"
        except Exception as e:
            result.error = f"Task execution error: {str(e)}"
            logger.error("Task execution error", task_id=task.task_id, error=str(e))

        return self._finalize_task(result, session, task, memory)

    def _finalize_task(
        self,
        result: TaskExecutionResult,
        session: SessionState,
        task: Task,
        memory: Optional[AgentMemory],
    ) -> TaskExecutionResult:
        """Finalize task execution and evaluate."""
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
