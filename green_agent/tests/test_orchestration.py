"""
Tests for Phase 9: Assessment Orchestration.

This module tests:
- Executor: A2A request routing and response parsing
- WebShopPlusAgent: Main orchestration loop
- Integration between components
"""

import asyncio
import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent import (
    AgentConfig,
    MockPurpleAgent,
    TaskExecutionResult,
    WebShopPlusAgent,
)
from src.evaluator import Evaluator
from src.executor import (
    Executor,
    ExecutorConfig,
    ExecutorResult,
    MultiAgentExecutor,
    RequestType,
)
from src.messenger import (
    A2AMessage,
    JSONRPCResponse,
    MessageRole,
    TaskState,
    create_text_message,
    extract_action_from_text,
    parse_action_from_response,
)
from src.models import (
    AgentMemory,
    AssessmentConfig,
    AssessmentResults,
    BudgetConstrainedTask,
    BudgetConstraints,
    BudgetEvaluationCriteria,
    CartState,
    Difficulty,
    EvaluationResult,
    OptimizationGoal,
    RequiredItem,
    SessionState,
    Task,
    TaskType,
)
from src.state_manager import StateManager
from src.task_generator import TaskGenerator


# =============================================================================
# Executor Tests
# =============================================================================


class TestExecutorConfig:
    """Tests for ExecutorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExecutorConfig()
        assert config.timeout == 120.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.action_timeout == 60.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ExecutorConfig(
            timeout=30.0,
            max_retries=5,
            retry_delay=2.0,
            action_timeout=15.0,
        )
        assert config.timeout == 30.0
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.action_timeout == 15.0


class TestExecutorResult:
    """Tests for ExecutorResult."""

    def test_default_result(self):
        """Test default result values."""
        result = ExecutorResult()
        assert result.action is None
        assert result.raw_response is None
        assert result.full_text == ""
        assert result.error is None
        assert result.timed_out is False

    def test_result_with_action(self):
        """Test result with action."""
        result = ExecutorResult(action="search[shoes]")
        assert result.action == "search[shoes]"

    def test_result_with_error(self):
        """Test result with error."""
        result = ExecutorResult(error="Connection failed", timed_out=True)
        assert result.error == "Connection failed"
        assert result.timed_out is True


class TestExecutor:
    """Tests for Executor class."""

    @pytest.fixture
    def executor_config(self):
        """Create executor config for tests."""
        return ExecutorConfig(
            timeout=10.0,
            action_timeout=5.0,
            max_retries=1,
        )

    @pytest.fixture
    def mock_client(self):
        """Create mock A2A client."""
        client = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_executor_context_manager(self, executor_config):
        """Test executor context manager."""
        executor = Executor(config=executor_config)
        assert executor._client is None

        async with executor:
            assert executor._client is not None
            assert executor._owned_client is True

        assert executor._client is None

    @pytest.mark.asyncio
    async def test_executor_with_provided_client(self, executor_config, mock_client):
        """Test executor with provided client."""
        executor = Executor(config=executor_config, client=mock_client)
        assert executor._client is mock_client
        assert executor._owned_client is False

    @pytest.mark.asyncio
    async def test_send_task_instruction(self, executor_config):
        """Test sending task instruction."""
        mock_response = JSONRPCResponse(
            result={
                "history": [
                    {
                        "role": "agent",
                        "parts": [{"kind": "text", "text": "search[running shoes]"}],
                    }
                ]
            },
            id="test-123",
        )

        with patch.object(Executor, "_execute_request", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = ExecutorResult(action="search[running shoes]")

            executor = Executor(config=executor_config)
            executor._client = AsyncMock()

            result = await executor.send_task_instruction(
                endpoint="http://agent:8001/a2a",
                instruction="Find running shoes under $100",
                task_id="task-1",
                context_id="ctx-1",
            )

            assert result.action == "search[running shoes]"

    @pytest.mark.asyncio
    async def test_send_observation(self, executor_config):
        """Test sending observation."""
        with patch.object(Executor, "_execute_request", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = ExecutorResult(action="click[buy now]")

            executor = Executor(config=executor_config)
            executor._client = AsyncMock()

            result = await executor.send_observation(
                endpoint="http://agent:8001/a2a",
                observation="Product page: Nike Running Shoes - $89.99",
                task_id="task-1",
                context_id="ctx-1",
                available_actions={"has_search_bar": True, "clickables": ["buy now", "back"]},
                reward=0.0,
                done=False,
            )

            assert result.action == "click[buy now]"

    @pytest.mark.asyncio
    async def test_send_error_notice(self, executor_config):
        """Test sending error notice."""
        with patch.object(Executor, "_execute_request", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = ExecutorResult(action="search[shoes]")

            executor = Executor(config=executor_config)
            executor._client = AsyncMock()

            result = await executor.send_error_notice(
                endpoint="http://agent:8001/a2a",
                error_message="Invalid action format",
                task_id="task-1",
                context_id="ctx-1",
            )

            # Should return result (may or may not have action)
            assert isinstance(result, ExecutorResult)

    @pytest.mark.asyncio
    async def test_extract_response_text(self, executor_config):
        """Test extracting text from response."""
        executor = Executor(config=executor_config)

        result = {
            "history": [
                {"role": "user", "parts": [{"kind": "text", "text": "Task instruction"}]},
                {"role": "agent", "parts": [{"kind": "text", "text": "I will search for shoes"}]},
            ],
            "artifacts": [
                {"parts": [{"kind": "text", "text": "search[shoes]"}]}
            ],
        }

        text = executor._extract_response_text(result)
        assert "I will search for shoes" in text
        assert "search[shoes]" in text


class TestMultiAgentExecutor:
    """Tests for MultiAgentExecutor."""

    @pytest.mark.asyncio
    async def test_multi_agent_context_manager(self):
        """Test multi-agent executor context manager."""
        executor = MultiAgentExecutor()

        async with executor:
            assert executor._client is not None

        assert executor._client is None

    @pytest.mark.asyncio
    async def test_get_executor(self):
        """Test getting executor for agent."""
        multi_exec = MultiAgentExecutor()
        multi_exec._client = AsyncMock()

        exec1 = multi_exec.get_executor("agent1")
        exec2 = multi_exec.get_executor("agent2")
        exec1_again = multi_exec.get_executor("agent1")

        assert exec1 is exec1_again  # Same instance
        assert exec1 is not exec2  # Different instances


# =============================================================================
# WebShopPlusAgent Tests
# =============================================================================


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_config(self):
        """Test default agent config."""
        config = AgentConfig()
        assert config.max_actions_per_task == 30
        assert config.task_timeout_seconds == 300.0
        assert config.action_timeout_seconds == 60.0
        assert config.max_retries_per_action == 3
        assert config.use_llm_evaluation is True
        assert config.webshop_mode == "preview"

    def test_custom_config(self):
        """Test custom agent config."""
        config = AgentConfig(
            max_actions_per_task=10,
            task_timeout_seconds=60.0,
            webshop_mode="full",
        )
        assert config.max_actions_per_task == 10
        assert config.task_timeout_seconds == 60.0
        assert config.webshop_mode == "full"


class TestTaskExecutionResult:
    """Tests for TaskExecutionResult."""

    def test_default_result(self):
        """Test default task execution result."""
        result = TaskExecutionResult(
            task_id="task-1",
            session_id="session-1",
        )
        assert result.task_id == "task-1"
        assert result.session_id == "session-1"
        assert result.completed is False
        assert result.total_reward == 0.0
        assert result.actions_taken == 0
        assert result.evaluation is None
        assert result.error is None
        assert result.timed_out is False

    def test_result_with_values(self):
        """Test task execution result with values."""
        eval_result = EvaluationResult(
            task_id="task-1",
            task_type=TaskType.BUDGET_CONSTRAINED,
            completed=True,
            success=True,
            overall_score=0.85,
        )

        result = TaskExecutionResult(
            task_id="task-1",
            session_id="session-1",
            completed=True,
            total_reward=0.9,
            actions_taken=5,
            evaluation=eval_result,
        )

        assert result.completed is True
        assert result.total_reward == 0.9
        assert result.actions_taken == 5
        assert result.evaluation.overall_score == 0.85


class TestWebShopPlusAgent:
    """Tests for WebShopPlusAgent."""

    @pytest.fixture
    def mock_task_generator(self):
        """Create mock task generator."""
        mock = MagicMock(spec=TaskGenerator)

        # Create test tasks
        task1 = BudgetConstrainedTask(
            task_id="budget-1",
            instruction="Find running shoes under $100",
            constraints=BudgetConstraints(
                budget=100.0,
                required_items=[
                    RequiredItem(category="shoes", attributes={"type": "running"})
                ],
                optimization_goal=OptimizationGoal.BALANCE,
            ),
        )

        mock.get_all_tasks.return_value = [task1]
        mock.get_tasks_by_type.return_value = [task1]

        return mock

    @pytest.fixture
    def mock_state_manager(self):
        """Create mock state manager."""
        mock = MagicMock(spec=StateManager)

        session = SessionState(
            session_id="test-session",
            task_id="budget-1",
            agent_id="test-agent",
        )
        mock.create_session.return_value = session
        mock.get_session.return_value = session
        mock.get_agent_memory.return_value = AgentMemory(agent_id="test-agent")

        return mock

    @pytest.fixture
    def mock_webshop(self):
        """Create mock WebShop wrapper."""
        from src.webshop_wrapper import StepResult

        mock = MagicMock()
        mock.reset.return_value = "Initial observation: Welcome to WebShop"
        mock.step.return_value = StepResult(
            observation="You clicked buy now",
            reward=0.9,
            done=True,
            info={},
        )
        mock.get_available_actions.return_value = {
            "has_search_bar": True,
            "clickables": ["buy now", "back"],
        }

        return mock

    @pytest.fixture
    def mock_evaluator(self):
        """Create mock evaluator."""
        mock = MagicMock(spec=Evaluator)

        result = EvaluationResult(
            task_id="budget-1",
            task_type=TaskType.BUDGET_CONSTRAINED,
            completed=True,
            success=True,
            overall_score=0.85,
        )
        mock.evaluate.return_value = result

        return mock

    def test_agent_initialization(self):
        """Test agent initialization."""
        config = AgentConfig(max_actions_per_task=10)
        agent = WebShopPlusAgent(config=config)

        assert agent.config.max_actions_per_task == 10
        assert agent._initialized is False

    def test_agent_lazy_properties(self):
        """Test lazy initialization of components."""
        agent = WebShopPlusAgent()

        # Access properties - they should create instances
        assert agent._task_generator is None
        task_gen = agent.task_generator
        assert agent._task_generator is not None

        assert agent._state_manager is None
        state_mgr = agent.state_manager
        assert agent._state_manager is not None

    def test_agent_with_provided_components(
        self,
        mock_task_generator,
        mock_state_manager,
        mock_evaluator,
    ):
        """Test agent with provided components."""
        agent = WebShopPlusAgent(
            task_generator=mock_task_generator,
            state_manager=mock_state_manager,
            evaluator=mock_evaluator,
        )

        assert agent.task_generator is mock_task_generator
        assert agent.state_manager is mock_state_manager
        assert agent.evaluator is mock_evaluator

    def test_select_tasks_all(self, mock_task_generator):
        """Test task selection with 'all' type."""
        agent = WebShopPlusAgent(task_generator=mock_task_generator)
        config = AssessmentConfig(task_types=["all"], num_tasks=10)

        tasks = agent._select_tasks(config)

        mock_task_generator.get_all_tasks.assert_called_once()
        assert len(tasks) <= 10

    def test_select_tasks_specific_type(self, mock_task_generator):
        """Test task selection with specific type."""
        agent = WebShopPlusAgent(task_generator=mock_task_generator)
        config = AssessmentConfig(task_types=["budget_constrained"], num_tasks=5)

        tasks = agent._select_tasks(config)

        mock_task_generator.get_tasks_by_type.assert_called_with("budget_constrained")

    def test_cancel_agent(self):
        """Test canceling agent."""
        agent = WebShopPlusAgent()
        assert agent._canceled is False

        agent.cancel()
        assert agent._canceled is True


class TestMockPurpleAgent:
    """Tests for MockPurpleAgent."""

    def test_default_actions(self):
        """Test default action sequence."""
        mock = MockPurpleAgent()
        assert mock.get_next_action("obs1") == "search[shoes]"
        assert mock.get_next_action("obs2") == "click[B07XYZ123]"
        assert mock.get_next_action("obs3") == "click[buy now]"

    def test_custom_actions(self):
        """Test custom action sequence."""
        mock = MockPurpleAgent(action_sequence=[
            "search[laptop]",
            "click[product-1]",
        ])
        assert mock.get_next_action("") == "search[laptop]"
        assert mock.get_next_action("") == "click[product-1]"
        # Falls back to default after sequence
        assert mock.get_next_action("") == "click[buy now]"

    def test_reset(self):
        """Test resetting action index."""
        mock = MockPurpleAgent()
        mock.get_next_action("")
        mock.get_next_action("")

        mock.reset()
        assert mock.get_next_action("") == "search[shoes]"


# =============================================================================
# Integration Tests
# =============================================================================


class TestOrchestrationIntegration:
    """Integration tests for orchestration flow."""

    @pytest.mark.asyncio
    async def test_finalize_task(self):
        """Test task finalization."""
        state_manager = StateManager()
        evaluator = Evaluator()

        # Create a session
        session = state_manager.create_session("budget-1", "test-agent")
        session.record_action("search[shoes]", "Found 10 products", 0.0)
        session.record_action("click[buy now]", "Purchase complete", 0.9)

        # Create a simple task
        task = BudgetConstrainedTask(
            task_id="budget-1",
            instruction="Find shoes under $50",
            constraints=BudgetConstraints(
                budget=50.0,
                required_items=[RequiredItem(category="shoes")],
            ),
        )

        # Create result
        result = TaskExecutionResult(
            task_id="budget-1",
            session_id=session.session_id,
            completed=True,
            total_reward=0.9,
            actions_taken=2,
        )

        # Create agent and finalize
        agent = WebShopPlusAgent(
            state_manager=state_manager,
            evaluator=evaluator,
        )
        agent._finalize_task(result, session, task, None)

        # Check that evaluation was added
        assert result.evaluation is not None
        assert result.evaluation.task_id == "budget-1"

    @pytest.mark.asyncio
    async def test_assessment_results_aggregation(self):
        """Test assessment results aggregation."""
        results = [
            EvaluationResult(
                task_id="task-1",
                task_type=TaskType.BUDGET_CONSTRAINED,
                completed=True,
                success=True,
                overall_score=0.8,
                time_elapsed_seconds=10.0,
            ),
            EvaluationResult(
                task_id="task-2",
                task_type=TaskType.BUDGET_CONSTRAINED,
                completed=True,
                success=False,
                overall_score=0.4,
                time_elapsed_seconds=15.0,
            ),
            EvaluationResult(
                task_id="task-3",
                task_type=TaskType.NEGATIVE_CONSTRAINT,
                completed=True,
                success=True,
                overall_score=0.9,
                time_elapsed_seconds=8.0,
            ),
        ]

        assessment = AssessmentResults(
            assessment_id="test-assessment",
            participants={"shopper": "http://agent:8001/a2a"},
            config=AssessmentConfig(num_tasks=3),
            results=results,
        )
        assessment.calculate_aggregate()

        assert assessment.aggregate.total_tasks == 3
        assert assessment.aggregate.successful_tasks == 2
        assert 0.69 <= assessment.aggregate.average_score <= 0.71  # ~0.7
        assert assessment.aggregate.average_time == 11.0

        # Check by task type
        assert "budget_constrained" in assessment.aggregate.by_task_type
        assert "negative_constraint" in assessment.aggregate.by_task_type
        assert assessment.aggregate.by_task_type["budget_constrained"]["count"] == 2
        assert assessment.aggregate.by_task_type["negative_constraint"]["count"] == 1


# =============================================================================
# Action Parsing Tests
# =============================================================================


class TestActionParsing:
    """Tests for action parsing utilities."""

    def test_extract_action_search(self):
        """Test extracting search action."""
        text = "I will search for running shoes. search[running shoes]"
        action = extract_action_from_text(text)
        assert action == "search[running shoes]"

    def test_extract_action_click(self):
        """Test extracting click action."""
        text = "Let me click on the buy button: click[buy now]"
        action = extract_action_from_text(text)
        assert action == "click[buy now]"

    def test_extract_action_case_insensitive(self):
        """Test case-insensitive action extraction."""
        text = "SEARCH[Laptop Computer]"
        action = extract_action_from_text(text)
        assert action == "search[Laptop Computer]"

    def test_extract_action_none(self):
        """Test when no action found."""
        text = "I'm thinking about what to do next"
        action = extract_action_from_text(text)
        assert action is None

    def test_parse_action_from_response_history(self):
        """Test parsing action from response with history."""
        response = JSONRPCResponse(
            result={
                "history": [
                    {"role": "user", "parts": [{"kind": "text", "text": "Task"}]},
                    {"role": "agent", "parts": [{"kind": "text", "text": "search[shoes]"}]},
                ]
            },
            id="test-1",
        )

        action = parse_action_from_response(response)
        assert action == "search[shoes]"

    def test_parse_action_from_response_no_action_in_message(self):
        """Test parsing when agent message has no action."""
        response = JSONRPCResponse(
            result={
                "history": [
                    {"role": "agent", "parts": [{"kind": "text", "text": "Thinking about options..."}]},
                ],
            },
            id="test-1",
        )

        action = parse_action_from_response(response)
        assert action is None  # No action in message means None

    def test_parse_action_from_response_error(self):
        """Test parsing action from error response."""
        response = JSONRPCResponse(
            error={"code": -32600, "message": "Invalid request"},
            id="test-1",
        )

        action = parse_action_from_response(response)
        assert action is None


# =============================================================================
# State Manager Integration Tests
# =============================================================================


class TestStateManagerIntegration:
    """Integration tests for state manager with orchestration."""

    def test_session_lifecycle(self):
        """Test complete session lifecycle."""
        state_manager = StateManager()

        # Create session
        session = state_manager.create_session("task-1", "agent-1")
        assert session.session_id in state_manager

        # Record actions
        state_manager.record_action(
            session.session_id,
            "search[shoes]",
            "Found 10 products",
            0.0,
        )
        state_manager.record_action(
            session.session_id,
            "click[buy now]",
            "Purchase complete",
            0.9,
        )

        # Complete session
        summary = state_manager.complete_session(session.session_id, "budget_constrained")

        assert session.completed is True
        assert session.actions_taken == 2
        assert summary.task_type == "budget_constrained"

    def test_agent_memory_tracking(self):
        """Test agent memory across sessions."""
        state_manager = StateManager()

        # Session 1
        session1 = state_manager.create_session("task-1", "agent-1")
        session1.preferences_established = {"color": "blue"}
        state_manager.complete_session(session1.session_id, "preference_memory")

        # Session 2
        session2 = state_manager.create_session("task-2", "agent-1")
        state_manager.complete_session(session2.session_id, "preference_memory")

        # Check memory
        memory = state_manager.get_agent_memory("agent-1")
        assert len(memory.sessions) == 2
        all_prefs = memory.get_all_preferences()
        assert "color" in all_prefs


# =============================================================================
# RequestType Tests
# =============================================================================


class TestRequestType:
    """Tests for RequestType enum."""

    def test_request_types(self):
        """Test all request type values."""
        assert RequestType.TASK_INSTRUCTION.value == "task_instruction"
        assert RequestType.OBSERVATION.value == "observation"
        assert RequestType.ERROR_NOTICE.value == "error_notice"
        assert RequestType.SESSION_END.value == "session_end"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
