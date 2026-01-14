"""
Tests for Phase 9: Assessment Orchestration.

This module tests:
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
    Task,
    TaskType,
)
from src.task_generator import TaskGenerator


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

        assert agent._webshop is None
        webshop = agent.webshop
        assert agent._webshop is not None

    def test_agent_with_provided_components(
        self,
        mock_task_generator,
        mock_evaluator,
    ):
        """Test agent with provided components."""
        agent = WebShopPlusAgent(
            task_generator=mock_task_generator,
            evaluator=mock_evaluator,
        )

        assert agent.task_generator is mock_task_generator
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

    @pytest.mark.asyncio
    async def test_agent_context_manager(self):
        """Test agent context manager."""
        agent = WebShopPlusAgent()
        assert agent._initialized is False

        async with agent:
            assert agent._initialized is True

        assert agent._initialized is False


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
        from src.webshop_mcp.session_state import SessionState as MCPSessionState
        evaluator = Evaluator()

        # Create a session state
        mcp_state = MCPSessionState(
            session_id="test-session",
            goal="Find shoes under $50",
            budget=50.0
        )
        mcp_state.history = [
            {"action": "search", "query": "shoes", "turn": 1},
            {"action": "click", "element_id": "buy-now", "turn": 2}
        ]
        mcp_state.completed = True

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
            session_id="test-session",
            completed=True,
            total_reward=0.9,
            actions_taken=2,
        )

        # Create agent and finalize
        agent = WebShopPlusAgent(
            evaluator=evaluator,
        )
        agent._finalize_task(result, task, mcp_state)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
