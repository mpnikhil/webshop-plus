"""
Tests for MCP-based task execution in WebShopPlusAgent.

Stage 7d of the AAA (A2A + MCP Agentification) implementation.
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent import (
    AgentConfig,
    TaskExecutionResult,
    WebShopPlusAgent,
)
from src.models import (
    BudgetConstrainedTask,
    BudgetConstraints,
    BudgetEvaluationCriteria,
    Difficulty,
    NegativeConstraintTask,
    NegativeConstraints,
    OptimizationGoal,
    RequiredItem,
    SessionState,
    TaskType,
)
from src.purple_client import PurpleAgentClient, TaskResult, TaskError
from a2a.types import TaskState


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mcp_agent_config():
    """Agent config for MCP-based execution."""
    return AgentConfig(
        mcp_host="localhost",
        mcp_port=8000,
        max_actions_per_task=10,
        default_budget=100.0,
    )


@pytest.fixture
def budget_task():
    """Sample budget-constrained task."""
    return BudgetConstrainedTask(
        task_id="budget-test-1",
        instruction="Find running shoes under $50",
        constraints=BudgetConstraints(
            budget=50.0,
            required_items=[
                RequiredItem(category="shoes", attributes={"type": "running"})
            ],
            optimization_goal=OptimizationGoal.BALANCE,
        ),
    )


@pytest.fixture
def negative_constraint_task():
    """Sample negative constraint task."""
    return NegativeConstraintTask(
        task_id="negative-test-1",
        instruction="Find a shirt that is NOT red",
        constraints=NegativeConstraints(
            forbidden_attributes=["red", "crimson"],
            required_attributes=["shirt", "cotton"],
            budget=75.0,
        ),
    )


@pytest.fixture
def mock_state_manager():
    """Mock state manager."""
    manager = MagicMock()
    session = SessionState(
        session_id="test-session-123",
        task_id="test-task",
        agent_id="test-agent",
    )
    manager.create_session.return_value = session
    manager.get_session.return_value = session
    return manager


@pytest.fixture
def mock_session_manager():
    """Mock MCP session manager."""
    manager = AsyncMock()

    # Mock MCP server returned by get_session
    mock_server = MagicMock()
    mock_server.is_completed.return_value = True
    mock_server.get_final_result.return_value = {
        "success": True,
        "turns_used": 5,
        "reward": 0.9,
    }

    manager.create_session.return_value = None
    manager.get_session.return_value = mock_server
    manager.cleanup_session.return_value = True

    return manager


@pytest.fixture
def mock_evaluator():
    """Mock evaluator."""
    from src.evaluator import Evaluator
    from src.models import EvaluationResult

    evaluator = MagicMock(spec=Evaluator)
    evaluator.evaluate.return_value = EvaluationResult(
        task_id="test-task",
        task_type=TaskType.BUDGET_CONSTRAINED,
        completed=True,
        success=True,
        overall_score=0.85,
    )
    return evaluator


# =============================================================================
# AgentConfig MCP Tests
# =============================================================================


class TestAgentConfigMCP:
    """Tests for MCP-related AgentConfig fields."""

    def test_mcp_config_fields(self):
        """MCP config fields have correct defaults."""
        config = AgentConfig()
        assert config.mcp_host == "localhost"
        assert config.mcp_port == 8000
        assert config.default_budget == 100.0

    def test_custom_mcp_config(self):
        """MCP config can be customized."""
        config = AgentConfig(
            mcp_host="mcp.example.com",
            mcp_port=9000,
            default_budget=200.0,
        )
        assert config.mcp_host == "mcp.example.com"
        assert config.mcp_port == 9000
        assert config.default_budget == 200.0


# =============================================================================
# Task Kickoff Data Extraction Tests
# =============================================================================


class TestExtractTaskKickoffData:
    """Tests for _extract_task_kickoff_data method."""

    def test_budget_task_extraction(self, mcp_agent_config, budget_task):
        """Extract goal, budget, constraints from budget task."""
        agent = WebShopPlusAgent(config=mcp_agent_config)
        goal, budget, constraints = agent._extract_task_kickoff_data(budget_task)

        assert goal == "Find running shoes under $50"
        assert budget == 50.0
        assert "category: shoes" in constraints
        assert "type: running" in constraints
        assert "optimization: balance" in constraints

    def test_negative_constraint_task_extraction(
        self, mcp_agent_config, negative_constraint_task
    ):
        """Extract data from negative constraint task."""
        agent = WebShopPlusAgent(config=mcp_agent_config)
        goal, budget, constraints = agent._extract_task_kickoff_data(
            negative_constraint_task
        )

        assert goal == "Find a shirt that is NOT red"
        assert budget == 75.0  # Uses task budget
        assert any("NOT:" in c for c in constraints)
        assert any("REQUIRE:" in c for c in constraints)

    def test_default_budget_used(self, mcp_agent_config):
        """Default budget used when not explicitly set."""
        agent = WebShopPlusAgent(config=mcp_agent_config)

        # Test with budget task that has explicit budget
        task = BudgetConstrainedTask(
            task_id="test",
            instruction="Test",
            constraints=BudgetConstraints(
                budget=75.0,
                required_items=[],
            ),
        )
        _, budget, _ = agent._extract_task_kickoff_data(task)
        assert budget == 75.0

        # Test with negative constraint task without budget (uses default)
        task2 = NegativeConstraintTask(
            task_id="test2",
            instruction="Test",
            constraints=NegativeConstraints(
                forbidden_attributes=["red"],
                budget=None,  # No budget specified
            ),
        )
        _, budget2, _ = agent._extract_task_kickoff_data(task2)
        assert budget2 == 100.0  # Default budget


# =============================================================================
# MCP URI Generation Tests
# =============================================================================


class TestGetMcpUri:
    """Tests for _get_mcp_uri method."""

    def test_mcp_uri_format(self, mcp_agent_config):
        """MCP URI has correct format."""
        agent = WebShopPlusAgent(config=mcp_agent_config)
        uri = agent._get_mcp_uri("session-123")
        assert uri == "http://localhost:8000/mcp/session-123"

    def test_custom_host_port(self):
        """MCP URI uses custom host and port."""
        config = AgentConfig(
            mcp_host="mcp.example.com",
            mcp_port=9000,
        )
        agent = WebShopPlusAgent(config=config)
        uri = agent._get_mcp_uri("abc")
        assert uri == "http://mcp.example.com:9000/mcp/abc"


# =============================================================================
# MCP Task Execution Tests
# =============================================================================


class TestExecuteTaskMCP:
    """Tests for _dispatch_task_to_purple method."""

    @pytest.mark.asyncio
    async def test_successful_mcp_execution(
        self,
        mcp_agent_config,
        budget_task,
        mock_state_manager,
        mock_session_manager,
        mock_evaluator,
    ):
        """Successful MCP task execution."""
        agent = WebShopPlusAgent(
            config=mcp_agent_config,
            state_manager=mock_state_manager,
            session_manager=mock_session_manager,
            evaluator=mock_evaluator,
        )

        # Mock PurpleAgentClient
        mock_task_result = TaskResult(
            success=True,
            task_id="purple-task-1",
            context_id="ctx-1",
            final_state=TaskState.completed,
            result_data={
                "cart": [{"name": "Shoes", "price": 45.0}],
                "total": 45.0,
            },
        )

        with patch.object(
            PurpleAgentClient, "__aenter__", new_callable=AsyncMock
        ) as mock_enter:
            mock_client = AsyncMock()
            mock_client.send_task.return_value = mock_task_result
            mock_enter.return_value = mock_client

            with patch.object(
                PurpleAgentClient, "__aexit__", new_callable=AsyncMock
            ):
                result = await agent._dispatch_task_to_purple(
                    task=budget_task,
                    shopper_endpoint="http://localhost:8001",
                    agent_id="test-agent",
                )

        assert result.completed is True
        assert result.error is None
        assert result.evaluation is not None

        # Verify MCP session was created
        mock_session_manager.create_session.assert_called_once()

        # Verify cleanup was called
        mock_session_manager.cleanup_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_execution_without_session_manager(
        self,
        mcp_agent_config,
        budget_task,
        mock_state_manager,
        mock_evaluator,
    ):
        """MCP execution works without session manager (no MCP URI)."""
        agent = WebShopPlusAgent(
            config=mcp_agent_config,
            state_manager=mock_state_manager,
            evaluator=mock_evaluator,
            session_manager=None,  # No session manager
        )

        mock_task_result = TaskResult(
            success=True,
            task_id="purple-task-1",
            context_id="ctx-1",
            final_state=TaskState.completed,
            result_data=None,
        )

        with patch.object(
            PurpleAgentClient, "__aenter__", new_callable=AsyncMock
        ) as mock_enter:
            mock_client = AsyncMock()
            mock_client.send_task.return_value = mock_task_result
            mock_enter.return_value = mock_client

            with patch.object(
                PurpleAgentClient, "__aexit__", new_callable=AsyncMock
            ):
                result = await agent._dispatch_task_to_purple(
                    task=budget_task,
                    shopper_endpoint="http://localhost:8001",
                    agent_id="test-agent",
                )

        assert result.completed is True
        # send_task should be called without mcp_uri
        mock_client.send_task.assert_called_once()
        call_kwargs = mock_client.send_task.call_args.kwargs
        assert call_kwargs["mcp_uri"] is None

    @pytest.mark.asyncio
    async def test_mcp_execution_failure(
        self,
        mcp_agent_config,
        budget_task,
        mock_state_manager,
        mock_evaluator,
    ):
        """MCP task execution handles failure."""
        # Create mock session manager that returns failure state
        mock_session_manager = AsyncMock()
        mock_server = MagicMock()
        mock_server.is_completed.return_value = True
        mock_server.get_final_result.return_value = {
            "success": False,  # Failed task
            "turns_used": 2,
        }
        mock_session_manager.create_session.return_value = None
        mock_session_manager.get_session.return_value = mock_server
        mock_session_manager.cleanup_session.return_value = True

        agent = WebShopPlusAgent(
            config=mcp_agent_config,
            state_manager=mock_state_manager,
            session_manager=mock_session_manager,
            evaluator=mock_evaluator,
        )

        mock_task_result = TaskResult(
            success=False,
            task_id="purple-task-1",
            context_id="ctx-1",
            final_state=TaskState.failed,
            error="Budget exceeded",
        )

        with patch.object(
            PurpleAgentClient, "__aenter__", new_callable=AsyncMock
        ) as mock_enter:
            mock_client = AsyncMock()
            mock_client.send_task.return_value = mock_task_result
            mock_enter.return_value = mock_client

            with patch.object(
                PurpleAgentClient, "__aexit__", new_callable=AsyncMock
            ) as mock_exit:
                mock_exit.return_value = None  # Don't suppress exceptions
                result = await agent._dispatch_task_to_purple(
                    task=budget_task,
                    shopper_endpoint="http://localhost:8001",
                    agent_id="test-agent",
                )

        assert result.completed is False
        assert "Budget exceeded" in result.error

    @pytest.mark.asyncio
    async def test_mcp_execution_client_error(
        self,
        mcp_agent_config,
        budget_task,
        mock_state_manager,
        mock_evaluator,
    ):
        """MCP execution handles PurpleAgentClient errors."""
        # Create mock session manager that's never accessed (due to early error)
        mock_session_manager = AsyncMock()
        mock_session_manager.create_session.return_value = None
        mock_session_manager.get_session.return_value = None
        mock_session_manager.cleanup_session.return_value = True

        agent = WebShopPlusAgent(
            config=mcp_agent_config,
            state_manager=mock_state_manager,
            session_manager=mock_session_manager,
            evaluator=mock_evaluator,
        )

        with patch.object(
            PurpleAgentClient, "__aenter__", new_callable=AsyncMock
        ) as mock_enter:
            mock_client = AsyncMock()
            mock_client.send_task.side_effect = TaskError("Connection refused")
            mock_enter.return_value = mock_client

            with patch.object(
                PurpleAgentClient, "__aexit__", new_callable=AsyncMock
            ) as mock_exit:
                mock_exit.return_value = None  # Don't suppress exceptions
                result = await agent._dispatch_task_to_purple(
                    task=budget_task,
                    shopper_endpoint="http://localhost:8001",
                    agent_id="test-agent",
                )

        assert result.error is not None
        assert "Purple agent error" in result.error or "Connection refused" in result.error


# =============================================================================
# Task Dispatch Tests
# =============================================================================


class TestTaskDispatch:
    """Tests for task dispatch in _execute_task."""

    @pytest.mark.asyncio
    async def test_dispatch_to_mcp(
        self,
        mcp_agent_config,
        budget_task,
        mock_state_manager,
        mock_evaluator,
    ):
        """Task dispatches to MCP path."""
        agent = WebShopPlusAgent(
            config=mcp_agent_config,
            state_manager=mock_state_manager,
            evaluator=mock_evaluator,
        )

        # Mock _dispatch_task_to_purple
        mock_result = TaskExecutionResult(
            task_id=budget_task.task_id,
            session_id="mcp-session",
            completed=True,
        )

        with patch.object(
            agent, "_dispatch_task_to_purple", new_callable=AsyncMock
        ) as mock_mcp:
            mock_mcp.return_value = mock_result

            result = await agent._execute_task(
                task=budget_task,
                shopper_endpoint="http://localhost:8001",
                agent_id="test-agent",
            )

        mock_mcp.assert_called_once_with(
            budget_task, "http://localhost:8001", "test-agent"
        )
        assert result.session_id == "mcp-session"


# =============================================================================
# Session Manager Integration Tests
# =============================================================================


class TestSessionManagerIntegration:
    """Tests for session manager integration."""

    @pytest.mark.asyncio
    async def test_mcp_session_created_with_correct_params(
        self,
        mcp_agent_config,
        budget_task,
        mock_state_manager,
        mock_session_manager,
        mock_evaluator,
    ):
        """MCP session is created with correct parameters."""
        agent = WebShopPlusAgent(
            config=mcp_agent_config,
            state_manager=mock_state_manager,
            session_manager=mock_session_manager,
            evaluator=mock_evaluator,
        )

        mock_task_result = TaskResult(
            success=True,
            task_id="t1",
            context_id="c1",
            final_state=TaskState.completed,
        )

        with patch.object(
            PurpleAgentClient, "__aenter__", new_callable=AsyncMock
        ) as mock_enter:
            mock_client = AsyncMock()
            mock_client.send_task.return_value = mock_task_result
            mock_enter.return_value = mock_client

            with patch.object(
                PurpleAgentClient, "__aexit__", new_callable=AsyncMock
            ):
                await agent._dispatch_task_to_purple(
                    task=budget_task,
                    shopper_endpoint="http://localhost:8001",
                    agent_id="test-agent",
                )

        # Check session was created with correct params
        call_kwargs = mock_session_manager.create_session.call_args.kwargs
        assert call_kwargs["goal"] == "Find running shoes under $50"
        assert call_kwargs["budget"] == 50.0
        assert call_kwargs["max_turns"] == 10
        assert "category: shoes" in call_kwargs["constraints"]

    @pytest.mark.asyncio
    async def test_mcp_result_extracted_from_session(
        self,
        mcp_agent_config,
        budget_task,
        mock_state_manager,
        mock_evaluator,
    ):
        """MCP session result is extracted and merged."""
        # Create mock session manager with specific result
        mock_session_manager = AsyncMock()
        mock_server = MagicMock()
        mock_server.is_completed.return_value = True
        mock_server.get_final_result.return_value = {
            "success": True,
            "turns_used": 3,
            "reward": 0.85,
        }
        mock_session_manager.create_session.return_value = None
        mock_session_manager.get_session.return_value = mock_server
        mock_session_manager.cleanup_session.return_value = True

        agent = WebShopPlusAgent(
            config=mcp_agent_config,
            state_manager=mock_state_manager,
            session_manager=mock_session_manager,
            evaluator=mock_evaluator,
        )

        mock_task_result = TaskResult(
            success=True,
            task_id="t1",
            context_id="c1",
            final_state=TaskState.completed,
        )

        with patch.object(
            PurpleAgentClient, "__aenter__", new_callable=AsyncMock
        ) as mock_enter:
            mock_client = AsyncMock()
            mock_client.send_task.return_value = mock_task_result
            mock_enter.return_value = mock_client

            with patch.object(
                PurpleAgentClient, "__aexit__", new_callable=AsyncMock
            ):
                result = await agent._dispatch_task_to_purple(
                    task=budget_task,
                    shopper_endpoint="http://localhost:8001",
                    agent_id="test-agent",
                )

        # Actions taken should come from MCP session result
        assert result.actions_taken == 3
        assert result.completed is True
        assert result.total_reward == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
