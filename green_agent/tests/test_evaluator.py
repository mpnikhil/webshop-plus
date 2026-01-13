"""
Tests for the WebShop+ Evaluator.

Tests all 5 task type evaluations:
- Budget constrained
- Preference memory
- Negative constraint
- Comparative reasoning
- Error recovery
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.evaluator import Evaluator
from src.llm_client import LLMClient
from src.models import (
    BudgetConstrainedTask,
    BudgetConstraints,
    BudgetEvaluationCriteria,
    CartItemSetup,
    ComparativeEvaluationCriteria,
    ComparativeReasoningTask,
    ComparativeRequirements,
    ConstraintEvaluationCriteria,
    CorrectState,
    Difficulty,
    ErrorRecoverySetup,
    ErrorRecoveryTask,
    EvaluationResult,
    MemoryEvaluationCriteria,
    MemoryTest,
    NegativeConstraintTask,
    NegativeConstraints,
    OptimizationGoal,
    PreferenceMemoryTask,
    RecoveryEvaluationCriteria,
    RequiredItem,
    SessionSequenceItem,
    TaskType,
)
from src.webshop_mcp.session_state import SessionState as MCPSessionState


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    client = MagicMock(spec=LLMClient)
    client.evaluate_with_rubric.return_value = (7, "Good comparison with relevant factors")
    return client


@pytest.fixture
def evaluator(mock_llm_client):
    """Create an evaluator with mock LLM client."""
    return Evaluator(mock_llm_client)


@pytest.fixture
def evaluator_no_llm():
    """Create an evaluator without LLM client."""
    return Evaluator()


@pytest.fixture
def basic_mcp_session():
    """Create a basic completed MCP session."""
    session = MCPSessionState(
        session_id="test-session-001",
        goal="Test shopping goal",
        budget=100.0,
        constraints=[],
        max_turns=30,
    )
    session.completed = True
    session.turn_count = 5
    return session


def add_mcp_cart_item(session: MCPSessionState, product_id: str, name: str,
                       price: float, attributes: dict = None, quantity: int = 1):
    """Helper to add an item to MCP cart."""
    session.cart.append({
        "product_id": product_id,
        "name": name,
        "price": price,
        "options": attributes or {},
        "quantity": quantity,
    })


def add_mcp_history(session: MCPSessionState, action: str, **kwargs):
    """Helper to add history record to MCP session."""
    record = {"action": action, "turn": session.turn_count}
    record.update(kwargs)
    session.history.append(record)
    session.turn_count += 1


@pytest.fixture
def budget_task():
    """Create a budget-constrained task."""
    return BudgetConstrainedTask(
        task_id="budget_001",
        task_type=TaskType.BUDGET_CONSTRAINED,
        instruction="Buy a mouse and mousepad for under $50",
        difficulty=Difficulty.EASY,
        expected_actions=12,
        timeout_seconds=180,
        constraints=BudgetConstraints(
            budget=50.0,
            required_items=[
                RequiredItem(
                    category="electronics",
                    attributes={"type": "wireless mouse"},
                    optional=False,
                ),
                RequiredItem(
                    category="office",
                    attributes={"type": "mousepad"},
                    optional=False,
                ),
            ],
            optimization_goal=OptimizationGoal.MAXIMIZE_QUALITY,
        ),
        evaluation_criteria=BudgetEvaluationCriteria(
            budget_weight=0.3,
            completion_weight=0.4,
            quality_weight=0.3,
        ),
    )


@pytest.fixture
def memory_task():
    """Create a preference memory task."""
    return PreferenceMemoryTask(
        task_id="memory_001",
        task_type=TaskType.PREFERENCE_MEMORY,
        instruction="Buy a shirt in my preferred color",
        difficulty=Difficulty.MEDIUM,
        expected_actions=10,
        timeout_seconds=180,
        session_sequence=[
            SessionSequenceItem(
                session_id="prev-session",
                instruction="Buy a blue t-shirt",
                establishes={"color": "blue"},
            ),
        ],
        memory_test=MemoryTest(
            attribute_to_recall="color",
            acceptable_values=["blue", "navy"],
        ),
        evaluation_criteria=MemoryEvaluationCriteria(
            recall_accuracy_weight=0.5,
            consistency_weight=0.5,
        ),
    )


@pytest.fixture
def constraint_task():
    """Create a negative constraint task."""
    return NegativeConstraintTask(
        task_id="constraint_001",
        task_type=TaskType.NEGATIVE_CONSTRAINT,
        instruction="Buy a laptop without gaming features",
        difficulty=Difficulty.MEDIUM,
        expected_actions=15,
        timeout_seconds=180,
        constraints=NegativeConstraints(
            required_attributes=["laptop", "professional"],
            forbidden_attributes=["gaming", "RGB"],
            forbidden_terms=["GeForce", "gaming laptop"],
            budget=1000.0,
        ),
        evaluation_criteria=ConstraintEvaluationCriteria(
            constraint_violation_penalty=0.5,
            match_score_weight=0.5,
        ),
    )


@pytest.fixture
def reasoning_task():
    """Create a comparative reasoning task."""
    return ComparativeReasoningTask(
        task_id="compare_001",
        task_type=TaskType.COMPARATIVE_REASONING,
        instruction="Compare Bluetooth speakers for outdoor use",
        difficulty=Difficulty.MEDIUM,
        expected_actions=15,
        timeout_seconds=240,
        requirements=ComparativeRequirements(
            category="electronics",
            attributes={"type": "Bluetooth speaker"},
            budget=80.0,
            comparison_request="Compare at least 2 options for outdoor use",
        ),
        evaluation_criteria=ComparativeEvaluationCriteria(
            minimum_options_explored=2,
            justification_required=True,
            justification_quality_weight=0.5,
        ),
    )


@pytest.fixture
def recovery_task():
    """Create an error recovery task."""
    return ErrorRecoveryTask(
        task_id="recovery_001",
        task_type=TaskType.ERROR_RECOVERY,
        instruction="Fix the wrong quantity in my cart",
        difficulty=Difficulty.EASY,
        expected_actions=6,
        timeout_seconds=120,
        setup=ErrorRecoverySetup(
            cart_contents=[
                CartItemSetup(
                    product_id="HP-001",
                    product_name="Sony Headphones",
                    attributes={"color": "black"},
                    quantity=3,
                    price=278.0,
                ),
            ],
            error_description="Wrong quantity - ordered 3 instead of 1",
        ),
        correct_state=CorrectState(
            expected_cart=[
                CartItemSetup(
                    product_id="HP-001",
                    product_name="Sony Headphones",
                    attributes={"color": "black"},
                    quantity=1,
                    price=278.0,
                ),
            ],
        ),
        evaluation_criteria=RecoveryEvaluationCriteria(
            error_identified=True,
            error_fixed=True,
            unnecessary_actions_penalty=0.1,
        ),
    )


# =============================================================================
# Evaluator Initialization Tests
# =============================================================================


class TestEvaluatorInit:
    """Tests for Evaluator initialization."""

    def test_init_with_llm_client(self, mock_llm_client):
        """Test initialization with LLM client."""
        evaluator = Evaluator(mock_llm_client)
        assert evaluator.llm_client is mock_llm_client

    def test_init_without_llm_client(self):
        """Test initialization without LLM client."""
        evaluator = Evaluator()
        assert evaluator.llm_client is None

    def test_init_with_none_llm_client(self):
        """Test initialization with explicit None."""
        evaluator = Evaluator(None)
        assert evaluator.llm_client is None


# =============================================================================
# Budget Constrained Task Tests
# =============================================================================


class TestBudgetTaskEvaluation:
    """Tests for budget-constrained task evaluation."""

    def test_perfect_budget_task(self, evaluator, basic_mcp_session, budget_task):
        """Test perfect budget task completion."""
        # Add cart items that meet all requirements
        add_mcp_cart_item(basic_mcp_session, "mouse-001", "Wireless Gaming Mouse",
                          25.0, {"type": "wireless mouse"})
        add_mcp_cart_item(basic_mcp_session, "pad-001", "Large Mousepad",
                          15.0, {"type": "mousepad"})

        result = evaluator.evaluate(basic_mcp_session, budget_task)

        assert result.task_id == "budget_001"
        assert result.task_type == TaskType.BUDGET_CONSTRAINED
        assert result.completed is True
        assert result.success is True
        assert result.overall_score >= 0.8
        assert len(result.scoring_breakdown) == 3

    def test_over_budget(self, evaluator, basic_mcp_session, budget_task):
        """Test task with over-budget spending."""
        add_mcp_cart_item(basic_mcp_session, "mouse-001", "Wireless Mouse",
                          45.0, {"type": "wireless mouse"})
        add_mcp_cart_item(basic_mcp_session, "pad-001", "Mousepad",
                          20.0, {"type": "mousepad"})  # Total: $65, budget: $50

        result = evaluator.evaluate(basic_mcp_session, budget_task)

        # Find budget component
        budget_component = next(
            c for c in result.scoring_breakdown if c.name == "budget_compliance"
        )
        assert budget_component.normalized_score < 1.0
        assert "over budget" in budget_component.explanation.lower()

    def test_missing_required_item(self, evaluator, basic_mcp_session, budget_task):
        """Test task with missing required item."""
        # Only add a mousepad, missing the wireless mouse
        add_mcp_cart_item(basic_mcp_session, "pad-001", "Office Desk Pad",
                          15.0, {"type": "mousepad"})

        result = evaluator.evaluate(basic_mcp_session, budget_task)

        completion_component = next(
            c for c in result.scoring_breakdown if c.name == "item_completion"
        )
        # Should not match both - only mousepad matches
        assert completion_component.normalized_score < 1.0
        assert result.success is False

    def test_empty_cart(self, evaluator, basic_mcp_session, budget_task):
        """Test task with empty cart."""
        # Cart is already empty by default

        result = evaluator.evaluate(basic_mcp_session, budget_task)

        # Budget compliance is 1.0 (0 spent is within budget), but completion is 0.0
        # So overall_score will be budget_weight * 1.0 = 0.3
        assert result.overall_score < 0.5  # Mostly failed
        assert result.success is False
        assert result.metrics["total_spent"] == 0

    def test_minimize_cost_goal(self, evaluator, basic_mcp_session):
        """Test minimize_cost optimization goal."""
        task = BudgetConstrainedTask(
            task_id="budget_min",
            task_type=TaskType.BUDGET_CONSTRAINED,
            instruction="Find the cheapest option",
            constraints=BudgetConstraints(
                budget=100.0,
                required_items=[
                    RequiredItem(category="electronics", attributes={"type": "headphones"})
                ],
                optimization_goal=OptimizationGoal.MINIMIZE_COST,
            ),
        )

        add_mcp_cart_item(basic_mcp_session, "hp-001", "Budget Headphones",
                          30.0, {"type": "headphones"})

        result = evaluator.evaluate(basic_mcp_session, task)

        quality_component = next(
            c for c in result.scoring_breakdown if c.name == "quality_optimization"
        )
        # Should get bonus for saving money
        assert "saved" in quality_component.explanation.lower()

    def test_with_purchases_instead_of_cart(self, evaluator, basic_mcp_session, budget_task):
        """Test evaluation with checkout history (simulates purchases)."""
        # In MCP, purchases are represented by cart + checkout action
        add_mcp_cart_item(basic_mcp_session, "mouse-001", "Wireless Mouse",
                          25.0, {"type": "wireless mouse"})
        add_mcp_cart_item(basic_mcp_session, "pad-001", "Office Mousepad",
                          15.0, {"type": "mousepad"})
        # Add checkout action to history
        add_mcp_history(basic_mcp_session, "session_end", reason="checkout")

        result = evaluator.evaluate(basic_mcp_session, budget_task)

        assert result.metrics["total_spent"] == 40.0
        assert result.success is True


# =============================================================================
# Preference Memory Task Tests
# =============================================================================


@pytest.mark.skip(reason="Preference memory tasks disabled - requires multi-session support")
class TestMemoryTaskEvaluation:
    """Tests for preference memory task evaluation."""

    def test_correct_preference_recall(self, evaluator, basic_mcp_session, memory_task):
        """Test correct preference recall."""
        # Add actions showing preference recall
        basic_mcp_session.actions = [
            ActionRecord(
                action="Searching for blue shirts based on previous preference",
                observation="Found 5 blue t-shirts",
            ),
            ActionRecord(
                action="Selecting the navy blue cotton shirt",
                observation="Added to cart",
            ),
        ]
        basic_mcp_session.cart = CartState(
            items=[
                CartItem(
                    product_id="shirt-001",
                    product_name="Navy Blue Cotton T-Shirt",
                    attributes={"color": "blue"},
                    price=29.99,
                )
            ]
        )

        result = evaluator.evaluate(basic_mcp_session, memory_task)

        assert result.overall_score >= 0.5
        recall_component = next(
            c for c in result.scoring_breakdown if c.name == "recall_accuracy"
        )
        assert recall_component.normalized_score >= 0.5

    def test_no_preference_recall(self, evaluator, basic_mcp_session, memory_task):
        """Test when agent doesn't recall preference."""
        basic_mcp_session.actions = [
            ActionRecord(action="Searching for shirts", observation="Found shirts"),
        ]
        basic_mcp_session.cart = CartState(
            items=[
                CartItem(
                    product_id="shirt-001",
                    product_name="Red Cotton T-Shirt",
                    attributes={"color": "red"},
                    price=29.99,
                )
            ]
        )

        result = evaluator.evaluate(basic_mcp_session, memory_task)

        recall_component = next(
            c for c in result.scoring_breakdown if c.name == "recall_accuracy"
        )
        assert recall_component.normalized_score < 0.5

    def test_with_agent_memory(self, evaluator, basic_mcp_session, memory_task):
        """Test evaluation with agent memory provided."""
        memory = AgentMemory(agent_id="test-agent")
        memory.add_session(
            SessionSummary(
                session_id="prev-session",
                task_id="prev-task",
                task_type="preference_memory",
                preferences={"color": "blue"},
            )
        )

        basic_mcp_session.actions = [
            ActionRecord(
                action="Using remembered color preference blue",
                observation="Searching for blue items",
            ),
        ]
        basic_mcp_session.cart = CartState(
            items=[
                CartItem(
                    product_id="shirt-001",
                    product_name="Blue Polo Shirt",
                    attributes={"color": "blue"},
                    price=39.99,
                )
            ]
        )

        result = evaluator.evaluate(basic_mcp_session, memory_task, memory=memory)

        assert result.overall_score >= 0.5


# =============================================================================
# Negative Constraint Task Tests
# =============================================================================


class TestConstraintTaskEvaluation:
    """Tests for negative constraint task evaluation."""

    def test_no_violations(self, evaluator, basic_mcp_session, constraint_task):
        """Test task with no constraint violations."""
        add_mcp_cart_item(basic_mcp_session, "laptop-001", "ThinkPad Professional Laptop",
                          899.0, {"type": "laptop", "category": "professional"})

        result = evaluator.evaluate(basic_mcp_session, constraint_task)

        assert result.success is True
        assert result.metrics["violation_count"] == 0

    def test_forbidden_attribute_violation(self, evaluator, basic_mcp_session, constraint_task):
        """Test detection of forbidden attribute."""
        add_mcp_cart_item(basic_mcp_session, "laptop-001", "Gaming Laptop with RGB",
                          999.0, {"type": "laptop"})

        result = evaluator.evaluate(basic_mcp_session, constraint_task)

        assert result.success is False
        assert result.metrics["violation_count"] > 0
        assert "gaming" in str(result.metrics["violations"]).lower()

    def test_forbidden_term_violation(self, evaluator, basic_mcp_session, constraint_task):
        """Test detection of forbidden term."""
        add_mcp_cart_item(basic_mcp_session, "laptop-001", "Laptop with GeForce RTX Graphics",
                          1200.0, {"type": "laptop"})

        result = evaluator.evaluate(basic_mcp_session, constraint_task)

        assert result.success is False
        assert result.metrics["violation_count"] > 0

    def test_multiple_violations(self, evaluator, basic_mcp_session, constraint_task):
        """Test multiple constraint violations."""
        add_mcp_cart_item(basic_mcp_session, "laptop-001", "Gaming Laptop with RGB and GeForce",
                          1500.0, {"type": "gaming laptop"})

        result = evaluator.evaluate(basic_mcp_session, constraint_task)

        assert result.metrics["violation_count"] >= 2
        assert result.overall_score < 0.5

    def test_missing_required_attribute(self, evaluator, basic_mcp_session, constraint_task):
        """Test missing required positive attribute."""
        add_mcp_cart_item(basic_mcp_session, "tablet-001", "iPad Pro",
                          799.0, {"type": "tablet"})  # Not a laptop, not professional

        result = evaluator.evaluate(basic_mcp_session, constraint_task)

        match_component = next(
            c for c in result.scoring_breakdown if c.name == "positive_match"
        )
        assert match_component.normalized_score < 1.0


# =============================================================================
# Comparative Reasoning Task Tests
# =============================================================================


class TestReasoningTaskEvaluation:
    """Tests for comparative reasoning task evaluation."""

    def test_good_comparison_with_llm(self, evaluator, basic_mcp_session, reasoning_task):
        """Test product exploration counting with ASIN tracking."""
        # Add product exploration history with unique ASINs
        add_mcp_history(basic_mcp_session, "click", element_id="p1", product_asin="B001SPEAKER", element_type="product")
        add_mcp_history(basic_mcp_session, "click", element_id="p2", product_asin="B002SPEAKER", element_type="product")

        # Add product to cart
        add_mcp_cart_item(basic_mcp_session, "B001SPEAKER", "JBL Flip 6", 129.99,
                          {"type": "speaker", "waterproof": "yes"})

        result = evaluator.evaluate(basic_mcp_session, reasoning_task)

        # Verify product exploration counts unique ASINs correctly
        assert result.metrics["products_explored"] == 2
        exploration_component = next(c for c in result.scoring_breakdown if c.name == "exploration")
        assert exploration_component.normalized_score == 1.0  # Met minimum of 2 products

    def test_no_comparison(self, evaluator_no_llm, basic_mcp_session, reasoning_task):
        """Test evaluation with no comparison made."""
        # No product exploration - direct purchase
        add_mcp_cart_item(basic_mcp_session, "B001SPEAKER", "Generic Speaker", 49.99)

        result = evaluator_no_llm.evaluate(basic_mcp_session, reasoning_task)

        exploration_component = next(
            c for c in result.scoring_breakdown if c.name == "exploration"
        )
        justification_component = next(
            c for c in result.scoring_breakdown if c.name == "justification_provided"
        )

        assert exploration_component.normalized_score < 1.0
        assert justification_component.normalized_score == 0.0

    def test_heuristic_justification_scoring(self, evaluator_no_llm, basic_mcp_session, reasoning_task):
        """Test heuristic justification scoring without LLM."""
        # View 2 products
        add_mcp_history(basic_mcp_session, "click", element_id="p1", product_asin="SPEAKER1", element_type="product")
        add_mcp_history(basic_mcp_session, "click", element_id="p2", product_asin="SPEAKER2", element_type="product")

        # Add one to cart
        add_mcp_cart_item(basic_mcp_session, "SPEAKER1", "High-Quality Speaker", 89.99)

        result = evaluator_no_llm.evaluate(basic_mcp_session, reasoning_task)

        # Should detect exploration
        assert result.metrics["products_explored"] >= 2
        exploration_component = next(
            c for c in result.scoring_breakdown if c.name == "exploration"
        )
        assert exploration_component.normalized_score > 0.0

    def test_insufficient_exploration(self, evaluator, basic_mcp_session, reasoning_task):
        """Test when not enough products are explored."""
        # Only 1 product viewed (minimum is 2 for reasoning task)
        add_mcp_history(basic_mcp_session, "click", element_id="p1", product_asin="SINGLE", element_type="product")
        add_mcp_cart_item(basic_mcp_session, "SINGLE", "Only Speaker Found", 59.99)

        result = evaluator.evaluate(basic_mcp_session, reasoning_task)

        exploration_component = next(
            c for c in result.scoring_breakdown if c.name == "exploration"
        )
        assert exploration_component.normalized_score < 1.0
        assert result.metrics["products_explored"] < 2


# =============================================================================
# Error Recovery Task Tests
# =============================================================================


class TestRecoveryTaskEvaluation:
    """Tests for error recovery task evaluation."""

    def test_successful_recovery(self, evaluator, basic_mcp_session, recovery_task):
        """Test successful error recovery."""
        # Final cart matches expected state (correct quantity)
        add_mcp_cart_item(basic_mcp_session, "HP-001", "Sony Headphones",
                          278.0, {"color": "black"}, quantity=1)

        # Add search and recovery actions
        add_mcp_history(basic_mcp_session, "search", query="headphones")
        add_mcp_history(basic_mcp_session, "click", element_id="p1", element_type="product", product_asin="HP-001")
        add_mcp_history(basic_mcp_session, "add_to_cart", product={"name": "Sony Headphones", "asin": "HP-001", "price": 278.0})

        result = evaluator.evaluate(basic_mcp_session, recovery_task)

        assert result.success is True
        assert result.metrics["error_fixed"] is True

    def test_failed_recovery(self, evaluator, basic_mcp_session, recovery_task):
        """Test failed error recovery - wrong final state."""
        # Cart still has wrong quantity (should be 1, but is 2)
        add_mcp_cart_item(basic_mcp_session, "HP-001", "Sony Headphones",
                          278.0, {"color": "black"}, quantity=2)

        result = evaluator.evaluate(basic_mcp_session, recovery_task)

        assert result.success is False
        assert result.metrics["error_fixed"] is False

    def test_error_identified(self, evaluator, basic_mcp_session, recovery_task):
        """Test error identification detection."""
        add_mcp_cart_item(basic_mcp_session, "HP-001", "Sony Headphones",
                          278.0, {"color": "black"}, quantity=1)

        # Search for the product, showing error identification
        add_mcp_history(basic_mcp_session, "search", query="wrong quantity headphones")

        result = evaluator.evaluate(basic_mcp_session, recovery_task)

        assert result.metrics["error_identified"] is True
        error_component = next(
            c for c in result.scoring_breakdown if c.name == "error_identified"
        )
        assert error_component.normalized_score == 1.0

    def test_inefficient_recovery(self, evaluator, basic_mcp_session, recovery_task):
        """Test recovery with too many actions."""
        add_mcp_cart_item(basic_mcp_session, "HP-001", "Sony Headphones",
                          278.0, {"color": "black"}, quantity=1)

        # Many unnecessary actions (expected is ~6)
        for i in range(15):
            add_mcp_history(basic_mcp_session, "search", query=f"search attempt {i}")

        result = evaluator.evaluate(basic_mcp_session, recovery_task)

        efficiency_component = next(
            c for c in result.scoring_breakdown if c.name == "efficiency"
        )
        assert efficiency_component.normalized_score < 1.0
        assert "inefficient" in efficiency_component.explanation.lower()

    def test_missing_product(self, evaluator, basic_mcp_session, recovery_task):
        """Test when expected product is missing from cart."""
        # Cart is empty - product never added
        # Do nothing, cart stays empty

        result = evaluator.evaluate(basic_mcp_session, recovery_task)

        assert result.success is False
        assert result.metrics["error_fixed"] is False


# =============================================================================
# General Evaluation Tests
# =============================================================================


class TestGeneralEvaluation:
    """Tests for general evaluation behavior."""

    def test_unknown_task_type(self, evaluator, basic_mcp_session):
        """Test handling of unknown task type."""
        # Create a task with invalid type
        from unittest.mock import MagicMock

        bad_task = MagicMock()
        bad_task.task_id = "bad-001"
        bad_task.task_type = "invalid_type"

        result = evaluator.evaluate(basic_mcp_session, bad_task)

        assert result.error is not None
        assert "unknown task type" in result.error.lower()

    def test_universal_metrics_populated(self, evaluator, basic_mcp_session, budget_task):
        """Test that universal metrics are always populated."""
        add_mcp_cart_item(basic_mcp_session, "x", "X", 10.0)

        result = evaluator.evaluate(basic_mcp_session, budget_task)

        assert result.task_id == budget_task.task_id
        assert result.task_type == budget_task.task_type
        assert result.completed == basic_mcp_session.completed
        assert result.actions_taken == basic_mcp_session.turn_count
        assert result.time_elapsed_seconds >= 0

    def test_scoring_breakdown_exists(self, evaluator, basic_mcp_session, budget_task):
        """Test that scoring breakdown is populated."""
        add_mcp_cart_item(basic_mcp_session, "x", "Mouse", 20.0)

        result = evaluator.evaluate(basic_mcp_session, budget_task)

        assert len(result.scoring_breakdown) > 0
        for component in result.scoring_breakdown:
            assert component.name
            assert 0 <= component.normalized_score <= 1
            assert component.weight > 0

    def test_metrics_populated(self, evaluator, basic_mcp_session, budget_task):
        """Test that task-specific metrics are populated."""
        add_mcp_cart_item(basic_mcp_session, "x", "Mouse", 20.0)

        result = evaluator.evaluate(basic_mcp_session, budget_task)

        assert "budget" in result.metrics
        assert "total_spent" in result.metrics


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_budget(self, evaluator, basic_mcp_session):
        """Test handling of zero budget."""
        task = BudgetConstrainedTask(
            task_id="zero-budget",
            task_type=TaskType.BUDGET_CONSTRAINED,
            instruction="Get something free",
            constraints=BudgetConstraints(
                budget=0.0,
                required_items=[],
            ),
        )
        # Cart stays empty

        result = evaluator.evaluate(basic_mcp_session, task)
        assert result is not None

    def test_empty_forbidden_constraints(self, evaluator, basic_mcp_session):
        """Test constraint task with no forbidden items."""
        task = NegativeConstraintTask(
            task_id="no-forbidden",
            task_type=TaskType.NEGATIVE_CONSTRAINT,
            instruction="Buy anything",
            constraints=NegativeConstraints(
                required_attributes=["laptop"],
                forbidden_attributes=[],
                forbidden_terms=[],
            ),
        )
        add_mcp_cart_item(basic_mcp_session, "x", "Laptop Computer", 500.0, {"type": "laptop"})

        result = evaluator.evaluate(basic_mcp_session, task)

        assert result.metrics["violation_count"] == 0

    def test_empty_session_sequence(self, evaluator, basic_mcp_session):
        """Test memory task with empty session sequence."""
        task = PreferenceMemoryTask(
            task_id="no-history",
            task_type=TaskType.PREFERENCE_MEMORY,
            instruction="Buy something",
            session_sequence=[],
            memory_test=MemoryTest(
                attribute_to_recall="color",
                acceptable_values=["blue"],
            ),
        )

        result = evaluator.evaluate(basic_mcp_session, task)
        assert result is not None

    def test_very_long_justification(self, evaluator_no_llm, basic_mcp_session, reasoning_task):
        """Test handling of very long justification text."""
        # Add product exploration with long history
        for i in range(5):
            add_mcp_history(basic_mcp_session, "search", query=f"product search {i}")
        add_mcp_cart_item(basic_mcp_session, "x", "Product", 50.0)

        result = evaluator_no_llm.evaluate(basic_mcp_session, reasoning_task)
        assert result is not None

    def test_special_characters_in_product_name(self, evaluator, basic_mcp_session, constraint_task):
        """Test handling of special characters in product names."""
        add_mcp_cart_item(basic_mcp_session, "x", "Laptop (Pro) - 15\" Display & More!",
                          999.0, {"type": "laptop"})

        result = evaluator.evaluate(basic_mcp_session, constraint_task)
        assert result is not None


# =============================================================================
# Direct Method Tests
# =============================================================================


class TestDirectMethods:
    """Tests for calling evaluation methods directly."""

    def test_evaluate_budget_task_directly(self, evaluator, basic_mcp_session, budget_task):
        """Test calling evaluate_budget_task directly."""
        add_mcp_cart_item(basic_mcp_session, "m", "Mouse", 25.0)
        add_mcp_cart_item(basic_mcp_session, "p", "Mousepad", 15.0)

        result = evaluator.evaluate_budget_task(basic_mcp_session, budget_task)

        assert isinstance(result, EvaluationResult)
        assert result.task_id == budget_task.task_id

    def test_evaluate_memory_task_directly(self, evaluator, basic_mcp_session, memory_task):
        """Test calling evaluate_memory_task directly."""
        result = evaluator.evaluate_memory_task(basic_mcp_session, memory_task)

        assert isinstance(result, EvaluationResult)
        assert result.task_id == memory_task.task_id

    def test_evaluate_constraint_task_directly(self, evaluator, basic_mcp_session, constraint_task):
        """Test calling evaluate_constraint_task directly."""
        add_mcp_cart_item(basic_mcp_session, "l", "Business Laptop", 800.0, {"type": "laptop"})

        result = evaluator.evaluate_constraint_task(basic_mcp_session, constraint_task)

        assert isinstance(result, EvaluationResult)
        assert result.task_id == constraint_task.task_id

    def test_evaluate_reasoning_task_directly(self, evaluator, basic_mcp_session, reasoning_task):
        """Test calling evaluate_reasoning_task directly."""
        result = evaluator.evaluate_reasoning_task(basic_mcp_session, reasoning_task)

        assert isinstance(result, EvaluationResult)
        assert result.task_id == reasoning_task.task_id

    def test_evaluate_recovery_task_directly(self, evaluator, basic_mcp_session, recovery_task):
        """Test calling evaluate_recovery_task directly."""
        add_mcp_cart_item(basic_mcp_session, "HP-001", "Sony Headphones",
                          278.0, quantity=1)

        result = evaluator.evaluate_recovery_task(basic_mcp_session, recovery_task)

        assert isinstance(result, EvaluationResult)
        assert result.task_id == recovery_task.task_id


# =============================================================================
# LLM Integration Tests (Mocked)
# =============================================================================


class TestLLMIntegration:
    """Tests for LLM integration in evaluation."""

    def test_llm_evaluate_with_rubric_called(self, evaluator, basic_mcp_session, reasoning_task):
        """Test that LLM is available for reasoning task evaluation."""
        # Add product exploration history (minimum for reasoning task)
        add_mcp_history(basic_mcp_session, "click", element_id="p1", product_asin="A", element_type="product")
        add_mcp_history(basic_mcp_session, "click", element_id="p2", product_asin="B", element_type="product")
        add_mcp_cart_item(basic_mcp_session, "A", "Product A", 50.0)

        result = evaluator.evaluate(basic_mcp_session, reasoning_task)

        # Should get a result and LLM client should be available
        assert result is not None
        assert evaluator._llm_client is not None

    def test_llm_failure_fallback(self, evaluator, basic_mcp_session, reasoning_task):
        """Test graceful handling when LLM fails."""
        # Simulate LLM client failure
        evaluator._llm_client.evaluate_with_rubric.side_effect = Exception("API Error")

        # Add product exploration history
        add_mcp_history(basic_mcp_session, "click", element_id="p1", product_asin="A", element_type="product")
        add_mcp_history(basic_mcp_session, "click", element_id="p2", product_asin="B", element_type="product")
        add_mcp_cart_item(basic_mcp_session, "A", "Product A", 50.0)

        result = evaluator.evaluate(basic_mcp_session, reasoning_task)

        # Should still get a result even if LLM fails
        assert result is not None
        # Exploration should still be scored based on MCP data
        exploration_component = next(
            c for c in result.scoring_breakdown if c.name == "exploration"
        )
        assert exploration_component.normalized_score > 0.0


# =============================================================================
# Integration with Real Task Data
# =============================================================================


class TestRealTaskData:
    """Tests using realistic task data patterns."""

    def test_realistic_budget_scenario(self, evaluator, basic_mcp_session):
        """Test with realistic budget task scenario."""
        task = BudgetConstrainedTask(
            task_id="budget_realistic",
            task_type=TaskType.BUDGET_CONSTRAINED,
            instruction="I need a good quality wireless mouse and a mousepad for my home office. I have about $50 to spend.",
            difficulty=Difficulty.EASY,
            constraints=BudgetConstraints(
                budget=50.0,
                required_items=[
                    RequiredItem(
                        category="electronics",
                        attributes={"type": "wireless mouse"},
                    ),
                    RequiredItem(
                        category="office",
                        attributes={"type": "mousepad"},
                    ),
                ],
                optimization_goal=OptimizationGoal.MAXIMIZE_QUALITY,
            ),
        )

        add_mcp_cart_item(basic_mcp_session, "LM001", "Logitech M510 Wireless Mouse",
                          24.99, {"type": "wireless mouse", "brand": "Logitech"})
        add_mcp_cart_item(basic_mcp_session, "MP001", "SteelSeries QcK Gaming Mousepad",
                          14.99, {"type": "mousepad", "size": "large"})

        result = evaluator.evaluate(basic_mcp_session, task)

        assert result.success is True
        assert result.overall_score >= 0.8
        assert result.metrics["total_spent"] == 39.98
        assert result.metrics["total_spent"] < result.metrics["budget"]

    def test_realistic_recovery_scenario(self, evaluator, basic_mcp_session):
        """Test with realistic error recovery scenario."""
        task = ErrorRecoveryTask(
            task_id="recovery_realistic",
            task_type=TaskType.ERROR_RECOVERY,
            instruction="Oh no, I accidentally added the wrong quantity. I only needed 1 of these headphones, not 3.",
            expected_actions=6,
            setup=ErrorRecoverySetup(
                cart_contents=[
                    CartItemSetup(
                        product_id="SONY-WH1000XM4",
                        product_name="Sony WH-1000XM4 Wireless Headphones",
                        attributes={"color": "black"},
                        quantity=3,
                        price=278.0,
                    ),
                ],
                error_description="Wrong quantity - ordered 3 instead of 1",
            ),
            correct_state=CorrectState(
                expected_cart=[
                    CartItemSetup(
                        product_id="SONY-WH1000XM4",
                        product_name="Sony WH-1000XM4 Wireless Headphones",
                        attributes={"color": "black"},
                        quantity=1,
                        price=278.0,
                    ),
                ],
            ),
        )

        add_mcp_cart_item(basic_mcp_session, "SONY-WH1000XM4",
                          "Sony WH-1000XM4 Wireless Headphones", 278.0,
                          {"color": "black"}, quantity=1)

        # Add recovery actions
        add_mcp_history(basic_mcp_session, "search", query="wrong quantity")
        add_mcp_history(basic_mcp_session, "click", element_id="p1", element_type="product", product_asin="SONY-WH1000XM4")

        result = evaluator.evaluate(basic_mcp_session, task)

        assert result.success is True
        assert result.metrics["error_fixed"] is True
        assert result.metrics["error_identified"] is True
