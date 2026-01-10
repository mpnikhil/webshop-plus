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
    ActionRecord,
    AgentMemory,
    BudgetConstrainedTask,
    BudgetConstraints,
    BudgetEvaluationCriteria,
    CartItem,
    CartItemSetup,
    CartState,
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
    PurchaseRecord,
    RecoveryEvaluationCriteria,
    RequiredItem,
    SessionSequenceItem,
    SessionState,
    SessionSummary,
    TaskType,
)


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
def basic_session():
    """Create a basic completed session."""
    session = SessionState(
        session_id="test-session-001",
        task_id="test-task-001",
        agent_id="test-agent",
        started_at=datetime(2024, 1, 1, 10, 0, 0),
    )
    session.completed = True
    session.ended_at = datetime(2024, 1, 1, 10, 5, 0)
    return session


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

    def test_perfect_budget_task(self, evaluator, basic_session, budget_task):
        """Test perfect budget task completion."""
        # Add cart items that meet all requirements
        basic_session.cart = CartState(
            items=[
                CartItem(
                    product_id="mouse-001",
                    product_name="Wireless Gaming Mouse",
                    attributes={"type": "wireless mouse"},
                    quantity=1,
                    price=25.0,
                ),
                CartItem(
                    product_id="pad-001",
                    product_name="Large Mousepad",
                    attributes={"type": "mousepad"},
                    quantity=1,
                    price=15.0,
                ),
            ]
        )

        result = evaluator.evaluate(basic_session, budget_task)

        assert result.task_id == "budget_001"
        assert result.task_type == TaskType.BUDGET_CONSTRAINED
        assert result.completed is True
        assert result.success is True
        assert result.overall_score >= 0.8
        assert len(result.scoring_breakdown) == 3

    def test_over_budget(self, evaluator, basic_session, budget_task):
        """Test task with over-budget spending."""
        basic_session.cart = CartState(
            items=[
                CartItem(
                    product_id="mouse-001",
                    product_name="Wireless Mouse",
                    attributes={"type": "wireless mouse"},
                    quantity=1,
                    price=45.0,
                ),
                CartItem(
                    product_id="pad-001",
                    product_name="Mousepad",
                    attributes={"type": "mousepad"},
                    quantity=1,
                    price=20.0,  # Total: $65, budget: $50
                ),
            ]
        )

        result = evaluator.evaluate(basic_session, budget_task)

        # Find budget component
        budget_component = next(
            c for c in result.scoring_breakdown if c.name == "budget_compliance"
        )
        assert budget_component.normalized_score < 1.0
        assert "over budget" in budget_component.explanation.lower()

    def test_missing_required_item(self, evaluator, basic_session, budget_task):
        """Test task with missing required item."""
        # Only add a mousepad, missing the wireless mouse
        basic_session.cart = CartState(
            items=[
                CartItem(
                    product_id="pad-001",
                    product_name="Office Desk Pad",
                    attributes={"type": "mousepad"},
                    quantity=1,
                    price=15.0,
                ),
                # Missing wireless mouse - the mousepad doesn't satisfy the mouse requirement
            ]
        )

        result = evaluator.evaluate(basic_session, budget_task)

        completion_component = next(
            c for c in result.scoring_breakdown if c.name == "item_completion"
        )
        # Should not match both - only mousepad matches
        assert completion_component.normalized_score < 1.0
        assert result.success is False

    def test_empty_cart(self, evaluator, basic_session, budget_task):
        """Test task with empty cart."""
        basic_session.cart = CartState(items=[])

        result = evaluator.evaluate(basic_session, budget_task)

        # Budget compliance is 1.0 (0 spent is within budget), but completion is 0.0
        # So overall_score will be budget_weight * 1.0 = 0.3
        assert result.overall_score < 0.5  # Mostly failed
        assert result.success is False
        assert result.metrics["total_spent"] == 0

    def test_minimize_cost_goal(self, evaluator, basic_session):
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

        basic_session.cart = CartState(
            items=[
                CartItem(
                    product_id="hp-001",
                    product_name="Budget Headphones",
                    attributes={"type": "headphones"},
                    price=30.0,
                )
            ]
        )

        result = evaluator.evaluate(basic_session, task)

        quality_component = next(
            c for c in result.scoring_breakdown if c.name == "quality_optimization"
        )
        # Should get bonus for saving money
        assert "saved" in quality_component.explanation.lower()

    def test_with_purchases_instead_of_cart(self, evaluator, basic_session, budget_task):
        """Test evaluation using purchases instead of cart."""
        basic_session.purchases = [
            PurchaseRecord(
                product_id="mouse-001",
                product_name="Wireless Mouse",
                attributes={"type": "wireless mouse"},
                price=25.0,
            ),
            PurchaseRecord(
                product_id="pad-001",
                product_name="Office Mousepad",
                attributes={"type": "mousepad"},
                price=15.0,
            ),
        ]

        result = evaluator.evaluate(basic_session, budget_task)

        assert result.metrics["total_spent"] == 40.0
        assert result.success is True


# =============================================================================
# Preference Memory Task Tests
# =============================================================================


class TestMemoryTaskEvaluation:
    """Tests for preference memory task evaluation."""

    def test_correct_preference_recall(self, evaluator, basic_session, memory_task):
        """Test correct preference recall."""
        # Add actions showing preference recall
        basic_session.actions = [
            ActionRecord(
                action="Searching for blue shirts based on previous preference",
                observation="Found 5 blue t-shirts",
            ),
            ActionRecord(
                action="Selecting the navy blue cotton shirt",
                observation="Added to cart",
            ),
        ]
        basic_session.cart = CartState(
            items=[
                CartItem(
                    product_id="shirt-001",
                    product_name="Navy Blue Cotton T-Shirt",
                    attributes={"color": "blue"},
                    price=29.99,
                )
            ]
        )

        result = evaluator.evaluate(basic_session, memory_task)

        assert result.overall_score >= 0.5
        recall_component = next(
            c for c in result.scoring_breakdown if c.name == "recall_accuracy"
        )
        assert recall_component.normalized_score >= 0.5

    def test_no_preference_recall(self, evaluator, basic_session, memory_task):
        """Test when agent doesn't recall preference."""
        basic_session.actions = [
            ActionRecord(action="Searching for shirts", observation="Found shirts"),
        ]
        basic_session.cart = CartState(
            items=[
                CartItem(
                    product_id="shirt-001",
                    product_name="Red Cotton T-Shirt",
                    attributes={"color": "red"},
                    price=29.99,
                )
            ]
        )

        result = evaluator.evaluate(basic_session, memory_task)

        recall_component = next(
            c for c in result.scoring_breakdown if c.name == "recall_accuracy"
        )
        assert recall_component.normalized_score < 0.5

    def test_with_agent_memory(self, evaluator, basic_session, memory_task):
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

        basic_session.actions = [
            ActionRecord(
                action="Using remembered color preference blue",
                observation="Searching for blue items",
            ),
        ]
        basic_session.cart = CartState(
            items=[
                CartItem(
                    product_id="shirt-001",
                    product_name="Blue Polo Shirt",
                    attributes={"color": "blue"},
                    price=39.99,
                )
            ]
        )

        result = evaluator.evaluate(basic_session, memory_task, memory=memory)

        assert result.overall_score >= 0.5


# =============================================================================
# Negative Constraint Task Tests
# =============================================================================


class TestConstraintTaskEvaluation:
    """Tests for negative constraint task evaluation."""

    def test_no_violations(self, evaluator, basic_session, constraint_task):
        """Test task with no constraint violations."""
        basic_session.cart = CartState(
            items=[
                CartItem(
                    product_id="laptop-001",
                    product_name="ThinkPad Professional Laptop",
                    attributes={"type": "laptop", "category": "professional"},
                    price=899.0,
                )
            ]
        )

        result = evaluator.evaluate(basic_session, constraint_task)

        assert result.success is True
        assert result.metrics["violation_count"] == 0

    def test_forbidden_attribute_violation(self, evaluator, basic_session, constraint_task):
        """Test detection of forbidden attribute."""
        basic_session.cart = CartState(
            items=[
                CartItem(
                    product_id="laptop-001",
                    product_name="Gaming Laptop with RGB",
                    attributes={"type": "laptop"},
                    price=999.0,
                )
            ]
        )

        result = evaluator.evaluate(basic_session, constraint_task)

        assert result.success is False
        assert result.metrics["violation_count"] > 0
        assert "gaming" in str(result.metrics["violations"]).lower()

    def test_forbidden_term_violation(self, evaluator, basic_session, constraint_task):
        """Test detection of forbidden term."""
        basic_session.cart = CartState(
            items=[
                CartItem(
                    product_id="laptop-001",
                    product_name="Laptop with GeForce RTX Graphics",
                    attributes={"type": "laptop"},
                    price=1200.0,
                )
            ]
        )

        result = evaluator.evaluate(basic_session, constraint_task)

        assert result.success is False
        assert result.metrics["violation_count"] > 0

    def test_multiple_violations(self, evaluator, basic_session, constraint_task):
        """Test multiple constraint violations."""
        basic_session.cart = CartState(
            items=[
                CartItem(
                    product_id="laptop-001",
                    product_name="Gaming Laptop with RGB and GeForce",
                    attributes={"type": "gaming laptop"},
                    price=1500.0,
                )
            ]
        )

        result = evaluator.evaluate(basic_session, constraint_task)

        assert result.metrics["violation_count"] >= 2
        assert result.overall_score < 0.5

    def test_missing_required_attribute(self, evaluator, basic_session, constraint_task):
        """Test missing required positive attribute."""
        basic_session.cart = CartState(
            items=[
                CartItem(
                    product_id="tablet-001",
                    product_name="iPad Pro",  # Not a laptop, not professional
                    attributes={"type": "tablet"},
                    price=799.0,
                )
            ]
        )

        result = evaluator.evaluate(basic_session, constraint_task)

        match_component = next(
            c for c in result.scoring_breakdown if c.name == "positive_match"
        )
        assert match_component.normalized_score < 1.0


# =============================================================================
# Comparative Reasoning Task Tests
# =============================================================================


class TestReasoningTaskEvaluation:
    """Tests for comparative reasoning task evaluation."""

    def test_good_comparison_with_llm(self, evaluator, basic_session, reasoning_task):
        """Test good comparison with LLM-as-judge."""
        basic_session.actions = [
            ActionRecord(
                action="view product/speaker-001",
                observation="JBL Flip 6 - $129.99, 4.8 stars, waterproof",
            ),
            ActionRecord(
                action="view product/speaker-002",
                observation="Sony SRS-XB13 - $49.99, 4.5 stars, portable",
            ),
            ActionRecord(
                action="I compared both speakers. The JBL Flip 6 is better for outdoor use "
                "because it has IP67 waterproof rating and louder sound, though more expensive. "
                "The Sony is more budget-friendly but less durable. "
                "I recommend the JBL Flip 6 for outdoor use.",
                observation="Purchase completed",
            ),
        ]

        result = evaluator.evaluate(basic_session, reasoning_task)

        # Should have good exploration and justification
        assert result.overall_score >= 0.5
        assert result.metrics["justification_provided"] is True

    def test_no_comparison(self, evaluator_no_llm, basic_session, reasoning_task):
        """Test evaluation with no comparison made."""
        basic_session.actions = [
            ActionRecord(action="Buy speaker", observation="Added to cart"),
        ]

        result = evaluator_no_llm.evaluate(basic_session, reasoning_task)

        exploration_component = next(
            c for c in result.scoring_breakdown if c.name == "exploration"
        )
        justification_component = next(
            c for c in result.scoring_breakdown if c.name == "justification_provided"
        )

        assert exploration_component.normalized_score < 1.0
        assert justification_component.normalized_score == 0.0

    def test_heuristic_justification_scoring(self, evaluator_no_llm, basic_session, reasoning_task):
        """Test heuristic justification scoring without LLM."""
        basic_session.actions = [
            ActionRecord(
                action="view product/SPEAKER1",
                observation="Found speaker 1",
            ),
            ActionRecord(
                action="view product/SPEAKER2",
                observation="Found speaker 2",
            ),
            ActionRecord(
                action="The first speaker is better than the second because it has higher ratings "
                "and better price value. I recommend choosing the first one for quality outdoor use.",
                observation="Done",
            ),
        ]

        result = evaluator_no_llm.evaluate(basic_session, reasoning_task)

        quality_component = next(
            c for c in result.scoring_breakdown if c.name == "justification_quality"
        )
        # Heuristic should detect comparison words
        assert quality_component.normalized_score > 0.0
        assert "heuristic" in quality_component.explanation.lower()

    def test_insufficient_exploration(self, evaluator, basic_session, reasoning_task):
        """Test when not enough products are explored."""
        basic_session.actions = [
            ActionRecord(
                action="view product/SINGLE",
                observation="Only one product viewed",
            ),
            ActionRecord(
                action="This is the only option I found",
                observation="Purchased",
            ),
        ]

        result = evaluator.evaluate(basic_session, reasoning_task)

        exploration_component = next(
            c for c in result.scoring_breakdown if c.name == "exploration"
        )
        assert exploration_component.normalized_score < 1.0


# =============================================================================
# Error Recovery Task Tests
# =============================================================================


class TestRecoveryTaskEvaluation:
    """Tests for error recovery task evaluation."""

    def test_successful_recovery(self, evaluator, basic_session, recovery_task):
        """Test successful error recovery."""
        # Final cart matches expected state
        basic_session.cart = CartState(
            items=[
                CartItem(
                    product_id="HP-001",
                    product_name="Sony Headphones",
                    attributes={"color": "black"},
                    quantity=1,
                    price=278.0,
                )
            ]
        )
        basic_session.actions = [
            ActionRecord(
                action="I see the wrong quantity. Let me fix this.",
                observation="Cart updated",
            ),
            ActionRecord(
                action="Change quantity to 1",
                observation="Quantity updated to 1",
            ),
        ]

        result = evaluator.evaluate(basic_session, recovery_task)

        assert result.success is True
        assert result.metrics["error_fixed"] is True

    def test_failed_recovery(self, evaluator, basic_session, recovery_task):
        """Test failed error recovery - wrong final state."""
        # Cart still has wrong quantity
        basic_session.cart = CartState(
            items=[
                CartItem(
                    product_id="HP-001",
                    product_name="Sony Headphones",
                    attributes={"color": "black"},
                    quantity=2,  # Should be 1
                    price=278.0,
                )
            ]
        )

        result = evaluator.evaluate(basic_session, recovery_task)

        assert result.success is False
        assert result.metrics["error_fixed"] is False

    def test_error_identified(self, evaluator, basic_session, recovery_task):
        """Test error identification detection."""
        basic_session.cart = CartState(
            items=[
                CartItem(
                    product_id="HP-001",
                    product_name="Sony Headphones",
                    attributes={"color": "black"},
                    quantity=1,
                    price=278.0,
                )
            ]
        )
        basic_session.actions = [
            ActionRecord(
                action="I notice there's a mistake - the wrong quantity was added",
                observation="Updating cart",
            ),
        ]

        result = evaluator.evaluate(basic_session, recovery_task)

        assert result.metrics["error_identified"] is True
        error_component = next(
            c for c in result.scoring_breakdown if c.name == "error_identified"
        )
        assert error_component.normalized_score == 1.0

    def test_inefficient_recovery(self, evaluator, basic_session, recovery_task):
        """Test recovery with too many actions."""
        basic_session.cart = CartState(
            items=[
                CartItem(
                    product_id="HP-001",
                    product_name="Sony Headphones",
                    attributes={"color": "black"},
                    quantity=1,
                    price=278.0,
                )
            ]
        )
        # Many unnecessary actions (expected is 6)
        basic_session.actions = [
            ActionRecord(action=f"Action {i}", observation=f"Result {i}")
            for i in range(15)
        ]

        result = evaluator.evaluate(basic_session, recovery_task)

        efficiency_component = next(
            c for c in result.scoring_breakdown if c.name == "efficiency"
        )
        assert efficiency_component.normalized_score < 1.0
        assert "inefficient" in efficiency_component.explanation.lower()

    def test_missing_product(self, evaluator, basic_session, recovery_task):
        """Test when expected product is missing from cart."""
        basic_session.cart = CartState(items=[])  # Empty cart

        result = evaluator.evaluate(basic_session, recovery_task)

        assert result.success is False
        assert result.metrics["error_fixed"] is False


# =============================================================================
# General Evaluation Tests
# =============================================================================


class TestGeneralEvaluation:
    """Tests for general evaluation behavior."""

    def test_unknown_task_type(self, evaluator, basic_session):
        """Test handling of unknown task type."""
        # Create a task with invalid type
        from unittest.mock import MagicMock

        bad_task = MagicMock()
        bad_task.task_id = "bad-001"
        bad_task.task_type = "invalid_type"

        result = evaluator.evaluate(basic_session, bad_task)

        assert result.error is not None
        assert "unknown task type" in result.error.lower()

    def test_universal_metrics_populated(self, evaluator, basic_session, budget_task):
        """Test that universal metrics are always populated."""
        basic_session.cart = CartState(
            items=[
                CartItem(product_id="x", product_name="X", price=10.0)
            ]
        )

        result = evaluator.evaluate(basic_session, budget_task)

        assert result.task_id == budget_task.task_id
        assert result.task_type == budget_task.task_type
        assert result.completed == basic_session.completed
        assert result.actions_taken == basic_session.actions_taken
        assert result.time_elapsed_seconds >= 0

    def test_scoring_breakdown_exists(self, evaluator, basic_session, budget_task):
        """Test that scoring breakdown is populated."""
        basic_session.cart = CartState(
            items=[
                CartItem(product_id="x", product_name="Mouse", price=20.0)
            ]
        )

        result = evaluator.evaluate(basic_session, budget_task)

        assert len(result.scoring_breakdown) > 0
        for component in result.scoring_breakdown:
            assert component.name
            assert 0 <= component.normalized_score <= 1
            assert component.weight > 0

    def test_metrics_populated(self, evaluator, basic_session, budget_task):
        """Test that task-specific metrics are populated."""
        basic_session.cart = CartState(
            items=[
                CartItem(product_id="x", product_name="Mouse", price=20.0)
            ]
        )

        result = evaluator.evaluate(basic_session, budget_task)

        assert "budget" in result.metrics
        assert "total_spent" in result.metrics


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_budget(self, evaluator, basic_session):
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
        basic_session.cart = CartState(items=[])

        result = evaluator.evaluate(basic_session, task)
        assert result is not None

    def test_empty_forbidden_constraints(self, evaluator, basic_session):
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
        basic_session.cart = CartState(
            items=[
                CartItem(product_id="x", product_name="Laptop Computer", price=500.0)
            ]
        )

        result = evaluator.evaluate(basic_session, task)

        assert result.metrics["violation_count"] == 0

    def test_empty_session_sequence(self, evaluator, basic_session):
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

        result = evaluator.evaluate(basic_session, task)
        assert result is not None

    def test_very_long_justification(self, evaluator_no_llm, basic_session, reasoning_task):
        """Test handling of very long justification text."""
        long_text = "I compared products thoroughly. " * 100
        basic_session.actions = [
            ActionRecord(action=long_text, observation="Done"),
        ]

        result = evaluator_no_llm.evaluate(basic_session, reasoning_task)
        assert result is not None

    def test_special_characters_in_product_name(self, evaluator, basic_session, constraint_task):
        """Test handling of special characters in product names."""
        basic_session.cart = CartState(
            items=[
                CartItem(
                    product_id="x",
                    product_name="Laptop (Pro) - 15\" Display & More!",
                    attributes={"type": "laptop"},
                    price=999.0,
                )
            ]
        )

        result = evaluator.evaluate(basic_session, constraint_task)
        assert result is not None


# =============================================================================
# Direct Method Tests
# =============================================================================


class TestDirectMethods:
    """Tests for calling evaluation methods directly."""

    def test_evaluate_budget_task_directly(self, evaluator, basic_session, budget_task):
        """Test calling evaluate_budget_task directly."""
        basic_session.cart = CartState(
            items=[
                CartItem(product_id="m", product_name="Mouse", price=25.0),
                CartItem(product_id="p", product_name="Mousepad", price=15.0),
            ]
        )

        result = evaluator.evaluate_budget_task(basic_session, budget_task)

        assert isinstance(result, EvaluationResult)
        assert result.task_id == budget_task.task_id

    def test_evaluate_memory_task_directly(self, evaluator, basic_session, memory_task):
        """Test calling evaluate_memory_task directly."""
        result = evaluator.evaluate_memory_task(basic_session, memory_task)

        assert isinstance(result, EvaluationResult)
        assert result.task_id == memory_task.task_id

    def test_evaluate_constraint_task_directly(self, evaluator, basic_session, constraint_task):
        """Test calling evaluate_constraint_task directly."""
        basic_session.cart = CartState(
            items=[
                CartItem(product_id="l", product_name="Business Laptop", price=800.0)
            ]
        )

        result = evaluator.evaluate_constraint_task(basic_session, constraint_task)

        assert isinstance(result, EvaluationResult)
        assert result.task_id == constraint_task.task_id

    def test_evaluate_reasoning_task_directly(self, evaluator, basic_session, reasoning_task):
        """Test calling evaluate_reasoning_task directly."""
        result = evaluator.evaluate_reasoning_task(basic_session, reasoning_task)

        assert isinstance(result, EvaluationResult)
        assert result.task_id == reasoning_task.task_id

    def test_evaluate_recovery_task_directly(self, evaluator, basic_session, recovery_task):
        """Test calling evaluate_recovery_task directly."""
        basic_session.cart = CartState(
            items=[
                CartItem(
                    product_id="HP-001",
                    product_name="Sony Headphones",
                    quantity=1,
                    price=278.0,
                )
            ]
        )

        result = evaluator.evaluate_recovery_task(basic_session, recovery_task)

        assert isinstance(result, EvaluationResult)
        assert result.task_id == recovery_task.task_id


# =============================================================================
# LLM Integration Tests (Mocked)
# =============================================================================


class TestLLMIntegration:
    """Tests for LLM integration in evaluation."""

    def test_llm_evaluate_with_rubric_called(self, mock_llm_client, evaluator, basic_session, reasoning_task):
        """Test that LLM evaluate_with_rubric is called for reasoning tasks."""
        basic_session.actions = [
            ActionRecord(action="view product/A", observation="Product A"),
            ActionRecord(action="view product/B", observation="Product B"),
            ActionRecord(
                action="I compared A and B. A is better because of the rating.",
                observation="Done",
            ),
        ]

        evaluator.evaluate(basic_session, reasoning_task)

        mock_llm_client.evaluate_with_rubric.assert_called_once()

    def test_llm_failure_fallback(self, evaluator, basic_session, reasoning_task):
        """Test fallback to heuristic when LLM fails."""
        evaluator._llm_client.evaluate_with_rubric.side_effect = Exception("API Error")

        basic_session.actions = [
            ActionRecord(action="view product/A", observation="Product A"),
            ActionRecord(action="view product/B", observation="Product B"),
            ActionRecord(
                action="A is better than B because of quality and price value.",
                observation="Done",
            ),
        ]

        result = evaluator.evaluate(basic_session, reasoning_task)

        # Should still get a result via heuristic
        quality_component = next(
            c for c in result.scoring_breakdown if c.name == "justification_quality"
        )
        assert "heuristic" in quality_component.explanation.lower()


# =============================================================================
# Integration with Real Task Data
# =============================================================================


class TestRealTaskData:
    """Tests using realistic task data patterns."""

    def test_realistic_budget_scenario(self, evaluator, basic_session):
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

        basic_session.cart = CartState(
            items=[
                CartItem(
                    product_id="LM001",
                    product_name="Logitech M510 Wireless Mouse",
                    attributes={"type": "wireless mouse", "brand": "Logitech"},
                    price=24.99,
                ),
                CartItem(
                    product_id="MP001",
                    product_name="SteelSeries QcK Gaming Mousepad",
                    attributes={"type": "mousepad", "size": "large"},
                    price=14.99,
                ),
            ]
        )

        result = evaluator.evaluate(basic_session, task)

        assert result.success is True
        assert result.overall_score >= 0.8
        assert result.metrics["total_spent"] == 39.98
        assert result.metrics["total_spent"] < result.metrics["budget"]

    def test_realistic_recovery_scenario(self, evaluator, basic_session):
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

        basic_session.cart = CartState(
            items=[
                CartItem(
                    product_id="SONY-WH1000XM4",
                    product_name="Sony WH-1000XM4 Wireless Headphones",
                    attributes={"color": "black"},
                    quantity=1,
                    price=278.0,
                )
            ]
        )
        basic_session.actions = [
            ActionRecord(
                action="I see the cart has 3 items when I only need 1. Let me fix this mistake.",
                observation="Cart view loaded",
            ),
            ActionRecord(
                action="update quantity to 1",
                observation="Quantity updated successfully",
            ),
        ]

        result = evaluator.evaluate(basic_session, task)

        assert result.success is True
        assert result.metrics["error_fixed"] is True
        assert result.metrics["error_identified"] is True
