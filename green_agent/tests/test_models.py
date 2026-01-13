"""
Tests for WebShop+ Pydantic models.

Tests cover:
- Task models (all 5 types)
- State models
- Evaluation models
- Memory models
- Helper functions
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.models import (
    # Enums
    TaskType,
    Difficulty,
    OptimizationGoal,
    # Task models
    BaseTask,
    BudgetConstrainedTask,
    PreferenceMemoryTask,
    NegativeConstraintTask,
    ComparativeReasoningTask,
    ErrorRecoveryTask,
    parse_task,
    # Nested task models
    RequiredItem,
    BudgetConstraints,
    SessionSequenceItem,
    MemoryTest,
    NegativeConstraints,
    ComparativeRequirements,
    CartItemSetup,
    ErrorRecoverySetup,
    CorrectState,
    # State models
    CartItem,
    CartState,
    ActionRecord,
    PurchaseRecord,
    SessionState,
    # Evaluation models
    ScoringComponent,
    EvaluationResult,
    # Memory models
    SessionSummary,
    AgentMemory,
    # Assessment models
    AssessmentConfig,
    AssessmentRequest,
    TaskUpdate,
    AggregateResults,
    AssessmentResults,
)


# =============================================================================
# Test Data Paths
# =============================================================================

TASKS_DIR = Path(__file__).parent.parent / "data" / "tasks"


# =============================================================================
# Task Model Tests
# =============================================================================


class TestTaskEnums:
    """Test enum definitions."""

    def test_task_types(self):
        assert TaskType.BUDGET_CONSTRAINED.value == "budget_constrained"
        assert TaskType.PREFERENCE_MEMORY.value == "preference_memory"
        assert TaskType.NEGATIVE_CONSTRAINT.value == "negative_constraint"
        assert TaskType.COMPARATIVE_REASONING.value == "comparative_reasoning"
        assert TaskType.ERROR_RECOVERY.value == "error_recovery"

    def test_difficulty_levels(self):
        assert Difficulty.EASY.value == "easy"
        assert Difficulty.MEDIUM.value == "medium"
        assert Difficulty.HARD.value == "hard"

    def test_optimization_goals(self):
        assert OptimizationGoal.MAXIMIZE_QUALITY.value == "maximize_quality"
        assert OptimizationGoal.MINIMIZE_COST.value == "minimize_cost"
        assert OptimizationGoal.BALANCE.value == "balance"


class TestBudgetConstrainedTask:
    """Test budget-constrained task model."""

    def test_create_minimal(self):
        task = BudgetConstrainedTask(
            task_id="budget_001",
            instruction="Buy items within budget",
            constraints=BudgetConstraints(
                budget=50.0,
                required_items=[
                    RequiredItem(category="electronics", attributes={"type": "mouse"})
                ],
                optimization_goal=OptimizationGoal.BALANCE,
            ),
        )
        assert task.task_id == "budget_001"
        assert task.task_type == TaskType.BUDGET_CONSTRAINED
        assert task.constraints.budget == 50.0

    def test_create_full(self):
        task = BudgetConstrainedTask(
            task_id="budget_002",
            instruction="Buy office supplies",
            difficulty=Difficulty.MEDIUM,
            expected_actions=15,
            timeout_seconds=240,
            constraints=BudgetConstraints(
                budget=100.0,
                required_items=[
                    RequiredItem(
                        category="office",
                        attributes={"type": "desk organizer"},
                        optional=False,
                    ),
                    RequiredItem(
                        category="office",
                        attributes={"type": "plant"},
                        optional=True,
                    ),
                ],
                optimization_goal=OptimizationGoal.MAXIMIZE_QUALITY,
            ),
        )
        assert task.difficulty == Difficulty.MEDIUM
        assert len(task.constraints.required_items) == 2
        assert task.constraints.required_items[1].optional is True

    def test_load_from_json_file(self):
        """Test loading actual task from JSON file."""
        json_path = TASKS_DIR / "budget_constrained.json"
        if json_path.exists():
            with open(json_path) as f:
                tasks_data = json.load(f)

            task = BudgetConstrainedTask(**tasks_data[0])
            assert task.task_id == "budget_001"
            assert task.constraints.budget > 0


class TestPreferenceMemoryTask:
    """Test preference memory task model."""

    def test_create_basic(self):
        task = PreferenceMemoryTask(
            task_id="memory_001",
            instruction="Buy same color as before",
            session_sequence=[
                SessionSequenceItem(
                    session_id="memory_001_s1",
                    instruction="Buy a blue shirt",
                    establishes={"color": "blue", "size": "medium"},
                )
            ],
            memory_test=MemoryTest(
                attribute_to_recall="color",
                acceptable_values=["blue", "navy blue"],
            ),
        )
        assert task.task_type == TaskType.PREFERENCE_MEMORY
        assert task.memory_test.attribute_to_recall == "color"

    def test_load_from_json_file(self):
        """Test loading actual task from JSON file."""
        json_path = TASKS_DIR / "preference_memory.json"
        if json_path.exists():
            with open(json_path) as f:
                tasks_data = json.load(f)

            task = PreferenceMemoryTask(**tasks_data[0])
            assert task.task_type == TaskType.PREFERENCE_MEMORY
            assert len(task.memory_test.acceptable_values) > 0


class TestNegativeConstraintTask:
    """Test negative constraint task model."""

    def test_create_basic(self):
        task = NegativeConstraintTask(
            task_id="constraint_001",
            instruction="Find fragrance-free moisturizer",
            constraints=NegativeConstraints(
                required_attributes=["moisturizer", "face"],
                forbidden_attributes=["fragrance"],
                forbidden_terms=["parfum", "scented"],
            ),
        )
        assert task.task_type == TaskType.NEGATIVE_CONSTRAINT
        assert "fragrance" in task.constraints.forbidden_attributes

    def test_with_budget(self):
        task = NegativeConstraintTask(
            task_id="constraint_002",
            instruction="Find cotton shirt under $40",
            constraints=NegativeConstraints(
                required_attributes=["t-shirt"],
                forbidden_attributes=["polyester"],
                forbidden_terms=["synthetic"],
                budget=40.0,
            ),
        )
        assert task.constraints.budget == 40.0

    def test_load_from_json_file(self):
        """Test loading actual task from JSON file."""
        json_path = TASKS_DIR / "negative_constraint.json"
        if json_path.exists():
            with open(json_path) as f:
                tasks_data = json.load(f)

            task = NegativeConstraintTask(**tasks_data[0])
            # Check that constraints are loaded - some tasks use forbidden_attributes instead of forbidden_terms
            assert len(task.constraints.forbidden_attributes) > 0 or len(task.constraints.forbidden_terms) > 0


class TestComparativeReasoningTask:
    """Test comparative reasoning task model."""

    def test_create_basic(self):
        task = ComparativeReasoningTask(
            task_id="compare_001",
            instruction="Compare Bluetooth speakers",
            requirements=ComparativeRequirements(
                category="electronics",
                attributes={"type": "Bluetooth speaker"},
                budget=80.0,
                comparison_request="Compare at least 2 options",
            ),
        )
        assert task.task_type == TaskType.COMPARATIVE_REASONING
        assert task.requirements.budget == 80.0

    def test_load_from_json_file(self):
        """Test loading actual task from JSON file."""
        json_path = TASKS_DIR / "comparative_reasoning.json"
        if json_path.exists():
            with open(json_path) as f:
                tasks_data = json.load(f)

            task = ComparativeReasoningTask(**tasks_data[0])
            assert task.evaluation_criteria.minimum_options_explored >= 2


class TestErrorRecoveryTask:
    """Test error recovery task model."""

    def test_create_basic(self):
        task = ErrorRecoveryTask(
            task_id="recovery_001",
            instruction="Fix wrong quantity",
            setup=ErrorRecoverySetup(
                cart_contents=[
                    CartItemSetup(
                        product_id="HP-001",
                        product_name="Sony Headphones",
                        attributes={"color": "black"},
                        quantity=3,
                        price=278.0,
                    )
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
                    )
                ]
            ),
        )
        assert task.task_type == TaskType.ERROR_RECOVERY
        assert task.setup.cart_contents[0].quantity == 3
        assert task.correct_state.expected_cart[0].quantity == 1

    def test_load_from_json_file(self):
        """Test loading actual task from JSON file."""
        json_path = TASKS_DIR / "error_recovery.json"
        if json_path.exists():
            with open(json_path) as f:
                tasks_data = json.load(f)

            task = ErrorRecoveryTask(**tasks_data[0])
            assert len(task.setup.cart_contents) > 0


class TestParseTask:
    """Test the parse_task helper function."""

    def test_parse_budget_task(self):
        data = {
            "task_id": "budget_001",
            "task_type": "budget_constrained",
            "instruction": "Buy items",
            "constraints": {
                "budget": 50.0,
                "required_items": [{"category": "test", "attributes": {}}],
                "optimization_goal": "balance",
            },
        }
        task = parse_task(data)
        assert isinstance(task, BudgetConstrainedTask)

    def test_parse_memory_task(self):
        data = {
            "task_id": "memory_001",
            "task_type": "preference_memory",
            "instruction": "Remember color",
            "session_sequence": [],
            "memory_test": {"attribute_to_recall": "color", "acceptable_values": ["blue"]},
        }
        task = parse_task(data)
        assert isinstance(task, PreferenceMemoryTask)

    def test_parse_constraint_task(self):
        data = {
            "task_id": "constraint_001",
            "task_type": "negative_constraint",
            "instruction": "Avoid allergens",
            "constraints": {"forbidden_terms": ["peanut"]},
        }
        task = parse_task(data)
        assert isinstance(task, NegativeConstraintTask)

    def test_parse_compare_task(self):
        data = {
            "task_id": "compare_001",
            "task_type": "comparative_reasoning",
            "instruction": "Compare options",
            "requirements": {
                "category": "electronics",
                "budget": 100.0,
                "comparison_request": "Compare 2 options",
            },
        }
        task = parse_task(data)
        assert isinstance(task, ComparativeReasoningTask)

    def test_parse_recovery_task(self):
        data = {
            "task_id": "recovery_001",
            "task_type": "error_recovery",
            "instruction": "Fix cart",
            "setup": {"cart_contents": [], "error_description": "wrong item"},
            "correct_state": {"expected_cart": []},
        }
        task = parse_task(data)
        assert isinstance(task, ErrorRecoveryTask)

    def test_parse_unknown_type_raises(self):
        data = {"task_id": "unknown_001", "task_type": "unknown_type", "instruction": "Test"}
        with pytest.raises(ValueError, match="Unknown task type"):
            parse_task(data)


# =============================================================================
# State Model Tests
# =============================================================================


class TestCartItem:
    """Test CartItem model."""

    def test_create_basic(self):
        item = CartItem(
            product_id="P001",
            product_name="Test Product",
            price=29.99,
        )
        assert item.quantity == 1
        assert item.total_price == 29.99

    def test_total_price_with_quantity(self):
        item = CartItem(
            product_id="P001",
            product_name="Test Product",
            quantity=3,
            price=10.0,
        )
        assert item.total_price == 30.0


class TestCartState:
    """Test CartState model."""

    def test_empty_cart(self):
        cart = CartState()
        assert cart.total == 0.0
        assert cart.item_count == 0
        assert len(cart.items) == 0

    def test_add_item(self):
        cart = CartState()
        cart.add_item(CartItem(product_id="P001", product_name="Item 1", price=10.0))
        assert cart.item_count == 1
        assert cart.total == 10.0

    def test_add_duplicate_item_increases_quantity(self):
        cart = CartState()
        cart.add_item(
            CartItem(product_id="P001", product_name="Item 1", attributes={"size": "M"}, price=10.0)
        )
        cart.add_item(
            CartItem(product_id="P001", product_name="Item 1", attributes={"size": "M"}, price=10.0)
        )
        assert len(cart.items) == 1
        assert cart.items[0].quantity == 2
        assert cart.total == 20.0

    def test_add_same_product_different_attributes(self):
        cart = CartState()
        cart.add_item(
            CartItem(product_id="P001", product_name="Item 1", attributes={"size": "M"}, price=10.0)
        )
        cart.add_item(
            CartItem(product_id="P001", product_name="Item 1", attributes={"size": "L"}, price=10.0)
        )
        assert len(cart.items) == 2

    def test_remove_item(self):
        cart = CartState()
        cart.add_item(CartItem(product_id="P001", product_name="Item 1", price=10.0))
        cart.add_item(CartItem(product_id="P002", product_name="Item 2", price=20.0))

        result = cart.remove_item("P001")
        assert result is True
        assert len(cart.items) == 1
        assert cart.total == 20.0

    def test_remove_nonexistent_item(self):
        cart = CartState()
        result = cart.remove_item("P999")
        assert result is False

    def test_clear(self):
        cart = CartState()
        cart.add_item(CartItem(product_id="P001", product_name="Item 1", price=10.0))
        cart.add_item(CartItem(product_id="P002", product_name="Item 2", price=20.0))
        cart.clear()
        assert cart.total == 0.0
        assert len(cart.items) == 0


class TestSessionState:
    """Test SessionState model."""

    def test_create_basic(self):
        session = SessionState(session_id="session_001", task_id="task_001")
        assert session.actions_taken == 0
        assert session.completed is False

    def test_record_action(self):
        session = SessionState(session_id="session_001", task_id="task_001")
        session.record_action("search[test]", "Results found", 0.5)

        assert session.actions_taken == 1
        assert session.current_observation == "Results found"
        assert session.actions[0].action == "search[test]"
        assert session.actions[0].reward == 0.5

    def test_complete(self):
        session = SessionState(session_id="session_001", task_id="task_001")
        session.complete()

        assert session.completed is True
        assert session.ended_at is not None

    def test_elapsed_seconds(self):
        session = SessionState(session_id="session_001", task_id="task_001")
        # Just verify it returns a non-negative number
        assert session.elapsed_seconds >= 0

    def test_to_summary(self):
        session = SessionState(
            session_id="session_001",
            task_id="task_001",
            preferences_established={"color": "blue"},
        )
        session.purchases.append(
            PurchaseRecord(product_id="P001", product_name="Test", price=10.0)
        )

        summary = session.to_summary()
        assert summary.session_id == "session_001"
        assert len(summary.purchases) == 1
        assert summary.preferences["color"] == "blue"


# =============================================================================
# Evaluation Model Tests
# =============================================================================


class TestScoringComponent:
    """Test ScoringComponent model."""

    def test_create_basic(self):
        component = ScoringComponent(
            name="budget_compliance",
            weight=0.3,
            raw_value=True,
            normalized_score=1.0,
            explanation="Under budget",
        )
        assert component.name == "budget_compliance"
        assert component.normalized_score == 1.0


class TestEvaluationResult:
    """Test EvaluationResult model."""

    def test_create_basic(self):
        result = EvaluationResult(
            task_id="task_001",
            task_type=TaskType.BUDGET_CONSTRAINED,
        )
        assert result.overall_score == 0.0
        assert result.completed is False

    def test_add_component(self):
        result = EvaluationResult(
            task_id="task_001",
            task_type=TaskType.BUDGET_CONSTRAINED,
        )
        result.add_component(
            name="budget_compliance",
            weight=0.3,
            raw_value=True,
            normalized_score=1.0,
        )
        assert len(result.scoring_breakdown) == 1

    def test_calculate_overall_score(self):
        result = EvaluationResult(
            task_id="task_001",
            task_type=TaskType.BUDGET_CONSTRAINED,
        )
        result.add_component("budget", 0.3, True, 1.0)
        result.add_component("completion", 0.4, 0.75, 0.75)
        result.add_component("quality", 0.3, 0.8, 0.8)

        score = result.calculate_overall_score()
        # (0.3*1.0 + 0.4*0.75 + 0.3*0.8) / 1.0 = 0.84
        assert abs(score - 0.84) < 0.001

    def test_calculate_overall_score_empty(self):
        result = EvaluationResult(
            task_id="task_001",
            task_type=TaskType.BUDGET_CONSTRAINED,
        )
        score = result.calculate_overall_score()
        assert score == 0.0


# =============================================================================
# Memory Model Tests
# =============================================================================


class TestAgentMemory:
    """Test AgentMemory model."""

    def test_create_empty(self):
        memory = AgentMemory(agent_id="agent_001")
        assert len(memory.sessions) == 0

    def test_add_session(self):
        memory = AgentMemory(agent_id="agent_001")
        memory.add_session(
            SessionSummary(
                session_id="s1",
                task_id="t1",
                task_type="budget_constrained",
            )
        )
        assert len(memory.sessions) == 1

    def test_get_sessions_by_type(self):
        memory = AgentMemory(agent_id="agent_001")
        memory.add_session(
            SessionSummary(session_id="s1", task_id="t1", task_type="budget_constrained")
        )
        memory.add_session(
            SessionSummary(session_id="s2", task_id="t2", task_type="preference_memory")
        )
        memory.add_session(
            SessionSummary(session_id="s3", task_id="t3", task_type="budget_constrained")
        )

        budget_sessions = memory.get_sessions_by_type("budget_constrained")
        assert len(budget_sessions) == 2

    def test_get_all_purchases(self):
        memory = AgentMemory(agent_id="agent_001")
        memory.add_session(
            SessionSummary(
                session_id="s1",
                task_id="t1",
                task_type="budget_constrained",
                purchases=[
                    PurchaseRecord(product_id="P1", product_name="Item 1", price=10.0)
                ],
            )
        )
        memory.add_session(
            SessionSummary(
                session_id="s2",
                task_id="t2",
                task_type="budget_constrained",
                purchases=[
                    PurchaseRecord(product_id="P2", product_name="Item 2", price=20.0)
                ],
            )
        )

        purchases = memory.get_all_purchases()
        assert len(purchases) == 2

    def test_get_all_preferences(self):
        memory = AgentMemory(agent_id="agent_001")
        memory.add_session(
            SessionSummary(
                session_id="s1",
                task_id="t1",
                task_type="preference_memory",
                preferences={"color": "blue"},
            )
        )
        memory.add_session(
            SessionSummary(
                session_id="s2",
                task_id="t2",
                task_type="preference_memory",
                preferences={"size": "medium", "brand": "Nike"},
            )
        )

        prefs = memory.get_all_preferences()
        assert prefs["color"] == "blue"
        assert prefs["size"] == "medium"
        assert prefs["brand"] == "Nike"

    def test_clear(self):
        memory = AgentMemory(agent_id="agent_001")
        memory.add_session(
            SessionSummary(session_id="s1", task_id="t1", task_type="budget_constrained")
        )
        memory.clear()
        assert len(memory.sessions) == 0


# =============================================================================
# Assessment Model Tests
# =============================================================================


class TestAssessmentConfig:
    """Test AssessmentConfig model."""

    def test_defaults(self):
        config = AssessmentConfig()
        assert config.task_types == ["all"]
        assert config.num_tasks == 80
        assert config.timeout_per_task == 300

    def test_custom(self):
        config = AssessmentConfig(
            task_types=["budget_constrained", "error_recovery"],
            num_tasks=10,
            timeout_per_task=120,
        )
        assert len(config.task_types) == 2


class TestAssessmentResults:
    """Test AssessmentResults model."""

    def test_calculate_aggregate(self):
        results = AssessmentResults(
            assessment_id="test_001",
            results=[
                EvaluationResult(
                    task_id="t1",
                    task_type=TaskType.BUDGET_CONSTRAINED,
                    overall_score=0.8,
                    success=True,
                    time_elapsed_seconds=30.0,
                ),
                EvaluationResult(
                    task_id="t2",
                    task_type=TaskType.BUDGET_CONSTRAINED,
                    overall_score=0.6,
                    success=True,
                    time_elapsed_seconds=40.0,
                ),
                EvaluationResult(
                    task_id="t3",
                    task_type=TaskType.ERROR_RECOVERY,
                    overall_score=0.9,
                    success=True,
                    time_elapsed_seconds=20.0,
                ),
            ],
        )
        results.calculate_aggregate()

        assert results.aggregate.total_tasks == 3
        assert results.aggregate.successful_tasks == 3
        assert abs(results.aggregate.average_score - 0.7667) < 0.01
        assert abs(results.aggregate.average_time - 30.0) < 0.01
        assert "budget_constrained" in results.aggregate.by_task_type
        assert results.aggregate.by_task_type["budget_constrained"]["count"] == 2


# =============================================================================
# Integration Tests - Load All Tasks
# =============================================================================


class TestLoadAllTasks:
    """Test loading all actual task files."""

    @pytest.fixture
    def all_task_files(self):
        """Get all task JSON files."""
        if not TASKS_DIR.exists():
            pytest.skip("Tasks directory not found")
        return list(TASKS_DIR.glob("*.json"))

    def test_all_task_files_load(self, all_task_files):
        """Verify all task files can be loaded and parsed."""
        total_tasks = 0
        for task_file in all_task_files:
            with open(task_file) as f:
                tasks_data = json.load(f)

            for task_data in tasks_data:
                task = parse_task(task_data)
                assert task.task_id is not None
                assert task.instruction is not None
                total_tasks += 1

        # Verify we have the expected number of tasks
        assert total_tasks == 80, f"Expected 80 tasks, found {total_tasks}"

    def test_task_type_counts(self, all_task_files):
        """Verify correct number of tasks per type."""
        type_counts = {
            "budget_constrained": 0,
            "preference_memory": 0,
            "negative_constraint": 0,
            "comparative_reasoning": 0,
            "error_recovery": 0,
        }

        for task_file in all_task_files:
            with open(task_file) as f:
                tasks_data = json.load(f)

            for task_data in tasks_data:
                task_type = task_data["task_type"]
                if task_type in type_counts:
                    type_counts[task_type] += 1

        assert type_counts["budget_constrained"] == 20
        assert type_counts["preference_memory"] == 15
        assert type_counts["negative_constraint"] == 20
        assert type_counts["comparative_reasoning"] == 15
        assert type_counts["error_recovery"] == 10
