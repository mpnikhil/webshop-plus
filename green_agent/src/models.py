"""
Pydantic models for WebShop+ green agent.

This module defines all data structures used throughout the WebShop+ evaluation system:
- Task models (base + 5 specialized types)
- State models (session, cart, actions)
- Evaluation models (results, scoring)
- Memory models (agent memory, session summaries)
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class TaskType(str, Enum):
    """Types of tasks supported by WebShop+."""

    BUDGET_CONSTRAINED = "budget_constrained"
    PREFERENCE_MEMORY = "preference_memory"
    NEGATIVE_CONSTRAINT = "negative_constraint"
    COMPARATIVE_REASONING = "comparative_reasoning"
    ERROR_RECOVERY = "error_recovery"


class Difficulty(str, Enum):
    """Task difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class OptimizationGoal(str, Enum):
    """Optimization goals for budget-constrained tasks."""

    MAXIMIZE_QUALITY = "maximize_quality"
    MINIMIZE_COST = "minimize_cost"
    BALANCE = "balance"


# =============================================================================
# Shared/Nested Models
# =============================================================================


class RequiredItem(BaseModel):
    """An item required for a budget-constrained task."""

    category: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    optional: bool = False


class BudgetConstraints(BaseModel):
    """Constraints for budget-constrained tasks."""

    budget: float
    required_items: list[RequiredItem]
    optimization_goal: OptimizationGoal = OptimizationGoal.BALANCE


class BudgetEvaluationCriteria(BaseModel):
    """Evaluation criteria for budget-constrained tasks."""

    budget_weight: float = 0.3
    completion_weight: float = 0.4
    quality_weight: float = 0.3


class SessionSequenceItem(BaseModel):
    """A session in a preference memory task sequence."""

    session_id: str
    instruction: str
    establishes: dict[str, Any] = Field(default_factory=dict)


class MemoryTest(BaseModel):
    """Memory test criteria for preference memory tasks."""

    attribute_to_recall: str
    acceptable_values: list[str]


class MemoryEvaluationCriteria(BaseModel):
    """Evaluation criteria for preference memory tasks."""

    recall_accuracy_weight: float = 0.5
    consistency_weight: float = 0.5


class NegativeConstraints(BaseModel):
    """Constraints for negative constraint tasks."""

    required_attributes: list[str] = Field(default_factory=list)
    forbidden_attributes: list[str] = Field(default_factory=list)
    forbidden_terms: list[str] = Field(default_factory=list)
    budget: Optional[float] = None


class ConstraintEvaluationCriteria(BaseModel):
    """Evaluation criteria for negative constraint tasks."""

    constraint_violation_penalty: float = 0.5
    match_score_weight: float = 0.5


class ComparativeRequirements(BaseModel):
    """Requirements for comparative reasoning tasks."""

    category: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    budget: float
    comparison_request: str


class ComparativeEvaluationCriteria(BaseModel):
    """Evaluation criteria for comparative reasoning tasks."""

    minimum_options_explored: int = 2
    justification_required: bool = True
    justification_quality_weight: float = 0.5


class CartItemSetup(BaseModel):
    """A cart item in error recovery task setup."""

    product_id: str
    product_name: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    quantity: int = 1
    price: float


class ErrorRecoverySetup(BaseModel):
    """Setup state for error recovery tasks."""

    cart_contents: list[CartItemSetup]
    error_description: str


class CorrectState(BaseModel):
    """Expected correct state after error recovery."""

    expected_cart: list[CartItemSetup]


class RecoveryEvaluationCriteria(BaseModel):
    """Evaluation criteria for error recovery tasks."""

    error_identified: bool = True
    error_fixed: bool = True
    unnecessary_actions_penalty: float = 0.1


# =============================================================================
# Task Models
# =============================================================================


class BaseTask(BaseModel):
    """Base task model with common fields."""

    task_id: str
    task_type: TaskType
    instruction: str
    difficulty: Difficulty = Difficulty.MEDIUM
    expected_actions: int = 15
    timeout_seconds: int = 180


class BudgetConstrainedTask(BaseTask):
    """Task testing budget management and multi-item purchasing."""

    task_type: Literal[TaskType.BUDGET_CONSTRAINED] = TaskType.BUDGET_CONSTRAINED
    constraints: BudgetConstraints
    evaluation_criteria: BudgetEvaluationCriteria = Field(
        default_factory=BudgetEvaluationCriteria
    )


class PreferenceMemoryTask(BaseTask):
    """Task testing preference recall across sessions."""

    task_type: Literal[TaskType.PREFERENCE_MEMORY] = TaskType.PREFERENCE_MEMORY
    session_sequence: list[SessionSequenceItem] = Field(default_factory=list)
    memory_test: MemoryTest
    evaluation_criteria: MemoryEvaluationCriteria = Field(
        default_factory=MemoryEvaluationCriteria
    )


class NegativeConstraintTask(BaseTask):
    """Task testing constraint satisfaction with forbidden attributes."""

    task_type: Literal[TaskType.NEGATIVE_CONSTRAINT] = TaskType.NEGATIVE_CONSTRAINT
    constraints: NegativeConstraints
    evaluation_criteria: ConstraintEvaluationCriteria = Field(
        default_factory=ConstraintEvaluationCriteria
    )


class ComparativeReasoningTask(BaseTask):
    """Task testing product comparison and justification."""

    task_type: Literal[TaskType.COMPARATIVE_REASONING] = TaskType.COMPARATIVE_REASONING
    requirements: ComparativeRequirements
    evaluation_criteria: ComparativeEvaluationCriteria = Field(
        default_factory=ComparativeEvaluationCriteria
    )


class ErrorRecoveryTask(BaseTask):
    """Task testing error identification and correction."""

    task_type: Literal[TaskType.ERROR_RECOVERY] = TaskType.ERROR_RECOVERY
    setup: ErrorRecoverySetup
    correct_state: CorrectState
    evaluation_criteria: RecoveryEvaluationCriteria = Field(
        default_factory=RecoveryEvaluationCriteria
    )


# Union type for all task types
Task = Union[
    BudgetConstrainedTask,
    PreferenceMemoryTask,
    NegativeConstraintTask,
    ComparativeReasoningTask,
    ErrorRecoveryTask,
]


def parse_task(data: dict[str, Any]) -> Task:
    """Parse a task dictionary into the appropriate Task model.

    Args:
        data: Dictionary containing task data with a 'task_type' field.

    Returns:
        The appropriate Task model instance.

    Raises:
        ValueError: If task_type is unknown.
    """
    task_type = data.get("task_type")

    if task_type == TaskType.BUDGET_CONSTRAINED.value:
        return BudgetConstrainedTask(**data)
    elif task_type == TaskType.PREFERENCE_MEMORY.value:
        return PreferenceMemoryTask(**data)
    elif task_type == TaskType.NEGATIVE_CONSTRAINT.value:
        return NegativeConstraintTask(**data)
    elif task_type == TaskType.COMPARATIVE_REASONING.value:
        return ComparativeReasoningTask(**data)
    elif task_type == TaskType.ERROR_RECOVERY.value:
        return ErrorRecoveryTask(**data)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


# =============================================================================
# State Models
# =============================================================================


class CartItem(BaseModel):
    """An item in the shopping cart."""

    product_id: str
    product_name: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    quantity: int = 1
    price: float

    @property
    def total_price(self) -> float:
        """Calculate total price for this item (price * quantity)."""
        return self.price * self.quantity


class CartState(BaseModel):
    """Current state of the shopping cart."""

    items: list[CartItem] = Field(default_factory=list)

    @property
    def total(self) -> float:
        """Calculate cart total."""
        return sum(item.total_price for item in self.items)

    @property
    def item_count(self) -> int:
        """Count total number of items (accounting for quantities)."""
        return sum(item.quantity for item in self.items)

    def add_item(self, item: CartItem) -> None:
        """Add an item to the cart."""
        # Check if item with same product_id and attributes exists
        for existing in self.items:
            if (
                existing.product_id == item.product_id
                and existing.attributes == item.attributes
            ):
                existing.quantity += item.quantity
                return
        self.items.append(item)

    def remove_item(self, product_id: str, attributes: Optional[dict] = None) -> bool:
        """Remove an item from the cart.

        Args:
            product_id: The product ID to remove.
            attributes: Optional attributes to match. If None, removes first match.

        Returns:
            True if item was removed, False if not found.
        """
        for i, item in enumerate(self.items):
            if item.product_id == product_id:
                if attributes is None or item.attributes == attributes:
                    self.items.pop(i)
                    return True
        return False

    def clear(self) -> None:
        """Clear all items from the cart."""
        self.items.clear()


class ActionRecord(BaseModel):
    """Record of a single action taken during a session."""

    timestamp: datetime = Field(default_factory=datetime.now)
    action: str
    observation: str
    reward: float = 0.0


class PurchaseRecord(BaseModel):
    """Record of a completed purchase."""

    product_id: str
    product_name: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    price: float
    purchased_at: datetime = Field(default_factory=datetime.now)


class SessionState(BaseModel):
    """State of a single assessment session."""

    session_id: str
    task_id: str
    agent_id: str = ""
    started_at: datetime = Field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    actions: list[ActionRecord] = Field(default_factory=list)
    current_observation: str = ""
    cart: CartState = Field(default_factory=CartState)
    purchases: list[PurchaseRecord] = Field(default_factory=list)
    preferences_established: dict[str, Any] = Field(default_factory=dict)
    completed: bool = False
    error: Optional[str] = None

    @property
    def actions_taken(self) -> int:
        """Count of actions taken."""
        return len(self.actions)

    @property
    def elapsed_seconds(self) -> float:
        """Time elapsed since session start."""
        end = self.ended_at or datetime.now()
        return (end - self.started_at).total_seconds()

    def record_action(self, action: str, observation: str, reward: float = 0.0) -> None:
        """Record an action and its result."""
        self.actions.append(
            ActionRecord(action=action, observation=observation, reward=reward)
        )
        self.current_observation = observation

    def complete(self) -> None:
        """Mark the session as completed."""
        self.completed = True
        self.ended_at = datetime.now()

    def to_summary(self) -> "SessionSummary":
        """Convert session state to a summary for agent memory."""
        return SessionSummary(
            session_id=self.session_id,
            task_id=self.task_id,
            task_type="",  # Will be filled by caller
            purchases=self.purchases,
            preferences=self.preferences_established,
        )


# =============================================================================
# Evaluation Models
# =============================================================================


class ScoringComponent(BaseModel):
    """A single component of an evaluation score."""

    name: str
    weight: float
    raw_value: Any
    normalized_score: float = Field(ge=0.0, le=1.0)
    explanation: str = ""


class EvaluationResult(BaseModel):
    """Result of evaluating a completed task."""

    task_id: str
    task_type: TaskType

    # Universal metrics
    completed: bool = False
    success: bool = False
    actions_taken: int = 0
    time_elapsed_seconds: float = 0.0

    # Aggregate score (0-1)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Type-specific metrics
    metrics: dict[str, Any] = Field(default_factory=dict)

    # Detailed scoring breakdown
    scoring_breakdown: list[ScoringComponent] = Field(default_factory=list)

    # Optional error message
    error: Optional[str] = None

    def add_component(
        self,
        name: str,
        weight: float,
        raw_value: Any,
        normalized_score: float,
        explanation: str = "",
    ) -> None:
        """Add a scoring component to the breakdown."""
        self.scoring_breakdown.append(
            ScoringComponent(
                name=name,
                weight=weight,
                raw_value=raw_value,
                normalized_score=normalized_score,
                explanation=explanation,
            )
        )

    def calculate_overall_score(self) -> float:
        """Calculate overall score from scoring components."""
        if not self.scoring_breakdown:
            return 0.0
        total_weight = sum(c.weight for c in self.scoring_breakdown)
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(
            c.weight * c.normalized_score for c in self.scoring_breakdown
        )
        self.overall_score = weighted_sum / total_weight
        return self.overall_score


# =============================================================================
# Memory Models
# =============================================================================


class SessionSummary(BaseModel):
    """Summary of a completed session for agent memory."""

    session_id: str
    task_id: str
    task_type: str = ""
    purchases: list[PurchaseRecord] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)
    completed_at: datetime = Field(default_factory=datetime.now)


class AgentMemory(BaseModel):
    """Memory of an agent's past sessions within an assessment."""

    agent_id: str
    sessions: list[SessionSummary] = Field(default_factory=list)

    def add_session(self, summary: SessionSummary) -> None:
        """Add a session summary to memory."""
        self.sessions.append(summary)

    def get_sessions_by_type(self, task_type: str) -> list[SessionSummary]:
        """Get all sessions of a specific task type."""
        return [s for s in self.sessions if s.task_type == task_type]

    def get_all_purchases(self) -> list[PurchaseRecord]:
        """Get all purchases across all sessions."""
        purchases = []
        for session in self.sessions:
            purchases.extend(session.purchases)
        return purchases

    def get_all_preferences(self) -> dict[str, Any]:
        """Get merged preferences from all sessions."""
        merged = {}
        for session in self.sessions:
            merged.update(session.preferences)
        return merged

    def clear(self) -> None:
        """Clear all memory."""
        self.sessions.clear()


# =============================================================================
# Assessment Models (for A2A communication)
# =============================================================================


class AssessmentConfig(BaseModel):
    """Configuration for an assessment run."""

    task_types: list[str] = Field(default_factory=lambda: ["all"])
    num_tasks: int = 80
    timeout_per_task: int = 300
    include_memory_tasks: bool = True


class AssessmentRequest(BaseModel):
    """Request to start an assessment."""

    participants: dict[str, str]  # role -> endpoint URL
    config: AssessmentConfig = Field(default_factory=AssessmentConfig)


class TaskUpdate(BaseModel):
    """Progress update during assessment."""

    task_id: str
    status: Literal["pending", "working", "completed", "failed"] = "working"
    current_task: Optional[str] = None
    tasks_completed: int = 0
    tasks_total: int = 0
    message: Optional[str] = None
    progress: float = 0.0


class AggregateResults(BaseModel):
    """Aggregate results across all tasks."""

    total_tasks: int = 0
    successful_tasks: int = 0
    average_score: float = 0.0
    average_time: float = 0.0
    by_task_type: dict[str, dict[str, Any]] = Field(default_factory=dict)


class AssessmentResults(BaseModel):
    """Final results of an assessment."""

    assessment_id: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    participants: dict[str, str] = Field(default_factory=dict)
    config: AssessmentConfig = Field(default_factory=AssessmentConfig)
    results: list[EvaluationResult] = Field(default_factory=list)
    aggregate: AggregateResults = Field(default_factory=AggregateResults)

    def calculate_aggregate(self) -> None:
        """Calculate aggregate statistics from results."""
        if not self.results:
            return

        self.aggregate.total_tasks = len(self.results)
        self.aggregate.successful_tasks = sum(1 for r in self.results if r.success)
        self.aggregate.average_score = sum(r.overall_score for r in self.results) / len(
            self.results
        )
        self.aggregate.average_time = sum(
            r.time_elapsed_seconds for r in self.results
        ) / len(self.results)

        # Calculate by task type
        by_type: dict[str, list[EvaluationResult]] = {}
        for result in self.results:
            task_type = result.task_type.value
            if task_type not in by_type:
                by_type[task_type] = []
            by_type[task_type].append(result)

        for task_type, type_results in by_type.items():
            self.aggregate.by_task_type[task_type] = {
                "avg_score": sum(r.overall_score for r in type_results)
                / len(type_results),
                "count": len(type_results),
                "success_rate": sum(1 for r in type_results if r.success)
                / len(type_results),
            }
