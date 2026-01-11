"""
Evaluator for WebShop+ green agent.

This module provides scoring logic for all 5 task types:
- budget_constrained: Budget compliance, item completion, quality optimization
- preference_memory: Preference recall accuracy and consistency
- negative_constraint: Constraint satisfaction with violation penalties
- comparative_reasoning: Product comparison with LLM-as-judge for justification
- error_recovery: Error identification and correction scoring
"""

import re
from typing import Any, Optional

from .llm_client import LLMClient
from .models import (
    AgentMemory,
    BudgetConstrainedTask,
    CartState,
    ComparativeReasoningTask,
    ErrorRecoveryTask,
    EvaluationResult,
    NegativeConstraintTask,
    PreferenceMemoryTask,
    SessionState,
    Task,
    TaskType,
)


class Evaluator:
    """
    Evaluator for WebShop+ assessment tasks.

    The Evaluator scores completed sessions against task requirements using
    task-specific evaluation logic. For comparative reasoning tasks, it uses
    LLM-as-judge via the provided LLMClient.

    Example:
        >>> from src.llm_client import LLMClient
        >>> from src.evaluator import Evaluator
        >>> client = LLMClient()
        >>> evaluator = Evaluator(client)
        >>> result = evaluator.evaluate(session, task)
        >>> print(f"Score: {result.overall_score}")
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        """
        Initialize the Evaluator.

        Args:
            llm_client: LLMClient for LLM-as-judge evaluations. If not provided,
                       comparative reasoning tasks will use a simpler heuristic.
        """
        self._llm_client = llm_client

    @property
    def llm_client(self) -> Optional[LLMClient]:
        """Return the configured LLM client."""
        return self._llm_client

    def evaluate(
        self,
        session: SessionState,
        task: Task,
        memory: Optional[AgentMemory] = None,
    ) -> EvaluationResult:
        """
        Evaluate a completed session against a task.

        Dispatches to the appropriate task-specific evaluator based on task type.

        Args:
            session: The completed session state with action history.
            task: The task that was being assessed.
            memory: Agent memory for preference_memory tasks (optional).

        Returns:
            EvaluationResult with scores and breakdown.
        """
        # Validate task_type is a known TaskType enum
        task_type = task.task_type
        if not isinstance(task_type, TaskType):
            # Try to convert string to TaskType
            try:
                task_type = TaskType(task_type)
            except (ValueError, KeyError):
                # Return error result for unknown task type
                return EvaluationResult(
                    task_id=getattr(task, "task_id", "unknown"),
                    task_type=TaskType.BUDGET_CONSTRAINED,  # Default for error case
                    completed=session.completed,
                    actions_taken=session.actions_taken,
                    time_elapsed_seconds=session.elapsed_seconds,
                    error=f"Unknown task type: {task.task_type}",
                )

        # Create base result with universal metrics
        result = EvaluationResult(
            task_id=task.task_id,
            task_type=task_type,
            completed=session.completed,
            actions_taken=session.actions_taken,
            time_elapsed_seconds=session.elapsed_seconds,
        )

        # Dispatch to type-specific evaluator
        if task_type == TaskType.BUDGET_CONSTRAINED:
            return self.evaluate_budget_task(session, task, result)
        elif task_type == TaskType.PREFERENCE_MEMORY:
            return self.evaluate_memory_task(session, task, memory, result)
        elif task_type == TaskType.NEGATIVE_CONSTRAINT:
            return self.evaluate_constraint_task(session, task, result)
        elif task_type == TaskType.COMPARATIVE_REASONING:
            return self.evaluate_reasoning_task(session, task, result)
        elif task_type == TaskType.ERROR_RECOVERY:
            return self.evaluate_recovery_task(session, task, result)
        else:
            result.error = f"Unknown task type: {task_type}"
            return result

    def evaluate_budget_task(
        self,
        session: SessionState,
        task: BudgetConstrainedTask,
        result: Optional[EvaluationResult] = None,
    ) -> EvaluationResult:
        """
        Evaluate a budget-constrained task.

        Scoring formula:
            overall = (budget_compliance * weight) + (item_completion * weight) + (quality * weight)

        Budget compliance: 1.0 if under budget, scales down if over
        Item completion: Fraction of required items purchased
        Quality: Based on optimization goal (ratings, price efficiency, or balanced)

        Args:
            session: Completed session with purchases/cart.
            task: The budget-constrained task definition.
            result: Optional pre-populated result (for dispatch pattern).

        Returns:
            EvaluationResult with budget task scoring.
        """
        if result is None:
            result = EvaluationResult(
                task_id=task.task_id,
                task_type=task.task_type,
                completed=session.completed,
                actions_taken=session.actions_taken,
                time_elapsed_seconds=session.elapsed_seconds,
            )

        budget = task.constraints.budget
        required_items = task.constraints.required_items
        optimization_goal = task.constraints.optimization_goal.value
        weights = task.evaluation_criteria

        # Calculate total spent from purchases or cart
        total_spent = self._get_total_spent(session)

        # 1. Budget compliance score
        if total_spent <= budget:
            budget_score = 1.0
            budget_explanation = f"Within budget: ${total_spent:.2f} / ${budget:.2f}"
        elif total_spent <= budget * 1.1:
            # Up to 10% over: partial credit
            overage = (total_spent - budget) / budget
            budget_score = 1.0 - overage
            budget_explanation = f"Slightly over budget: ${total_spent:.2f} / ${budget:.2f} ({overage*100:.1f}% over)"
        else:
            # More than 10% over: steep penalty
            overage = (total_spent - budget) / budget
            budget_score = max(0.0, 1.0 - overage * 2)
            budget_explanation = f"Over budget: ${total_spent:.2f} / ${budget:.2f} ({overage*100:.1f}% over)"

        result.add_component(
            name="budget_compliance",
            weight=weights.budget_weight,
            raw_value=total_spent,
            normalized_score=budget_score,
            explanation=budget_explanation,
        )

        # 2. Item completion score
        purchased_items = self._get_purchased_items(session)
        required_count = len([r for r in required_items if not r.optional])
        optional_count = len([r for r in required_items if r.optional])

        matched_required = 0
        matched_optional = 0

        for req in required_items:
            if self._item_matches_requirement(purchased_items, req):
                if req.optional:
                    matched_optional += 1
                else:
                    matched_required += 1

        if required_count > 0:
            completion_score = matched_required / required_count
            # Bonus for optional items (up to 10%)
            if optional_count > 0:
                optional_bonus = (matched_optional / optional_count) * 0.1
                completion_score = min(1.0, completion_score + optional_bonus)
        else:
            completion_score = 1.0 if matched_optional > 0 else 0.0

        completion_explanation = f"Matched {matched_required}/{required_count} required items"
        if optional_count > 0:
            completion_explanation += f", {matched_optional}/{optional_count} optional"

        result.add_component(
            name="item_completion",
            weight=weights.completion_weight,
            raw_value={"required": matched_required, "optional": matched_optional},
            normalized_score=completion_score,
            explanation=completion_explanation,
        )

        # 3. Quality/optimization score
        quality_score, quality_explanation = self._calculate_quality_score(
            session, budget, optimization_goal, total_spent
        )

        result.add_component(
            name="quality_optimization",
            weight=weights.quality_weight,
            raw_value=optimization_goal,
            normalized_score=quality_score,
            explanation=quality_explanation,
        )

        # Calculate overall and determine success
        result.calculate_overall_score()
        result.success = (
            budget_score >= 0.8 and completion_score >= 0.8 and result.overall_score >= 0.6
        )

        # Add metrics
        result.metrics = {
            "budget": budget,
            "total_spent": total_spent,
            "required_items_matched": matched_required,
            "required_items_total": required_count,
            "optimization_goal": optimization_goal,
        }

        return result

    def evaluate_memory_task(
        self,
        session: SessionState,
        task: PreferenceMemoryTask,
        memory: Optional[AgentMemory] = None,
        result: Optional[EvaluationResult] = None,
    ) -> EvaluationResult:
        """
        Evaluate a preference memory task.

        Scoring formula:
            overall = (recall_accuracy * weight) + (consistency * weight)

        Recall accuracy: Did the agent recall the established preference correctly?
        Consistency: Is the purchase consistent with remembered preferences?

        Args:
            session: Completed session with actions and purchases.
            task: The preference memory task definition.
            memory: Agent memory with previous session preferences.
            result: Optional pre-populated result.

        Returns:
            EvaluationResult with memory task scoring.
        """
        if result is None:
            result = EvaluationResult(
                task_id=task.task_id,
                task_type=task.task_type,
                completed=session.completed,
                actions_taken=session.actions_taken,
                time_elapsed_seconds=session.elapsed_seconds,
            )

        weights = task.evaluation_criteria
        memory_test = task.memory_test

        # Get expected preferences from previous sessions in the sequence
        expected_preferences = self._extract_expected_preferences(task)

        # Check agent's actions/messages for preference recall
        recall_score, recall_explanation = self._check_preference_recall(
            session,
            memory_test.attribute_to_recall,
            memory_test.acceptable_values,
            memory,
        )

        result.add_component(
            name="recall_accuracy",
            weight=weights.recall_accuracy_weight,
            raw_value=memory_test.attribute_to_recall,
            normalized_score=recall_score,
            explanation=recall_explanation,
        )

        # Check purchase consistency with remembered preferences
        consistency_score, consistency_explanation = self._check_preference_consistency(
            session,
            expected_preferences,
            memory,
        )

        result.add_component(
            name="preference_consistency",
            weight=weights.consistency_weight,
            raw_value=expected_preferences,
            normalized_score=consistency_score,
            explanation=consistency_explanation,
        )

        result.calculate_overall_score()
        result.success = recall_score >= 0.5 and consistency_score >= 0.5

        result.metrics = {
            "attribute_tested": memory_test.attribute_to_recall,
            "acceptable_values": memory_test.acceptable_values,
            "expected_preferences": expected_preferences,
            "recall_detected": recall_score > 0,
        }

        return result

    def evaluate_constraint_task(
        self,
        session: SessionState,
        task: NegativeConstraintTask,
        result: Optional[EvaluationResult] = None,
    ) -> EvaluationResult:
        """
        Evaluate a negative constraint task.

        Scoring formula:
            violation_penalty = min(1.0, violations * 0.25)  # Each violation costs 25%
            overall = positive_match * (1 - violation_penalty)

        Args:
            session: Completed session with purchases.
            task: The negative constraint task definition.
            result: Optional pre-populated result.

        Returns:
            EvaluationResult with constraint task scoring.
        """
        if result is None:
            result = EvaluationResult(
                task_id=task.task_id,
                task_type=task.task_type,
                completed=session.completed,
                actions_taken=session.actions_taken,
                time_elapsed_seconds=session.elapsed_seconds,
            )

        constraints = task.constraints
        weights = task.evaluation_criteria
        purchased_items = self._get_purchased_items(session)

        # Check for constraint violations
        violations = []
        for item in purchased_items:
            item_violations = self._check_constraint_violations(
                item,
                constraints.forbidden_attributes,
                constraints.forbidden_terms,
            )
            violations.extend(item_violations)

        violation_count = len(violations)
        violation_penalty = min(1.0, violation_count * weights.constraint_violation_penalty)

        violation_score = 1.0 - violation_penalty
        if violations:
            violation_explanation = f"{violation_count} violation(s): {', '.join(violations[:3])}"
            if len(violations) > 3:
                violation_explanation += f" (+{len(violations) - 3} more)"
        else:
            violation_explanation = "No constraint violations detected"

        result.add_component(
            name="constraint_satisfaction",
            weight=1.0 - weights.match_score_weight,
            raw_value=violation_count,
            normalized_score=violation_score,
            explanation=violation_explanation,
        )

        # Check positive match (required attributes)
        match_score, match_explanation = self._check_positive_match(
            purchased_items,
            constraints.required_attributes,
        )

        result.add_component(
            name="positive_match",
            weight=weights.match_score_weight,
            raw_value=constraints.required_attributes,
            normalized_score=match_score,
            explanation=match_explanation,
        )

        # Apply violation penalty to overall match
        result.calculate_overall_score()

        # Apply additional penalty: violations reduce overall score
        if violation_count > 0:
            result.overall_score = result.overall_score * (1 - violation_penalty)

        result.success = violation_count == 0 and match_score >= 0.5

        result.metrics = {
            "violations": violations,
            "violation_count": violation_count,
            "forbidden_attributes": constraints.forbidden_attributes,
            "forbidden_terms": constraints.forbidden_terms,
            "required_attributes": constraints.required_attributes,
        }

        return result

    def evaluate_reasoning_task(
        self,
        session: SessionState,
        task: ComparativeReasoningTask,
        result: Optional[EvaluationResult] = None,
    ) -> EvaluationResult:
        """
        Evaluate a comparative reasoning task using LLM-as-judge.

        Scoring formula:
            overall = (exploration * 0.3) + (justification_provided * 0.2) + (justification_quality * 0.5)

        Args:
            session: Completed session with action trace.
            task: The comparative reasoning task definition.
            result: Optional pre-populated result.

        Returns:
            EvaluationResult with reasoning task scoring.
        """
        if result is None:
            result = EvaluationResult(
                task_id=task.task_id,
                task_type=task.task_type,
                completed=session.completed,
                actions_taken=session.actions_taken,
                time_elapsed_seconds=session.elapsed_seconds,
            )

        requirements = task.requirements
        criteria = task.evaluation_criteria

        # 1. Exploration score: Did the agent explore enough options?
        products_explored = self._count_products_explored(session)
        min_required = criteria.minimum_options_explored

        if products_explored >= min_required:
            exploration_score = 1.0
            exploration_explanation = f"Explored {products_explored} products (min: {min_required})"
        elif products_explored > 0:
            exploration_score = products_explored / min_required
            exploration_explanation = f"Only explored {products_explored}/{min_required} required products"
        else:
            exploration_score = 0.0
            exploration_explanation = "No products explored"

        result.add_component(
            name="exploration",
            weight=0.3,
            raw_value=products_explored,
            normalized_score=exploration_score,
            explanation=exploration_explanation,
        )

        # 2. Justification provided: Did the agent provide reasoning?
        justification = self._extract_justification(session)
        justification_provided = len(justification) > 50  # At least some explanation

        if justification_provided:
            provided_score = 1.0
            provided_explanation = "Agent provided comparison justification"
        else:
            provided_score = 0.0
            provided_explanation = "No justification provided"

        result.add_component(
            name="justification_provided",
            weight=0.2,
            raw_value=justification_provided,
            normalized_score=provided_score,
            explanation=provided_explanation,
        )

        # 3. Justification quality: LLM-as-judge
        if justification_provided and self._llm_client is not None:
            quality_score, quality_explanation = self._evaluate_justification_quality(
                justification,
                requirements.comparison_request,
                requirements.category,
            )
        elif justification_provided:
            # Fallback heuristic without LLM
            quality_score, quality_explanation = self._heuristic_justification_score(
                justification
            )
        else:
            quality_score = 0.0
            quality_explanation = "No justification to evaluate"

        result.add_component(
            name="justification_quality",
            weight=criteria.justification_quality_weight,
            raw_value=justification[:200] if justification else "",
            normalized_score=quality_score,
            explanation=quality_explanation,
        )

        result.calculate_overall_score()
        result.success = exploration_score >= 0.5 and result.overall_score >= 0.5

        result.metrics = {
            "products_explored": products_explored,
            "minimum_required": min_required,
            "justification_provided": justification_provided,
            "justification_length": len(justification),
            "comparison_request": requirements.comparison_request,
        }

        return result

    def evaluate_recovery_task(
        self,
        session: SessionState,
        task: ErrorRecoveryTask,
        result: Optional[EvaluationResult] = None,
    ) -> EvaluationResult:
        """
        Evaluate an error recovery task.

        Scoring formula:
            action_penalty = max(0, (actions_taken - expected) / expected) * penalty_weight
            overall = (error_fixed * 0.7) + (error_identified * 0.1) + ((1 - action_penalty) * 0.2)

        Args:
            session: Completed session with final cart state.
            task: The error recovery task definition.
            result: Optional pre-populated result.

        Returns:
            EvaluationResult with recovery task scoring.
        """
        if result is None:
            result = EvaluationResult(
                task_id=task.task_id,
                task_type=task.task_type,
                completed=session.completed,
                actions_taken=session.actions_taken,
                time_elapsed_seconds=session.elapsed_seconds,
            )

        setup = task.setup
        correct_state = task.correct_state
        criteria = task.evaluation_criteria

        # Build expected cart from correct_state
        expected_cart = self._build_cart_from_setup(correct_state.expected_cart)

        # 1. Error fixed: Does the final cart match expected?
        error_fixed, fix_explanation = self._compare_carts(
            session.cart, expected_cart
        )

        result.add_component(
            name="error_fixed",
            weight=0.7,
            raw_value=error_fixed,
            normalized_score=1.0 if error_fixed else 0.0,
            explanation=fix_explanation,
        )

        # 2. Error identified: Did the agent acknowledge the error?
        error_identified = self._check_error_identified(session, setup.error_description)

        result.add_component(
            name="error_identified",
            weight=0.1,
            raw_value=error_identified,
            normalized_score=1.0 if error_identified else 0.5,
            explanation="Agent acknowledged error" if error_identified else "Error acknowledgment unclear",
        )

        # 3. Efficiency: Action penalty for excessive actions
        expected_actions = task.expected_actions
        actual_actions = session.actions_taken

        if actual_actions <= expected_actions:
            efficiency_score = 1.0
            efficiency_explanation = f"Efficient: {actual_actions} actions (expected {expected_actions})"
        else:
            overage = (actual_actions - expected_actions) / expected_actions
            penalty = min(1.0, overage * criteria.unnecessary_actions_penalty * 5)
            efficiency_score = 1.0 - penalty
            efficiency_explanation = f"Inefficient: {actual_actions} actions (expected {expected_actions})"

        result.add_component(
            name="efficiency",
            weight=0.2,
            raw_value=actual_actions,
            normalized_score=efficiency_score,
            explanation=efficiency_explanation,
        )

        result.calculate_overall_score()
        result.success = error_fixed and session.completed

        result.metrics = {
            "error_description": setup.error_description,
            "error_fixed": error_fixed,
            "error_identified": error_identified,
            "actions_taken": actual_actions,
            "expected_actions": expected_actions,
            "initial_cart_items": len(setup.cart_contents),
            "expected_cart_items": len(correct_state.expected_cart),
            "final_cart_items": len(session.cart.items),
        }

        return result

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _get_total_spent(self, session: SessionState) -> float:
        """Get total amount spent from purchases or cart."""
        if session.purchases:
            return sum(p.price for p in session.purchases)
        return session.cart.total

    def _get_purchased_items(self, session: SessionState) -> list[dict[str, Any]]:
        """Get list of purchased/carted items as dicts."""
        items = []

        # Check purchases first
        for purchase in session.purchases:
            items.append({
                "product_id": purchase.product_id,
                "product_name": purchase.product_name,
                "attributes": purchase.attributes,
                "price": purchase.price,
            })

        # Fall back to cart if no purchases
        if not items:
            for cart_item in session.cart.items:
                items.append({
                    "product_id": cart_item.product_id,
                    "product_name": cart_item.product_name,
                    "attributes": cart_item.attributes,
                    "price": cart_item.price,
                    "quantity": cart_item.quantity,
                })

        return items

    def _item_matches_requirement(
        self,
        items: list[dict[str, Any]],
        requirement: Any,
    ) -> bool:
        """Check if any purchased item matches a required item specification."""
        req_category = requirement.category.lower()
        req_attrs = requirement.attributes

        for item in items:
            name = item.get("product_name", "").lower()
            attrs = item.get("attributes", {})

            # Check if category matches (in name or attributes)
            category_match = req_category in name or any(
                req_category in str(v).lower() for v in attrs.values()
            )

            # Check if required attributes match
            # ALL required attributes must be satisfied for attr_match to be True
            attr_match = len(req_attrs) > 0  # Need at least one attribute to check
            for key, value in req_attrs.items():
                value_lower = str(value).lower()
                found = False

                # Check if the required value appears in item name
                if value_lower in name:
                    found = True
                else:
                    # Check if the required value appears in any attribute value
                    for attr_val in attrs.values():
                        if value_lower in str(attr_val).lower():
                            found = True
                            break

                if not found:
                    attr_match = False
                    break

            # Item matches if it satisfies BOTH category AND attributes,
            # or if it satisfies all required attributes (which implies correct type)
            if (category_match and attr_match) or (attr_match and len(req_attrs) > 0):
                return True

        return False

    def _calculate_quality_score(
        self,
        session: SessionState,
        budget: float,
        optimization_goal: str,
        total_spent: float,
    ) -> tuple[float, str]:
        """Calculate quality score based on optimization goal."""
        if optimization_goal == "maximize_quality":
            # Quality is maximized when spending most of budget on rated items
            if total_spent > 0:
                efficiency = min(1.0, total_spent / budget)
                return efficiency, f"Quality optimization: {efficiency*100:.0f}% of budget utilized"
            return 0.0, "No purchases made"

        elif optimization_goal == "minimize_cost":
            # Quality is maximized when getting items under budget
            if total_spent > 0 and total_spent <= budget:
                savings = (budget - total_spent) / budget
                score = 0.5 + (savings * 0.5)  # Base + bonus for savings
                return score, f"Cost minimization: saved ${budget - total_spent:.2f} ({savings*100:.0f}%)"
            elif total_spent > budget:
                return 0.3, "Over budget - cost not minimized"
            return 0.0, "No purchases made"

        else:  # balance
            # Balance between cost and quality
            if total_spent > 0 and total_spent <= budget:
                utilization = total_spent / budget
                # Optimal is around 70-90% of budget
                if 0.7 <= utilization <= 0.9:
                    score = 1.0
                elif utilization < 0.7:
                    score = 0.5 + (utilization / 0.7) * 0.5
                else:
                    score = 0.9 + (1.0 - utilization) / 0.1 * 0.1
                return score, f"Balanced: used {utilization*100:.0f}% of budget"
            elif total_spent > budget:
                return 0.3, "Over budget"
            return 0.0, "No purchases made"

    def _extract_expected_preferences(self, task: PreferenceMemoryTask) -> dict[str, Any]:
        """Extract preferences that should be remembered from task sequence."""
        preferences = {}
        for session_item in task.session_sequence:
            preferences.update(session_item.establishes)
        return preferences

    def _check_preference_recall(
        self,
        session: SessionState,
        attribute: str,
        acceptable_values: list[str],
        memory: Optional[AgentMemory],
    ) -> tuple[float, str]:
        """Check if the agent correctly recalled a preference."""
        # Look for mentions of the attribute or acceptable values in actions
        action_text = " ".join(a.action.lower() for a in session.actions)
        observation_text = " ".join(a.observation.lower() for a in session.actions)
        all_text = action_text + " " + observation_text

        attribute_lower = attribute.lower()
        values_lower = [v.lower() for v in acceptable_values]

        # Check for explicit recall
        attribute_mentioned = attribute_lower in all_text
        value_mentioned = any(v in all_text for v in values_lower)

        if attribute_mentioned and value_mentioned:
            return 1.0, f"Recalled {attribute} with correct value"
        elif value_mentioned:
            return 0.8, f"Used correct value for {attribute} (implicit recall)"
        elif attribute_mentioned:
            return 0.4, f"Mentioned {attribute} but incorrect value"
        else:
            # Check memory-based recall
            if memory:
                remembered = memory.get_all_preferences()
                if attribute_lower in str(remembered).lower():
                    return 0.5, f"Preference {attribute} in memory but not explicitly used"

            return 0.0, f"No recall of {attribute} detected"

    def _check_preference_consistency(
        self,
        session: SessionState,
        expected_preferences: dict[str, Any],
        memory: Optional[AgentMemory],
    ) -> tuple[float, str]:
        """Check if purchases are consistent with remembered preferences."""
        if not expected_preferences:
            return 1.0, "No preferences to check"

        items = self._get_purchased_items(session)
        if not items:
            return 0.0, "No purchases to check consistency"

        matches = 0
        total = len(expected_preferences)

        for key, value in expected_preferences.items():
            key_lower = key.lower()
            value_lower = str(value).lower()

            for item in items:
                # Check in product name
                if value_lower in item.get("product_name", "").lower():
                    matches += 1
                    break

                # Check in attributes
                item_attrs = item.get("attributes", {})
                for attr_key, attr_val in item_attrs.items():
                    if key_lower in attr_key.lower() and value_lower in str(attr_val).lower():
                        matches += 1
                        break

        if total > 0:
            score = matches / total
            return score, f"Consistency: {matches}/{total} preferences matched"
        return 1.0, "No preferences to verify"

    def _check_constraint_violations(
        self,
        item: dict[str, Any],
        forbidden_attributes: list[str],
        forbidden_terms: list[str],
    ) -> list[str]:
        """Check a purchased item for constraint violations."""
        violations = []
        item_text = (
            item.get("product_name", "").lower()
            + " "
            + " ".join(str(v).lower() for v in item.get("attributes", {}).values())
        )

        for attr in forbidden_attributes:
            if attr.lower() in item_text:
                violations.append(f"forbidden attribute: {attr}")

        for term in forbidden_terms:
            if term.lower() in item_text:
                violations.append(f"forbidden term: {term}")

        return violations

    def _check_positive_match(
        self,
        items: list[dict[str, Any]],
        required_attributes: list[str],
    ) -> tuple[float, str]:
        """Check if purchased items have required positive attributes."""
        if not required_attributes:
            return 1.0, "No required attributes specified"

        if not items:
            return 0.0, "No purchases to check"

        # Combine all item text
        all_item_text = ""
        for item in items:
            all_item_text += item.get("product_name", "").lower() + " "
            all_item_text += " ".join(str(v).lower() for v in item.get("attributes", {}).values())
            all_item_text += " "

        matched = 0
        for attr in required_attributes:
            if attr.lower() in all_item_text:
                matched += 1

        score = matched / len(required_attributes)
        return score, f"Matched {matched}/{len(required_attributes)} required attributes"

    def _count_products_explored(self, session: SessionState) -> int:
        """Count unique products explored during the session."""
        product_ids = set()
        product_patterns = [
            r"product[_/]?(\w+)",
            r"item[_/]?(\w+)",
            r"asin[=:]?\s*([A-Z0-9]+)",
            r"view.*?(\w{8,})",
        ]

        for action in session.actions:
            action_text = action.action + " " + action.observation

            for pattern in product_patterns:
                matches = re.findall(pattern, action_text, re.IGNORECASE)
                product_ids.update(matches)

            # Also count "click" or "view" actions as exploration
            if any(word in action.action.lower() for word in ["click", "view", "select", "open"]):
                # Extract any identifier-like strings
                ids = re.findall(r"\b[A-Z0-9]{6,}\b", action_text)
                product_ids.update(ids)

        # Reasonable minimum
        return max(len(product_ids), len([a for a in session.actions if "product" in a.observation.lower()]))

    def _extract_justification(self, session: SessionState) -> str:
        """Extract agent's justification/reasoning from session."""
        justification_parts = []

        comparison_keywords = [
            "because", "since", "therefore", "better", "worse",
            "compared", "versus", "vs", "recommend", "choose",
            "prefer", "rating", "review", "price", "quality",
            "feature", "advantage", "disadvantage", "pros", "cons",
        ]

        for action in session.actions:
            action_text = action.action.lower()

            # Check if action contains comparison/justification language
            if any(keyword in action_text for keyword in comparison_keywords):
                justification_parts.append(action.action)

            # Also check for longer explanatory text
            if len(action.action) > 100:
                justification_parts.append(action.action)

        return " ".join(justification_parts)

    def _evaluate_justification_quality(
        self,
        justification: str,
        comparison_request: str,
        category: str,
    ) -> tuple[float, str]:
        """Use LLM-as-judge to evaluate justification quality."""
        rubric = f"""
Evaluate the quality of this product comparison justification.

The agent was asked to: {comparison_request}
Product category: {category}

A good justification should:
1. Compare specific features relevant to the request
2. Consider price-to-value ratio
3. Reference actual product attributes (ratings, specs, reviews)
4. Provide a clear recommendation with reasoning
5. Address the specific use case mentioned

Score from 0-10 where:
- 0-2: No real comparison or generic/irrelevant text
- 3-4: Basic comparison but missing key factors
- 5-6: Decent comparison with some relevant factors
- 7-8: Good comparison covering most important factors
- 9-10: Excellent comparison with thorough, relevant analysis
"""

        try:
            score, explanation = self._llm_client.evaluate_with_rubric(
                content=justification,
                rubric=rubric,
                max_score=10,
            )
            normalized_score = score / 10.0
            return normalized_score, explanation
        except Exception:
            # Fall back to heuristic if LLM fails
            return self._heuristic_justification_score(justification)

    def _heuristic_justification_score(
        self,
        justification: str,
    ) -> tuple[float, str]:
        """Simple heuristic for justification quality without LLM."""
        if not justification:
            return 0.0, "No justification provided"

        score = 0.0
        reasons = []

        # Length-based score
        length = len(justification)
        if length > 200:
            score += 0.3
            reasons.append("detailed")
        elif length > 100:
            score += 0.2
            reasons.append("moderate length")

        # Check for comparison language
        comparison_words = ["better", "worse", "compared", "versus", "vs", "than"]
        if any(word in justification.lower() for word in comparison_words):
            score += 0.3
            reasons.append("comparative language")

        # Check for specific attributes
        attribute_words = ["price", "rating", "review", "feature", "quality", "value"]
        matches = sum(1 for word in attribute_words if word in justification.lower())
        if matches >= 2:
            score += 0.2
            reasons.append(f"{matches} attribute mentions")

        # Check for recommendation
        if any(word in justification.lower() for word in ["recommend", "choose", "pick", "best"]):
            score += 0.2
            reasons.append("clear recommendation")

        score = min(1.0, score)
        explanation = "Heuristic: " + ", ".join(reasons) if reasons else "Weak justification"
        return score, explanation

    def _build_cart_from_setup(self, cart_setup: list) -> CartState:
        """Build a CartState from cart setup items."""
        from .models import CartItem, CartItemSetup

        cart = CartState()
        for setup_item in cart_setup:
            if isinstance(setup_item, CartItemSetup):
                item = CartItem(
                    product_id=setup_item.product_id,
                    product_name=setup_item.product_name,
                    attributes=setup_item.attributes,
                    quantity=setup_item.quantity,
                    price=setup_item.price,
                )
            else:
                # Already a CartItem or dict
                item = CartItem(**setup_item) if isinstance(setup_item, dict) else setup_item
            cart.add_item(item)
        return cart

    def _compare_carts(
        self,
        actual: CartState,
        expected: CartState,
    ) -> tuple[bool, str]:
        """Compare actual cart to expected cart state."""
        # Compare items
        if len(actual.items) != len(expected.items):
            return False, f"Item count mismatch: {len(actual.items)} vs {len(expected.items)} expected"

        # Match items by product_id
        actual_by_id = {item.product_id: item for item in actual.items}
        expected_by_id = {item.product_id: item for item in expected.items}

        for prod_id, expected_item in expected_by_id.items():
            if prod_id not in actual_by_id:
                return False, f"Missing expected product: {prod_id}"

            actual_item = actual_by_id[prod_id]

            # Check quantity
            if actual_item.quantity != expected_item.quantity:
                return False, f"Quantity mismatch for {prod_id}: {actual_item.quantity} vs {expected_item.quantity}"

        # Check for unexpected items
        for prod_id in actual_by_id:
            if prod_id not in expected_by_id:
                return False, f"Unexpected product in cart: {prod_id}"

        return True, "Cart matches expected state"

    def _check_error_identified(
        self,
        session: SessionState,
        error_description: str,
    ) -> bool:
        """Check if the agent acknowledged the error in their actions."""
        error_keywords = error_description.lower().split()
        action_text = " ".join(a.action.lower() for a in session.actions)

        # Look for acknowledgment keywords
        acknowledgment_words = [
            "wrong", "error", "mistake", "incorrect", "fix",
            "remove", "change", "adjust", "update", "correct",
        ]

        acknowledged = any(word in action_text for word in acknowledgment_words)

        # Also check if they mentioned specifics from the error
        specific_match = sum(1 for word in error_keywords if word in action_text and len(word) > 3)

        return acknowledged or specific_match >= 2
