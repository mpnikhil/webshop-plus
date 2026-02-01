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

import structlog

from .llm_client import LLMClient

logger = structlog.get_logger()
from .models import (
    BudgetConstrainedTask,
    CartItem,
    CartState,
    ComparativeReasoningTask,
    ErrorRecoveryTask,
    EvaluationResult,
    NegativeConstraintTask,
    PreferenceMemoryTask,
    Task,
    TaskType,
)
from .webshop_mcp.session_state import SessionState as MCPSessionState


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
        mcp_state: MCPSessionState,
        task: Task,
    ) -> EvaluationResult:
        """
        Evaluate a completed session against a task using MCP state.

        Dispatches to the appropriate task-specific evaluator based on task type.

        Args:
            mcp_state: The MCP session state with cart, history, and turn count.
            task: The task that was being assessed.

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
                    completed=mcp_state.completed,
                    actions_taken=mcp_state.turn_count,
                    time_elapsed_seconds=0.0,  # TODO: Track timing in MCP state
                    error=f"Unknown task type: {task.task_type}",
                )

        # Create base result with universal metrics from MCP state
        result = EvaluationResult(
            task_id=task.task_id,
            task_type=task_type,
            completed=mcp_state.completed,
            actions_taken=mcp_state.turn_count,
            time_elapsed_seconds=0.0,  # TODO: Track timing in MCP state
        )

        # Dispatch to type-specific evaluator
        if task_type == TaskType.BUDGET_CONSTRAINED:
            return self.evaluate_budget_task(mcp_state, task, result)
        elif task_type == TaskType.PREFERENCE_MEMORY:
            return self.evaluate_memory_task(mcp_state, task, result)
        elif task_type == TaskType.NEGATIVE_CONSTRAINT:
            return self.evaluate_constraint_task(mcp_state, task, result)
        elif task_type == TaskType.COMPARATIVE_REASONING:
            return self.evaluate_reasoning_task(mcp_state, task, result)
        elif task_type == TaskType.ERROR_RECOVERY:
            return self.evaluate_recovery_task(mcp_state, task, result)
        else:
            result.error = f"Unknown task type: {task_type}"
            return result

    def evaluate_budget_task(
        self,
        mcp_state: MCPSessionState,
        task: BudgetConstrainedTask,
        result: Optional[EvaluationResult] = None,
    ) -> EvaluationResult:
        """
        Evaluate a budget-constrained task using MCP state.

        Scoring formula:
            overall = (budget_compliance * weight) + (item_completion * weight) + (quality * weight)

        Budget compliance: 1.0 if under budget, scales down if over
        Item completion: Fraction of required items purchased
        Quality: Based on optimization goal (ratings, price efficiency, or balanced)

        Args:
            mcp_state: MCP session state with cart and history.
            task: The budget-constrained task definition.
            result: Optional pre-populated result (for dispatch pattern).

        Returns:
            EvaluationResult with budget task scoring.
        """
        if result is None:
            result = EvaluationResult(
                task_id=task.task_id,
                task_type=task.task_type,
                completed=mcp_state.completed,
                actions_taken=mcp_state.turn_count,
                time_elapsed_seconds=0.0,
            )

        budget = task.constraints.budget
        required_items = task.constraints.required_items
        optimization_goal = task.constraints.optimization_goal.value
        weights = task.evaluation_criteria

        # Calculate total spent from MCP cart
        total_spent = self._get_total_spent(mcp_state)

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
        purchased_items = self._get_purchased_items(mcp_state)
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
            budget, optimization_goal, total_spent
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
        # Task is successful if overall score is high enough
        result.success = result.overall_score >= 0.7

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
        mcp_state: MCPSessionState,
        task: PreferenceMemoryTask,
        result: Optional[EvaluationResult] = None,
    ) -> EvaluationResult:
        """
        Evaluate a preference memory task using MCP state.

        Scoring formula:
            overall = recall_accuracy * recall_weight + consistency * consistency_weight

        Args:
            mcp_state: MCP session state with cart.
            task: The preference memory task definition.
            result: Optional pre-populated result.

        Returns:
            EvaluationResult with memory task scoring.
        """
        if result is None:
            result = EvaluationResult(
                task_id=task.task_id,
                task_type=task.task_type,
                completed=mcp_state.completed,
                actions_taken=mcp_state.turn_count,
                time_elapsed_seconds=0.0,
            )

        memory_test = task.memory_test
        weights = task.evaluation_criteria

        # 1. Check recall accuracy (Did they buy an item matching the recalled attribute?)
        recall_score, recall_explanation = self._check_preference_recall(
            mcp_state,
            task,
        )

        result.add_component(
            name="recall_accuracy",
            weight=weights.recall_accuracy_weight,
            raw_value=memory_test.acceptable_values,
            normalized_score=recall_score,
            explanation=recall_explanation,
        )

        # 2. Consistency (For now, just a placeholder or mapped to recall)
        # In single-shot simplified memory, consistency is implicitly tested by the recall itself
        consistency_score = recall_score
        consistency_explanation = "Consistency implied by accurate recall in single-session test"

        result.add_component(
            name="consistency",
            weight=weights.consistency_weight,
            raw_value=None,
            normalized_score=consistency_score,
            explanation=consistency_explanation,
        )

        result.calculate_overall_score()
        result.success = result.overall_score >= 0.7

        result.metrics = {
            "attribute_to_recall": memory_test.attribute_to_recall,
            "acceptable_values": memory_test.acceptable_values,
        }

        return result

    def evaluate_constraint_task(
        self,
        mcp_state: MCPSessionState,
        task: NegativeConstraintTask,
        result: Optional[EvaluationResult] = None,
    ) -> EvaluationResult:
        """
        Evaluate a negative constraint task using MCP state.

        Scoring formula:
            violation_penalty = min(1.0, violations * 0.25)  # Each violation costs 25%
            overall = positive_match * (1 - violation_penalty)

        Args:
            mcp_state: MCP session state with cart.
            task: The negative constraint task definition.
            result: Optional pre-populated result.

        Returns:
            EvaluationResult with constraint task scoring.
        """
        if result is None:
            result = EvaluationResult(
                task_id=task.task_id,
                task_type=task.task_type,
                completed=mcp_state.completed,
                actions_taken=mcp_state.turn_count,
                time_elapsed_seconds=0.0,
            )

        constraints = task.constraints
        weights = task.evaluation_criteria
        purchased_items = self._get_purchased_items(mcp_state)

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
        mcp_state: MCPSessionState,
        task: ComparativeReasoningTask,
        result: Optional[EvaluationResult] = None,
    ) -> EvaluationResult:
        """
        Evaluate a comparative reasoning task using MCP state.

        Scoring formula:
            overall = (exploration * 0.3) + (justification_provided * 0.2) + (justification_quality * 0.5)

        Args:
            mcp_state: MCP session state with history.
            task: The comparative reasoning task definition.
            result: Optional pre-populated result.

        Returns:
            EvaluationResult with reasoning task scoring.
        """
        if result is None:
            result = EvaluationResult(
                task_id=task.task_id,
                task_type=task.task_type,
                completed=mcp_state.completed,
                actions_taken=mcp_state.turn_count,
                time_elapsed_seconds=0.0,
            )

        requirements = task.requirements
        criteria = task.evaluation_criteria

        # 1. Exploration score: Did the agent explore enough options?
        products_explored = self._count_products_explored(mcp_state)
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
        justification = self._extract_justification(mcp_state)
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
        mcp_state: MCPSessionState,
        task: ErrorRecoveryTask,
        result: Optional[EvaluationResult] = None,
    ) -> EvaluationResult:
        """
        Evaluate an error recovery task using MCP state.

        Scoring formula:
            action_penalty = max(0, (actions_taken - expected) / expected) * penalty_weight
            overall = (error_fixed * 0.7) + (error_identified * 0.1) + ((1 - action_penalty) * 0.2)

        Args:
            mcp_state: MCP session state with cart.
            task: The error recovery task definition.
            result: Optional pre-populated result.

        Returns:
            EvaluationResult with recovery task scoring.
        """
        if result is None:
            result = EvaluationResult(
                task_id=task.task_id,
                task_type=task.task_type,
                completed=mcp_state.completed,
                actions_taken=mcp_state.turn_count,
                time_elapsed_seconds=0.0,
            )

        setup = task.setup
        correct_state = task.correct_state
        criteria = task.evaluation_criteria

        # Build expected cart from correct_state
        expected_cart = self._build_cart_from_setup(correct_state.expected_cart)

        # 1. Error fixed: Does the final cart match expected?
        # Build MCP cart representation for comparison
        mcp_cart = self._build_cart_from_mcp(mcp_state.cart)
        error_fixed, fix_explanation = self._compare_carts(
            mcp_cart, expected_cart
        )

        result.add_component(
            name="error_fixed",
            weight=0.7,
            raw_value=error_fixed,
            normalized_score=1.0 if error_fixed else 0.0,
            explanation=fix_explanation,
        )

        # 2. Error identified: Did the agent acknowledge the error?
        error_identified = self._check_error_identified(mcp_state, setup.error_description)

        result.add_component(
            name="error_identified",
            weight=0.1,
            raw_value=error_identified,
            normalized_score=1.0 if error_identified else 0.5,
            explanation="Agent acknowledged error" if error_identified else "Error acknowledgment unclear",
        )

        # 3. Efficiency: Action penalty for excessive actions
        expected_actions = task.expected_actions
        actual_actions = result.actions_taken  # From mcp_state.turn_count

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
        result.success = error_fixed and mcp_state.completed

        result.metrics = {
            "error_description": setup.error_description,
            "error_fixed": error_fixed,
            "error_identified": error_identified,
            "actions_taken": actual_actions,
            "expected_actions": expected_actions,
            "initial_cart_items": len(setup.cart_contents),
            "expected_cart_items": len(correct_state.expected_cart),
            "final_cart_items": len(mcp_state.cart),
        }

        return result

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _get_total_spent(self, mcp_state: MCPSessionState) -> float:
        """Get total amount spent from MCP cart."""
        return mcp_state.get_cart_total()

    def _get_purchased_items(self, mcp_state: MCPSessionState) -> list[dict[str, Any]]:
        """Get list of cart items from MCP state as dicts."""
        items = []

        # MCP cart already contains items as dicts
        for cart_item in mcp_state.cart:
            # Combine catalog attributes (for matching) with selected options
            attributes = {}

            # First add catalog attributes from WebShop product data
            catalog_attrs = cart_item.get("catalog_attributes", {})
            if catalog_attrs:
                attributes.update(catalog_attrs)

            # Then add selected options (color, size, etc.)
            options = cart_item.get("options", {})
            if options:
                attributes.update(options)

            items.append({
                "product_id": cart_item.get("product_id", ""),
                "product_name": cart_item.get("name", ""),
                "attributes": attributes,
                "price": cart_item.get("price", 0.0),
                "quantity": cart_item.get("quantity", 1),
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
            item_text = name + " " + " ".join(str(v).lower() for v in attrs.values())

            # Check if category matches (in name or attributes)
            category_match = req_category in item_text
            
            # Try LLM fallback for category if simple match fails
            if not category_match and self._llm_client:
                if self._llm_check_positive_match(item_text, req_category):
                    category_match = True

            # Check if required attributes match
            # ALL required attributes must be satisfied
            attr_match = True 
            if req_attrs:
                for key, value in req_attrs.items():
                    value_lower = str(value).lower()
                    found = False

                    # Check simple containment
                    if value_lower in item_text:
                        found = True
                    # Try LLM fallback for attribute if simple match fails
                    elif self._llm_client and self._llm_check_positive_match(item_text, value_lower):
                        found = True

                    if not found:
                        attr_match = False
                        break

            # Item matches if it satisfies BOTH category AND attributes
            if category_match and attr_match:
                return True

        return False

        return False

    def _calculate_quality_score(
        self,
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
        """Extract preferences that should be remembered from task sequence (stub)."""
        return {}

    def _format_user_history(self, task: PreferenceMemoryTask) -> str:
        """Format user history from task session sequence."""
        if task.user_history_text:
            return task.user_history_text

        history_lines = []
        for i, session in enumerate(task.session_sequence):
            history_lines.append(f"Session {i+1}:")
            history_lines.append(f"  Request: {session.instruction}")
            if session.establishes:
                preferences = ", ".join(f"{k}={v}" for k, v in session.establishes.items())
                history_lines.append(f"  Outcome: User established preference for [{preferences}]")
        return "\n".join(history_lines)

    def _check_preference_recall(
        self,
        mcp_state: MCPSessionState,
        task: PreferenceMemoryTask,
    ) -> tuple[float, str]:
        """Check if the agent correctly recalled a preference using LLM-as-a-judge.

        Constructs a prompt with history and all purchased items, asking LLM to verify overall match.
        """
        purchased_items = self._get_purchased_items(mcp_state)
        if not purchased_items:
            return 0.0, "No items purchased to verify preference"

        # Summarize all purchased items
        items_summary = []
        for i, item in enumerate(purchased_items):
            item_desc = (
                item.get("product_name", "")
                + " "
                + " ".join(str(v) for v in item.get("attributes", {}).values())
            )
            items_summary.append(f"Item {i+1}: {item_desc}")
        
        all_items_text = "\n".join(items_summary)
        user_history = self._format_user_history(task)
        attribute = task.memory_test.attribute_to_recall
        
        # If no LLM, fallback to simple string match on ANY item
        if self._llm_client is None:
            for item_text in items_summary:
                for val in task.memory_test.acceptable_values:
                    if val.lower() in item_text.lower():
                        return 1.0, f"Match found (fallback): {val}"
            return 0.0, "No match found in any item (LLM unavailable)"

        acceptable_str = ", ".join(task.memory_test.acceptable_values)

        prompt = f"""You are evaluating if a shopping agent correctly recalled a user's preference.

USER HISTORY:
{user_history}

MEMORY TEST:
The user has an established preference for: {attribute}
Acceptable Matches (Synonyms): {acceptable_str}

PURCHASED ITEMS:
{all_items_text}

QUESTION:
Based on the User History, did the agent respect the user's preference for '{attribute}' in its purchases?

History Interpretation Guide:
- The history shows outcomes like "Outcome: User established preference for [key=value]".
- If the key indicates a negative (e.g., "dislikes", "avoid", "returned", "hates"), then the value is FORBIDDEN.
- Otherwise, the value is REQUIRED (or a synonym).

Examples:
- [fit=slim fit] -> Item MUST be slim fit.
- [dislikes=polyester] -> Item MUST NOT be polyester.

Reply YES if the purchases respect the preference.
Reply NO if the purchases violate or ignore the preference.
Only reply YES or NO.
"""

        try:
            messages = [{"role": "user", "content": prompt}]
            # Enable thinking to encourage reasoning, allowing tokens for it
            response = self._llm_client.complete(messages, max_tokens=1024, thinking=True)
            
            clean_response = response.strip().lower()
            
            # Check if response starts with YES (simple case)
            if clean_response.startswith("yes"):
                return 1.0, "LLM confirmed preference recall"
                
            # Check if response ends with YES (CoT case)
            # Allow for boundary, colon, asterisk before YES, and punctuation after
            if re.search(r"(\b|:|\*)yes\W*$", clean_response):
                return 1.0, "LLM confirmed preference recall"

            return 0.0, "LLM rejected preference match"
        except Exception as e:
            logger.error("LLM preference check failed", error=str(e))
            return 0.0, f"Evaluation error: {str(e)}"

    def _check_preference_consistency(
        self,
        mcp_state: MCPSessionState,
        expected_preferences: dict[str, Any],
    ) -> tuple[float, str]:
        """Check if purchases are consistent with remembered preferences (stub)."""
        return 0.0, "Preference memory tasks not supported"

    def _llm_check_constraint_violation(
        self,
        item_text: str,
        forbidden_attribute: str,
    ) -> bool:
        """Use LLM to check if an item violates a negative constraint."""
        if self._llm_client is None:
            return True  # Fallback to strict string match assumption

        prompt = f"""Does the following product actually HAVE the attribute "{forbidden_attribute}"?

Product: {item_text}

Reply YES if the product has this attribute.
Reply NO if the product does NOT have this attribute (e.g., if it says "not {forbidden_attribute}" or is a different style).
Only reply YES or NO."""

        try:
            messages = [{"role": "user", "content": prompt}]
            logger.debug("LLM positive match prompt", prompt=prompt)
            response = self._llm_client.complete(messages, max_tokens=1024)
            logger.debug("LLM positive match response", response=response)
            return "yes" in response.lower()
        except Exception as e:
            logger.error("LLM positive match failed", error=str(e))
            return False
            return True

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
                # String match found, verify with LLM if available
                if self._llm_client is None or self._llm_check_constraint_violation(item_text, attr):
                    violations.append(f"forbidden attribute: {attr}")

        for term in forbidden_terms:
            if term.lower() in item_text:
                # String match found, verify with LLM if available
                if self._llm_client is None or self._llm_check_constraint_violation(item_text, term):
                    violations.append(f"forbidden term: {term}")

        return violations

    def _llm_check_positive_match(
        self,
        item_text: str,
        required_attribute: str,
    ) -> bool:
        """Use LLM to check if an item semantically matches a requirement."""
        if self._llm_client is None:
            return False

        prompt = f"""Does the following product satisfy the requirement "{required_attribute}"?

Product: {item_text}

Reply YES if it clearly satisfies the requirement or is a synonym/specific type of it.
Reply NO if it does not match or is ambiguous.
Only reply YES or NO."""

        try:
            messages = [{"role": "user", "content": prompt}]
            logger.debug("LLM positive match prompt", prompt=prompt)
            response = self._llm_client.complete(messages, max_tokens=1024)
            logger.debug("LLM positive match response", response=response)
            return "yes" in response.lower()
        except Exception as e:
            logger.error("LLM positive match failed", error=str(e))
            return False
            return False

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
            # Try simple string matching first
            if attr.lower() in all_item_text:
                matched += 1
            # Fallback to LLM semantic matching if available
            elif self._llm_client is not None and self._llm_check_positive_match(all_item_text, attr):
                matched += 1

        score = matched / len(required_attributes)
        return score, f"Matched {matched}/{len(required_attributes)} required attributes"

    def _count_products_explored(self, mcp_state: MCPSessionState) -> int:
        """Count unique products explored from MCP history."""
        product_asins = set()

        # Count click actions on product elements
        for record in mcp_state.history:
            action_type = record.get("action", "")

            if action_type == "click":
                # Use the actual product ASIN if available (added by click tool for product clicks)
                product_asin = record.get("product_asin")
                if product_asin:
                    product_asins.add(product_asin)
                else:
                    # Fallback: count positional element IDs (p1, p2, etc.)
                    # Note: This is less accurate since element IDs are reused across searches
                    element_id = record.get("element_id", "")
                    if element_id and element_id.startswith("p") and len(element_id) <= 4:
                        product_asins.add(element_id)

            elif action_type == "add_to_cart":
                product = record.get("product", {})
                product_id = product.get("product_id") or product.get("asin", "")
                if product_id:
                    product_asins.add(product_id)

        return len(product_asins)

    def _extract_justification(self, mcp_state: MCPSessionState) -> str:
        """Extract agent's justification/reasoning from session state.

        Uses reasoning_summary captured from purple agent's A2A completion.
        Falls back to MCP history summary if no reasoning available.
        """
        # Use captured reasoning from purple agent if available
        if mcp_state.reasoning_summary:
            return mcp_state.reasoning_summary

        # Fallback: summarize actions from MCP history
        if mcp_state.history:
            actions = []
            for record in mcp_state.history:
                action_type = record.get("action", "")
                if action_type == "search":
                    actions.append(f"searched for '{record.get('query', '')}'")
                elif action_type == "click":
                    actions.append(f"clicked {record.get('element_id', '')}")
                elif action_type == "add_to_cart":
                    product = record.get("product", {})
                    actions.append(f"added {product.get('name', 'item')} to cart")
            return "Actions taken: " + ", ".join(actions) if actions else ""
        return ""

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

    def _build_cart_from_mcp(self, mcp_cart: list[dict[str, Any]]) -> CartState:
        """Build a CartState from MCP cart items."""
        cart = CartState()
        for cart_dict in mcp_cart:
            # Combine catalog attributes with user-selected options
            attributes = {}
            # First add catalog attributes (product features)
            if "catalog_attributes" in cart_dict:
                attributes.update(cart_dict["catalog_attributes"])
            
            # Then add/override with selected options
            if "options" in cart_dict:
                attributes.update(cart_dict["options"])

            item = CartItem(
                product_id=cart_dict.get("product_id", ""),
                product_name=cart_dict.get("name", ""),
                attributes=attributes,
                quantity=cart_dict.get("quantity", 1),
                price=cart_dict.get("price", 0.0),
            )
            cart.add_item(item)
        return cart

    def _build_cart_from_setup(self, cart_setup: list) -> CartState:
        """Build a CartState from cart setup items."""
        from .models import CartItemSetup

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

    def _llm_compare_carts(
        self,
        actual: CartState,
        expected: CartState,
    ) -> tuple[bool, str]:
        """Use LLM to compare actual and expected carts semantically."""
        if self._llm_client is None:
            return False, "LLM client not available for semantic comparison"

        # Format carts for prompt
        def format_cart(cart):
            lines = []
            for item in cart.items:
                attrs = ", ".join(f"{k}: {v}" for k, v in item.attributes.items())
                lines.append(f"- Qty {item.quantity}: {item.product_name} ({attrs})")
            return "\n".join(lines) or "Empty Cart"

        prompt = f"""Compare these two shopping carts. The 'Expected Cart' defines the requirements, but the 'Actual Cart' might contain real products that satisfy those requirements even if names/IDs differ.

EXPECTED CART (Goal):
{format_cart(expected)}

ACTUAL CART (Result):
{format_cart(actual)}

Does the Actual Cart satisfy the requirements of the Expected Cart?
Ignore minor name differences or ID mismatches. Focus on whether the correct TYPES of items with correct ATTRIBUTES were purchased in the correct QUANTITIES.

Reply YES if it matches.
Reply NO if items are wrong, missing, or have incorrect attributes.
Only reply YES or NO."""

        try:
            messages = [{"role": "user", "content": prompt}]
            logger.debug("LLM cart comparison prompt", prompt=prompt)
            response = self._llm_client.complete(messages, max_tokens=1024)
            logger.debug("LLM cart comparison response", response=response)
            match = "yes" in response.lower()
            return match, "LLM confirmed match" if match else "LLM rejected match"
        except Exception as e:
            logger.error("LLM cart comparison failed", error=str(e))
            return False, f"LLM evaluation failed: {str(e)}"

    def _compare_carts(
        self,
        actual: CartState,
        expected: CartState,
    ) -> tuple[bool, str]:
        """Compare actual cart to expected cart state with LLM fallback."""
        # Try strict matching first (fast path)
        strict_match, strict_reason = self._strict_compare_carts(actual, expected)
        if strict_match:
            return True, strict_reason

        # Fallback to LLM semantic matching if strict match fails
        if self._llm_client:
            return self._llm_compare_carts(actual, expected)

        return False, strict_reason

    def _strict_compare_carts(
        self,
        actual: CartState,
        expected: CartState,
    ) -> tuple[bool, str]:
        """Compare actual cart to expected cart state using strict ID matching."""
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
        mcp_state: MCPSessionState,
        error_description: str,
    ) -> bool:
        """Check if the agent acknowledged the error using LLM-as-judge.

        Uses the agent's reasoning (if available) and actions to determine
        if the error was properly identified before attempting to fix it.
        """
        # If we have reasoning from the agent, use LLM to evaluate
        if mcp_state.reasoning_summary and self._llm_client is not None:
            return self._llm_check_error_identification(
                mcp_state.reasoning_summary,
                error_description,
                mcp_state.history,
            )

        # Fallback heuristic when no LLM or reasoning available
        # Check if agent took corrective actions (removed items or searched for alternatives)
        has_removal = any(
            record.get("action") == "remove_from_cart"
            for record in mcp_state.history
        )

        # Check if searches relate to the correct item (not the error)
        # Extract what the correct item should be from error description
        search_queries = [
            record.get("query", "").lower()
            for record in mcp_state.history
            if record.get("action") == "search"
        ]

        # If they removed something and searched, likely identified the error
        return has_removal or len(search_queries) > 0

    def _llm_check_error_identification(
        self,
        reasoning: str,
        error_description: str,
        history: list[dict],
    ) -> bool:
        """Use LLM to evaluate if agent identified the error.

        Args:
            reasoning: Agent's reasoning/justification text.
            error_description: Description of the error to identify.
            history: MCP action history.

        Returns:
            True if the agent demonstrated understanding of the error.
        """
        # Build action summary
        actions_summary = []
        for record in history:
            action = record.get("action", "")
            if action == "search":
                actions_summary.append(f"- Searched for: {record.get('query', '')}")
            elif action == "remove_from_cart":
                removed = record.get("removed_product", {}).get("name", "item")
                actions_summary.append(f"- Removed from cart: {removed}")
            elif action == "add_to_cart":
                added = record.get("product", {}).get("name", "item")
                actions_summary.append(f"- Added to cart: {added}")

        actions_text = "\n".join(actions_summary) if actions_summary else "No actions recorded"

        prompt = f"""Evaluate whether the shopping agent identified and understood the error.

ERROR DESCRIPTION: {error_description}

AGENT'S REASONING:
{reasoning}

AGENT'S ACTIONS:
{actions_text}

Did the agent demonstrate that they understood what was wrong?
Consider:
1. Did their reasoning mention or acknowledge the error?
2. Did their actions show they understood what needed to be fixed?
3. Did they search for or add the correct item (not the wrong one)?

Answer with just "yes" or "no"."""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self._llm_client.complete(messages, max_tokens=1024)
            return "yes" in response.lower()
        except Exception:
            # Fallback to True if LLM fails (benefit of doubt)
            return True
