"""Session state management for MCP server sessions.

This module provides the SessionState class that tracks the state of a single
assessment session, including cart contents, current page, visible elements,
turn count, and action history.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionState:
    """Tracks state for a single assessment session.

    Attributes:
        session_id: Unique identifier for this session.
        goal: The shopping task goal (e.g., "Find running shoes under $50").
        budget: Maximum allowed spending amount.
        constraints: List of constraints (e.g., ["no synthetic materials"]).
        max_turns: Maximum number of actions before session terminates.
        cart: List of products added to cart.
        current_page: Current page state ("home", "search_results", "product_detail").
        visible_elements: Dict mapping element IDs to element data.
        selected_options: Dict mapping option types to selected values.
        turn_count: Number of actions taken so far.
        completed: Whether the session has ended.
        history: List of action records for evaluation.
    """

    # Task definition (immutable after creation)
    session_id: str
    goal: str
    budget: float
    constraints: list[str] = field(default_factory=list)
    max_turns: int = 30

    # Runtime state (mutable)
    cart: list[dict[str, Any]] = field(default_factory=list)
    current_page: str = "home"
    visible_elements: dict[str, dict[str, Any]] = field(default_factory=dict)
    selected_options: dict[str, str] = field(default_factory=dict)
    turn_count: int = 0
    completed: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)
    # Agent reasoning captured from purple agent's A2A completion
    reasoning_summary: str = ""

    def get_cart_total(self) -> float:
        """Calculate total price of items in cart.

        Returns:
            Sum of all item prices in the cart.
        """
        return sum(item.get("price", 0.0) for item in self.cart)

    def is_over_budget(self) -> bool:
        """Check if cart total exceeds budget.

        Returns:
            True if cart total is greater than budget.
        """
        return self.get_cart_total() > self.budget

    def add_to_cart(self, product: dict[str, Any]) -> dict[str, Any]:
        """Add product to cart with current selected options.

        Args:
            product: Product dict with at least "name" and "price" keys.

        Returns:
            Status dict with cart state after addition.
        """
        # Create cart item with selected options
        cart_item = {
            "name": product.get("name", "Unknown"),
            "price": product.get("price", 0.0),
            "product_id": product.get("product_id", product.get("asin", "")),
            "options": dict(self.selected_options),
        }

        self.cart.append(cart_item)

        # Record in history
        self.history.append({
            "action": "add_to_cart",
            "product": cart_item,
            "turn": self.turn_count,
        })

        # Clear selected options for next product
        self.selected_options.clear()

        total = self.get_cart_total()
        return {
            "added": cart_item["name"],
            "cart_total": total,
            "budget": self.budget,
            "over_budget": total > self.budget,
            "cart_size": len(self.cart),
        }

    def remove_from_cart(self, item_index: int) -> dict[str, Any]:
        """Remove item from cart by index.

        Args:
            item_index: Index of item to remove (0-based).

        Returns:
            Status dict with cart state after removal, or error if invalid index.
        """
        if item_index < 0 or item_index >= len(self.cart):
            return {
                "error": f"Invalid index {item_index}. Cart has {len(self.cart)} items.",
                "cart_size": len(self.cart),
            }

        removed_item = self.cart.pop(item_index)

        # Record in history
        self.history.append({
            "action": "remove_from_cart",
            "item_index": item_index,
            "removed_product": removed_item,
            "turn": self.turn_count,
        })

        total = self.get_cart_total()
        return {
            "removed": removed_item["name"],
            "cart_total": total,
            "budget": self.budget,
            "over_budget": total > self.budget,
            "cart_size": len(self.cart),
        }

    def select_option(self, option_type: str, value: str) -> dict[str, Any]:
        """Select a product option (e.g., size, color).

        Args:
            option_type: Type of option (e.g., "size", "color").
            value: Selected value (e.g., "10", "blue").

        Returns:
            Status dict with currently selected options.
        """
        self.selected_options[option_type] = value
        self.history.append({
            "action": "select_option",
            "option_type": option_type,
            "value": value,
            "turn": self.turn_count,
        })
        return {
            "selected": {option_type: value},
            "all_selections": dict(self.selected_options),
        }

    def increment_turn(self) -> bool:
        """Increment turn count and check if max exceeded.

        Returns:
            True if turn count now exceeds max_turns, False otherwise.
        """
        self.turn_count += 1
        return self.turn_count > self.max_turns

    def mark_completed(self, reason: str) -> None:
        """Mark session as completed.

        Args:
            reason: Why the session ended (e.g., "checkout", "max_turns").
        """
        self.completed = True
        self.history.append({
            "action": "session_end",
            "reason": reason,
            "turn": self.turn_count,
        })

    def get_summary(self) -> dict[str, Any]:
        """Get summary of session state for evaluation.

        Returns:
            Dict containing key session metrics.
        """
        return {
            "session_id": self.session_id,
            "goal": self.goal,
            "budget": self.budget,
            "constraints": self.constraints,
            "cart": self.cart,
            "cart_total": self.get_cart_total(),
            "over_budget": self.is_over_budget(),
            "turns_used": self.turn_count,
            "max_turns": self.max_turns,
            "completed": self.completed,
            "history_length": len(self.history),
        }
