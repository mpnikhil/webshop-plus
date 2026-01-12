"""Tests for MCP checkout tool.

These tests verify the checkout() tool:
- Returns terminal evaluation with success/failure
- Scores correctly based on cart state and budget
- Marks session as completed
- Returns all required metadata
"""

from dataclasses import dataclass
from typing import Any

import pytest

from src.webshop_mcp.session_state import SessionState
from src.webshop_mcp.server import (
    checkout,
    current_session_id,
    register_session,
    unregister_session,
    _session_states,
    _webshop_interfaces,
)


@dataclass
class MockStepResult:
    """Mock result from WebShop step() call."""

    observation: str
    reward: float = 0.0
    done: bool = False
    info: dict = None

    def __post_init__(self):
        if self.info is None:
            self.info = {}


class MockWebShop:
    """Mock WebShop environment for testing checkout."""

    def __init__(
        self,
        prices: dict | None = None,
    ):
        self._product_prices = prices or {}
        self.step_calls: list[str] = []
        self.reset_calls: list[str] = []

    def reset(self, session: str | None = None) -> str:
        """Mock reset."""
        self.reset_calls.append(session)
        return "Welcome to WebShop"

    def step(self, action: str) -> MockStepResult:
        """Mock step."""
        self.step_calls.append(action)
        return MockStepResult(observation="")

    def get_available_actions(self) -> dict:
        """Return empty available actions."""
        return {"has_search_bar": True, "clickables": []}

    @property
    def product_prices(self) -> dict[str, float]:
        """Return configured product prices."""
        return self._product_prices

    @property
    def product_item_dict(self) -> dict[str, dict]:
        """Return empty product dict."""
        return {}


@pytest.fixture(autouse=True)
def cleanup_global_state():
    """Clean up global session state before and after each test."""
    _session_states.clear()
    _webshop_interfaces.clear()
    yield
    _session_states.clear()
    _webshop_interfaces.clear()
    # Reset contextvar if set
    try:
        current_session_id.get()
        # Can't reset without token, so just clear state
    except LookupError:
        pass


def setup_session(
    session_id: str = "test_checkout",
    goal: str = "Find running shoes under $50",
    budget: float = 50.0,
    webshop: MockWebShop | None = None,
    max_turns: int = 30,
    constraints: list[str] | None = None,
):
    """Set up a session for testing.

    Args:
        session_id: Session ID to use.
        goal: Shopping goal.
        budget: Budget limit.
        webshop: Mock WebShop instance.
        max_turns: Maximum turns allowed.
        constraints: Optional constraints.

    Returns:
        Tuple of (state, token) where token can be used to reset contextvar.
    """
    state = SessionState(
        session_id=session_id,
        goal=goal,
        budget=budget,
        max_turns=max_turns,
        constraints=constraints or [],
    )
    register_session(session_id, state, webshop)
    token = current_session_id.set(session_id)
    return state, token


class TestCheckoutSuccess:
    """Test checkout with valid cart within budget."""

    def test_checkout_success_single_item(self):
        """Checkout succeeds with one item under budget."""
        webshop = MockWebShop(prices={"B001": 29.99, "B002": 39.99, "B003": 59.99})
        state, token = setup_session(webshop=webshop)
        try:
            # Add item to cart
            state.cart.append({
                "name": "Running Shoes",
                "price": 29.99,
                "product_id": "B001",
            })

            # Call checkout
            result = checkout()

            assert result["terminated"] is True
            assert result["reason"] == "checkout"
            assert result["success"] is True
            assert result["score"] == 1.0
            assert result["total"] == 29.99
            assert result["budget"] == 50.0
            assert result["budget_remaining"] == 20.01
            assert "failure_reason" not in result
        finally:
            current_session_id.reset(token)

    def test_checkout_success_multiple_items(self):
        """Checkout succeeds with multiple items under budget."""
        webshop = MockWebShop(prices={"B001": 25.00, "B002": 10.00})
        state, token = setup_session(webshop=webshop)
        try:
            state.cart.append({"name": "Shoes", "price": 25.00, "product_id": "B001"})
            state.cart.append({"name": "Socks", "price": 10.00, "product_id": "B002"})

            result = checkout()

            assert result["success"] is True
            assert result["score"] == 1.0
            assert result["total"] == 35.00
            assert result["cart_size"] == 2
            assert result["budget_remaining"] == 15.00
        finally:
            current_session_id.reset(token)

    def test_checkout_success_exact_budget(self):
        """Checkout succeeds when total equals budget exactly."""
        webshop = MockWebShop(prices={"B001": 50.00})
        state, token = setup_session(webshop=webshop)
        try:
            state.cart.append({"name": "Shoes", "price": 50.00, "product_id": "B001"})

            result = checkout()

            assert result["success"] is True
            assert result["score"] == 1.0
            assert result["total"] == 50.00
            assert result["budget_remaining"] == 0.0
        finally:
            current_session_id.reset(token)


class TestCheckoutEmptyCart:
    """Test checkout with empty cart."""

    def test_checkout_empty_cart_fails(self):
        """Checkout fails with empty cart."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop)
        try:
            # Cart is empty by default

            result = checkout()

            assert result["terminated"] is True
            assert result["reason"] == "checkout"
            assert result["success"] is False
            assert result["failure_reason"] == "empty_cart"
            assert result["score"] == 0.0
            assert result["total"] == 0.0
            assert result["cart_size"] == 0
        finally:
            current_session_id.reset(token)

    def test_checkout_empty_cart_no_budget_remaining(self):
        """Empty cart doesn't show budget_remaining (failure case)."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop)
        try:
            result = checkout()

            assert "budget_remaining" not in result
            assert "over_budget_by" not in result
        finally:
            current_session_id.reset(token)


class TestCheckoutOverBudget:
    """Test checkout when cart exceeds budget."""

    def test_checkout_over_budget_single_item(self):
        """Checkout fails when single item exceeds budget."""
        webshop = MockWebShop(prices={"B003": 75.00})
        state, token = setup_session(webshop=webshop)
        try:
            state.cart.append({"name": "Premium Shoes", "price": 75.00, "product_id": "B003"})

            result = checkout()

            assert result["terminated"] is True
            assert result["reason"] == "checkout"
            assert result["success"] is False
            assert result["failure_reason"] == "budget_exceeded"
            assert result["score"] == 0.3
            assert result["total"] == 75.00
            assert result["over_budget_by"] == 25.00
        finally:
            current_session_id.reset(token)

    def test_checkout_over_budget_multiple_items(self):
        """Checkout fails when combined items exceed budget."""
        webshop = MockWebShop(prices={"B001": 35.00, "B002": 40.00})
        state, token = setup_session(webshop=webshop)
        try:
            state.cart.append({"name": "Shoes", "price": 35.00, "product_id": "B001"})
            state.cart.append({"name": "Sneakers", "price": 40.00, "product_id": "B002"})

            result = checkout()

            assert result["success"] is False
            assert result["failure_reason"] == "budget_exceeded"
            assert result["score"] == 0.3
            assert result["total"] == 75.00
            assert result["over_budget_by"] == 25.00
            assert result["cart_size"] == 2
        finally:
            current_session_id.reset(token)

    def test_checkout_slightly_over_budget(self):
        """Checkout fails even when only slightly over budget."""
        webshop = MockWebShop(prices={"B001": 50.01})
        state, token = setup_session(webshop=webshop)
        try:
            state.cart.append({"name": "Shoes", "price": 50.01, "product_id": "B001"})

            result = checkout()

            assert result["success"] is False
            assert result["failure_reason"] == "budget_exceeded"
            assert result["score"] == 0.3
            assert result["over_budget_by"] == pytest.approx(0.01, abs=0.001)
        finally:
            current_session_id.reset(token)


class TestCheckoutMarksCompleted:
    """Test that checkout marks session as completed."""

    def test_checkout_marks_session_completed(self):
        """Session is marked completed after checkout."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop)
        try:
            assert state.completed is False

            checkout()

            assert state.completed is True
        finally:
            current_session_id.reset(token)

    def test_checkout_records_in_history(self):
        """Checkout is recorded in session history."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop)
        try:
            initial_history_length = len(state.history)

            checkout()

            assert len(state.history) == initial_history_length + 1
            last_entry = state.history[-1]
            assert last_entry["action"] == "session_end"
            assert last_entry["reason"] == "checkout"
        finally:
            current_session_id.reset(token)


class TestCheckoutReturnsMetadata:
    """Test that checkout returns all required metadata."""

    def test_checkout_returns_session_id(self):
        """Checkout returns session_id."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop)
        try:
            state.cart.append({"name": "Shoes", "price": 30.00, "product_id": "B001"})

            result = checkout()

            assert result["session_id"] == "test_checkout"
        finally:
            current_session_id.reset(token)

    def test_checkout_returns_turns_used(self):
        """Checkout returns turns_used."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop)
        try:
            state.turn_count = 5
            state.cart.append({"name": "Shoes", "price": 30.00, "product_id": "B001"})

            result = checkout()

            assert result["turns_used"] == 5
            assert result["max_turns"] == 30
        finally:
            current_session_id.reset(token)

    def test_checkout_returns_cart_contents(self):
        """Checkout returns full cart contents."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop)
        try:
            state.cart.append({"name": "Shoes", "price": 30.00, "product_id": "B001"})
            state.cart.append({"name": "Socks", "price": 5.00, "product_id": "B002"})

            result = checkout()

            assert result["cart"] == state.cart
            assert len(result["cart"]) == 2
        finally:
            current_session_id.reset(token)

    def test_checkout_returns_history_length(self):
        """Checkout returns history length for evaluation context."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop)
        try:
            state.history.append({"action": "search", "query": "shoes"})
            state.history.append({"action": "click", "element_id": "p1"})
            state.cart.append({"name": "Shoes", "price": 30.00, "product_id": "B001"})

            result = checkout()

            # History includes the checkout itself
            assert result["history_length"] == 3
        finally:
            current_session_id.reset(token)


class TestCheckoutDoesNotIncrementTurn:
    """Verify checkout behavior with turn count."""

    def test_checkout_does_not_increment_turn(self):
        """Checkout does not increment turn count (it's terminal)."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop)
        try:
            state.turn_count = 5
            state.cart.append({"name": "Shoes", "price": 30.00, "product_id": "B001"})

            checkout()

            # Checkout is terminal, doesn't use a turn
            assert state.turn_count == 5
        finally:
            current_session_id.reset(token)


class TestCheckoutWithDifferentBudgets:
    """Test checkout with various budget scenarios."""

    def test_checkout_with_zero_budget(self):
        """Checkout with zero budget."""
        webshop = MockWebShop()
        state, token = setup_session(
            session_id="zero_budget",
            goal="Free items only",
            budget=0.0,
            webshop=webshop,
        )
        try:
            # Add free item
            state.cart.append({"name": "Free Sample", "price": 0.0, "product_id": "FREE"})

            result = checkout()

            assert result["success"] is True
            assert result["score"] == 1.0
            assert result["total"] == 0.0
        finally:
            current_session_id.reset(token)

    def test_checkout_with_large_budget(self):
        """Checkout with large budget."""
        webshop = MockWebShop()
        state, token = setup_session(
            session_id="large_budget",
            goal="Buy anything",
            budget=10000.0,
            webshop=webshop,
        )
        try:
            state.cart.append({"name": "Expensive Item", "price": 500.00, "product_id": "B001"})

            result = checkout()

            assert result["success"] is True
            assert result["budget_remaining"] == 9500.0
        finally:
            current_session_id.reset(token)


class TestCheckoutScoring:
    """Verify scoring logic for different checkout scenarios."""

    def test_score_success_is_1_0(self):
        """Success score is exactly 1.0."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop)
        try:
            state.cart.append({"name": "Shoes", "price": 30.00, "product_id": "B001"})

            result = checkout()

            assert result["score"] == 1.0
        finally:
            current_session_id.reset(token)

    def test_score_empty_cart_is_0_0(self):
        """Empty cart score is exactly 0.0."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop)
        try:
            result = checkout()

            assert result["score"] == 0.0
        finally:
            current_session_id.reset(token)

    def test_score_over_budget_is_0_3(self):
        """Over budget score is exactly 0.3."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop)
        try:
            state.cart.append({"name": "Expensive", "price": 100.00, "product_id": "B001"})

            result = checkout()

            assert result["score"] == 0.3
        finally:
            current_session_id.reset(token)


class TestCheckoutTerminalStatus:
    """Verify checkout is always terminal."""

    def test_terminated_is_always_true(self):
        """Terminated is True regardless of outcome."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop)
        try:
            # Test empty cart
            result1 = checkout()
            assert result1["terminated"] is True
        finally:
            current_session_id.reset(token)

    def test_reason_is_always_checkout(self):
        """Reason is always 'checkout'."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop)
        try:
            state.cart.append({"name": "Shoes", "price": 30.00, "product_id": "B001"})

            result = checkout()

            assert result["reason"] == "checkout"
        finally:
            current_session_id.reset(token)
