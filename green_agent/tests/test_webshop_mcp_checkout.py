"""Tests for MCP checkout tool.

These tests verify the checkout() tool in WebShopMCPServer:
- Returns terminal evaluation with success/failure
- Scores correctly based on cart state and budget
- Marks session as completed
- Returns all required metadata
"""

from dataclasses import dataclass
from typing import Any

import pytest

from src.webshop_mcp.server import WebShopMCPServer
from src.webshop_mcp.session_state import SessionState


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


@pytest.fixture
def state():
    """Create a default session state."""
    return SessionState(
        session_id="test_checkout",
        goal="Find running shoes under $50",
        budget=50.0,
        constraints=["no synthetic"],
        max_turns=30,
    )


@pytest.fixture
def mock_webshop():
    """Create a mock WebShop."""
    return MockWebShop(prices={"B001": 29.99, "B002": 39.99, "B003": 59.99})


@pytest.fixture
def server(state, mock_webshop):
    """Create a WebShopMCPServer with mocked WebShop."""
    return WebShopMCPServer(state, webshop=mock_webshop)


class TestCheckoutSuccess:
    """Test checkout with valid cart within budget."""

    def test_checkout_success_single_item(self, server, state):
        """Checkout succeeds with one item under budget."""
        # Add item to cart
        state.cart.append({
            "name": "Running Shoes",
            "price": 29.99,
            "product_id": "B001",
        })

        # Call checkout
        result = server.mcp._tool_manager._tools["checkout"].fn()

        assert result["terminated"] is True
        assert result["reason"] == "checkout"
        assert result["success"] is True
        assert result["score"] == 1.0
        assert result["total"] == 29.99
        assert result["budget"] == 50.0
        assert result["budget_remaining"] == 20.01
        assert "failure_reason" not in result

    def test_checkout_success_multiple_items(self, server, state):
        """Checkout succeeds with multiple items under budget."""
        state.cart.append({"name": "Shoes", "price": 25.00, "product_id": "B001"})
        state.cart.append({"name": "Socks", "price": 10.00, "product_id": "B002"})

        result = server.mcp._tool_manager._tools["checkout"].fn()

        assert result["success"] is True
        assert result["score"] == 1.0
        assert result["total"] == 35.00
        assert result["cart_size"] == 2
        assert result["budget_remaining"] == 15.00

    def test_checkout_success_exact_budget(self, server, state):
        """Checkout succeeds when total equals budget exactly."""
        state.cart.append({"name": "Shoes", "price": 50.00, "product_id": "B001"})

        result = server.mcp._tool_manager._tools["checkout"].fn()

        assert result["success"] is True
        assert result["score"] == 1.0
        assert result["total"] == 50.00
        assert result["budget_remaining"] == 0.0


class TestCheckoutEmptyCart:
    """Test checkout with empty cart."""

    def test_checkout_empty_cart_fails(self, server, state):
        """Checkout fails with empty cart."""
        # Cart is empty by default

        result = server.mcp._tool_manager._tools["checkout"].fn()

        assert result["terminated"] is True
        assert result["reason"] == "checkout"
        assert result["success"] is False
        assert result["failure_reason"] == "empty_cart"
        assert result["score"] == 0.0
        assert result["total"] == 0.0
        assert result["cart_size"] == 0

    def test_checkout_empty_cart_no_budget_remaining(self, server, state):
        """Empty cart doesn't show budget_remaining (failure case)."""
        result = server.mcp._tool_manager._tools["checkout"].fn()

        assert "budget_remaining" not in result
        assert "over_budget_by" not in result


class TestCheckoutOverBudget:
    """Test checkout when cart exceeds budget."""

    def test_checkout_over_budget_single_item(self, server, state):
        """Checkout fails when single item exceeds budget."""
        state.cart.append({"name": "Premium Shoes", "price": 75.00, "product_id": "B003"})

        result = server.mcp._tool_manager._tools["checkout"].fn()

        assert result["terminated"] is True
        assert result["reason"] == "checkout"
        assert result["success"] is False
        assert result["failure_reason"] == "budget_exceeded"
        assert result["score"] == 0.3
        assert result["total"] == 75.00
        assert result["over_budget_by"] == 25.00

    def test_checkout_over_budget_multiple_items(self, server, state):
        """Checkout fails when combined items exceed budget."""
        state.cart.append({"name": "Shoes", "price": 35.00, "product_id": "B001"})
        state.cart.append({"name": "Sneakers", "price": 40.00, "product_id": "B002"})

        result = server.mcp._tool_manager._tools["checkout"].fn()

        assert result["success"] is False
        assert result["failure_reason"] == "budget_exceeded"
        assert result["score"] == 0.3
        assert result["total"] == 75.00
        assert result["over_budget_by"] == 25.00
        assert result["cart_size"] == 2

    def test_checkout_slightly_over_budget(self, server, state):
        """Checkout fails even when only slightly over budget."""
        state.cart.append({"name": "Shoes", "price": 50.01, "product_id": "B001"})

        result = server.mcp._tool_manager._tools["checkout"].fn()

        assert result["success"] is False
        assert result["failure_reason"] == "budget_exceeded"
        assert result["score"] == 0.3
        assert result["over_budget_by"] == pytest.approx(0.01, abs=0.001)


class TestCheckoutMarksCompleted:
    """Test that checkout marks session as completed."""

    def test_checkout_marks_session_completed(self, server, state):
        """Session is marked completed after checkout."""
        assert state.completed is False

        server.mcp._tool_manager._tools["checkout"].fn()

        assert state.completed is True

    def test_checkout_records_in_history(self, server, state):
        """Checkout is recorded in session history."""
        initial_history_length = len(state.history)

        server.mcp._tool_manager._tools["checkout"].fn()

        assert len(state.history) == initial_history_length + 1
        last_entry = state.history[-1]
        assert last_entry["action"] == "session_end"
        assert last_entry["reason"] == "checkout"


class TestCheckoutReturnsMetadata:
    """Test that checkout returns all required metadata."""

    def test_checkout_returns_session_id(self, server, state):
        """Checkout returns session_id."""
        state.cart.append({"name": "Shoes", "price": 30.00, "product_id": "B001"})

        result = server.mcp._tool_manager._tools["checkout"].fn()

        assert result["session_id"] == "test_checkout"

    def test_checkout_returns_turns_used(self, server, state):
        """Checkout returns turns_used."""
        state.turn_count = 5
        state.cart.append({"name": "Shoes", "price": 30.00, "product_id": "B001"})

        result = server.mcp._tool_manager._tools["checkout"].fn()

        assert result["turns_used"] == 5
        assert result["max_turns"] == 30

    def test_checkout_returns_cart_contents(self, server, state):
        """Checkout returns full cart contents."""
        state.cart.append({"name": "Shoes", "price": 30.00, "product_id": "B001"})
        state.cart.append({"name": "Socks", "price": 5.00, "product_id": "B002"})

        result = server.mcp._tool_manager._tools["checkout"].fn()

        assert result["cart"] == state.cart
        assert len(result["cart"]) == 2

    def test_checkout_returns_history_length(self, server, state):
        """Checkout returns history length for evaluation context."""
        state.history.append({"action": "search", "query": "shoes"})
        state.history.append({"action": "click", "element_id": "p1"})
        state.cart.append({"name": "Shoes", "price": 30.00, "product_id": "B001"})

        result = server.mcp._tool_manager._tools["checkout"].fn()

        # History includes the checkout itself
        assert result["history_length"] == 3


class TestCheckoutDoesNotIncrementTurn:
    """Verify checkout behavior with turn count."""

    def test_checkout_does_not_increment_turn(self, server, state):
        """Checkout does not increment turn count (it's terminal)."""
        state.turn_count = 5
        state.cart.append({"name": "Shoes", "price": 30.00, "product_id": "B001"})

        server.mcp._tool_manager._tools["checkout"].fn()

        # Checkout is terminal, doesn't use a turn
        assert state.turn_count == 5


class TestCheckoutWithDifferentBudgets:
    """Test checkout with various budget scenarios."""

    def test_checkout_with_zero_budget(self, mock_webshop):
        """Checkout with zero budget."""
        state = SessionState(
            session_id="zero_budget",
            goal="Free items only",
            budget=0.0,
        )
        server = WebShopMCPServer(state, webshop=mock_webshop)

        # Add free item
        state.cart.append({"name": "Free Sample", "price": 0.0, "product_id": "FREE"})

        result = server.mcp._tool_manager._tools["checkout"].fn()

        assert result["success"] is True
        assert result["score"] == 1.0
        assert result["total"] == 0.0

    def test_checkout_with_large_budget(self, mock_webshop):
        """Checkout with large budget."""
        state = SessionState(
            session_id="large_budget",
            goal="Buy anything",
            budget=10000.0,
        )
        server = WebShopMCPServer(state, webshop=mock_webshop)

        state.cart.append({"name": "Expensive Item", "price": 500.00, "product_id": "B001"})

        result = server.mcp._tool_manager._tools["checkout"].fn()

        assert result["success"] is True
        assert result["budget_remaining"] == 9500.0


class TestCheckoutScoring:
    """Verify scoring logic for different checkout scenarios."""

    def test_score_success_is_1_0(self, server, state):
        """Success score is exactly 1.0."""
        state.cart.append({"name": "Shoes", "price": 30.00, "product_id": "B001"})

        result = server.mcp._tool_manager._tools["checkout"].fn()

        assert result["score"] == 1.0

    def test_score_empty_cart_is_0_0(self, server, state):
        """Empty cart score is exactly 0.0."""
        result = server.mcp._tool_manager._tools["checkout"].fn()

        assert result["score"] == 0.0

    def test_score_over_budget_is_0_3(self, server, state):
        """Over budget score is exactly 0.3."""
        state.cart.append({"name": "Expensive", "price": 100.00, "product_id": "B001"})

        result = server.mcp._tool_manager._tools["checkout"].fn()

        assert result["score"] == 0.3


class TestCheckoutTerminalStatus:
    """Verify checkout is always terminal."""

    def test_terminated_is_always_true(self, server, state):
        """Terminated is True regardless of outcome."""
        # Test empty cart
        result1 = server.mcp._tool_manager._tools["checkout"].fn()
        assert result1["terminated"] is True

    def test_reason_is_always_checkout(self, server, state):
        """Reason is always 'checkout'."""
        state.cart.append({"name": "Shoes", "price": 30.00, "product_id": "B001"})

        result = server.mcp._tool_manager._tools["checkout"].fn()

        assert result["reason"] == "checkout"
