"""Tests for SessionState class.

Tests cover:
- Initial state creation
- Cart operations (add, total, budget check)
- Option selection
- Turn counting and max turns
- Session completion
- Summary generation
"""

import pytest
from src.mcp.session_state import SessionState


class TestSessionStateInitialization:
    """Tests for SessionState initialization."""

    def test_initial_state(self):
        """Test default initial state values."""
        state = SessionState(
            session_id="test-123",
            goal="Find running shoes under $50",
            budget=50.0,
        )

        assert state.session_id == "test-123"
        assert state.goal == "Find running shoes under $50"
        assert state.budget == 50.0
        assert state.constraints == []
        assert state.max_turns == 30
        assert state.cart == []
        assert state.current_page == "home"
        assert state.visible_elements == {}
        assert state.selected_options == {}
        assert state.turn_count == 0
        assert state.completed is False
        assert state.history == []

    def test_initial_state_with_constraints(self):
        """Test state with constraints specified."""
        state = SessionState(
            session_id="test-456",
            goal="Find cotton shirt",
            budget=30.0,
            constraints=["no synthetic", "blue color"],
            max_turns=20,
        )

        assert state.constraints == ["no synthetic", "blue color"]
        assert state.max_turns == 20


class TestCartOperations:
    """Tests for cart-related operations."""

    @pytest.fixture
    def state(self):
        """Create a fresh session state for testing."""
        return SessionState(
            session_id="cart-test",
            goal="Buy shoes",
            budget=100.0,
        )

    def test_add_to_cart(self, state):
        """Test adding a product to cart."""
        product = {"name": "Running Shoes", "price": 45.99, "asin": "ABC123"}
        result = state.add_to_cart(product)

        assert result["added"] == "Running Shoes"
        assert result["cart_total"] == 45.99
        assert result["budget"] == 100.0
        assert result["over_budget"] is False
        assert result["cart_size"] == 1
        assert len(state.cart) == 1
        assert state.cart[0]["name"] == "Running Shoes"
        assert state.cart[0]["price"] == 45.99

    def test_add_to_cart_with_options(self, state):
        """Test adding product with selected options."""
        state.select_option("size", "10")
        state.select_option("color", "blue")

        product = {"name": "Sneakers", "price": 55.00}
        result = state.add_to_cart(product)

        assert state.cart[0]["options"] == {"size": "10", "color": "blue"}
        # Options should be cleared after add
        assert state.selected_options == {}

    def test_add_to_cart_over_budget(self, state):
        """Test detecting over-budget cart."""
        state.add_to_cart({"name": "Expensive Shoes", "price": 80.0})
        result = state.add_to_cart({"name": "More Shoes", "price": 30.0})

        assert result["cart_total"] == 110.0
        assert result["over_budget"] is True

    def test_get_cart_total(self, state):
        """Test cart total calculation."""
        state.add_to_cart({"name": "Item 1", "price": 25.0})
        state.add_to_cart({"name": "Item 2", "price": 35.0})
        state.add_to_cart({"name": "Item 3", "price": 15.0})

        assert state.get_cart_total() == 75.0

    def test_get_cart_total_empty(self, state):
        """Test cart total with empty cart."""
        assert state.get_cart_total() == 0.0

    def test_is_over_budget_false(self, state):
        """Test is_over_budget when under budget."""
        state.add_to_cart({"name": "Cheap Shoes", "price": 50.0})
        assert state.is_over_budget() is False

    def test_is_over_budget_true(self, state):
        """Test is_over_budget when over budget."""
        state.add_to_cart({"name": "Expensive Shoes", "price": 150.0})
        assert state.is_over_budget() is True

    def test_is_over_budget_at_exactly_budget(self, state):
        """Test is_over_budget at exact budget amount."""
        state.add_to_cart({"name": "Exact Shoes", "price": 100.0})
        assert state.is_over_budget() is False


class TestOptionSelection:
    """Tests for product option selection."""

    @pytest.fixture
    def state(self):
        return SessionState(session_id="opt-test", goal="Buy shirt", budget=50.0)

    def test_select_option(self, state):
        """Test selecting a single option."""
        result = state.select_option("size", "M")

        assert result["selected"] == {"size": "M"}
        assert result["all_selections"] == {"size": "M"}
        assert state.selected_options == {"size": "M"}

    def test_select_multiple_options(self, state):
        """Test selecting multiple options."""
        state.select_option("size", "L")
        result = state.select_option("color", "red")

        assert result["all_selections"] == {"size": "L", "color": "red"}

    def test_override_option(self, state):
        """Test overriding a previously selected option."""
        state.select_option("size", "M")
        result = state.select_option("size", "L")

        assert result["all_selections"] == {"size": "L"}


class TestTurnCounting:
    """Tests for turn counting and limits."""

    @pytest.fixture
    def state(self):
        return SessionState(
            session_id="turn-test",
            goal="Shop",
            budget=100.0,
            max_turns=5,
        )

    def test_increment_turn(self, state):
        """Test incrementing turn count."""
        exceeded = state.increment_turn()

        assert exceeded is False
        assert state.turn_count == 1

    def test_increment_turn_multiple(self, state):
        """Test multiple turn increments."""
        for _ in range(4):
            state.increment_turn()

        assert state.turn_count == 4

    def test_max_turns_not_exceeded(self, state):
        """Test max turns boundary - at limit."""
        for _ in range(5):
            exceeded = state.increment_turn()

        # At max_turns (5), not exceeded yet
        assert exceeded is False
        assert state.turn_count == 5

    def test_max_turns_exceeded(self, state):
        """Test max turns exceeded detection."""
        for _ in range(5):
            state.increment_turn()

        # 6th turn exceeds max_turns of 5
        exceeded = state.increment_turn()

        assert exceeded is True
        assert state.turn_count == 6


class TestSessionCompletion:
    """Tests for session completion."""

    @pytest.fixture
    def state(self):
        return SessionState(session_id="end-test", goal="Shop", budget=100.0)

    def test_mark_completed_checkout(self, state):
        """Test marking session complete via checkout."""
        state.increment_turn()
        state.mark_completed("checkout")

        assert state.completed is True
        assert state.history[-1]["action"] == "session_end"
        assert state.history[-1]["reason"] == "checkout"

    def test_mark_completed_max_turns(self, state):
        """Test marking session complete via max turns."""
        state.mark_completed("max_turns")

        assert state.completed is True
        assert state.history[-1]["reason"] == "max_turns"


class TestSessionSummary:
    """Tests for session summary generation."""

    def test_get_summary_initial(self):
        """Test summary of fresh session."""
        state = SessionState(
            session_id="sum-test",
            goal="Find shoes",
            budget=75.0,
            constraints=["leather only"],
        )

        summary = state.get_summary()

        assert summary["session_id"] == "sum-test"
        assert summary["goal"] == "Find shoes"
        assert summary["budget"] == 75.0
        assert summary["constraints"] == ["leather only"]
        assert summary["cart"] == []
        assert summary["cart_total"] == 0.0
        assert summary["over_budget"] is False
        assert summary["turns_used"] == 0
        assert summary["max_turns"] == 30
        assert summary["completed"] is False
        assert summary["history_length"] == 0

    def test_get_summary_after_shopping(self):
        """Test summary after shopping activity."""
        state = SessionState(
            session_id="shop-test",
            goal="Buy sneakers",
            budget=80.0,
        )

        state.increment_turn()
        state.select_option("size", "9")
        state.add_to_cart({"name": "Nike Air", "price": 65.0})
        state.increment_turn()
        state.mark_completed("checkout")

        summary = state.get_summary()

        assert summary["cart_total"] == 65.0
        assert summary["over_budget"] is False
        assert summary["turns_used"] == 2
        assert summary["completed"] is True
        assert len(summary["cart"]) == 1
        assert summary["history_length"] == 3  # select + add + session_end


class TestHistoryTracking:
    """Tests for action history tracking."""

    @pytest.fixture
    def state(self):
        return SessionState(session_id="hist-test", goal="Shop", budget=100.0)

    def test_history_tracks_add_to_cart(self, state):
        """Test that add_to_cart is recorded in history."""
        state.increment_turn()
        state.add_to_cart({"name": "Shoes", "price": 50.0})

        assert len(state.history) == 1
        assert state.history[0]["action"] == "add_to_cart"
        assert state.history[0]["turn"] == 1

    def test_history_tracks_option_selection(self, state):
        """Test that option selection is recorded in history."""
        state.increment_turn()
        state.select_option("color", "green")

        assert len(state.history) == 1
        assert state.history[0]["action"] == "select_option"
        assert state.history[0]["option_type"] == "color"
        assert state.history[0]["value"] == "green"

    def test_history_order(self, state):
        """Test that history maintains correct order."""
        state.increment_turn()
        state.select_option("size", "8")
        state.add_to_cart({"name": "Boot", "price": 80.0})
        state.mark_completed("checkout")

        assert len(state.history) == 3
        assert state.history[0]["action"] == "select_option"
        assert state.history[1]["action"] == "add_to_cart"
        assert state.history[2]["action"] == "session_end"
