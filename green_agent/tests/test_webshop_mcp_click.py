"""Tests for MCP click tool.

These tests verify the click() tool:
- Validates element IDs from visible_elements
- Handles product clicks (show product page)
- Handles option selection
- Handles add to cart
- Handles navigation (next/prev page)
- Increments turn count
- Handles max turns exceeded
"""

from dataclasses import dataclass
from typing import Any

import pytest

from src.webshop_mcp.session_state import SessionState
from src.webshop_mcp.server import (
    click,
    current_session_id,
    register_session,
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
    """Mock WebShop environment for testing."""

    def __init__(
        self,
        search_results_html: str = "",
        available_actions: dict | None = None,
        prices: dict | None = None,
        product_items: dict | None = None,
    ):
        self.search_results_html = search_results_html
        self._available_actions = available_actions or {
            "has_search_bar": True,
            "clickables": [],
        }
        self._product_prices = prices or {}
        self._product_item_dict = product_items or {}
        self.step_calls: list[str] = []
        self.reset_calls: list[str] = []

    def reset(self, session: str | None = None) -> str:
        """Mock reset."""
        self.reset_calls.append(session)
        return "Welcome to WebShop"

    def step(self, action: str) -> MockStepResult:
        """Mock step - return configured HTML."""
        self.step_calls.append(action)
        return MockStepResult(observation=self.search_results_html)

    def get_available_actions(self) -> dict:
        """Return configured available actions."""
        return self._available_actions

    @property
    def product_prices(self) -> dict[str, float]:
        """Return configured product prices."""
        return self._product_prices

    @property
    def product_item_dict(self) -> dict[str, dict]:
        """Return configured product info."""
        return self._product_item_dict


@pytest.fixture(autouse=True)
def cleanup_global_state():
    """Clean up global session state before and after each test."""
    _session_states.clear()
    _webshop_interfaces.clear()
    yield
    _session_states.clear()
    _webshop_interfaces.clear()


def create_search_results_html(products: list[dict]) -> str:
    """Create mock HTML for search results."""
    items = []
    for p in products:
        items.append(f'''
        <div class="list-group-item">
            <h4>{p["name"]}</h4>
            <h5>${p["price"]:.2f}</h5>
            <a class="product-link">{p["asin"]}</a>
        </div>
        ''')
    return f'<div class="list-group">{"".join(items)}</div>'


def setup_session(
    session_id: str = "test-123",
    goal: str = "Find shoes",
    budget: float = 100.0,
    webshop: MockWebShop | None = None,
    max_turns: int = 30,
):
    """Set up a session for testing."""
    state = SessionState(
        session_id=session_id,
        goal=goal,
        budget=budget,
        max_turns=max_turns,
    )
    register_session(session_id, state, webshop)
    token = current_session_id.set(session_id)
    return state, token


class TestClickValidatesElementId:
    """Test that click validates element IDs from visible_elements."""

    def test_click_invalid_element_returns_error(self):
        """Click on unknown element should return error with available elements."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop)
        try:
            # Set up some visible elements
            state.visible_elements = {
                "p1": {"type": "product", "asin": "B001", "data": {}},
                "p2": {"type": "product", "asin": "B002", "data": {}},
            }

            result = click("invalid_id")

            assert "error" in result
            assert "invalid_id" in result["error"]
            assert "available_elements" in result
            assert "p1" in result["available_elements"]
            assert "p2" in result["available_elements"]
        finally:
            current_session_id.reset(token)

    def test_click_valid_element_succeeds(self):
        """Click on valid element should not return error."""
        webshop = MockWebShop(product_items={"B001": {}})
        state, token = setup_session(webshop=webshop)
        try:
            state.visible_elements = {
                "p1": {
                    "type": "product",
                    "asin": "B001",
                    "data": {"name": "Shoes", "price": 49.99, "asin": "B001"},
                },
            }

            result = click("p1")

            assert "error" not in result
            assert result["page"] == "product_detail"
        finally:
            current_session_id.reset(token)

    def test_click_records_error_in_history(self):
        """Invalid click should be recorded in history with error."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop)
        try:
            state.visible_elements = {}

            click("nonexistent")

            # Check history has error entry
            error_entry = next(
                (h for h in state.history if h.get("error") == "element_not_found"),
                None,
            )
            assert error_entry is not None
            assert error_entry["element_id"] == "nonexistent"
        finally:
            current_session_id.reset(token)


class TestClickOnProduct:
    """Test clicking on a product from search results."""

    def test_click_product_shows_product_page(self):
        """Clicking a product should show product detail page."""
        webshop = MockWebShop(product_items={"B001": {"name": "Running Shoes"}})
        state, token = setup_session(webshop=webshop)
        try:
            state.visible_elements = {
                "p1": {
                    "type": "product",
                    "asin": "B001",
                    "data": {"name": "Running Shoes", "price": 49.99, "asin": "B001"},
                },
            }

            result = click("p1")

            assert result["page"] == "product_detail"
            assert result["product"]["name"] == "Running Shoes"
            assert result["product"]["price"] == 49.99
        finally:
            current_session_id.reset(token)

    def test_click_product_updates_current_page(self):
        """Clicking product should update current_page state."""
        webshop = MockWebShop(product_items={"B001": {}})
        state, token = setup_session(webshop=webshop)
        try:
            state.current_page = "search_results"
            state.visible_elements = {
                "p1": {
                    "type": "product",
                    "asin": "B001",
                    "data": {"name": "Shoes", "price": 49.99, "asin": "B001"},
                },
            }

            click("p1")

            assert state.current_page == "product_detail"
        finally:
            current_session_id.reset(token)

    def test_click_product_shows_add_to_cart_action(self):
        """Product page should include add_to_cart action."""
        webshop = MockWebShop(product_items={"B001": {}})
        state, token = setup_session(webshop=webshop)
        try:
            state.visible_elements = {
                "p1": {
                    "type": "product",
                    "asin": "B001",
                    "data": {"name": "Shoes", "price": 49.99, "asin": "B001"},
                },
            }

            result = click("p1")

            action_ids = [a["id"] for a in result["actions"]]
            assert "add_to_cart" in action_ids
        finally:
            current_session_id.reset(token)

    def test_click_product_shows_options(self):
        """Product page should show available options."""
        webshop = MockWebShop(
            product_items={
                "B001": {
                    "name": "Shoes",
                    "size": ["8", "9", "10"],
                    "color": ["black", "white"],
                }
            }
        )
        state, token = setup_session(webshop=webshop)
        try:
            state.visible_elements = {
                "p1": {
                    "type": "product",
                    "asin": "B001",
                    "data": {"name": "Shoes", "price": 49.99, "asin": "B001"},
                },
            }

            result = click("p1")

            # Should have options
            assert "options" in result
            assert len(result["options"]) > 0

            # Check size options exist
            size_options = [o for o in result["options"] if o["type"] == "size"]
            assert len(size_options) == 3
        finally:
            current_session_id.reset(token)


class TestClickOnOption:
    """Test clicking on product options."""

    def test_click_option_selects_it(self):
        """Clicking an option should select it."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop)
        try:
            state.visible_elements = {
                "size_10": {
                    "type": "option",
                    "option_type": "size",
                    "value": "10",
                    "product_asin": "B001",
                },
            }

            result = click("size_10")

            assert result["action"] == "option_selected"
            assert result["option_type"] == "size"
            assert result["value"] == "10"
        finally:
            current_session_id.reset(token)

    def test_click_option_updates_selected_options(self):
        """Option selection should update state.selected_options."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop)
        try:
            state.visible_elements = {
                "color_blue": {
                    "type": "option",
                    "option_type": "color",
                    "value": "blue",
                    "product_asin": "B001",
                },
            }

            click("color_blue")

            assert state.selected_options.get("color") == "blue"
        finally:
            current_session_id.reset(token)


class TestClickAddToCart:
    """Test clicking add to cart."""

    def test_click_add_to_cart_adds_product(self):
        """Clicking add_to_cart should add product to cart."""
        webshop = MockWebShop(prices={"B001": 49.99})
        state, token = setup_session(webshop=webshop)
        try:
            state.visible_elements = {
                "add_to_cart": {
                    "type": "add_to_cart",
                    "product": {"name": "Running Shoes", "price": 49.99},
                    "asin": "B001",
                },
            }

            result = click("add_to_cart")

            assert result["action"] == "added_to_cart"
            assert result["added"] == "Running Shoes"
            assert len(state.cart) == 1
        finally:
            current_session_id.reset(token)

    def test_add_to_cart_updates_cart_total(self):
        """Adding to cart should update cart_total."""
        webshop = MockWebShop(prices={"B001": 49.99})
        state, token = setup_session(webshop=webshop)
        try:
            state.visible_elements = {
                "add_to_cart": {
                    "type": "add_to_cart",
                    "product": {"name": "Shoes", "price": 49.99},
                    "asin": "B001",
                },
            }

            result = click("add_to_cart")

            assert result["cart_total"] == 49.99
            assert result["cart_size"] == 1
        finally:
            current_session_id.reset(token)

    def test_add_to_cart_warns_over_budget(self):
        """Adding item over budget should show warning."""
        webshop = MockWebShop(prices={"B001": 59.99})
        state, token = setup_session(webshop=webshop, budget=40.0)
        try:
            state.visible_elements = {
                "add_to_cart": {
                    "type": "add_to_cart",
                    "product": {"name": "Expensive Shoes", "price": 59.99},
                    "asin": "B001",
                },
            }

            result = click("add_to_cart")

            assert result["over_budget"] is True
            assert result["warning"] is not None
            assert "budget" in result["warning"].lower()
        finally:
            current_session_id.reset(token)


class TestClickNavigation:
    """Test clicking navigation elements."""

    def test_click_next_page(self):
        """Clicking next_page should navigate to next results."""
        new_products = [{"asin": "B003", "name": "Page 2 Shoes", "price": 39.99}]
        html = create_search_results_html(new_products)
        webshop = MockWebShop(
            search_results_html=html,
            available_actions={"clickables": ["< prev"]},
            prices={"B003": 39.99},
        )
        state, token = setup_session(webshop=webshop)
        try:
            state.visible_elements = {
                "next_page": {
                    "type": "navigation",
                    "action": "next",
                },
            }

            result = click("next_page")

            assert result["page"] == "search_results"
            assert result["action"] == "next_page"
            assert len(result["products"]) == 1
        finally:
            current_session_id.reset(token)

    def test_click_back_to_results(self):
        """Clicking back_to_results should return to search results."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop)
        try:
            state.current_page = "product_detail"
            state.visible_elements = {
                "back_to_results": {
                    "type": "navigation",
                    "action": "back",
                },
            }

            result = click("back_to_results")

            assert result["page"] == "search_results"
            assert state.current_page == "search_results"
        finally:
            current_session_id.reset(token)


class TestClickIncrementsTurnCount:
    """Test that click increments turn count."""

    def test_click_increments_turn(self):
        """Each click should increment turn_count by 1."""
        webshop = MockWebShop(product_items={"B001": {}})
        state, token = setup_session(webshop=webshop)
        try:
            state.visible_elements = {
                "p1": {
                    "type": "product",
                    "asin": "B001",
                    "data": {"name": "Shoes", "price": 49.99, "asin": "B001"},
                },
            }

            assert state.turn_count == 0
            click("p1")
            assert state.turn_count == 1
        finally:
            current_session_id.reset(token)

    def test_invalid_click_also_increments_turn(self):
        """Invalid click should still increment turn (action was taken)."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop)
        try:
            state.visible_elements = {}

            assert state.turn_count == 0
            click("invalid")
            assert state.turn_count == 1
        finally:
            current_session_id.reset(token)


class TestClickMaxTurnsExceeded:
    """Test that click handles max turns exceeded."""

    def test_click_returns_terminal_when_max_turns_exceeded(self):
        """When max turns exceeded, return terminal response."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop, max_turns=2)
        try:
            state.turn_count = 2  # At max
            state.visible_elements = {
                "p1": {"type": "product", "asin": "B001", "data": {}},
            }

            result = click("p1")

            assert result["terminated"] is True
            assert result["reason"] == "max_turns_exceeded"
            assert result["score"] == 0.2
        finally:
            current_session_id.reset(token)

    def test_max_turns_marks_session_completed(self):
        """When max turns exceeded, session should be marked completed."""
        webshop = MockWebShop()
        state, token = setup_session(webshop=webshop, max_turns=1)
        try:
            state.turn_count = 1
            state.visible_elements = {"p1": {"type": "product", "asin": "B001", "data": {}}}

            click("p1")

            assert state.completed is True
        finally:
            current_session_id.reset(token)


class TestClickReturnsMetadata:
    """Test that click returns useful metadata."""

    def test_click_returns_turn_info(self):
        """Click result should include turn count and remaining turns."""
        webshop = MockWebShop(product_items={"B001": {}})
        state, token = setup_session(webshop=webshop, max_turns=10)
        try:
            state.visible_elements = {
                "p1": {
                    "type": "product",
                    "asin": "B001",
                    "data": {"name": "Shoes", "price": 49.99, "asin": "B001"},
                },
            }

            result = click("p1")

            assert result["turn"] == 1
            assert result["turns_remaining"] == 9
        finally:
            current_session_id.reset(token)

    def test_click_returns_budget_info(self):
        """Click result should include budget info."""
        webshop = MockWebShop(product_items={"B001": {}})
        state, token = setup_session(webshop=webshop, budget=75.50)
        try:
            state.visible_elements = {
                "p1": {
                    "type": "product",
                    "asin": "B001",
                    "data": {"name": "Shoes", "price": 49.99, "asin": "B001"},
                },
            }

            result = click("p1")

            assert result["budget"] == 75.50
            assert "cart_total" in result
        finally:
            current_session_id.reset(token)
