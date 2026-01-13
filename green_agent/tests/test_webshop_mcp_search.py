"""Tests for MCP search tool.

These tests verify the search() tool:
- Returns products with element IDs
- Updates visible_elements in session state
- Increments turn count
- Handles max turns exceeded
"""

from dataclasses import dataclass
from typing import Any

import pytest

from src.webshop_mcp.session_state import SessionState
from src.webshop_mcp.server import (
    search,
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
    """Mock WebShop environment for testing."""

    def __init__(
        self,
        search_results_html: str = "",
        available_actions: dict | None = None,
        prices: dict | None = None,
    ):
        self.search_results_html = search_results_html
        self._available_actions = available_actions or {
            "has_search_bar": True,
            "clickables": [],
        }
        self._product_prices = prices or {}
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


def create_search_results_html(products: list[dict]) -> str:
    """Create mock [SEP]-delimited text for search results.
    
    WebShop text environment returns [SEP]-delimited format, not HTML.
    Format: Instruction [SEP] ... [SEP] ASIN [SEP] Name [SEP] Price [SEP] ...

    Args:
        products: List of dicts with 'asin', 'name', 'price' keys.

    Returns:
        [SEP]-delimited string mimicking WebShop search results.
    """
    parts = [
        "Instruction: Find running shoes",
        "Back to Search",
        "Page 1 (Total results: {})".format(len(products)),
    ]
    
    # Add products in [SEP] format: ASIN [SEP] Name [SEP] Price
    for p in products:
        parts.extend([
            p["asin"],
            p["name"],
            "${:.2f}".format(p["price"]),
        ])
    
    # Add navigation if multiple products
    if len(products) > 0:
        parts.append("Next >")
    
    return " [SEP] ".join(parts)


def setup_session(
    session_id: str = "test-123",
    goal: str = "Find running shoes",
    budget: float = 100.0,
    webshop: MockWebShop | None = None,
    max_turns: int = 30,
):
    """Set up a session for testing.

    Args:
        session_id: Session ID to use.
        goal: Shopping goal.
        budget: Budget limit.
        webshop: Mock WebShop instance.
        max_turns: Maximum turns allowed.

    Returns:
        Tuple of (state, token) where token can be used to reset contextvar.
    """
    state = SessionState(
        session_id=session_id,
        goal=goal,
        budget=budget,
        max_turns=max_turns,
    )
    register_session(session_id, state, webshop)
    token = current_session_id.set(session_id)
    return state, token


class TestSearchReturnsProductsWithIds:
    """Test that search returns structured products with element IDs."""

    def test_search_returns_products_list(self):
        """Search should return a products list."""
        products = [
            {"asin": "B001234567", "name": "Running Shoes", "price": 49.99},
            {"asin": "B002345678", "name": "Trail Runners", "price": 59.99},
        ]
        html = create_search_results_html(products)
        webshop = MockWebShop(search_results_html=html)

        state, token = setup_session(webshop=webshop)
        try:
            result = search("running shoes")

            assert "products" in result
            assert len(result["products"]) == 2
        finally:
            current_session_id.reset(token)

    def test_search_products_have_element_ids(self):
        """Each product should have an element ID like 'p1', 'p2'."""
        products = [
            {"asin": "B001234567", "name": "Running Shoes", "price": 49.99},
            {"asin": "B002345678", "name": "Trail Runners", "price": 59.99},
            {"asin": "B003456789", "name": "Sneakers", "price": 39.99},
        ]
        html = create_search_results_html(products)
        webshop = MockWebShop(search_results_html=html)

        state, token = setup_session(webshop=webshop)
        try:
            result = search("shoes")

            # Check IDs are p1, p2, p3
            ids = [p["id"] for p in result["products"]]
            assert ids == ["p1", "p2", "p3"]
        finally:
            current_session_id.reset(token)

    def test_search_products_have_name_and_price(self):
        """Each product should include name and price."""
        products = [
            {"asin": "B001234567", "name": "Running Shoes", "price": 49.99},
        ]
        html = create_search_results_html(products)
        webshop = MockWebShop(search_results_html=html)

        state, token = setup_session(webshop=webshop)
        try:
            result = search("shoes")

            product = result["products"][0]
            assert "name" in product
            assert "price" in product
            assert product["name"] == "Running Shoes"
        finally:
            current_session_id.reset(token)

    def test_search_uses_webshop_prices_when_available(self):
        """Search should use product_prices from webshop if available."""
        products = [
            {"asin": "B001234567", "name": "Running Shoes", "price": 49.99},
        ]
        html = create_search_results_html(products)
        # WebShop has a different price for this ASIN
        webshop = MockWebShop(
            search_results_html=html,
            prices={"B001": 45.00},
        )

        state, token = setup_session(webshop=webshop)
        try:
            result = search("shoes")

            # Should use the price from webshop.product_prices
            assert result["products"][0]["price"] == 45.00
        finally:
            current_session_id.reset(token)


class TestSearchUpdatesVisibleElements:
    """Test that search updates visible_elements in session state."""

    def test_search_updates_visible_elements(self):
        """Search should update state.visible_elements with product info."""
        products = [
            {"asin": "B001234567", "name": "Running Shoes", "price": 49.99},
        ]
        html = create_search_results_html(products)
        webshop = MockWebShop(search_results_html=html)

        state, token = setup_session(webshop=webshop)
        try:
            search("shoes")

            assert "p1" in state.visible_elements
            assert state.visible_elements["p1"]["type"] == "product"
            assert state.visible_elements["p1"]["asin"] == "B001"
        finally:
            current_session_id.reset(token)

    def test_search_clears_previous_visible_elements(self):
        """New search should clear previous visible elements."""
        products = [
            {"asin": "B001234567", "name": "Running Shoes", "price": 49.99},
        ]
        html = create_search_results_html(products)
        webshop = MockWebShop(search_results_html=html)

        state, token = setup_session(webshop=webshop)
        try:
            # Add some pre-existing elements
            state.visible_elements = {"old_element": {"type": "old"}}

            search("shoes")

            # Old elements should be gone
            assert "old_element" not in state.visible_elements
            assert "p1" in state.visible_elements
        finally:
            current_session_id.reset(token)


class TestSearchTurnCount:
    """Test that search increments turn count."""

    def test_search_increments_turn_count(self):
        """Each search should increment turn count."""
        products = [{"asin": "B001", "name": "Running Shoes", "price": 49.99}]
        html = create_search_results_html(products)
        webshop = MockWebShop(search_results_html=html)

        state, token = setup_session(webshop=webshop)
        try:
            assert state.turn_count == 0

            search("shoes")
            assert state.turn_count == 1

            search("sneakers")
            assert state.turn_count == 2
        finally:
            current_session_id.reset(token)

    def test_search_returns_turn_info(self):
        """Search result should include turn information."""
        products = [{"asin": "B001", "name": "Running Shoes", "price": 49.99}]
        html = create_search_results_html(products)
        webshop = MockWebShop(search_results_html=html)

        state, token = setup_session(webshop=webshop, max_turns=10)
        try:
            result = search("shoes")

            assert "turn" in result
            assert "turns_remaining" in result
            assert result["turn"] == 1
            assert result["turns_remaining"] == 9
        finally:
            current_session_id.reset(token)


class TestSearchMaxTurns:
    """Test max turns handling in search."""

    def test_search_terminates_at_max_turns(self):
        """Search should return terminal state when max turns exceeded."""
        products = [{"asin": "B001", "name": "Running Shoes", "price": 49.99}]
        html = create_search_results_html(products)
        webshop = MockWebShop(search_results_html=html)

        state, token = setup_session(webshop=webshop, max_turns=2)
        try:
            # First two searches are fine
            search("shoes")
            search("sneakers")

            # Third search should terminate
            result = search("boots")

            assert result["terminated"] is True
            assert result["reason"] == "max_turns_exceeded"
            assert "score" in result
        finally:
            current_session_id.reset(token)

    def test_search_marks_session_completed_at_max_turns(self):
        """Session should be marked completed when max turns hit."""
        products = [{"asin": "B001", "name": "Running Shoes", "price": 49.99}]
        html = create_search_results_html(products)
        webshop = MockWebShop(search_results_html=html)

        state, token = setup_session(webshop=webshop, max_turns=1)
        try:
            assert not state.completed

            # This search hits max turns
            search("shoes")
            search("more")

            assert state.completed
        finally:
            current_session_id.reset(token)


class TestSearchBudgetInfo:
    """Test that search includes budget information."""

    def test_search_includes_budget(self):
        """Search result should include budget info."""
        products = [{"asin": "B001", "name": "Running Shoes", "price": 49.99}]
        html = create_search_results_html(products)
        webshop = MockWebShop(search_results_html=html)

        state, token = setup_session(webshop=webshop, budget=50.0)
        try:
            result = search("shoes")

            assert "budget" in result
            assert result["budget"] == 50.0
        finally:
            current_session_id.reset(token)

    def test_search_includes_cart_total(self):
        """Search result should include current cart total."""
        products = [{"asin": "B001", "name": "Running Shoes", "price": 49.99}]
        html = create_search_results_html(products)
        webshop = MockWebShop(search_results_html=html)

        state, token = setup_session(webshop=webshop)
        try:
            result = search("shoes")

            assert "cart_total" in result
            assert result["cart_total"] == 0.0  # Empty cart
        finally:
            current_session_id.reset(token)


class TestSearchPagination:
    """Test search pagination actions."""

    def test_search_includes_next_page_action(self):
        """Search should include next page if available."""
        products = [{"asin": "B001234567", "name": "Running Shoes", "price": 49.99}]
        sep_text = create_search_results_html(products)
        webshop = MockWebShop(
            search_results_html=sep_text,
            available_actions={"clickables": ["next >"]},
        )

        state, token = setup_session(webshop=webshop)
        try:
            result = search("shoes")

            actions = [a["id"] for a in result.get("available_actions", [])]
            assert "next_page" in actions
        finally:
            current_session_id.reset(token)

    def test_search_includes_prev_page_action(self):
        """Search should include prev page if available."""
        products = [{"asin": "B001234567", "name": "Running Shoes", "price": 49.99}]
        sep_text = create_search_results_html(products)
        webshop = MockWebShop(
            search_results_html=sep_text,
            available_actions={"clickables": ["< prev"]},
        )

        state, token = setup_session(webshop=webshop)
        try:
            result = search("shoes")

            actions = [a["id"] for a in result.get("available_actions", [])]
            assert "prev_page" in actions
        finally:
            current_session_id.reset(token)


class TestSearchWebShopIntegration:
    """Test search integration with WebShop environment."""

    def test_search_calls_webshop_step(self):
        """Search should call webshop.step with search action."""
        products = [{"asin": "B001", "name": "Running Shoes", "price": 49.99}]
        html = create_search_results_html(products)
        webshop = MockWebShop(search_results_html=html)

        state, token = setup_session(webshop=webshop)
        try:
            search("running shoes")

            assert len(webshop.step_calls) == 1
            assert "search[running shoes]" in webshop.step_calls[0]
        finally:
            current_session_id.reset(token)

    def test_search_records_history(self):
        """Search should record action in session history."""
        products = [{"asin": "B001", "name": "Running Shoes", "price": 49.99}]
        html = create_search_results_html(products)
        webshop = MockWebShop(search_results_html=html)

        state, token = setup_session(webshop=webshop)
        try:
            search("running shoes")

            assert len(state.history) == 1
            assert state.history[0]["action"] == "search"
            assert state.history[0]["query"] == "running shoes"
        finally:
            current_session_id.reset(token)
