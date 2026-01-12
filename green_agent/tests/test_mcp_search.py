"""Tests for MCP search tool.

These tests verify the search() tool in WebShopMCPServer:
- Returns products with element IDs
- Updates visible_elements in session state
- Increments turn count
- Handles max turns exceeded
"""

from dataclasses import dataclass
from typing import Any

import pytest

from src.mcp.server import WebShopMCPServer
from src.mcp.session_state import SessionState


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


def create_search_results_html(products: list[dict]) -> str:
    """Create mock HTML for search results.

    Args:
        products: List of dicts with 'asin', 'name', 'price' keys.

    Returns:
        HTML string mimicking WebShop search results.
    """
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


class TestSearchReturnsProductsWithIds:
    """Test that search returns structured products with element IDs."""

    def test_search_returns_products_list(self):
        """Search should return a products list."""
        products = [
            {"asin": "B001", "name": "Running Shoes", "price": 49.99},
            {"asin": "B002", "name": "Trail Runners", "price": 59.99},
        ]
        html = create_search_results_html(products)

        state = SessionState(
            session_id="test-123",
            goal="Find running shoes",
            budget=100.0,
        )
        webshop = MockWebShop(search_results_html=html)
        server = WebShopMCPServer(state, webshop=webshop)

        # Get the search tool and call it
        result = server.mcp._tool_manager._tools["search"].fn("running shoes")

        assert "products" in result
        assert len(result["products"]) == 2

    def test_search_products_have_element_ids(self):
        """Each product should have an element ID like 'p1', 'p2'."""
        products = [
            {"asin": "B001", "name": "Running Shoes", "price": 49.99},
            {"asin": "B002", "name": "Trail Runners", "price": 59.99},
            {"asin": "B003", "name": "Sneakers", "price": 39.99},
        ]
        html = create_search_results_html(products)

        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        webshop = MockWebShop(search_results_html=html)
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["search"].fn("shoes")

        # Check IDs are p1, p2, p3
        ids = [p["id"] for p in result["products"]]
        assert ids == ["p1", "p2", "p3"]

    def test_search_products_have_name_and_price(self):
        """Each product should include name and price."""
        products = [
            {"asin": "B001", "name": "Running Shoes", "price": 49.99},
        ]
        html = create_search_results_html(products)

        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        webshop = MockWebShop(search_results_html=html)
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["search"].fn("shoes")

        product = result["products"][0]
        assert "name" in product
        assert "price" in product
        assert product["name"] == "Running Shoes"

    def test_search_uses_webshop_prices_when_available(self):
        """Search should use product_prices from webshop if available."""
        products = [
            {"asin": "B001", "name": "Running Shoes", "price": 49.99},
        ]
        html = create_search_results_html(products)

        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        # WebShop has a different price for this ASIN
        webshop = MockWebShop(
            search_results_html=html,
            prices={"B001": 45.00},
        )
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["search"].fn("shoes")

        # Should use the price from webshop.product_prices
        assert result["products"][0]["price"] == 45.00


class TestSearchUpdatesVisibleElements:
    """Test that search updates visible_elements in session state."""

    def test_search_populates_visible_elements(self):
        """After search, visible_elements should contain product entries."""
        products = [
            {"asin": "B001", "name": "Running Shoes", "price": 49.99},
            {"asin": "B002", "name": "Trail Runners", "price": 59.99},
        ]
        html = create_search_results_html(products)

        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        webshop = MockWebShop(search_results_html=html)
        server = WebShopMCPServer(state, webshop=webshop)

        server.mcp._tool_manager._tools["search"].fn("shoes")

        # Check visible_elements has p1, p2
        assert "p1" in state.visible_elements
        assert "p2" in state.visible_elements

    def test_visible_elements_contain_product_data(self):
        """Visible elements should store product type and data."""
        products = [
            {"asin": "B001", "name": "Running Shoes", "price": 49.99},
        ]
        html = create_search_results_html(products)

        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        webshop = MockWebShop(search_results_html=html)
        server = WebShopMCPServer(state, webshop=webshop)

        server.mcp._tool_manager._tools["search"].fn("shoes")

        elem = state.visible_elements["p1"]
        assert elem["type"] == "product"
        assert elem["asin"] == "B001"
        assert "data" in elem

    def test_search_clears_previous_visible_elements(self):
        """New search should clear previous visible elements."""
        products = [
            {"asin": "B001", "name": "Shoes", "price": 49.99},
        ]
        html = create_search_results_html(products)

        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        # Pre-populate with old elements
        state.visible_elements = {
            "old_element": {"type": "product", "asin": "OLD"},
        }

        webshop = MockWebShop(search_results_html=html)
        server = WebShopMCPServer(state, webshop=webshop)

        server.mcp._tool_manager._tools["search"].fn("shoes")

        # Old element should be gone
        assert "old_element" not in state.visible_elements
        # New element should be present
        assert "p1" in state.visible_elements

    def test_search_updates_current_page(self):
        """Search should update current_page to 'search_results'."""
        products = [{"asin": "B001", "name": "Shoes", "price": 49.99}]
        html = create_search_results_html(products)

        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.current_page = "home"

        webshop = MockWebShop(search_results_html=html)
        server = WebShopMCPServer(state, webshop=webshop)

        server.mcp._tool_manager._tools["search"].fn("shoes")

        assert state.current_page == "search_results"


class TestSearchIncrementsTurnCount:
    """Test that search increments turn count."""

    def test_search_increments_turn(self):
        """Each search should increment turn_count by 1."""
        html = create_search_results_html([])

        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        assert state.turn_count == 0

        webshop = MockWebShop(search_results_html=html)
        server = WebShopMCPServer(state, webshop=webshop)

        server.mcp._tool_manager._tools["search"].fn("shoes")
        assert state.turn_count == 1

        server.mcp._tool_manager._tools["search"].fn("sneakers")
        assert state.turn_count == 2

    def test_search_records_in_history(self):
        """Search actions should be recorded in history."""
        html = create_search_results_html([])

        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )

        webshop = MockWebShop(search_results_html=html)
        server = WebShopMCPServer(state, webshop=webshop)

        server.mcp._tool_manager._tools["search"].fn("running shoes")

        # Check history
        assert len(state.history) >= 1
        search_entry = next(
            (h for h in state.history if h.get("action") == "search"), None
        )
        assert search_entry is not None
        assert search_entry["query"] == "running shoes"


class TestSearchMaxTurnsExceeded:
    """Test that search handles max turns exceeded."""

    def test_search_returns_terminal_when_max_turns_exceeded(self):
        """When max turns exceeded, return terminal response."""
        html = create_search_results_html([])

        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
            max_turns=2,
        )
        # Set turn count to max
        state.turn_count = 2

        webshop = MockWebShop(search_results_html=html)
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["search"].fn("shoes")

        assert result["terminated"] is True
        assert result["reason"] == "max_turns_exceeded"
        assert result["score"] == 0.2

    def test_max_turns_marks_session_completed(self):
        """When max turns exceeded, session should be marked completed."""
        html = create_search_results_html([])

        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
            max_turns=1,
        )
        state.turn_count = 1

        webshop = MockWebShop(search_results_html=html)
        server = WebShopMCPServer(state, webshop=webshop)

        server.mcp._tool_manager._tools["search"].fn("shoes")

        assert state.completed is True

    def test_normal_search_does_not_terminate(self):
        """Normal search within turn limit should not terminate."""
        products = [{"asin": "B001", "name": "Shoes", "price": 49.99}]
        html = create_search_results_html(products)

        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
            max_turns=10,
        )

        webshop = MockWebShop(search_results_html=html)
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["search"].fn("shoes")

        assert "terminated" not in result or result.get("terminated") is False
        assert state.completed is False


class TestSearchReturnsPaginationActions:
    """Test that search returns pagination actions when available."""

    def test_search_includes_next_page_action(self):
        """When 'Next >' is available, include it in actions."""
        products = [{"asin": "B001", "name": "Shoes", "price": 49.99}]
        html = create_search_results_html(products)

        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        webshop = MockWebShop(
            search_results_html=html,
            available_actions={
                "has_search_bar": False,
                "clickables": ["next >", "b001"],
            },
        )
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["search"].fn("shoes")

        assert "actions" in result
        action_ids = [a["id"] for a in result["actions"]]
        assert "next_page" in action_ids

    def test_next_page_added_to_visible_elements(self):
        """Next page action should be in visible_elements."""
        products = [{"asin": "B001", "name": "Shoes", "price": 49.99}]
        html = create_search_results_html(products)

        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        webshop = MockWebShop(
            search_results_html=html,
            available_actions={
                "has_search_bar": False,
                "clickables": ["next >"],
            },
        )
        server = WebShopMCPServer(state, webshop=webshop)

        server.mcp._tool_manager._tools["search"].fn("shoes")

        assert "next_page" in state.visible_elements
        assert state.visible_elements["next_page"]["type"] == "navigation"

    def test_search_includes_prev_page_action(self):
        """When '< Prev' is available, include it in actions."""
        products = [{"asin": "B001", "name": "Shoes", "price": 49.99}]
        html = create_search_results_html(products)

        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        webshop = MockWebShop(
            search_results_html=html,
            available_actions={
                "has_search_bar": False,
                "clickables": ["< prev", "next >"],
            },
        )
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["search"].fn("shoes")

        action_ids = [a["id"] for a in result["actions"]]
        assert "prev_page" in action_ids
        assert "next_page" in action_ids


class TestSearchReturnsMetadata:
    """Test that search returns useful metadata."""

    def test_search_returns_query(self):
        """Result should include the query that was searched."""
        html = create_search_results_html([])

        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        webshop = MockWebShop(search_results_html=html)
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["search"].fn("blue running shoes")

        assert result["query"] == "blue running shoes"

    def test_search_returns_page_type(self):
        """Result should include page type."""
        html = create_search_results_html([])

        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        webshop = MockWebShop(search_results_html=html)
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["search"].fn("shoes")

        assert result["page"] == "search_results"

    def test_search_returns_turn_info(self):
        """Result should include turn count and remaining turns."""
        html = create_search_results_html([])

        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
            max_turns=10,
        )
        webshop = MockWebShop(search_results_html=html)
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["search"].fn("shoes")

        assert result["turn"] == 1
        assert result["turns_remaining"] == 9

    def test_search_returns_budget_info(self):
        """Result should include budget and cart total."""
        html = create_search_results_html([])

        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=75.50,
        )
        webshop = MockWebShop(search_results_html=html)
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["search"].fn("shoes")

        assert result["budget"] == 75.50
        assert result["cart_total"] == 0.0
