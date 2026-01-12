"""Tests for MCP click tool.

These tests verify the click() tool in WebShopMCPServer:
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


class TestClickValidatesElementId:
    """Test that click validates element IDs from visible_elements."""

    def test_click_invalid_element_returns_error(self):
        """Click on unknown element should return error with available elements."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        # Set up some visible elements
        state.visible_elements = {
            "p1": {"type": "product", "asin": "B001", "data": {}},
            "p2": {"type": "product", "asin": "B002", "data": {}},
        }

        webshop = MockWebShop()
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["click"].fn("invalid_id")

        assert "error" in result
        assert "invalid_id" in result["error"]
        assert "available_elements" in result
        assert "p1" in result["available_elements"]
        assert "p2" in result["available_elements"]

    def test_click_valid_element_succeeds(self):
        """Click on valid element should not return error."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.visible_elements = {
            "p1": {
                "type": "product",
                "asin": "B001",
                "data": {"name": "Shoes", "price": 49.99, "asin": "B001"},
            },
        }

        webshop = MockWebShop(product_items={"B001": {}})
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["click"].fn("p1")

        assert "error" not in result
        assert result["page"] == "product_detail"

    def test_click_records_error_in_history(self):
        """Invalid click should be recorded in history with error."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.visible_elements = {}

        webshop = MockWebShop()
        server = WebShopMCPServer(state, webshop=webshop)

        server.mcp._tool_manager._tools["click"].fn("nonexistent")

        # Check history has error entry
        error_entry = next(
            (h for h in state.history if h.get("error") == "element_not_found"),
            None,
        )
        assert error_entry is not None
        assert error_entry["element_id"] == "nonexistent"


class TestClickOnProduct:
    """Test clicking on a product from search results."""

    def test_click_product_shows_product_page(self):
        """Clicking a product should show product detail page."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.visible_elements = {
            "p1": {
                "type": "product",
                "asin": "B001",
                "data": {"name": "Running Shoes", "price": 49.99, "asin": "B001"},
            },
        }

        webshop = MockWebShop(product_items={"B001": {"name": "Running Shoes"}})
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["click"].fn("p1")

        assert result["page"] == "product_detail"
        assert result["product"]["name"] == "Running Shoes"
        assert result["product"]["price"] == 49.99

    def test_click_product_updates_current_page(self):
        """Clicking product should update current_page state."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.current_page = "search_results"
        state.visible_elements = {
            "p1": {
                "type": "product",
                "asin": "B001",
                "data": {"name": "Shoes", "price": 49.99, "asin": "B001"},
            },
        }

        webshop = MockWebShop(product_items={"B001": {}})
        server = WebShopMCPServer(state, webshop=webshop)

        server.mcp._tool_manager._tools["click"].fn("p1")

        assert state.current_page == "product_detail"

    def test_click_product_shows_add_to_cart_action(self):
        """Product page should include add_to_cart action."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.visible_elements = {
            "p1": {
                "type": "product",
                "asin": "B001",
                "data": {"name": "Shoes", "price": 49.99, "asin": "B001"},
            },
        }

        webshop = MockWebShop(product_items={"B001": {}})
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["click"].fn("p1")

        action_ids = [a["id"] for a in result["actions"]]
        assert "add_to_cart" in action_ids

    def test_click_product_shows_options(self):
        """Product page should show available options."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.visible_elements = {
            "p1": {
                "type": "product",
                "asin": "B001",
                "data": {"name": "Shoes", "price": 49.99, "asin": "B001"},
            },
        }

        # Product with size options
        webshop = MockWebShop(
            product_items={
                "B001": {
                    "name": "Shoes",
                    "size": ["8", "9", "10"],
                    "color": ["black", "white"],
                }
            }
        )
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["click"].fn("p1")

        # Should have options
        assert "options" in result
        assert len(result["options"]) > 0

        # Check size options exist
        size_options = [o for o in result["options"] if o["type"] == "size"]
        assert len(size_options) == 3

    def test_click_product_populates_option_elements(self):
        """Product options should be in visible_elements."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.visible_elements = {
            "p1": {
                "type": "product",
                "asin": "B001",
                "data": {"name": "Shoes", "price": 49.99, "asin": "B001"},
            },
        }

        webshop = MockWebShop(
            product_items={"B001": {"size": ["10", "11"]}}
        )
        server = WebShopMCPServer(state, webshop=webshop)

        server.mcp._tool_manager._tools["click"].fn("p1")

        # Should have size option elements
        assert "size_10" in state.visible_elements
        assert "size_11" in state.visible_elements
        assert state.visible_elements["size_10"]["type"] == "option"


class TestClickOnOption:
    """Test clicking on product options."""

    def test_click_option_selects_it(self):
        """Clicking an option should select it."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.visible_elements = {
            "size_10": {
                "type": "option",
                "option_type": "size",
                "value": "10",
                "product_asin": "B001",
            },
        }

        webshop = MockWebShop()
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["click"].fn("size_10")

        assert result["action"] == "option_selected"
        assert result["option_type"] == "size"
        assert result["value"] == "10"

    def test_click_option_updates_selected_options(self):
        """Option selection should update state.selected_options."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.visible_elements = {
            "color_blue": {
                "type": "option",
                "option_type": "color",
                "value": "blue",
                "product_asin": "B001",
            },
        }

        webshop = MockWebShop()
        server = WebShopMCPServer(state, webshop=webshop)

        server.mcp._tool_manager._tools["click"].fn("color_blue")

        assert state.selected_options.get("color") == "blue"

    def test_click_multiple_options(self):
        """Should be able to select multiple option types."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.visible_elements = {
            "size_10": {
                "type": "option",
                "option_type": "size",
                "value": "10",
                "product_asin": "B001",
            },
            "color_blue": {
                "type": "option",
                "option_type": "color",
                "value": "blue",
                "product_asin": "B001",
            },
        }

        webshop = MockWebShop()
        server = WebShopMCPServer(state, webshop=webshop)

        server.mcp._tool_manager._tools["click"].fn("size_10")
        # Turn count already incremented, add back an element for second click
        state.visible_elements["color_blue"] = {
            "type": "option",
            "option_type": "color",
            "value": "blue",
            "product_asin": "B001",
        }
        server.mcp._tool_manager._tools["click"].fn("color_blue")

        assert state.selected_options.get("size") == "10"
        assert state.selected_options.get("color") == "blue"

    def test_option_returns_all_selections(self):
        """Option click should return all current selections."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.selected_options = {"size": "9"}
        state.visible_elements = {
            "color_red": {
                "type": "option",
                "option_type": "color",
                "value": "red",
                "product_asin": "B001",
            },
        }

        webshop = MockWebShop()
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["click"].fn("color_red")

        assert "size" in result["selected_options"]
        assert "color" in result["selected_options"]


class TestClickAddToCart:
    """Test clicking add to cart."""

    def test_click_add_to_cart_adds_product(self):
        """Clicking add_to_cart should add product to cart."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.visible_elements = {
            "add_to_cart": {
                "type": "add_to_cart",
                "product": {"name": "Running Shoes", "price": 49.99},
                "asin": "B001",
            },
        }

        webshop = MockWebShop(prices={"B001": 49.99})
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["click"].fn("add_to_cart")

        assert result["action"] == "added_to_cart"
        assert result["added"] == "Running Shoes"
        assert len(state.cart) == 1

    def test_add_to_cart_updates_cart_total(self):
        """Adding to cart should update cart_total."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.visible_elements = {
            "add_to_cart": {
                "type": "add_to_cart",
                "product": {"name": "Shoes", "price": 49.99},
                "asin": "B001",
            },
        }

        webshop = MockWebShop(prices={"B001": 49.99})
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["click"].fn("add_to_cart")

        assert result["cart_total"] == 49.99
        assert result["cart_size"] == 1

    def test_add_to_cart_warns_over_budget(self):
        """Adding item over budget should show warning."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=40.0,  # Budget less than price
        )
        state.visible_elements = {
            "add_to_cart": {
                "type": "add_to_cart",
                "product": {"name": "Expensive Shoes", "price": 59.99},
                "asin": "B001",
            },
        }

        webshop = MockWebShop(prices={"B001": 59.99})
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["click"].fn("add_to_cart")

        assert result["over_budget"] is True
        assert result["warning"] is not None
        assert "budget" in result["warning"].lower()

    def test_add_to_cart_clears_selected_options(self):
        """Adding to cart should clear selected options."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.selected_options = {"size": "10", "color": "blue"}
        state.visible_elements = {
            "add_to_cart": {
                "type": "add_to_cart",
                "product": {"name": "Shoes", "price": 49.99},
                "asin": "B001",
            },
        }

        webshop = MockWebShop(prices={"B001": 49.99})
        server = WebShopMCPServer(state, webshop=webshop)

        server.mcp._tool_manager._tools["click"].fn("add_to_cart")

        # Selected options should be cleared after adding to cart
        assert len(state.selected_options) == 0

    def test_add_to_cart_uses_webshop_price(self):
        """Add to cart should use price from WebShop, not cached."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.visible_elements = {
            "add_to_cart": {
                "type": "add_to_cart",
                "product": {"name": "Shoes", "price": 49.99},  # Cached price
                "asin": "B001",
            },
        }

        # WebShop has different price
        webshop = MockWebShop(prices={"B001": 45.00})
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["click"].fn("add_to_cart")

        assert result["cart_total"] == 45.00
        assert state.cart[0]["price"] == 45.00


class TestClickNavigation:
    """Test clicking navigation elements."""

    def test_click_next_page(self):
        """Clicking next_page should navigate to next results."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.visible_elements = {
            "next_page": {
                "type": "navigation",
                "action": "next",
            },
        }

        # Mock WebShop returns new products
        new_products = [
            {"asin": "B003", "name": "Page 2 Shoes", "price": 39.99},
        ]
        html = create_search_results_html(new_products)

        webshop = MockWebShop(
            search_results_html=html,
            available_actions={"clickables": ["< prev"]},
            prices={"B003": 39.99},
        )
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["click"].fn("next_page")

        assert result["page"] == "search_results"
        assert result["action"] == "next_page"
        assert len(result["products"]) == 1

    def test_click_prev_page(self):
        """Clicking prev_page should navigate to previous results."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.visible_elements = {
            "prev_page": {
                "type": "navigation",
                "action": "prev",
            },
        }

        new_products = [
            {"asin": "B001", "name": "Page 1 Shoes", "price": 49.99},
        ]
        html = create_search_results_html(new_products)

        webshop = MockWebShop(
            search_results_html=html,
            available_actions={"clickables": ["next >"]},
            prices={"B001": 49.99},
        )
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["click"].fn("prev_page")

        assert result["page"] == "search_results"
        assert result["action"] == "prev_page"

    def test_click_back_to_results(self):
        """Clicking back_to_results should return to search results."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.current_page = "product_detail"
        state.visible_elements = {
            "back_to_results": {
                "type": "navigation",
                "action": "back",
            },
        }

        webshop = MockWebShop()
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["click"].fn("back_to_results")

        assert result["page"] == "search_results"
        assert state.current_page == "search_results"

    def test_navigation_calls_webshop_step(self):
        """Navigation should call WebShop step with correct action."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.visible_elements = {
            "next_page": {"type": "navigation", "action": "next"},
        }

        webshop = MockWebShop(search_results_html="<div></div>")
        server = WebShopMCPServer(state, webshop=webshop)

        server.mcp._tool_manager._tools["click"].fn("next_page")

        # Should have called step with click[next >]
        assert len(webshop.step_calls) == 1
        assert "next >" in webshop.step_calls[0]


class TestClickIncrementsTurnCount:
    """Test that click increments turn count."""

    def test_click_increments_turn(self):
        """Each click should increment turn_count by 1."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.visible_elements = {
            "p1": {
                "type": "product",
                "asin": "B001",
                "data": {"name": "Shoes", "price": 49.99, "asin": "B001"},
            },
        }

        webshop = MockWebShop(product_items={"B001": {}})
        server = WebShopMCPServer(state, webshop=webshop)

        assert state.turn_count == 0
        server.mcp._tool_manager._tools["click"].fn("p1")
        assert state.turn_count == 1

    def test_invalid_click_also_increments_turn(self):
        """Invalid click should still increment turn (action was taken)."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.visible_elements = {}

        webshop = MockWebShop()
        server = WebShopMCPServer(state, webshop=webshop)

        assert state.turn_count == 0
        server.mcp._tool_manager._tools["click"].fn("invalid")
        assert state.turn_count == 1

    def test_click_records_in_history(self):
        """Click actions should be recorded in history."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.visible_elements = {
            "p1": {
                "type": "product",
                "asin": "B001",
                "data": {"name": "Shoes", "price": 49.99, "asin": "B001"},
            },
        }

        webshop = MockWebShop(product_items={"B001": {}})
        server = WebShopMCPServer(state, webshop=webshop)

        server.mcp._tool_manager._tools["click"].fn("p1")

        click_entry = next(
            (h for h in state.history if h.get("action") == "click"), None
        )
        assert click_entry is not None
        assert click_entry["element_id"] == "p1"
        assert click_entry["element_type"] == "product"


class TestClickMaxTurnsExceeded:
    """Test that click handles max turns exceeded."""

    def test_click_returns_terminal_when_max_turns_exceeded(self):
        """When max turns exceeded, return terminal response."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
            max_turns=2,
        )
        state.turn_count = 2  # At max
        state.visible_elements = {
            "p1": {"type": "product", "asin": "B001", "data": {}},
        }

        webshop = MockWebShop()
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["click"].fn("p1")

        assert result["terminated"] is True
        assert result["reason"] == "max_turns_exceeded"
        assert result["score"] == 0.2

    def test_max_turns_marks_session_completed(self):
        """When max turns exceeded, session should be marked completed."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
            max_turns=1,
        )
        state.turn_count = 1
        state.visible_elements = {"p1": {"type": "product", "asin": "B001", "data": {}}}

        webshop = MockWebShop()
        server = WebShopMCPServer(state, webshop=webshop)

        server.mcp._tool_manager._tools["click"].fn("p1")

        assert state.completed is True

    def test_normal_click_does_not_terminate(self):
        """Normal click within turn limit should not terminate."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
            max_turns=10,
        )
        state.visible_elements = {
            "p1": {
                "type": "product",
                "asin": "B001",
                "data": {"name": "Shoes", "price": 49.99, "asin": "B001"},
            },
        }

        webshop = MockWebShop(product_items={"B001": {}})
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["click"].fn("p1")

        assert "terminated" not in result or result.get("terminated") is False
        assert state.completed is False


class TestClickReturnsMetadata:
    """Test that click returns useful metadata."""

    def test_click_returns_turn_info(self):
        """Click result should include turn count and remaining turns."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
            max_turns=10,
        )
        state.visible_elements = {
            "p1": {
                "type": "product",
                "asin": "B001",
                "data": {"name": "Shoes", "price": 49.99, "asin": "B001"},
            },
        }

        webshop = MockWebShop(product_items={"B001": {}})
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["click"].fn("p1")

        assert result["turn"] == 1
        assert result["turns_remaining"] == 9

    def test_click_returns_budget_info(self):
        """Click result should include budget info."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=75.50,
        )
        state.visible_elements = {
            "p1": {
                "type": "product",
                "asin": "B001",
                "data": {"name": "Shoes", "price": 49.99, "asin": "B001"},
            },
        }

        webshop = MockWebShop(product_items={"B001": {}})
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["click"].fn("p1")

        assert result["budget"] == 75.50
        assert "cart_total" in result

    def test_error_click_returns_available_elements(self):
        """Error click should list available elements."""
        state = SessionState(
            session_id="test-123",
            goal="Find shoes",
            budget=100.0,
        )
        state.visible_elements = {
            "p1": {"type": "product", "asin": "B001", "data": {}},
            "next_page": {"type": "navigation", "action": "next"},
        }

        webshop = MockWebShop()
        server = WebShopMCPServer(state, webshop=webshop)

        result = server.mcp._tool_manager._tools["click"].fn("wrong_id")

        assert "p1" in result["available_elements"]
        assert "next_page" in result["available_elements"]
