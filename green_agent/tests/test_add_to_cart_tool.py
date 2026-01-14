
import pytest
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add green_agent root to path so we can import src
GREEN_AGENT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(GREEN_AGENT_DIR))

from src.webshop_mcp.session_state import SessionState
from src.webshop_mcp.server import add_to_cart

# Mock get_current_state since it uses contextvars
@pytest.fixture
def mock_get_current_state():
    with patch("src.webshop_mcp.server.get_current_state") as mock:
        yield mock

@pytest.fixture
def mock_webshop_interface():
    with patch("src.webshop_mcp.server._get_webshop") as mock:
        webshop = MagicMock()
        webshop.product_prices = {"B001": 50.0}
        webshop.product_item_dict = {"B001": {"name": "Test Product"}}
        mock.return_value = webshop
        yield mock

def test_add_to_cart_success(mock_get_current_state, mock_webshop_interface):
    """Test successful add_to_cart execution."""
    # Setup state
    state = SessionState(
        session_id="test_session",
        goal="buy something",
        budget=100.0,
        constraints=[],
        max_turns=10
    )
    
    # Simulate being on a product page
    product_data = {"name": "Test Product", "price": 50.0, "asin": "B001"}
    state.visible_elements["add_to_cart"] = {
        "type": "add_to_cart",
        "product": product_data,
        "asin": "B001"
    }
    
    mock_get_current_state.return_value = state
    
    # Call the tool
    result = add_to_cart()
    
    # Verify results
    assert result["status"] == "added_to_cart"
    assert len(state.cart) == 1
    assert state.cart[0]["product_id"] == "B001"
    assert state.turn_count == 1
    
    # Verify history
    # The tool logs a "click" action (history[-2])
    # The state.add_to_cart method logs an "add_to_cart" action (history[-1])
    click_history = state.history[-2]
    assert click_history["action"] == "click" # Should log as click for consistency
    assert click_history["element_id"] == "add_to_cart"

def test_add_to_cart_failure_no_element(mock_get_current_state):
    """Test add_to_cart failure when not on product page."""
    # Setup state without visible elements
    state = SessionState(
        session_id="test_session",
        goal="buy something",
        budget=100.0,
        constraints=[],
        max_turns=10
    )
    
    mock_get_current_state.return_value = state
    
    # Call the tool
    result = add_to_cart()
    
    # Verify failure
    assert "error" in result
    assert "not available" in result["error"]
    assert state.turn_count == 1 # Should still increment turn
    
    # Verify history log
    last_history = state.history[-1]
    assert last_history["action"] == "add_to_cart"
    assert last_history["error"] == "action_not_available"

def test_add_to_cart_turn_limit(mock_get_current_state):
    """Test add_to_cart respects turn limits."""
    # Setup state at limit
    state = SessionState(
        session_id="test_session",
        goal="buy something",
        budget=100.0,
        constraints=[],
        max_turns=5
    )
    state.turn_count = 5
    
    mock_get_current_state.return_value = state
    
    # Call the tool
    result = add_to_cart()
    
    # Verify termination
    assert result.get("terminated") is True
    assert result.get("reason") == "max_turns_exceeded"
