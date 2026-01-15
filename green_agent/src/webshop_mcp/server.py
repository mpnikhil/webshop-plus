"""FastMCP server for WebShop+ tool execution.

This module provides a single global FastMCP server that handles shopping tools
(search, click, checkout) for all sessions. Session isolation is achieved via
a contextvar that tracks the current session_id.

Architecture:
- ONE FastMCP instance (mcp) handles all MCP requests
- Session state is stored in _session_states dict, keyed by session_id
- current_session_id contextvar is set by the HTTP router before each request
- Tools read the contextvar to get the current session's state

This approach is required because FastMCP's session_manager.run() must be
called from Starlette's lifespan context, not per-session.
"""

import asyncio
import contextvars
from typing import Any, Protocol

import structlog
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from .session_state import SessionState

logger = structlog.get_logger()

# =============================================================================
# Session Context Management
# =============================================================================

# Contextvar for tracking current session_id during request handling
current_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_session_id"
)

# Global registry of session states (session_id -> SessionState)
_session_states: dict[str, SessionState] = {}

# Global registry of WebShop interfaces (session_id -> WebShopInterface)
_webshop_interfaces: dict[str, "WebShopInterface"] = {}

# Global registry of completion events (session_id -> asyncio.Event)
_completion_events: dict[str, asyncio.Event] = {}

# Global registry of final results (session_id -> dict)
_final_results: dict[str, dict[str, Any]] = {}


def register_session(
    session_id: str,
    state: SessionState,
    webshop: "WebShopInterface | None" = None,
) -> None:
    """Register a new session with the global MCP server.

    Args:
        session_id: Unique session identifier.
        state: SessionState instance for this session.
        webshop: Optional WebShop interface (for testing).
    """
    _session_states[session_id] = state
    if webshop is not None:
        _webshop_interfaces[session_id] = webshop
    _completion_events[session_id] = asyncio.Event()
    logger.info("Session registered", session_id=session_id)


def unregister_session(session_id: str) -> None:
    """Unregister a session from the global MCP server.

    Args:
        session_id: Session identifier to remove.
    """
    _session_states.pop(session_id, None)
    _webshop_interfaces.pop(session_id, None)
    _completion_events.pop(session_id, None)
    _final_results.pop(session_id, None)
    logger.info("Session unregistered", session_id=session_id)


def get_session_state(session_id: str) -> SessionState | None:
    """Get session state by ID.

    Args:
        session_id: Session identifier.

    Returns:
        SessionState if found, None otherwise.
    """
    return _session_states.get(session_id)


def is_session_registered(session_id: str) -> bool:
    """Check if a session is registered.

    Args:
        session_id: Session identifier.

    Returns:
        True if session exists.
    """
    result = session_id in _session_states
    logger.debug(
        "is_session_registered check",
        session_id=session_id,
        result=result,
        all_sessions=list(_session_states.keys()),
    )
    return result


def get_current_state() -> SessionState:
    """Get the SessionState for the current request context.

    Uses the current_session_id contextvar to look up the right state.

    Returns:
        SessionState for current session.

    Raises:
        ValueError: If no session is set or session not found.
    """
    try:
        session_id = current_session_id.get()
    except LookupError:
        raise ValueError("No session ID set in current context")

    state = _session_states.get(session_id)
    if state is None:
        raise ValueError(f"Session '{session_id}' not found")

    return state


def _get_webshop(session_id: str) -> "WebShopInterface":
    """Get or create WebShop interface for a session.

    Args:
        session_id: Session identifier.

    Returns:
        WebShop interface for the session.
    """
    if session_id in _webshop_interfaces:
        return _webshop_interfaces[session_id]

    # Lazy import to avoid loading WebShop unless needed
    from ..webshop_wrapper import WebShopWrapper

    webshop = WebShopWrapper(mode="preview")
    webshop.reset(session=session_id)
    _webshop_interfaces[session_id] = webshop
    return webshop


# =============================================================================
# WebShop Interface Protocol
# =============================================================================


class WebShopInterface(Protocol):
    """Protocol for WebShop environment interface."""

    def reset(self, session: str | None = None) -> str:
        """Reset the environment and start a new session."""
        ...

    def step(self, action: str) -> Any:
        """Take an action in the environment."""
        ...

    def get_available_actions(self) -> dict:
        """Get available actions at current state."""
        ...

    @property
    def product_prices(self) -> dict[str, float]:
        """Get product prices by ASIN."""
        ...

    @property
    def product_item_dict(self) -> dict[str, dict]:
        """Get product info by ASIN."""
        ...


# =============================================================================
# Global FastMCP Server
# =============================================================================

# Disable DNS rebinding protection for MCP in trusted environment
# MCP sessions are already isolated via unique session IDs
# Purple agents are authenticated via A2A protocol before receiving MCP URIs
transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=False
)

mcp = FastMCP(
    "WebShop MCP Server",
    transport_security=transport_security,
)

# Add logging for MCP server initialization
logger.info("FastMCP server created", transport_security_enabled=transport_security.enable_dns_rebinding_protection)


# =============================================================================
# Helper Functions for Tools
# =============================================================================


def _terminal_max_turns(state: SessionState) -> dict[str, Any]:
    """Return terminal response when max turns exceeded.

    Args:
        state: Current session state.

    Returns:
        Evaluation dict with terminated=True and score=0.2.
    """
    state.mark_completed("max_turns_exceeded")
    result = {
        "terminated": True,
        "reason": "max_turns_exceeded",
        "turns_used": state.turn_count,
        "max_turns": state.max_turns,
        "cart": state.cart,
        "cart_total": state.get_cart_total(),
        "score": 0.2,
    }
    _signal_completion(state.session_id, result)
    return result


def _signal_completion(session_id: str, result: dict[str, Any]) -> None:
    """Signal that a session has completed (internal).

    Args:
        session_id: Session that completed.
        result: Final evaluation result.
    """
    _final_results[session_id] = result
    if session_id in _completion_events:
        _completion_events[session_id].set()


def signal_completion(session_id: str, result: dict[str, Any]) -> None:
    """Signal that a session has completed (public API).

    Args:
        session_id: Session that completed.
        result: Final evaluation result.
    """
    _signal_completion(session_id, result)


async def wait_for_completion(
    session_id: str, timeout: float | None = None
) -> dict[str, Any]:
    """Wait for a session to complete.

    Args:
        session_id: Session to wait for.
        timeout: Maximum wait time in seconds.

    Returns:
        Final evaluation result.

    Raises:
        asyncio.TimeoutError: If timeout reached.
        ValueError: If session not found.
    """
    if session_id not in _completion_events:
        raise ValueError(f"Session '{session_id}' not found")

    # Check if already completed
    if session_id in _final_results:
        return _final_results[session_id]

    # Wait for completion
    await asyncio.wait_for(_completion_events[session_id].wait(), timeout=timeout)
    return _final_results.get(session_id, {})


def get_final_result(session_id: str) -> dict[str, Any] | None:
    """Get final result for a completed session.

    Args:
        session_id: Session identifier.

    Returns:
        Final result if completed, None otherwise.
    """
    return _final_results.get(session_id)


def is_session_completed(session_id: str) -> bool:
    """Check if session is completed.

    Args:
        session_id: Session identifier.

    Returns:
        True if session has completed.
    """
    # Check if we have a final result
    if session_id in _final_results:
        return True
    # Also check the state's completed attribute
    state = _session_states.get(session_id)
    if state and state.completed:
        return True
    return False


def _parse_search_results(text: str) -> list[dict[str, Any]]:
    """Parse search results from WebShop to extract product info.

    WebShop returns results in a [SEP]-delimited format like:
    Instruction: [SEP] ... [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP]
    Next > [SEP] B09N9T673Q [SEP] Product Name [SEP] $100.0 [SEP] ...

    The pattern repeats: ASIN [SEP] Name [SEP] Price [SEP]

    Args:
        text: [SEP]-separated string from WebShop search results.

    Returns:
        List of product dicts with asin, name, price, and price_display.
    """
    import re

    products = []

    # Split by [SEP]
    parts = [p.strip() for p in text.split("[SEP]") if p.strip()]

    # Find where products start (after navigation elements)
    # Products have pattern: ASIN (like B09N9T673Q), Name, Price
    # ASIN pattern: starts with B and has 10 alphanumeric chars
    asin_pattern = re.compile(r"^B[A-Z0-9]{9,}$")
    price_pattern = re.compile(r"^\$[\d.,]+(?:\s+to\s+\$[\d.,]+)?$")

    i = 0
    while i < len(parts):
        part = parts[i]

        # Check if this looks like an ASIN
        if asin_pattern.match(part):
            asin = part

            # Next part should be product name
            name = parts[i + 1] if i + 1 < len(parts) else asin

            # Part after name should be price
            price_text = "$0.00"
            if i + 2 < len(parts):
                potential_price = parts[i + 2]
                if price_pattern.match(potential_price):
                    price_text = potential_price

            price = _parse_price(price_text)

            products.append({
                "asin": asin,
                "name": name,
                "price": price,
                "price_display": price_text,
            })

            # Skip to after the price (ASIN + name + price = 3 parts)
            i += 3
        else:
            i += 1

    return products


def _parse_price(price_text: str) -> float:
    """Parse price string to float.

    Handles formats like "$10.99", "$10 to $20" (takes lower bound).

    Args:
        price_text: Price string from HTML.

    Returns:
        Price as float.
    """
    import re

    # Find all price values in the string
    prices = re.findall(r"\$?([\d,]+\.?\d*)", price_text.replace(",", ""))
    if prices:
        try:
            return float(prices[0])
        except ValueError:
            return 0.0
    return 0.0


def _show_product_page(
    state: SessionState, element: dict[str, Any]
) -> dict[str, Any]:
    """Show product detail page with options.

    Args:
        state: Current session state.
        element: Product element from visible_elements.

    Returns:
        Product page response with options and actions.
    """
    product = element["data"]
    asin = element.get("asin", product.get("asin", ""))

    # Update page state
    state.current_page = "product_detail"
    state.visible_elements = {}

    # Get product details from WebShop if available
    webshop = _get_webshop(state.session_id)
    product_info = webshop.product_item_dict.get(asin, {})

    # Build options from product attributes
    options = []
    option_groups = {}

    # Extract options from product_info
    if product_info:
        # WebShop products have options in various forms
        for key in ["size", "color", "style", "pack"]:
            if key in product_info:
                values = product_info[key]
                if isinstance(values, list):
                    option_groups[key] = values
                elif isinstance(values, str):
                    option_groups[key] = [values]

        # Also check for 'options' dict
        if "options" in product_info:
            for opt_type, opt_values in product_info["options"].items():
                if isinstance(opt_values, list):
                    option_groups[opt_type] = opt_values

    # Create element IDs for each option
    for opt_type, values in option_groups.items():
        for value in values:
            # Create clean element ID
            eid = f"{opt_type}_{value}".lower().replace(" ", "_").replace("-", "_")
            eid = "".join(c for c in eid if c.isalnum() or c == "_")

            state.visible_elements[eid] = {
                "type": "option",
                "option_type": opt_type,
                "value": value,
                "product_asin": asin,
            }
            options.append({
                "id": eid,
                "type": opt_type,
                "label": str(value),
            })

    # Add "add to cart" action
    state.visible_elements["add_to_cart"] = {
        "type": "add_to_cart",
        "product": product,
        "asin": asin,
    }

    # Add "back to results" action
    state.visible_elements["back_to_results"] = {
        "type": "navigation",
        "action": "back",
    }

    # Get current selections display
    selected_display = {k: v for k, v in state.selected_options.items()}

    # Truncate name for compact response
    short_name = product.get("name", "Unknown")[:50]

    # Prepare detailed product data for fairness (agent needs same data as evaluator)
    attributes = product_info.get("Attributes", []) if product_info else []
    catalog_attributes: dict[str, Any] = {}
    description = ""
    category = ""

    if product_info:
        category = product_info.get("category", "")

        if isinstance(product_info.get("attributes"), dict):
            catalog_attributes.update(product_info["attributes"])

        if isinstance(attributes, list):
            catalog_attributes["attributes_list"] = attributes

        if isinstance(product_info.get("description"), str):
            description = product_info["description"][:400]

    # Keep response compact for LLM context window, but enriched with details
    response = {
        "page": state.current_page,
        "product": short_name,
        "price": product.get("price", 0.0),
        "attributes": attributes,
        "category": category,
        "catalog_attributes": catalog_attributes,
        "description": description,
        "options": options,
        "selected_options": selected_display,
        "actions": ["add_to_cart", "back_to_results"],
        "hint": "Click add_to_cart to add this product, then checkout to complete.",
        "turn": state.turn_count,
        "turns_remaining": state.max_turns - state.turn_count,
        "budget": state.budget,
        "cart_total": state.get_cart_total(),
    }

    logger.info(
        "MCP click(product) response",
        session_id=state.session_id,
        product=short_name[:30],
        price=product.get("price"),
    )

    return response


def _select_option(state: SessionState, element: dict[str, Any]) -> dict[str, Any]:
    """Select a product option.

    Args:
        state: Current session state.
        element: Option element from visible_elements.

    Returns:
        Updated state with selection confirmation.
    """
    option_type = element["option_type"]
    value = element["value"]

    # Record selection in state
    result = state.select_option(option_type, value)

    return {
        "page": "product_detail",
        "action": "option_selected",
        "option_type": option_type,
        "value": value,
        "selected_options": result["all_selections"],
        "turn": state.turn_count,
        "turns_remaining": state.max_turns - state.turn_count,
        "budget": state.budget,
        "cart_total": state.get_cart_total(),
    }


def _add_to_cart(state: SessionState, element: dict[str, Any]) -> dict[str, Any]:
    """Add current product to cart.

    Args:
        state: Current session state.
        element: Add to cart element from visible_elements.

    Returns:
        Cart update confirmation.
    """
    product = element["product"]
    asin = element.get("asin", product.get("asin", ""))

    # Get actual price and product info from WebShop
    webshop = _get_webshop(state.session_id)
    price = webshop.product_prices.get(asin, product.get("price", 0.0))

    # Get catalog attributes from WebShop for evaluation matching
    product_info = webshop.product_item_dict.get(asin, {})
    catalog_attributes = {}

    # Extract relevant attributes for evaluation (category, attributes, etc.)
    if product_info:
        # Get product category for category matching
        if "category" in product_info:
            catalog_attributes["category"] = product_info["category"]

        # Get product Attributes list (e.g., ["gym workout", "running shorts"])
        # This is the same data purple agent sees on product detail page
        if "Attributes" in product_info:
            attrs_list = product_info["Attributes"]
            if isinstance(attrs_list, list):
                # Convert list to dict for evaluator matching
                for attr in attrs_list:
                    if isinstance(attr, str):
                        # Split compound attributes like "gym workout" into searchable keys
                        catalog_attributes[attr] = attr

        # Get product attributes dict (if exists - contains things like "sole", "care", etc.)
        if "attributes" in product_info:
            catalog_attributes.update(product_info["attributes"])

        # Also include raw attribute strings from product data
        for key in ["instruction", "name", "description"]:
            if key in product_info and isinstance(product_info[key], str):
                catalog_attributes[f"_raw_{key}"] = product_info[key]

    # Create product with current price and catalog attributes
    cart_product = {
        "name": product.get("name", "Unknown"),
        "price": price,
        "asin": asin,
        "product_id": asin,
        "catalog_attributes": catalog_attributes,
    }

    # Add to cart (this also clears selected_options)
    result = state.add_to_cart(cart_product)

    # Keep response compact for LLM context window
    response = {
        "page": state.current_page,
        "status": "added_to_cart",
        "cart_total": result["cart_total"],
        "budget": state.budget,
        "hint": "Now call checkout() to complete purchase.",
        "turn": state.turn_count,
        "turns_remaining": state.max_turns - state.turn_count,
    }

    logger.info(
        "MCP add_to_cart response",
        session_id=state.session_id,
        product=result["added"][:30] if result["added"] else "None",
        cart_size=result["cart_size"],
        cart_total=result["cart_total"],
        over_budget=result["over_budget"],
    )

    return response


def _navigate(state: SessionState, element: dict[str, Any]) -> dict[str, Any]:
    """Handle navigation actions (next/prev page, back to results).

    Args:
        state: Current session state.
        element: Navigation element from visible_elements.

    Returns:
        New page state after navigation.
    """
    action = element.get("action", "")
    webshop = _get_webshop(state.session_id)

    if action == "next":
        # Navigate to next search results page
        result = webshop.step("click[next >]")
        return _process_search_results_page(state, result.observation, "next_page")

    elif action == "prev":
        # Navigate to previous search results page
        result = webshop.step("click[< prev]")
        return _process_search_results_page(state, result.observation, "prev_page")

    elif action == "back":
        # Back to search results - re-render last search
        state.current_page = "search_results"
        return {
            "page": "search_results",
            "action": "navigated_back",
            "message": "Returned to search results. Use search() to find products.",
            "turn": state.turn_count,
            "turns_remaining": state.max_turns - state.turn_count,
            "budget": state.budget,
            "cart_total": state.get_cart_total(),
        }

    else:
        return {
            "error": f"Unknown navigation action '{action}'",
            "turn": state.turn_count,
        }


def _process_search_results_page(
    state: SessionState, html: str, nav_action: str
) -> dict[str, Any]:
    """Process search results page HTML after navigation.

    Args:
        state: Current session state.
        html: HTML from WebShop after navigation.
        nav_action: The navigation action that was taken.

    Returns:
        Structured search results response.
    """
    # Update page state
    state.current_page = "search_results"
    state.visible_elements = {}

    # Parse products from HTML
    raw_products = _parse_search_results(html)

    # Get prices from webshop
    webshop = _get_webshop(state.session_id)
    webshop_prices = getattr(webshop, "product_prices", {})

    # Build structured response
    products = []
    for i, product in enumerate(raw_products):
        element_id = f"p{i + 1}"
        asin = product["asin"]
        price = webshop_prices.get(asin, product["price"])

        state.visible_elements[element_id] = {
            "type": "product",
            "asin": asin,
            "data": {
                "asin": asin,
                "name": product["name"],
                "price": price,
                "price_display": product.get("price_display", f"${price:.2f}"),
            },
        }

        products.append({
            "id": element_id,
            "name": product["name"],
            "price": price,
            "price_display": product.get("price_display", f"${price:.2f}"),
        })

    # Check for pagination actions
    actions = []
    available = webshop.get_available_actions()
    clickables = [c.lower() for c in available.get("clickables", [])]

    if "next >" in clickables:
        state.visible_elements["next_page"] = {
            "type": "navigation",
            "action": "next",
        }
        actions.append({"id": "next_page", "label": "Next Page"})

    if "< prev" in clickables:
        state.visible_elements["prev_page"] = {
            "type": "navigation",
            "action": "prev",
        }
        actions.append({"id": "prev_page", "label": "Previous Page"})

    return {
        "page": "search_results",
        "action": nav_action,
        "products": products,
        "product_count": len(products),
        "actions": actions,
        "turn": state.turn_count,
        "turns_remaining": state.max_turns - state.turn_count,
        "budget": state.budget,
        "cart_total": state.get_cart_total(),
    }


# =============================================================================
# MCP Tools
# =============================================================================


@mcp.tool()
def search(query: str) -> dict[str, Any]:
    """Search the store catalog for products.

    Args:
        query: Search terms (e.g., "running shoes", "blue cotton shirt")

    Returns:
        Structured results with clickable element IDs:
        - page: Current page type ("search_results")
        - query: The search query used
        - products: List of products with id, name, price
        - actions: Available navigation actions
        - turn: Current turn number
        - budget: Remaining budget info
    """
    logger.info("search() tool called", query=query)

    try:
        state = get_current_state()
    except Exception as e:
        logger.exception("Failed to get current state", error=str(e), query=query)
        raise

    logger.info("Got session state", session_id=state.session_id)

    # Check turn limit first
    if state.increment_turn():
        return _terminal_max_turns(state)

    # Record search in history
    state.history.append({
        "action": "search",
        "query": query,
        "turn": state.turn_count,
    })

    # Execute search via WebShop
    try:
        logger.info("About to call _get_webshop", session_id=state.session_id)
        webshop = _get_webshop(state.session_id)
        logger.info("Got WebShop interface, calling step()")
        result = webshop.step(f"search[{query}]")
        logger.info("WebShop step() completed", observation_len=len(result.observation) if hasattr(result, 'observation') else 0)
    except Exception as e:
        logger.exception("Error during WebShop search", error=str(e), query=query, session_id=state.session_id)
        raise

    # Update page state
    state.current_page = "search_results"
    state.visible_elements = {}

    # Parse HTML to get product info
    raw_products = _parse_search_results(result.observation)

    # Get actual prices from webshop price dict if available
    webshop_prices = getattr(webshop, "product_prices", {})

    # Build structured response with element IDs
    # Limit to 5 products to keep response compact for LLM context
    products = []
    for i, product in enumerate(raw_products[:5]):
        element_id = f"p{i + 1}"
        asin = product["asin"]

        # Use actual price from webshop if available
        price = webshop_prices.get(asin, product["price"])

        # Store in visible elements for click() to use
        state.visible_elements[element_id] = {
            "type": "product",
            "asin": asin,
            "data": {
                "asin": asin,
                "name": product["name"],
                "price": price,
                "price_display": product.get("price_display", f"${price:.2f}"),
            },
        }

        # Truncate name to 50 chars for compact response
        short_name = product["name"][:50] + "..." if len(product["name"]) > 50 else product["name"]
        products.append({
            "id": element_id,
            "name": short_name,
            "price": price,
        })

    # Check for pagination actions
    actions = []
    available = webshop.get_available_actions()
    clickables = [c.lower() for c in available.get("clickables", [])]

    if "next >" in clickables:
        state.visible_elements["next_page"] = {
            "type": "navigation",
            "action": "next",
        }
        actions.append({"id": "next_page", "label": "Next Page"})

    if "< prev" in clickables:
        state.visible_elements["prev_page"] = {
            "type": "navigation",
            "action": "prev",
        }
        actions.append({"id": "prev_page", "label": "Previous Page"})

    # Keep response compact for LLM context window
    # Strong action directive for LLM
    if products:
        first_affordable = next((p for p in products if p["price"] <= state.budget), products[0])
        next_action = f"NEXT: click(\"{first_affordable['id']}\") - DO NOT search again!"
    else:
        next_action = "No products found. Try a different search term."

    result = {
        "page": state.current_page,
        "products": products,
        "available_actions": actions,
        "next_action": next_action,
        "turn": state.turn_count,
        "turns_remaining": state.max_turns - state.turn_count,
        "budget": state.budget,
        "cart_total": state.get_cart_total(),
    }

    logger.info(
        "MCP search() response",
        session_id=state.session_id,
        query=query,
        product_count=len(products),
        turn=state.turn_count,
    )

    return result


@mcp.tool()
def click(element_id: str) -> dict[str, Any]:
    """Click on an element by its ID from previous observation.

    Args:
        element_id: ID from previous observation (e.g., "p1", "size_10")

    Returns:
        New page state with available element IDs, or error dict
        if element not found.
    """
    state = get_current_state()

    # Check turn limit first
    if state.increment_turn():
        return _terminal_max_turns(state)

    # Validate element exists
    if element_id not in state.visible_elements:
        # Record failed click in history
        state.history.append({
            "action": "click",
            "element_id": element_id,
            "error": "element_not_found",
            "turn": state.turn_count,
        })
        return {
            "error": f"Unknown element '{element_id}'",
            "available_elements": list(state.visible_elements.keys()),
            "turn": state.turn_count,
            "turns_remaining": state.max_turns - state.turn_count,
        }

    element = state.visible_elements[element_id]

    # Record click in history
    history_record = {
        "action": "click",
        "element_id": element_id,
        "element_type": element["type"],
        "turn": state.turn_count,
    }

    # For product clicks, also store the ASIN so evaluator can count unique products
    if element["type"] == "product":
        asin = element.get("asin") or element.get("data", {}).get("asin", "")
        if asin:
            history_record["product_asin"] = asin

    state.history.append(history_record)

    # Dispatch based on element type
    if element["type"] == "product":
        return _show_product_page(state, element)
    elif element["type"] == "option":
        return _select_option(state, element)
    elif element["type"] == "add_to_cart":
        return _add_to_cart(state, element)
    elif element["type"] == "navigation":
        return _navigate(state, element)
    else:
        return {
            "error": f"Unknown element type '{element['type']}'",
            "element_id": element_id,
            "turn": state.turn_count,
        }


@mcp.tool()
def add_to_cart() -> dict[str, Any]:
    """Add the current product to cart.

    Use this when viewing a product detail page to add it to your cart.
    This is equivalent to click("add_to_cart").

    Returns:
        Cart update confirmation or error if not on product page.
    """
    state = get_current_state()

    # Check turn limit first
    if state.increment_turn():
        return _terminal_max_turns(state)

    # Validate action exists
    if "add_to_cart" not in state.visible_elements:
        # Record failed action in history
        state.history.append({
            "action": "add_to_cart",
            "error": "action_not_available",
            "turn": state.turn_count,
        })
        return {
            "error": "add_to_cart action not available. You must be on a product page.",
            "page": state.current_page,
            "turn": state.turn_count,
            "turns_remaining": state.max_turns - state.turn_count,
        }

    element = state.visible_elements["add_to_cart"]

    # Record action in history
    state.history.append({
        "action": "click",  # Log as click to match evaluation expectations
        "element_id": "add_to_cart",
        "element_type": "add_to_cart",
        "turn": state.turn_count,
        "product_asin": element.get("asin") or element.get("product", {}).get("asin", "")
    })

    return _add_to_cart(state, element)


@mcp.tool()
def checkout() -> dict[str, Any]:
    """Complete purchase and end session. TERMINAL action.

    This is a terminal action - calling it ends the session and
    returns the final evaluation with success/failure and score.

    Scoring:
    - Empty cart: score = 0.0 (failure_reason: "empty_cart")
    - Over budget: score = 0.3 (failure_reason: "budget_exceeded")
    - Success (items in cart, within budget): score = 1.0

    Returns:
        Evaluation dict with:
        - terminated: Always True (this is a terminal action)
        - reason: "checkout"
        - cart: List of items in cart
        - total: Total cart amount
        - budget: The budget constraint
        - turns_used: Number of turns used
        - success: Boolean indicating if purchase was successful
        - score: Numeric score (0.0, 0.3, or 1.0)
        - failure_reason: Only present if success is False
    """
    state = get_current_state()

    # Mark session as completed
    state.mark_completed("checkout")

    # Calculate total
    total = state.get_cart_total()

    # Build base evaluation
    evaluation: dict[str, Any] = {
        "terminated": True,
        "reason": "checkout",
        "session_id": state.session_id,
        "cart": state.cart,
        "cart_size": len(state.cart),
        "total": total,
        "budget": state.budget,
        "turns_used": state.turn_count,
        "max_turns": state.max_turns,
    }

    # Determine success/failure and score
    if not state.cart:
        # Empty cart - worst outcome
        evaluation["success"] = False
        evaluation["failure_reason"] = "empty_cart"
        evaluation["score"] = 0.0
    elif total > state.budget:
        # Over budget - partial failure
        evaluation["success"] = False
        evaluation["failure_reason"] = "budget_exceeded"
        evaluation["over_budget_by"] = total - state.budget
        evaluation["score"] = 0.3
    else:
        # Success - items in cart and within budget
        evaluation["success"] = True
        evaluation["budget_remaining"] = state.budget - total
        evaluation["score"] = 1.0

    # Add history summary for evaluation context
    evaluation["history_length"] = len(state.history)

    logger.info(
        "MCP checkout() response",
        session_id=state.session_id,
        success=evaluation.get("success"),
        score=evaluation.get("score"),
        cart_size=len(state.cart),
        total=total,
        budget=state.budget,
        turns_used=state.turn_count,
    )

    # Signal completion to any waiters
    _signal_completion(state.session_id, evaluation)

    return evaluation


@mcp.tool()
def view_cart() -> dict[str, Any]:
    """View current cart contents with item indices for removal.

    Use this to see what's in your cart before checkout or to identify
    items to remove with remove_from_cart().

    Returns:
        - items: List of cart items with index, name, price, options
        - cart_total: Total price of all items
        - budget: Budget constraint
        - budget_remaining: How much budget is left
        - turn: Current turn number
    """
    state = get_current_state()

    # Check turn limit
    if state.increment_turn():
        return _terminal_max_turns(state)

    # Record in history
    state.history.append({
        "action": "view_cart",
        "turn": state.turn_count,
    })

    # Build indexed item list
    items = []
    for i, item in enumerate(state.cart):
        items.append({
            "index": i,
            "name": item.get("name", "Unknown"),
            "price": item.get("price", 0.0),
            "options": item.get("options", {}),
            "product_id": item.get("product_id", ""),
            "catalog_attributes": item.get("catalog_attributes", {}),
        })

    total = state.get_cart_total()

    logger.info(
        "MCP view_cart() response",
        session_id=state.session_id,
        cart_size=len(items),
        cart_total=total,
    )

    return {
        "items": items,
        "cart_total": total,
    }


@mcp.tool()
def remove_from_cart(item_index: int) -> dict[str, Any]:
    """Remove an item from the cart by its index.

    Use view_cart() first to see item indices.

    Args:
        item_index: Index of item to remove (0-based, from view_cart)

    Returns:
        - removed: Name of removed item
        - cart_total: Updated total
        - cart_size: Number of items remaining
        - error: Only present if removal failed
    """
    state = get_current_state()

    # Check turn limit
    if state.increment_turn():
        return _terminal_max_turns(state)

    # Attempt removal
    result = state.remove_from_cart(item_index)

    if "error" in result:
        logger.warning(
            "MCP remove_from_cart() failed",
            session_id=state.session_id,
            item_index=item_index,
            error=result["error"],
        )
        return {
            "error": result["error"],
            "cart_size": result["cart_size"],
        }

    logger.info(
        "MCP remove_from_cart() response",
        session_id=state.session_id,
        removed=result["removed"],
        cart_size=result["cart_size"],
        cart_total=result["cart_total"],
    )

    return {
        "removed": result["removed"],
        "cart_total": result["cart_total"],
        "cart_size": result["cart_size"],
    }


# =============================================================================
# App Factory
# =============================================================================


def get_mcp_app() -> Any:
    """Get the global MCP app for mounting.

    Returns:
        Starlette app that handles MCP requests at /mcp endpoint.
    """
    return mcp.streamable_http_app()
