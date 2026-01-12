"""FastMCP server for WebShop+ tool execution.

This module provides the WebShopMCPServer class that exposes shopping tools
(search, click, checkout) via the MCP protocol for use by shopping agents.

Tools return structured responses with element IDs that agents use for
subsequent interactions. The server is session-scoped - each assessment
session gets its own server instance with isolated state.
"""

from typing import Any, Protocol

from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

from .session_state import SessionState


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


class WebShopMCPServer:
    """Session-scoped MCP server for WebShop tools.

    This server provides three tools for shopping:
    - search(query): Search the product catalog
    - click(element_id): Click on an element from previous observation
    - checkout(): Complete purchase (terminal action)

    Each instance is bound to a SessionState that tracks cart, budget,
    turn count, and visible elements.

    Example:
        state = SessionState(
            session_id="abc123",
            goal="Find running shoes under $50",
            budget=50.0,
            constraints=["no synthetic"],
        )
        server = WebShopMCPServer(state)
        app = server.get_app()  # Mount in Starlette/FastAPI
    """

    def __init__(
        self,
        state: SessionState,
        webshop: WebShopInterface | None = None,
    ):
        """Initialize MCP server with session state.

        Args:
            state: SessionState instance for tracking session data.
            webshop: Optional WebShop environment interface. If not provided,
                a WebShopWrapper will be created lazily on first tool use.
        """
        self.state = state
        self._webshop = webshop
        self._webshop_initialized = False
        self.mcp = FastMCP(f"WebShop-{state.session_id}")
        self._register_tools()

    def _get_webshop(self) -> WebShopInterface:
        """Get or create the WebShop environment interface.

        Returns:
            WebShop environment interface.

        Raises:
            RuntimeError: If WebShop environment cannot be initialized.
        """
        if self._webshop is None:
            # Lazy import to avoid loading WebShop unless needed
            from ..webshop_wrapper import WebShopWrapper

            self._webshop = WebShopWrapper(mode="preview")

        if not self._webshop_initialized:
            # Reset the environment for this session
            self._webshop.reset(session=self.state.session_id)
            self._webshop_initialized = True

        return self._webshop

    def _terminal_max_turns(self) -> dict[str, Any]:
        """Return terminal response when max turns exceeded.

        Returns:
            Evaluation dict with terminated=True and score=0.2.
        """
        self.state.mark_completed("max_turns_exceeded")
        return {
            "terminated": True,
            "reason": "max_turns_exceeded",
            "turns_used": self.state.turn_count,
            "max_turns": self.state.max_turns,
            "cart": self.state.cart,
            "cart_total": self.state.get_cart_total(),
            "score": 0.2,
        }

    def _parse_search_results(self, html: str) -> list[dict[str, Any]]:
        """Parse search results HTML to extract product info.

        Args:
            html: HTML string from WebShop search results.

        Returns:
            List of product dicts with asin, name, price, and image.
        """
        soup = BeautifulSoup(html, "html.parser")
        products = []

        # Find product links
        product_links = soup.find_all(class_="product-link")

        for link in product_links:
            asin = link.get_text().strip().upper()

            # Find parent product container for more details
            parent = link.find_parent(class_="list-group-item")
            if parent:
                # Extract product title
                title_elem = parent.find("h4")
                name = title_elem.get_text().strip() if title_elem else asin

                # Extract price - look for price tag
                price_elem = parent.find("h5")
                price_text = price_elem.get_text().strip() if price_elem else "$0.00"

                # Parse price - handle ranges like "$10 to $20"
                price = self._parse_price(price_text)

                products.append({
                    "asin": asin,
                    "name": name,
                    "price": price,
                    "price_display": price_text,
                })
            else:
                products.append({
                    "asin": asin,
                    "name": asin,
                    "price": 0.0,
                    "price_display": "$0.00",
                })

        return products

    def _parse_price(self, price_text: str) -> float:
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

    def _show_product_page(self, element: dict[str, Any]) -> dict[str, Any]:
        """Show product detail page with options.

        Args:
            element: Product element from visible_elements.

        Returns:
            Product page response with options and actions.
        """
        product = element["data"]
        asin = element.get("asin", product.get("asin", ""))

        # Update page state
        self.state.current_page = "product_detail"
        self.state.visible_elements = {}

        # Get product details from WebShop if available
        webshop = self._get_webshop()
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

                self.state.visible_elements[eid] = {
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
        self.state.visible_elements["add_to_cart"] = {
            "type": "add_to_cart",
            "product": product,
            "asin": asin,
        }

        # Add "back to results" action
        self.state.visible_elements["back_to_results"] = {
            "type": "navigation",
            "action": "back",
        }

        # Get current selections display
        selected_display = {k: v for k, v in self.state.selected_options.items()}

        return {
            "page": "product_detail",
            "product": {
                "name": product.get("name", "Unknown"),
                "price": product.get("price", 0.0),
                "price_display": product.get("price_display", f"${product.get('price', 0):.2f}"),
                "asin": asin,
            },
            "options": options,
            "selected_options": selected_display,
            "actions": [
                {"id": "add_to_cart", "label": "Add to Cart"},
                {"id": "back_to_results", "label": "Back to Results"},
            ],
            "turn": self.state.turn_count,
            "turns_remaining": self.state.max_turns - self.state.turn_count,
            "budget": self.state.budget,
            "cart_total": self.state.get_cart_total(),
        }

    def _select_option(self, element: dict[str, Any]) -> dict[str, Any]:
        """Select a product option.

        Args:
            element: Option element from visible_elements.

        Returns:
            Updated state with selection confirmation.
        """
        option_type = element["option_type"]
        value = element["value"]

        # Record selection in state
        result = self.state.select_option(option_type, value)

        return {
            "page": "product_detail",
            "action": "option_selected",
            "option_type": option_type,
            "value": value,
            "selected_options": result["all_selections"],
            "turn": self.state.turn_count,
            "turns_remaining": self.state.max_turns - self.state.turn_count,
            "budget": self.state.budget,
            "cart_total": self.state.get_cart_total(),
        }

    def _add_to_cart(self, element: dict[str, Any]) -> dict[str, Any]:
        """Add current product to cart.

        Args:
            element: Add to cart element from visible_elements.

        Returns:
            Cart update confirmation.
        """
        product = element["product"]
        asin = element.get("asin", product.get("asin", ""))

        # Get actual price from WebShop
        webshop = self._get_webshop()
        price = webshop.product_prices.get(asin, product.get("price", 0.0))

        # Create product with current price
        cart_product = {
            "name": product.get("name", "Unknown"),
            "price": price,
            "asin": asin,
            "product_id": asin,
        }

        # Add to cart (this also clears selected_options)
        result = self.state.add_to_cart(cart_product)

        # Determine if still on product page or should show cart summary
        return {
            "page": "product_detail",
            "action": "added_to_cart",
            "added": result["added"],
            "cart_total": result["cart_total"],
            "cart_size": result["cart_size"],
            "budget": self.state.budget,
            "budget_remaining": self.state.budget - result["cart_total"],
            "over_budget": result["over_budget"],
            "warning": "Cart total exceeds budget!" if result["over_budget"] else None,
            "turn": self.state.turn_count,
            "turns_remaining": self.state.max_turns - self.state.turn_count,
        }

    def _navigate(self, element: dict[str, Any]) -> dict[str, Any]:
        """Handle navigation actions (next/prev page, back to results).

        Args:
            element: Navigation element from visible_elements.

        Returns:
            New page state after navigation.
        """
        action = element.get("action", "")
        webshop = self._get_webshop()

        if action == "next":
            # Navigate to next search results page
            result = webshop.step("click[next >]")
            return self._process_search_results_page(result.observation, "next_page")

        elif action == "prev":
            # Navigate to previous search results page
            result = webshop.step("click[< prev]")
            return self._process_search_results_page(result.observation, "prev_page")

        elif action == "back":
            # Back to search results - re-render last search
            self.state.current_page = "search_results"
            # We need to return to search results, but we don't have the last query
            # For now, return a message indicating the action
            return {
                "page": "search_results",
                "action": "navigated_back",
                "message": "Returned to search results. Use search() to find products.",
                "turn": self.state.turn_count,
                "turns_remaining": self.state.max_turns - self.state.turn_count,
                "budget": self.state.budget,
                "cart_total": self.state.get_cart_total(),
            }

        else:
            return {
                "error": f"Unknown navigation action '{action}'",
                "turn": self.state.turn_count,
            }

    def _process_search_results_page(
        self, html: str, nav_action: str
    ) -> dict[str, Any]:
        """Process search results page HTML after navigation.

        Args:
            html: HTML from WebShop after navigation.
            nav_action: The navigation action that was taken.

        Returns:
            Structured search results response.
        """
        # Update page state
        self.state.current_page = "search_results"
        self.state.visible_elements = {}

        # Parse products from HTML
        raw_products = self._parse_search_results(html)

        # Get prices from webshop
        webshop = self._get_webshop()
        webshop_prices = getattr(webshop, "product_prices", {})

        # Build structured response
        products = []
        for i, product in enumerate(raw_products):
            element_id = f"p{i + 1}"
            asin = product["asin"]
            price = webshop_prices.get(asin, product["price"])

            self.state.visible_elements[element_id] = {
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
            self.state.visible_elements["next_page"] = {
                "type": "navigation",
                "action": "next",
            }
            actions.append({"id": "next_page", "label": "Next Page"})

        if "< prev" in clickables:
            self.state.visible_elements["prev_page"] = {
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
            "turn": self.state.turn_count,
            "turns_remaining": self.state.max_turns - self.state.turn_count,
            "budget": self.state.budget,
            "cart_total": self.state.get_cart_total(),
        }

    def _register_tools(self) -> None:
        """Register all shopping tools with the MCP server."""
        # Store reference to self for use in closures
        server = self

        @self.mcp.tool()
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
            # Check turn limit first
            if server.state.increment_turn():
                return server._terminal_max_turns()

            # Record search in history
            server.state.history.append({
                "action": "search",
                "query": query,
                "turn": server.state.turn_count,
            })

            # Execute search via WebShop
            webshop = server._get_webshop()
            result = webshop.step(f"search[{query}]")

            # Update page state
            server.state.current_page = "search_results"
            server.state.visible_elements = {}

            # Parse HTML to get product info
            raw_products = server._parse_search_results(result.observation)

            # Get actual prices from webshop price dict if available
            webshop_prices = getattr(webshop, "product_prices", {})

            # Build structured response with element IDs
            products = []
            for i, product in enumerate(raw_products):
                element_id = f"p{i + 1}"
                asin = product["asin"]

                # Use actual price from webshop if available
                price = webshop_prices.get(asin, product["price"])

                # Store in visible elements for click() to use
                server.state.visible_elements[element_id] = {
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
                server.state.visible_elements["next_page"] = {
                    "type": "navigation",
                    "action": "next",
                }
                actions.append({"id": "next_page", "label": "Next Page"})

            if "< prev" in clickables:
                server.state.visible_elements["prev_page"] = {
                    "type": "navigation",
                    "action": "prev",
                }
                actions.append({"id": "prev_page", "label": "Previous Page"})

            return {
                "page": "search_results",
                "query": query,
                "products": products,
                "product_count": len(products),
                "actions": actions,
                "turn": server.state.turn_count,
                "turns_remaining": server.state.max_turns - server.state.turn_count,
                "budget": server.state.budget,
                "cart_total": server.state.get_cart_total(),
            }

        @self.mcp.tool()
        def click(element_id: str) -> dict[str, Any]:
            """Click on an element by its ID from previous observation.

            Args:
                element_id: ID from previous observation (e.g., "p1", "size_10")

            Returns:
                New page state with available element IDs, or error dict
                if element not found.
            """
            # Check turn limit first
            if server.state.increment_turn():
                return server._terminal_max_turns()

            # Validate element exists
            if element_id not in server.state.visible_elements:
                # Record failed click in history
                server.state.history.append({
                    "action": "click",
                    "element_id": element_id,
                    "error": "element_not_found",
                    "turn": server.state.turn_count,
                })
                return {
                    "error": f"Unknown element '{element_id}'",
                    "available_elements": list(server.state.visible_elements.keys()),
                    "turn": server.state.turn_count,
                    "turns_remaining": server.state.max_turns - server.state.turn_count,
                }

            element = server.state.visible_elements[element_id]

            # Record click in history
            server.state.history.append({
                "action": "click",
                "element_id": element_id,
                "element_type": element["type"],
                "turn": server.state.turn_count,
            })

            # Dispatch based on element type
            if element["type"] == "product":
                return server._show_product_page(element)
            elif element["type"] == "option":
                return server._select_option(element)
            elif element["type"] == "add_to_cart":
                return server._add_to_cart(element)
            elif element["type"] == "navigation":
                return server._navigate(element)
            else:
                return {
                    "error": f"Unknown element type '{element['type']}'",
                    "element_id": element_id,
                    "turn": server.state.turn_count,
                }

        @self.mcp.tool()
        def checkout() -> dict[str, Any]:
            """Complete purchase and end session. TERMINAL action.

            This is a terminal action - calling it ends the session and
            returns the final evaluation with success/failure and score.

            Returns:
                Evaluation dict with terminated=True, cart contents,
                total, budget comparison, and score.
            """
            # Placeholder - will be implemented in Stage 5
            return {
                "error": "Not implemented",
                "message": "checkout() will be implemented in Stage 5",
                "session_id": server.state.session_id,
            }

    def get_app(self) -> Any:
        """Return Starlette app for mounting.

        The returned app handles MCP protocol via streamable HTTP transport.
        Mount this at a path like "/mcp/{session_id}" in your main server.

        Returns:
            ASGI application that handles MCP requests.
        """
        return self.mcp.streamable_http_app()
