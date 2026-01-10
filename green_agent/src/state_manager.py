"""
State Manager for WebShop+ green agent.

This module provides session tracking, cart parsing, and agent memory management.
It coordinates state across multiple assessment sessions and maintains agent
preferences for multi-session tasks like preference_memory.
"""

import re
import uuid
from typing import Optional

from bs4 import BeautifulSoup

from .models import (
    AgentMemory,
    CartItem,
    CartItemSetup,
    CartState,
    SessionState,
    SessionSummary,
)


class StateManager:
    """
    Manages session state, cart parsing, and agent memory for WebShop+ assessments.

    The StateManager is responsible for:
    - Creating and tracking assessment sessions
    - Recording actions and observations within sessions
    - Parsing cart state from WebShop HTML observations
    - Maintaining agent memory across sessions (for preference recall)
    - Injecting cart state for error recovery tasks
    """

    def __init__(self) -> None:
        """Initialize the StateManager with empty session and memory stores."""
        self._sessions: dict[str, SessionState] = {}
        self._agent_memories: dict[str, AgentMemory] = {}
        self._injected_cart: Optional[CartState] = None

    def create_session(self, task_id: str, agent_id: str = "") -> SessionState:
        """
        Create a new assessment session.

        Args:
            task_id: The ID of the task being assessed.
            agent_id: The ID of the agent being assessed (optional).

        Returns:
            A new SessionState with a unique session_id.
        """
        session_id = str(uuid.uuid4())
        session = SessionState(
            session_id=session_id,
            task_id=task_id,
            agent_id=agent_id,
        )

        # If there's an injected cart for error recovery, apply it
        if self._injected_cart is not None:
            session.cart = self._injected_cart.model_copy(deep=True)
            self._injected_cart = None  # Clear after use

        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> SessionState:
        """
        Get a session by its ID.

        Args:
            session_id: The unique identifier of the session.

        Returns:
            The SessionState for the given ID.

        Raises:
            KeyError: If the session ID is not found.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        return self._sessions[session_id]

    def record_action(
        self,
        session_id: str,
        action: str,
        observation: str,
        reward: float = 0.0,
    ) -> None:
        """
        Record an action and its resulting observation in a session.

        Args:
            session_id: The session to record the action in.
            action: The action taken by the agent.
            observation: The observation/result of the action.
            reward: The reward received (default 0.0).

        Raises:
            KeyError: If the session ID is not found.
        """
        session = self.get_session(session_id)
        session.record_action(action, observation, reward)

        # Try to parse cart from observation and update session cart
        try:
            cart = self.parse_cart_from_observation(observation)
            if cart.items:  # Only update if we found cart items
                session.cart = cart
        except Exception:
            # Cart parsing may fail for non-cart pages, that's okay
            pass

    def parse_cart_from_observation(self, observation: str) -> CartState:
        """
        Parse cart contents from a WebShop HTML observation.

        WebShop cart observations contain HTML with cart items. This method
        extracts product information from various possible HTML structures.

        Args:
            observation: The HTML observation string from WebShop.

        Returns:
            A CartState with parsed items (may be empty if no cart found).
        """
        cart = CartState()

        if not observation or not observation.strip():
            return cart

        soup = BeautifulSoup(observation, "html.parser")

        # Try multiple possible cart container selectors
        cart_containers = (
            soup.find_all("div", class_="cart")
            or soup.find_all("div", class_="cart-container")
            or soup.find_all("div", id="cart")
            or soup.find_all("table", class_="cart")
        )

        for container in cart_containers:
            items = self._parse_cart_container(container)
            for item in items:
                cart.add_item(item)

        # Also try parsing individual item divs if no cart container found
        if not cart.items:
            items = soup.find_all("div", class_="item")
            for item_div in items:
                item = self._parse_item_div(item_div)
                if item:
                    cart.add_item(item)

        # Try parsing cart rows in table format
        if not cart.items:
            rows = soup.find_all("tr", class_="cart-item")
            for row in rows:
                item = self._parse_table_row(row)
                if item:
                    cart.add_item(item)

        return cart

    def _parse_cart_container(self, container) -> list[CartItem]:
        """Parse cart items from a cart container element."""
        items = []

        # Find all item divs within the container
        item_divs = container.find_all("div", class_="item") or container.find_all(
            "div", class_="cart-item"
        )

        for item_div in item_divs:
            item = self._parse_item_div(item_div)
            if item:
                items.append(item)

        return items

    def _parse_item_div(self, item_div) -> Optional[CartItem]:
        """Parse a single cart item from an item div element."""
        try:
            # Try various selectors for product name
            name_elem = (
                item_div.find("span", class_="product-name")
                or item_div.find("div", class_="product-name")
                or item_div.find("a", class_="product-name")
                or item_div.find("h4")
                or item_div.find("h3")
            )
            product_name = name_elem.get_text(strip=True) if name_elem else ""

            if not product_name:
                return None

            # Try various selectors for price
            price_elem = (
                item_div.find("span", class_="price")
                or item_div.find("div", class_="price")
                or item_div.find("span", class_="product-price")
            )
            price_text = price_elem.get_text(strip=True) if price_elem else "0"
            price = self._parse_price(price_text)

            # Try various selectors for quantity
            qty_elem = (
                item_div.find("span", class_="quantity")
                or item_div.find("input", class_="quantity")
                or item_div.find("div", class_="quantity")
            )
            if qty_elem:
                if qty_elem.name == "input":
                    quantity = int(qty_elem.get("value", 1))
                else:
                    qty_text = qty_elem.get_text(strip=True)
                    quantity = self._parse_quantity(qty_text)
            else:
                quantity = 1

            # Try to get product ID from data attribute or link
            product_id = item_div.get("data-product-id", "")
            if not product_id:
                link = item_div.find("a")
                if link and link.get("href"):
                    # Extract product ID from URL like /product/ABC123
                    href = link.get("href", "")
                    match = re.search(r"/product/([^/\?]+)", href)
                    if match:
                        product_id = match.group(1)

            if not product_id:
                # Generate a product ID from the name
                product_id = re.sub(r"[^a-zA-Z0-9]", "-", product_name)[:20]

            # Parse attributes if present
            attributes = self._parse_attributes(item_div)

            return CartItem(
                product_id=product_id,
                product_name=product_name,
                attributes=attributes,
                quantity=quantity,
                price=price,
            )
        except Exception:
            return None

    def _parse_table_row(self, row) -> Optional[CartItem]:
        """Parse a cart item from a table row element."""
        try:
            cells = row.find_all("td")
            if len(cells) < 2:
                return None

            # Typical layout: name, qty, price or similar
            product_name = cells[0].get_text(strip=True)
            if not product_name:
                return None

            price = 0.0
            quantity = 1
            product_id = row.get("data-product-id", "")

            for cell in cells[1:]:
                text = cell.get_text(strip=True)
                if "$" in text:
                    price = self._parse_price(text)
                elif text.isdigit():
                    quantity = int(text)

            if not product_id:
                product_id = re.sub(r"[^a-zA-Z0-9]", "-", product_name)[:20]

            return CartItem(
                product_id=product_id,
                product_name=product_name,
                quantity=quantity,
                price=price,
            )
        except Exception:
            return None

    def _parse_price(self, price_text: str) -> float:
        """Parse a price string like '$29.99' into a float."""
        # Remove currency symbols and whitespace
        clean = re.sub(r"[^\d.]", "", price_text)
        try:
            return float(clean) if clean else 0.0
        except ValueError:
            return 0.0

    def _parse_quantity(self, qty_text: str) -> int:
        """Parse a quantity string into an integer."""
        # Extract digits from text like "Qty: 2" or "x2" or just "2"
        match = re.search(r"(\d+)", qty_text)
        return int(match.group(1)) if match else 1

    def _parse_attributes(self, item_div) -> dict:
        """Parse product attributes from an item div."""
        attributes = {}

        # Look for attribute spans/divs
        attr_elems = item_div.find_all("span", class_="attribute") or item_div.find_all(
            "div", class_="attribute"
        )
        for elem in attr_elems:
            text = elem.get_text(strip=True)
            if ":" in text:
                key, value = text.split(":", 1)
                attributes[key.strip().lower()] = value.strip()

        # Also check for common specific attributes
        color_elem = item_div.find(class_="color") or item_div.find(
            attrs={"data-color": True}
        )
        if color_elem:
            if color_elem.get("data-color"):
                attributes["color"] = color_elem.get("data-color")
            else:
                attributes["color"] = color_elem.get_text(strip=True)

        size_elem = item_div.find(class_="size") or item_div.find(
            attrs={"data-size": True}
        )
        if size_elem:
            if size_elem.get("data-size"):
                attributes["size"] = size_elem.get("data-size")
            else:
                attributes["size"] = size_elem.get_text(strip=True)

        return attributes

    def get_agent_memory(self, agent_id: str) -> AgentMemory:
        """
        Get the memory for a specific agent.

        Creates an empty AgentMemory if one doesn't exist for the agent.

        Args:
            agent_id: The unique identifier of the agent.

        Returns:
            The AgentMemory for the agent.
        """
        if agent_id not in self._agent_memories:
            self._agent_memories[agent_id] = AgentMemory(agent_id=agent_id)
        return self._agent_memories[agent_id]

    def update_agent_memory(
        self, agent_id: str, summary: SessionSummary
    ) -> None:
        """
        Update an agent's memory with a session summary.

        Args:
            agent_id: The unique identifier of the agent.
            summary: The session summary to add to memory.
        """
        memory = self.get_agent_memory(agent_id)
        memory.add_session(summary)

    def inject_cart_state(self, cart: CartState) -> None:
        """
        Inject a cart state for the next session (used for error recovery tasks).

        The injected cart will be applied to the next session created and then
        cleared. This allows setting up specific cart states for error recovery
        testing.

        Args:
            cart: The CartState to inject into the next session.
        """
        self._injected_cart = cart.model_copy(deep=True)

    def inject_cart_from_setup(self, cart_contents: list[CartItemSetup]) -> None:
        """
        Inject a cart state from task setup data.

        Convenience method for converting CartItemSetup list (from task JSON)
        to a CartState and injecting it.

        Args:
            cart_contents: List of CartItemSetup from an ErrorRecoveryTask.
        """
        cart = CartState()
        for setup_item in cart_contents:
            item = CartItem(
                product_id=setup_item.product_id,
                product_name=setup_item.product_name,
                attributes=setup_item.attributes,
                quantity=setup_item.quantity,
                price=setup_item.price,
            )
            cart.add_item(item)
        self.inject_cart_state(cart)

    def complete_session(
        self, session_id: str, task_type: str = ""
    ) -> SessionSummary:
        """
        Mark a session as complete and create a summary for agent memory.

        Args:
            session_id: The session to complete.
            task_type: The type of task (for memory classification).

        Returns:
            A SessionSummary of the completed session.

        Raises:
            KeyError: If the session ID is not found.
        """
        session = self.get_session(session_id)
        session.complete()

        summary = session.to_summary()
        summary.task_type = task_type

        # Update agent memory if agent_id is set
        if session.agent_id:
            self.update_agent_memory(session.agent_id, summary)

        return summary

    def get_all_sessions(self) -> list[SessionState]:
        """Get all sessions managed by this StateManager."""
        return list(self._sessions.values())

    def get_sessions_by_agent(self, agent_id: str) -> list[SessionState]:
        """Get all sessions for a specific agent."""
        return [s for s in self._sessions.values() if s.agent_id == agent_id]

    def clear(self) -> None:
        """Clear all sessions and agent memories."""
        self._sessions.clear()
        self._agent_memories.clear()
        self._injected_cart = None

    def __len__(self) -> int:
        """Return the number of active sessions."""
        return len(self._sessions)

    def __contains__(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self._sessions
