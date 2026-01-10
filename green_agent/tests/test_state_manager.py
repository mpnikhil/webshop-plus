"""
Tests for StateManager.

Tests session creation, action recording, cart parsing, agent memory,
and cart injection for error recovery tasks.
"""

import pytest

from src.models import (
    AgentMemory,
    CartItem,
    CartItemSetup,
    CartState,
    PurchaseRecord,
    SessionState,
    SessionSummary,
)
from src.state_manager import StateManager


class TestStateManagerInit:
    """Tests for StateManager initialization."""

    def test_init_creates_empty_stores(self):
        """StateManager should start with empty session and memory stores."""
        sm = StateManager()
        assert len(sm) == 0
        assert sm.get_all_sessions() == []

    def test_multiple_instances_independent(self):
        """Multiple StateManager instances should be independent."""
        sm1 = StateManager()
        sm2 = StateManager()

        sm1.create_session("task_1", "agent_1")
        assert len(sm1) == 1
        assert len(sm2) == 0


class TestSessionCreation:
    """Tests for session creation and retrieval."""

    def test_create_session_returns_session_state(self):
        """create_session should return a SessionState."""
        sm = StateManager()
        session = sm.create_session("task_001", "agent_001")

        assert isinstance(session, SessionState)
        assert session.task_id == "task_001"
        assert session.agent_id == "agent_001"
        assert session.session_id is not None
        assert len(session.session_id) == 36  # UUID format

    def test_create_session_without_agent_id(self):
        """create_session should work without agent_id."""
        sm = StateManager()
        session = sm.create_session("task_001")

        assert session.agent_id == ""
        assert session.task_id == "task_001"

    def test_create_multiple_sessions(self):
        """Should be able to create multiple sessions."""
        sm = StateManager()
        s1 = sm.create_session("task_001", "agent_001")
        s2 = sm.create_session("task_002", "agent_001")
        s3 = sm.create_session("task_003", "agent_002")

        assert len(sm) == 3
        assert s1.session_id != s2.session_id
        assert s2.session_id != s3.session_id

    def test_get_session_returns_correct_session(self):
        """get_session should return the correct session by ID."""
        sm = StateManager()
        s1 = sm.create_session("task_001", "agent_001")
        s2 = sm.create_session("task_002", "agent_002")

        retrieved = sm.get_session(s1.session_id)
        assert retrieved.session_id == s1.session_id
        assert retrieved.task_id == "task_001"

    def test_get_session_not_found(self):
        """get_session should raise KeyError for unknown session."""
        sm = StateManager()
        with pytest.raises(KeyError, match="Session not found"):
            sm.get_session("nonexistent-session-id")

    def test_session_in_manager(self):
        """__contains__ should check if session exists."""
        sm = StateManager()
        session = sm.create_session("task_001")

        assert session.session_id in sm
        assert "nonexistent" not in sm

    def test_create_session_with_empty_cart(self):
        """New sessions should have an empty cart by default."""
        sm = StateManager()
        session = sm.create_session("task_001")

        assert session.cart is not None
        assert len(session.cart.items) == 0


class TestActionRecording:
    """Tests for action recording within sessions."""

    def test_record_action_adds_to_session(self):
        """record_action should add an action to the session."""
        sm = StateManager()
        session = sm.create_session("task_001")

        sm.record_action(session.session_id, "search[laptop]", "Results: ...", 0.0)

        assert len(session.actions) == 1
        assert session.actions[0].action == "search[laptop]"
        assert session.actions[0].observation == "Results: ..."

    def test_record_multiple_actions(self):
        """Should be able to record multiple actions."""
        sm = StateManager()
        session = sm.create_session("task_001")

        sm.record_action(session.session_id, "search[laptop]", "Results: ...", 0.0)
        sm.record_action(session.session_id, "click[item1]", "Product page", 0.1)
        sm.record_action(session.session_id, "buy", "Purchase complete", 1.0)

        assert len(session.actions) == 3
        assert session.actions_taken == 3

    def test_record_action_updates_current_observation(self):
        """record_action should update current_observation."""
        sm = StateManager()
        session = sm.create_session("task_001")

        sm.record_action(session.session_id, "search[laptop]", "First observation")
        assert session.current_observation == "First observation"

        sm.record_action(session.session_id, "click[item1]", "Second observation")
        assert session.current_observation == "Second observation"

    def test_record_action_session_not_found(self):
        """record_action should raise KeyError for unknown session."""
        sm = StateManager()
        with pytest.raises(KeyError):
            sm.record_action("nonexistent", "action", "observation")

    def test_record_action_with_reward(self):
        """record_action should store the reward."""
        sm = StateManager()
        session = sm.create_session("task_001")

        sm.record_action(session.session_id, "buy", "Complete", 1.0)
        assert session.actions[0].reward == 1.0


class TestCartParsing:
    """Tests for cart parsing from HTML observations."""

    def test_parse_empty_observation(self):
        """parse_cart should handle empty observation."""
        sm = StateManager()
        cart = sm.parse_cart_from_observation("")
        assert len(cart.items) == 0

    def test_parse_none_observation(self):
        """parse_cart should handle None-like observation."""
        sm = StateManager()
        cart = sm.parse_cart_from_observation("   ")
        assert len(cart.items) == 0

    def test_parse_cart_div_with_items(self):
        """parse_cart should extract items from cart div."""
        sm = StateManager()
        html = """
        <div class="cart">
            <div class="item">
                <span class="product-name">Sony Headphones</span>
                <span class="price">$299.99</span>
                <span class="quantity">2</span>
            </div>
        </div>
        """
        cart = sm.parse_cart_from_observation(html)

        assert len(cart.items) == 1
        assert cart.items[0].product_name == "Sony Headphones"
        assert cart.items[0].price == 299.99
        assert cart.items[0].quantity == 2

    def test_parse_multiple_cart_items(self):
        """parse_cart should extract multiple items."""
        sm = StateManager()
        html = """
        <div class="cart">
            <div class="item">
                <span class="product-name">Laptop</span>
                <span class="price">$999.99</span>
                <span class="quantity">1</span>
            </div>
            <div class="item">
                <span class="product-name">Mouse</span>
                <span class="price">$49.99</span>
                <span class="quantity">1</span>
            </div>
        </div>
        """
        cart = sm.parse_cart_from_observation(html)

        assert len(cart.items) == 2
        assert cart.total == 999.99 + 49.99

    def test_parse_cart_with_attributes(self):
        """parse_cart should extract product attributes."""
        sm = StateManager()
        html = """
        <div class="cart">
            <div class="item">
                <span class="product-name">T-Shirt</span>
                <span class="price">$19.99</span>
                <span class="color" data-color="blue">Blue</span>
                <span class="size" data-size="M">Medium</span>
            </div>
        </div>
        """
        cart = sm.parse_cart_from_observation(html)

        assert len(cart.items) == 1
        assert cart.items[0].attributes.get("color") == "blue"
        assert cart.items[0].attributes.get("size") == "M"

    def test_parse_cart_with_product_link(self):
        """parse_cart should extract product ID from link."""
        sm = StateManager()
        html = """
        <div class="cart">
            <div class="item">
                <a href="/product/ABC123" class="product-name">Wireless Mouse</a>
                <span class="price">$29.99</span>
            </div>
        </div>
        """
        cart = sm.parse_cart_from_observation(html)

        assert len(cart.items) == 1
        assert cart.items[0].product_id == "ABC123"

    def test_parse_cart_with_data_product_id(self):
        """parse_cart should use data-product-id attribute."""
        sm = StateManager()
        html = """
        <div class="cart">
            <div class="item" data-product-id="PROD-456">
                <span class="product-name">Keyboard</span>
                <span class="price">$79.99</span>
            </div>
        </div>
        """
        cart = sm.parse_cart_from_observation(html)

        assert len(cart.items) == 1
        assert cart.items[0].product_id == "PROD-456"

    def test_parse_cart_quantity_input(self):
        """parse_cart should handle quantity in input field."""
        sm = StateManager()
        html = """
        <div class="cart">
            <div class="item">
                <span class="product-name">Notebook</span>
                <span class="price">$5.99</span>
                <input class="quantity" type="number" value="3">
            </div>
        </div>
        """
        cart = sm.parse_cart_from_observation(html)

        assert len(cart.items) == 1
        assert cart.items[0].quantity == 3

    def test_parse_cart_quantity_text_formats(self):
        """parse_cart should handle various quantity text formats."""
        sm = StateManager()

        test_cases = [
            ("Qty: 5", 5),
            ("x2", 2),
            ("3", 3),
            ("Quantity: 4", 4),
        ]

        for qty_text, expected in test_cases:
            html = f"""
            <div class="cart">
                <div class="item">
                    <span class="product-name">Item</span>
                    <span class="price">$10.00</span>
                    <span class="quantity">{qty_text}</span>
                </div>
            </div>
            """
            cart = sm.parse_cart_from_observation(html)
            assert cart.items[0].quantity == expected, f"Failed for '{qty_text}'"

    def test_parse_cart_price_formats(self):
        """parse_cart should handle various price formats."""
        sm = StateManager()

        test_cases = [
            ("$29.99", 29.99),
            ("29.99", 29.99),
            ("$1,299.00", 1299.00),
            ("USD 49.99", 49.99),
        ]

        for price_text, expected in test_cases:
            html = f"""
            <div class="cart">
                <div class="item">
                    <span class="product-name">Item</span>
                    <span class="price">{price_text}</span>
                </div>
            </div>
            """
            cart = sm.parse_cart_from_observation(html)
            assert cart.items[0].price == expected, f"Failed for '{price_text}'"

    def test_parse_cart_table_format(self):
        """parse_cart should handle table-based cart layout."""
        sm = StateManager()
        html = """
        <table class="cart">
            <tr class="cart-item" data-product-id="TBL-001">
                <td>Table Lamp</td>
                <td>2</td>
                <td>$45.00</td>
            </tr>
        </table>
        """
        cart = sm.parse_cart_from_observation(html)

        assert len(cart.items) == 1
        assert cart.items[0].product_name == "Table Lamp"
        assert cart.items[0].quantity == 2
        assert cart.items[0].price == 45.00

    def test_parse_cart_alternative_container(self):
        """parse_cart should handle cart-container class."""
        sm = StateManager()
        html = """
        <div class="cart-container">
            <div class="item">
                <span class="product-name">Monitor</span>
                <span class="price">$399.99</span>
            </div>
        </div>
        """
        cart = sm.parse_cart_from_observation(html)

        assert len(cart.items) == 1
        assert cart.items[0].product_name == "Monitor"

    def test_parse_cart_id_selector(self):
        """parse_cart should handle cart by id."""
        sm = StateManager()
        html = """
        <div id="cart">
            <div class="item">
                <span class="product-name">Desk</span>
                <span class="price">$199.99</span>
            </div>
        </div>
        """
        cart = sm.parse_cart_from_observation(html)

        assert len(cart.items) == 1
        assert cart.items[0].product_name == "Desk"

    def test_parse_cart_updates_session_cart(self):
        """record_action should update session cart when cart HTML is present."""
        sm = StateManager()
        session = sm.create_session("task_001")

        html = """
        <div class="cart">
            <div class="item">
                <span class="product-name">Chair</span>
                <span class="price">$149.99</span>
                <span class="quantity">1</span>
            </div>
        </div>
        """

        sm.record_action(session.session_id, "view_cart", html)

        assert len(session.cart.items) == 1
        assert session.cart.items[0].product_name == "Chair"

    def test_parse_non_cart_observation_doesnt_clear_cart(self):
        """Non-cart observations shouldn't clear existing cart."""
        sm = StateManager()
        session = sm.create_session("task_001")

        # First, add items to cart via observation
        cart_html = """
        <div class="cart">
            <div class="item">
                <span class="product-name">Product</span>
                <span class="price">$50.00</span>
            </div>
        </div>
        """
        sm.record_action(session.session_id, "view_cart", cart_html)
        assert len(session.cart.items) == 1

        # Then record a non-cart observation
        sm.record_action(session.session_id, "search[shoes]", "Search results...")
        # Cart should not be cleared (still has 1 item)
        assert len(session.cart.items) == 1


class TestAgentMemory:
    """Tests for agent memory management."""

    def test_get_agent_memory_creates_new(self):
        """get_agent_memory should create new memory for unknown agent."""
        sm = StateManager()
        memory = sm.get_agent_memory("agent_001")

        assert isinstance(memory, AgentMemory)
        assert memory.agent_id == "agent_001"
        assert len(memory.sessions) == 0

    def test_get_agent_memory_returns_existing(self):
        """get_agent_memory should return existing memory."""
        sm = StateManager()
        memory1 = sm.get_agent_memory("agent_001")
        memory1.add_session(
            SessionSummary(
                session_id="s1",
                task_id="t1",
                task_type="budget_constrained",
            )
        )

        memory2 = sm.get_agent_memory("agent_001")
        assert len(memory2.sessions) == 1
        assert memory1 is memory2

    def test_update_agent_memory(self):
        """update_agent_memory should add session summary to memory."""
        sm = StateManager()
        summary = SessionSummary(
            session_id="sess_001",
            task_id="task_001",
            task_type="preference_memory",
            preferences={"color": "blue", "brand": "Nike"},
        )

        sm.update_agent_memory("agent_001", summary)
        memory = sm.get_agent_memory("agent_001")

        assert len(memory.sessions) == 1
        assert memory.sessions[0].preferences["color"] == "blue"

    def test_agent_memory_accumulates_sessions(self):
        """Agent memory should accumulate multiple session summaries."""
        sm = StateManager()

        for i in range(3):
            summary = SessionSummary(
                session_id=f"sess_{i}",
                task_id=f"task_{i}",
                task_type="budget_constrained",
            )
            sm.update_agent_memory("agent_001", summary)

        memory = sm.get_agent_memory("agent_001")
        assert len(memory.sessions) == 3

    def test_different_agents_have_separate_memory(self):
        """Different agents should have independent memory."""
        sm = StateManager()

        sm.update_agent_memory(
            "agent_001",
            SessionSummary(session_id="s1", task_id="t1", preferences={"a": 1}),
        )
        sm.update_agent_memory(
            "agent_002",
            SessionSummary(session_id="s2", task_id="t2", preferences={"b": 2}),
        )

        mem1 = sm.get_agent_memory("agent_001")
        mem2 = sm.get_agent_memory("agent_002")

        assert len(mem1.sessions) == 1
        assert len(mem2.sessions) == 1
        assert mem1.sessions[0].preferences["a"] == 1
        assert mem2.sessions[0].preferences["b"] == 2


class TestCartInjection:
    """Tests for cart injection (error recovery tasks)."""

    def test_inject_cart_state(self):
        """inject_cart_state should set up cart for next session."""
        sm = StateManager()

        cart = CartState()
        cart.add_item(
            CartItem(
                product_id="HP-001",
                product_name="Headphones",
                price=299.99,
                quantity=2,
            )
        )

        sm.inject_cart_state(cart)
        session = sm.create_session("recovery_001")

        assert len(session.cart.items) == 1
        assert session.cart.items[0].product_name == "Headphones"
        assert session.cart.items[0].quantity == 2

    def test_inject_cart_clears_after_use(self):
        """Injected cart should only apply to the next session."""
        sm = StateManager()

        cart = CartState()
        cart.add_item(
            CartItem(product_id="P1", product_name="Product", price=10.0)
        )

        sm.inject_cart_state(cart)
        s1 = sm.create_session("recovery_001")
        s2 = sm.create_session("task_002")

        assert len(s1.cart.items) == 1
        assert len(s2.cart.items) == 0  # No injection for second session

    def test_inject_cart_from_setup(self):
        """inject_cart_from_setup should convert CartItemSetup list."""
        sm = StateManager()

        cart_contents = [
            CartItemSetup(
                product_id="HP-001",
                product_name="Sony WH-1000XM4 Wireless Headphones",
                attributes={"color": "black"},
                quantity=3,
                price=278.00,
            ),
            CartItemSetup(
                product_id="WB-042",
                product_name="Hydro Flask 32oz Water Bottle",
                attributes={"color": "pacific"},
                quantity=1,
                price=44.95,
            ),
        ]

        sm.inject_cart_from_setup(cart_contents)
        session = sm.create_session("recovery_001")

        assert len(session.cart.items) == 2
        assert session.cart.items[0].product_id == "HP-001"
        assert session.cart.items[0].quantity == 3
        assert session.cart.items[1].product_id == "WB-042"
        assert session.cart.total == (278.00 * 3) + 44.95

    def test_inject_cart_deep_copy(self):
        """Injected cart should be a deep copy (independent)."""
        sm = StateManager()

        cart = CartState()
        cart.add_item(
            CartItem(product_id="P1", product_name="Product", price=10.0, quantity=1)
        )

        sm.inject_cart_state(cart)
        session = sm.create_session("task_001")

        # Modify original cart
        cart.items[0].quantity = 5

        # Session cart should not be affected
        assert session.cart.items[0].quantity == 1


class TestSessionCompletion:
    """Tests for session completion and summary generation."""

    def test_complete_session_marks_complete(self):
        """complete_session should mark the session as completed."""
        sm = StateManager()
        session = sm.create_session("task_001", "agent_001")

        sm.complete_session(session.session_id, "budget_constrained")

        assert session.completed is True
        assert session.ended_at is not None

    def test_complete_session_returns_summary(self):
        """complete_session should return a SessionSummary."""
        sm = StateManager()
        session = sm.create_session("task_001", "agent_001")
        session.preferences_established = {"color": "blue"}

        summary = sm.complete_session(session.session_id, "preference_memory")

        assert isinstance(summary, SessionSummary)
        assert summary.session_id == session.session_id
        assert summary.task_id == "task_001"
        assert summary.task_type == "preference_memory"
        assert summary.preferences["color"] == "blue"

    def test_complete_session_updates_agent_memory(self):
        """complete_session should update agent memory when agent_id is set."""
        sm = StateManager()
        session = sm.create_session("task_001", "agent_001")

        sm.complete_session(session.session_id, "budget_constrained")

        memory = sm.get_agent_memory("agent_001")
        assert len(memory.sessions) == 1
        assert memory.sessions[0].task_id == "task_001"

    def test_complete_session_no_memory_update_without_agent(self):
        """complete_session should not update memory if no agent_id."""
        sm = StateManager()
        session = sm.create_session("task_001")  # No agent_id

        sm.complete_session(session.session_id, "budget_constrained")

        # No agent memory should be created
        assert len(sm._agent_memories) == 0

    def test_complete_session_includes_purchases(self):
        """Session summary should include purchases from the session."""
        sm = StateManager()
        session = sm.create_session("task_001", "agent_001")
        session.purchases.append(
            PurchaseRecord(
                product_id="P1",
                product_name="Laptop",
                price=999.99,
            )
        )

        summary = sm.complete_session(session.session_id)

        assert len(summary.purchases) == 1
        assert summary.purchases[0].product_name == "Laptop"


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_get_all_sessions(self):
        """get_all_sessions should return all sessions."""
        sm = StateManager()
        sm.create_session("task_001", "agent_001")
        sm.create_session("task_002", "agent_001")
        sm.create_session("task_003", "agent_002")

        sessions = sm.get_all_sessions()
        assert len(sessions) == 3

    def test_get_sessions_by_agent(self):
        """get_sessions_by_agent should filter by agent ID."""
        sm = StateManager()
        sm.create_session("task_001", "agent_001")
        sm.create_session("task_002", "agent_001")
        sm.create_session("task_003", "agent_002")

        agent1_sessions = sm.get_sessions_by_agent("agent_001")
        agent2_sessions = sm.get_sessions_by_agent("agent_002")

        assert len(agent1_sessions) == 2
        assert len(agent2_sessions) == 1

    def test_get_sessions_by_agent_empty(self):
        """get_sessions_by_agent should return empty list for unknown agent."""
        sm = StateManager()
        sm.create_session("task_001", "agent_001")

        sessions = sm.get_sessions_by_agent("unknown_agent")
        assert sessions == []

    def test_clear(self):
        """clear should remove all sessions and memories."""
        sm = StateManager()
        sm.create_session("task_001", "agent_001")
        sm.update_agent_memory(
            "agent_001",
            SessionSummary(session_id="s1", task_id="t1"),
        )
        sm.inject_cart_state(CartState())

        sm.clear()

        assert len(sm) == 0
        assert sm.get_all_sessions() == []
        assert len(sm._agent_memories) == 0
        assert sm._injected_cart is None

    def test_len(self):
        """__len__ should return session count."""
        sm = StateManager()
        assert len(sm) == 0

        sm.create_session("task_001")
        assert len(sm) == 1

        sm.create_session("task_002")
        assert len(sm) == 2


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""

    def test_preference_memory_workflow(self):
        """Test complete preference memory task workflow."""
        sm = StateManager()

        # Session 1: Establish preferences
        s1 = sm.create_session("pref_001_s1", "shopper_agent")
        s1.preferences_established = {
            "brand": "Nike",
            "size": "M",
            "color": "blue",
        }
        sm.record_action(s1.session_id, "search[nike shoes]", "Results...")
        sm.record_action(s1.session_id, "buy", "Purchased!")
        sm.complete_session(s1.session_id, "preference_memory")

        # Session 2: Test recall
        s2 = sm.create_session("pref_001_s2", "shopper_agent")

        # Agent should be able to access memory
        memory = sm.get_agent_memory("shopper_agent")
        prefs = memory.get_all_preferences()

        assert prefs["brand"] == "Nike"
        assert prefs["size"] == "M"
        assert prefs["color"] == "blue"

    def test_error_recovery_workflow(self):
        """Test complete error recovery task workflow."""
        sm = StateManager()

        # Setup cart with errors (from task JSON)
        cart_contents = [
            CartItemSetup(
                product_id="HP-001",
                product_name="Sony WH-1000XM4 Wireless Headphones",
                attributes={"color": "black"},
                quantity=3,  # Error: should be 1
                price=278.00,
            )
        ]

        # Inject the cart
        sm.inject_cart_from_setup(cart_contents)

        # Create session (cart is pre-populated)
        session = sm.create_session("recovery_001", "shopper_agent")

        # Verify cart has the setup data
        assert len(session.cart.items) == 1
        assert session.cart.items[0].quantity == 3
        assert session.cart.total == 278.00 * 3

        # Agent actions to fix
        sm.record_action(session.session_id, "update_quantity[HP-001, 1]", "Updated")

        # Complete and evaluate
        sm.complete_session(session.session_id, "error_recovery")
        assert session.completed is True

    def test_multi_task_assessment(self):
        """Test running multiple tasks for one agent."""
        sm = StateManager()
        agent_id = "test_agent"

        # Run 5 different tasks
        task_types = [
            "budget_constrained",
            "preference_memory",
            "negative_constraint",
            "comparative_reasoning",
            "error_recovery",
        ]

        for i, task_type in enumerate(task_types):
            session = sm.create_session(f"task_{i}", agent_id)
            sm.record_action(session.session_id, "action", "result")
            sm.complete_session(session.session_id, task_type)

        # Verify all sessions recorded
        assert len(sm) == 5

        # Verify agent memory accumulated
        memory = sm.get_agent_memory(agent_id)
        assert len(memory.sessions) == 5

        # Verify sessions can be filtered by type
        budget_sessions = memory.get_sessions_by_type("budget_constrained")
        assert len(budget_sessions) == 1
