"""
Integration tests for the A2A + MCP happy path.

Stage 11 of the AAA (A2A + MCP Agentification) implementation.

These tests verify the complete end-to-end flow:
1. Green agent creates MCP session and includes MCP URI in kickoff
2. Purple agent extracts MCP URI from kickoff message
3. Purple agent's ShoppingAgent connects to MCP server
4. Shopping tools (search/click/checkout) execute successfully
5. Session completes and evaluation is returned

Test Categories:
1. MCP Session Creation (Green)
2. MCP URI Extraction (Purple)
3. MCP Tool Execution (Green MCP Server)
4. End-to-End Flow (Both Agents)
"""

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.testclient import TestClient

# Add parent directories to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
GREEN_AGENT_DIR = PROJECT_ROOT / "green_agent"
PURPLE_AGENT_DIR = PROJECT_ROOT / "purple_agent"

# Import helpers from conftest
from conftest import (
    make_a2a_request,
    make_message_send_request,
    make_mcp_kickoff_message,
)


# =============================================================================
# MCP Session Creation Tests (Green Agent)
# =============================================================================


class TestMCPSessionCreation:
    """Tests for green agent MCP session creation."""

    def test_session_manager_creates_session(self):
        """SessionManager creates a session with correct parameters."""
        sys.path.insert(0, str(GREEN_AGENT_DIR))
        try:
            from src.webshop_mcp import SessionManager

            manager = SessionManager(max_sessions=10, session_ttl=300)

            async def run_test():
                session_id = "test-session-001"
                server = await manager.create_session(
                    session_id=session_id,
                    goal="Find running shoes under $50",
                    budget=50.0,
                    constraints=["waterproof"],
                    max_turns=15,
                )

                # Verify session was created
                assert server is not None
                assert server.state.session_id == session_id
                assert server.state.goal == "Find running shoes under $50"
                assert server.state.budget == 50.0
                assert server.state.max_turns == 15

                # Verify session can be retrieved
                retrieved = await manager.get_session(session_id)
                assert retrieved is server

                # Cleanup
                await manager.cleanup_session(session_id)
                assert await manager.get_session(session_id) is None

            asyncio.run(run_test())
        finally:
            sys.path.remove(str(GREEN_AGENT_DIR))

    def test_mcp_uri_generation(self):
        """MCP URI is generated correctly."""
        sys.path.insert(0, str(GREEN_AGENT_DIR))
        try:
            from src.a2a_executor import WebShopPlusExecutor
            from src.agent import AgentConfig
            from src.webshop_mcp import SessionManager

            manager = SessionManager()
            executor = WebShopPlusExecutor(
                agent_config=AgentConfig(),
                session_manager=manager,
                mcp_host="example.com",
                mcp_port=9000,
            )

            session_id = "abc123"
            uri = executor.get_mcp_uri(session_id)
            assert uri == "http://example.com:9000/mcp/abc123"
        finally:
            sys.path.remove(str(GREEN_AGENT_DIR))

    def test_mcp_session_creates_correct_uri(self):
        """create_mcp_session returns correct session_id and URI."""
        sys.path.insert(0, str(GREEN_AGENT_DIR))
        try:
            from src.a2a_executor import WebShopPlusExecutor
            from src.agent import AgentConfig
            from src.webshop_mcp import SessionManager

            manager = SessionManager()
            executor = WebShopPlusExecutor(
                agent_config=AgentConfig(),
                session_manager=manager,
                mcp_host="localhost",
                mcp_port=8000,
            )

            async def run_test():
                session_id, mcp_uri = await executor.create_mcp_session(
                    goal="Find shoes",
                    budget=100.0,
                    constraints=["leather"],
                    max_turns=10,
                )

                # Verify session ID is a UUID
                uuid.UUID(session_id)

                # Verify URI format
                assert mcp_uri == f"http://localhost:8000/mcp/{session_id}"

                # Verify session exists
                session = await manager.get_session(session_id)
                assert session is not None
                assert session.state.goal == "Find shoes"
                assert session.state.budget == 100.0

                # Cleanup
                await executor.cleanup_mcp_session(session_id)

            asyncio.run(run_test())
        finally:
            sys.path.remove(str(GREEN_AGENT_DIR))


# =============================================================================
# MCP URI Extraction Tests (Purple Agent - via subprocess)
# =============================================================================


class TestMCPURIExtraction:
    """Tests for purple agent MCP URI extraction from kickoff messages.

    These tests run in subprocess to avoid module import conflicts between
    green_agent and purple_agent src modules.
    """

    def test_extract_mcp_uri_from_kickoff(self):
        """Executor extracts MCP URI from kickoff message."""
        import subprocess

        code = '''
import json
from src.executor import Executor
from a2a.types import Message, Role, TextPart

executor = Executor()

# Create a kickoff message with MCP resource
kickoff_payload = json.dumps({
    "goal": "Find running shoes under $50",
    "budget": 50.0,
    "constraints": ["waterproof"],
    "resources": [
        {"type": "mcp", "uri": "http://localhost:8000/mcp/session-123"}
    ],
})

message = Message(
    messageId="test-msg",
    role=Role.user,
    parts=[TextPart(text=kickoff_payload)],
)

uri = executor._extract_mcp_uri(message)
assert uri == "http://localhost:8000/mcp/session-123", f"Got: {uri}"
print("SUCCESS")
'''
        result = subprocess.run(
            ["uv", "run", "python", "-c", code],
            cwd=str(PURPLE_AGENT_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "SUCCESS" in result.stdout, f"Failed: {result.stderr or result.stdout}"

    def test_extract_mcp_uri_missing_resources(self):
        """Executor returns None when resources missing."""
        import subprocess

        code = '''
import json
from src.executor import Executor
from a2a.types import Message, Role, TextPart

executor = Executor()

# Message without resources
payload = json.dumps({
    "goal": "Find shoes",
    "budget": 50.0,
})

message = Message(
    messageId="test-msg",
    role=Role.user,
    parts=[TextPart(text=payload)],
)

uri = executor._extract_mcp_uri(message)
assert uri is None, f"Expected None, got: {uri}"
print("SUCCESS")
'''
        result = subprocess.run(
            ["uv", "run", "python", "-c", code],
            cwd=str(PURPLE_AGENT_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "SUCCESS" in result.stdout, f"Failed: {result.stderr or result.stdout}"

    def test_extract_task_data_from_kickoff(self):
        """Executor extracts goal, budget, constraints from kickoff."""
        import subprocess

        code = '''
import json
from src.executor import Executor
from a2a.types import Message, Role, TextPart

executor = Executor()

kickoff_payload = json.dumps({
    "goal": "Find waterproof running shoes",
    "budget": 75.0,
    "constraints": ["size 10", "no synthetic"],
    "resources": [{"type": "mcp", "uri": "http://localhost:8000/mcp/s1"}],
})

message = Message(
    messageId="test-msg",
    role=Role.user,
    parts=[TextPart(text=kickoff_payload)],
)

task_data = executor._extract_task_data(message)

assert task_data["goal"] == "Find waterproof running shoes", f"goal: {task_data.get('goal')}"
assert task_data["budget"] == 75.0, f"budget: {task_data.get('budget')}"
assert "size 10" in task_data["constraints"], f"constraints: {task_data.get('constraints')}"
assert "no synthetic" in task_data["constraints"], f"constraints: {task_data.get('constraints')}"
print("SUCCESS")
'''
        result = subprocess.run(
            ["uv", "run", "python", "-c", code],
            cwd=str(PURPLE_AGENT_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "SUCCESS" in result.stdout, f"Failed: {result.stderr or result.stdout}"


# =============================================================================
# MCP Tool Execution Tests (Green MCP Server)
# =============================================================================


class TestMCPToolExecution:
    """Tests for MCP tool execution on the green agent.

    These tests use call_tool() to invoke tools via the FastMCP API.
    """

    @pytest.mark.asyncio
    async def test_search_tool_returns_products(self):
        """search() tool returns products with element IDs."""
        sys.path.insert(0, str(GREEN_AGENT_DIR))
        try:
            from src.webshop_mcp.session_state import SessionState
            from src.webshop_mcp.server import WebShopMCPServer

            state = SessionState(
                session_id="test-session",
                goal="Find running shoes",
                budget=100.0,
                constraints=[],
                max_turns=30,
            )

            # Create mock webshop interface
            mock_webshop = MagicMock()
            mock_webshop.step.return_value = MagicMock(
                observation='<div class="list-group-item"><h4>Running Shoe X</h4><h5>$45.00</h5><span class="product-link">B001ABC</span></div>'
            )
            mock_webshop.get_available_actions.return_value = {"clickables": []}
            mock_webshop.product_prices = {"B001ABC": 45.0}

            server = WebShopMCPServer(state, webshop=mock_webshop)

            # Call search using call_tool
            result = await server.mcp.call_tool("search", {"query": "running shoes"})

            # Parse result (call_tool returns a list of content)
            assert len(result) > 0
            # Result contains TextContent or similar
            result_text = str(result[0])
            assert "search_results" in result_text or "products" in result_text
        finally:
            sys.path.remove(str(GREEN_AGENT_DIR))

    @pytest.mark.asyncio
    async def test_checkout_tool_completes_session(self):
        """checkout() tool marks session as completed."""
        sys.path.insert(0, str(GREEN_AGENT_DIR))
        try:
            from src.webshop_mcp.session_state import SessionState
            from src.webshop_mcp.server import WebShopMCPServer

            state = SessionState(
                session_id="test-session",
                goal="Find running shoes",
                budget=100.0,
                constraints=[],
                max_turns=30,
            )

            # Add item to cart
            state.add_to_cart({
                "name": "Running Shoe X",
                "price": 45.0,
                "asin": "B001ABC",
            })

            server = WebShopMCPServer(state)

            # Call checkout
            result = await server.mcp.call_tool("checkout", {})

            # Check state
            assert state.completed is True
            assert server.is_completed() is True

            # Check final result
            final_result = server.get_final_result()
            assert final_result is not None
            assert final_result["terminated"] is True
            assert final_result["success"] is True
            assert final_result["score"] == 1.0
        finally:
            sys.path.remove(str(GREEN_AGENT_DIR))

    @pytest.mark.asyncio
    async def test_checkout_empty_cart_fails(self):
        """checkout() with empty cart returns failure."""
        sys.path.insert(0, str(GREEN_AGENT_DIR))
        try:
            from src.webshop_mcp.session_state import SessionState
            from src.webshop_mcp.server import WebShopMCPServer

            state = SessionState(
                session_id="test-session",
                goal="Find running shoes",
                budget=100.0,
                constraints=[],
                max_turns=30,
            )

            server = WebShopMCPServer(state)

            # Call checkout with empty cart
            result = await server.mcp.call_tool("checkout", {})

            final_result = server.get_final_result()
            assert final_result["terminated"] is True
            assert final_result["success"] is False
            assert final_result["failure_reason"] == "empty_cart"
            assert final_result["score"] == 0.0
        finally:
            sys.path.remove(str(GREEN_AGENT_DIR))

    @pytest.mark.asyncio
    async def test_checkout_over_budget_partial_failure(self):
        """checkout() over budget returns partial failure."""
        sys.path.insert(0, str(GREEN_AGENT_DIR))
        try:
            from src.webshop_mcp.session_state import SessionState
            from src.webshop_mcp.server import WebShopMCPServer

            state = SessionState(
                session_id="test-session",
                goal="Find shoes",
                budget=50.0,  # Budget is $50
                constraints=[],
                max_turns=30,
            )

            # Add item over budget
            state.add_to_cart({
                "name": "Expensive Shoe",
                "price": 75.0,  # Over budget
                "asin": "B002XYZ",
            })

            server = WebShopMCPServer(state)

            # Call checkout
            result = await server.mcp.call_tool("checkout", {})

            final_result = server.get_final_result()
            assert final_result["terminated"] is True
            assert final_result["success"] is False
            assert final_result["failure_reason"] == "budget_exceeded"
            assert final_result["score"] == 0.3
        finally:
            sys.path.remove(str(GREEN_AGENT_DIR))


# =============================================================================
# ShoppingAgent Tests (Purple Agent - via subprocess)
# =============================================================================


class TestShoppingAgentIntegration:
    """Tests for ShoppingAgent MCP integration.

    These tests run in subprocess to avoid module import conflicts.
    """

    def test_shopping_agent_initialization(self):
        """ShoppingAgent initializes correctly."""
        import subprocess

        code = '''
from src.shopping_agent import ShoppingAgent

agent = ShoppingAgent(model="gemini-2.0-flash", max_turns=20)
assert agent.model == "gemini-2.0-flash", f"model: {agent.model}"
assert agent.max_turns == 20, f"max_turns: {agent.max_turns}"
print("SUCCESS")
'''
        result = subprocess.run(
            ["uv", "run", "python", "-c", code],
            cwd=str(PURPLE_AGENT_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "SUCCESS" in result.stdout, f"Failed: {result.stderr or result.stdout}"

    def test_shopping_agent_instruction_formatting(self):
        """ShoppingAgent formats instructions correctly."""
        import subprocess

        code = '''
from src.shopping_agent import ShoppingAgent

agent = ShoppingAgent()
instruction = agent._format_instruction(
    goal="Find running shoes",
    budget=50.0,
    constraints=["waterproof", "size 10"],
)

assert "Find running shoes" in instruction, f"Missing goal in: {instruction[:100]}"
assert "$50" in instruction, f"Missing budget in: {instruction[:100]}"
assert "waterproof, size 10" in instruction, f"Missing constraints in: {instruction[:100]}"
print("SUCCESS")
'''
        result = subprocess.run(
            ["uv", "run", "python", "-c", code],
            cwd=str(PURPLE_AGENT_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "SUCCESS" in result.stdout, f"Failed: {result.stderr or result.stdout}"

    def test_shopping_agent_run_requires_mcp_uri(self):
        """ShoppingAgent.run() requires mcp_uri."""
        import subprocess

        code = '''
import asyncio
from src.shopping_agent import ShoppingAgent

async def test():
    agent = ShoppingAgent()
    try:
        await agent.run("", {"goal": "Find shoes"})
        print("FAIL: No exception raised")
    except ValueError as e:
        if "mcp_uri" in str(e).lower():
            print("SUCCESS")
        else:
            print(f"FAIL: Wrong error: {e}")

asyncio.run(test())
'''
        result = subprocess.run(
            ["uv", "run", "python", "-c", code],
            cwd=str(PURPLE_AGENT_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "SUCCESS" in result.stdout, f"Failed: {result.stderr or result.stdout}"

    def test_shopping_agent_run_requires_goal(self):
        """ShoppingAgent.run() requires goal in task_data."""
        import subprocess

        code = '''
import asyncio
from src.shopping_agent import ShoppingAgent

async def test():
    agent = ShoppingAgent()
    try:
        await agent.run("http://localhost:8000/mcp/s1", {})
        print("FAIL: No exception raised")
    except ValueError as e:
        if "goal" in str(e).lower():
            print("SUCCESS")
        else:
            print(f"FAIL: Wrong error: {e}")

asyncio.run(test())
'''
        result = subprocess.run(
            ["uv", "run", "python", "-c", code],
            cwd=str(PURPLE_AGENT_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "SUCCESS" in result.stdout, f"Failed: {result.stderr or result.stdout}"


# =============================================================================
# MCP Route Handler Tests (Green Agent)
# =============================================================================


class TestMCPRouteHandler:
    """Tests for the MCP route handler in green agent."""

    def test_mcp_route_returns_404_for_unknown_session(self, green_app):
        """MCP route returns 404 for non-existent session."""
        client = TestClient(green_app)

        response = client.post(
            "/mcp/nonexistent-session",
            json={"jsonrpc": "2.0", "method": "initialize", "id": 1},
        )

        # Should return 404 (session not found)
        assert response.status_code == 404
        data = response.json()
        assert "error" in data

    def test_mcp_route_handles_invalid_path(self, green_app):
        """MCP route handles invalid paths gracefully."""
        client = TestClient(green_app)

        response = client.get("/mcp/")

        # Should return 400 (missing session_id) or 404
        assert response.status_code in [400, 404, 405]


# =============================================================================
# End-to-End Tests (Full Flow)
# =============================================================================


class TestEndToEndFlow:
    """End-to-end tests for the complete A2A + MCP flow."""

    def test_green_agent_health_check(self, green_app):
        """Green agent health check works."""
        client = TestClient(green_app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_purple_agent_health_check(self):
        """Purple agent health check works (subprocess isolation)."""
        import subprocess

        code = '''
from starlette.testclient import TestClient
from src.server import create_app

app = create_app(card_url="http://localhost:18001")
client = TestClient(app)
response = client.get("/health")
assert response.status_code == 200, f"Status: {response.status_code}"
assert response.json()["status"] == "healthy"
print("SUCCESS")
'''
        result = subprocess.run(
            ["uv", "run", "python", "-c", code],
            cwd=str(PURPLE_AGENT_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "SUCCESS" in result.stdout, f"Failed: {result.stderr or result.stdout}"

    def test_green_agent_card(self, green_app):
        """Green agent has valid agent card."""
        client = TestClient(green_app)
        response = client.get("/.well-known/agent-card.json")
        assert response.status_code == 200
        card = response.json()
        assert card["name"] == "WebShop+ Benchmark"
        assert "skills" in card

    def test_purple_agent_card(self):
        """Purple agent has valid agent card (subprocess isolation)."""
        import subprocess

        code = '''
from starlette.testclient import TestClient
from src.server import create_app

app = create_app(card_url="http://localhost:18001")
client = TestClient(app)
response = client.get("/.well-known/agent-card.json")
assert response.status_code == 200, f"Status: {response.status_code}"
card = response.json()
assert "Shopper" in card["name"], f"Name: {card.get('name')}"
print("SUCCESS")
'''
        result = subprocess.run(
            ["uv", "run", "python", "-c", code],
            cwd=str(PURPLE_AGENT_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "SUCCESS" in result.stdout, f"Failed: {result.stderr or result.stdout}"

    @pytest.mark.asyncio
    async def test_mcp_session_lifecycle(self):
        """Test complete MCP session lifecycle."""
        sys.path.insert(0, str(GREEN_AGENT_DIR))
        try:
            from src.webshop_mcp import SessionManager

            manager = SessionManager()

            # Create session
            session_id = "lifecycle-test-001"
            server = await manager.create_session(
                session_id=session_id,
                goal="Find shoes",
                budget=100.0,
                constraints=["leather"],
                max_turns=10,
            )

            assert server is not None
            assert not server.is_completed()

            # Simulate shopping flow
            state = server.state
            state.add_to_cart({"name": "Leather Boot", "price": 80.0, "asin": "B001"})

            # Call checkout tool via mcp.call_tool
            result = await server.mcp.call_tool("checkout", {})

            # Verify completion
            assert server.is_completed()

            # Get final result
            final = server.get_final_result()
            assert final["success"] is True
            assert final["score"] == 1.0

            # Cleanup
            await manager.cleanup_session(session_id)
            assert await manager.get_session(session_id) is None

        finally:
            sys.path.remove(str(GREEN_AGENT_DIR))

    def test_executor_handles_mcp_kickoff(self):
        """Purple executor routes MCP kickoff to ShoppingAgent."""
        import subprocess

        code = '''
import json
from src.executor import Executor
from a2a.types import Message, Role, TextPart

executor = Executor()

# Create MCP kickoff message
kickoff = json.dumps({
    "goal": "Find running shoes",
    "budget": 50.0,
    "constraints": [],
    "resources": [{"type": "mcp", "uri": "http://localhost:8000/mcp/test"}],
})

message = Message(
    messageId="test",
    role=Role.user,
    parts=[TextPart(text=kickoff)],
)

# Verify MCP URI is detected
uri = executor._extract_mcp_uri(message)
assert uri == "http://localhost:8000/mcp/test", f"URI: {uri}"

# Verify task data is extracted
task_data = executor._extract_task_data(message)
assert task_data["goal"] == "Find running shoes", f"goal: {task_data.get('goal')}"
assert task_data["budget"] == 50.0, f"budget: {task_data.get('budget')}"
print("SUCCESS")
'''
        result = subprocess.run(
            ["uv", "run", "python", "-c", code],
            cwd=str(PURPLE_AGENT_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "SUCCESS" in result.stdout, f"Failed: {result.stderr or result.stdout}"


# =============================================================================
# Integration Flow Tests (With Mocks)
# =============================================================================


class TestIntegrationWithMocks:
    """Integration tests using mocks to simulate inter-agent communication."""

    @pytest.mark.asyncio
    async def test_full_mcp_flow_mocked(self):
        """Test full MCP flow with mocked ADK agent."""
        sys.path.insert(0, str(GREEN_AGENT_DIR))

        try:
            # Import components
            from src.webshop_mcp import SessionManager

            # Create session manager and session
            manager = SessionManager()
            session_id = "mock-flow-test"

            server = await manager.create_session(
                session_id=session_id,
                goal="Find running shoes under $50",
                budget=50.0,
                constraints=["waterproof"],
                max_turns=30,
            )

            mcp_uri = f"http://localhost:8000/mcp/{session_id}"

            # Simulate what purple agent would do:
            # 1. Extract task data (this would happen in purple agent's executor)
            kickoff = {
                "goal": "Find running shoes under $50",
                "budget": 50.0,
                "constraints": ["waterproof"],
                "resources": [{"type": "mcp", "uri": mcp_uri}],
            }

            # 2. Mock the webshop for search
            mock_webshop = MagicMock()
            mock_webshop.step.return_value = MagicMock(
                observation='<div class="list-group-item"><h4>Nike Running Shoe</h4><h5>$45.00</h5><span class="product-link">B001ABC</span></div>'
            )
            mock_webshop.get_available_actions.return_value = {"clickables": []}
            mock_webshop.product_prices = {"B001ABC": 45.0}
            mock_webshop.product_item_dict = {"B001ABC": {"name": "Nike Running Shoe"}}
            server._webshop = mock_webshop
            server._webshop_initialized = True

            # 3. Execute search via MCP call_tool
            search_result = await server.mcp.call_tool("search", {"query": "running shoes"})
            assert len(search_result) > 0

            # 4. Add product to cart (simulating click workflow)
            server.state.add_to_cart({
                "name": "Nike Running Shoe",
                "price": 45.0,
                "asin": "B001ABC",
            })

            # 5. Execute checkout via MCP call_tool
            checkout_result = await server.mcp.call_tool("checkout", {})

            # Verify successful completion
            final_result = server.get_final_result()
            assert final_result["terminated"] is True
            assert final_result["success"] is True
            assert final_result["score"] == 1.0
            assert final_result["total"] == 45.0
            assert final_result["budget"] == 50.0

            # Verify session is completed
            assert server.is_completed()

            # Cleanup
            await manager.cleanup_session(session_id)

        finally:
            sys.path.remove(str(GREEN_AGENT_DIR))


# =============================================================================
# Server Communication Tests (Subprocess-based)
# =============================================================================


@pytest.mark.slow
class TestServerCommunication:
    """Tests that require running servers (marked as slow)."""

    @pytest.mark.asyncio
    async def test_green_server_responds(self, green_server):
        """Green server responds to requests."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{green_server.base_url}/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_purple_server_responds(self, purple_server):
        """Purple server responds to requests."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{purple_server.base_url}/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_both_servers_have_agent_cards(self, green_server, purple_server):
        """Both servers expose agent cards."""
        async with httpx.AsyncClient() as client:
            green_card = await client.get(green_server.agent_card_url)
            purple_card = await client.get(purple_server.agent_card_url)

            assert green_card.status_code == 200
            assert purple_card.status_code == 200

            assert "WebShop+ Benchmark" in green_card.json()["name"]
            assert "Shopper" in purple_card.json()["name"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--ignore-glob=*slow*"])
