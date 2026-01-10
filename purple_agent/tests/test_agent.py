"""
Tests for WebShop+ Purple Agent.

This module contains comprehensive tests for:
- Messenger utilities (A2A protocol)
- ShopperAgent (shopping logic)
- Server endpoints (A2A server)
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

# =============================================================================
# Messenger Tests
# =============================================================================


class TestMessengerModels:
    """Test A2A protocol models."""

    def test_task_state_enum(self):
        """Test TaskState enum values."""
        from src.messenger import TaskState

        assert TaskState.SUBMITTED == "submitted"
        assert TaskState.WORKING == "working"
        assert TaskState.COMPLETED == "completed"
        assert TaskState.FAILED == "failed"

    def test_message_role_enum(self):
        """Test MessageRole enum values."""
        from src.messenger import MessageRole

        assert MessageRole.USER == "user"
        assert MessageRole.AGENT == "agent"
        assert MessageRole.SYSTEM == "system"

    def test_a2a_message_creation(self):
        """Test A2AMessage model."""
        from src.messenger import A2AMessage, MessageRole

        msg = A2AMessage(
            role=MessageRole.USER,
            parts=[{"kind": "text", "text": "Hello"}],
        )
        assert msg.role == MessageRole.USER
        assert len(msg.parts) == 1
        assert msg.parts[0]["text"] == "Hello"
        assert msg.messageId is not None

    def test_task_status_creation(self):
        """Test TaskStatus model."""
        from src.messenger import TaskStatus, TaskState

        status = TaskStatus(state=TaskState.WORKING, message="Processing...")
        assert status.state == TaskState.WORKING
        assert status.message == "Processing..."
        assert status.timestamp is not None

    def test_a2a_task_creation(self):
        """Test A2ATask model."""
        from src.messenger import A2ATask, TaskStatus, TaskState

        task = A2ATask(
            status=TaskStatus(state=TaskState.COMPLETED),
        )
        assert task.id is not None
        assert task.contextId is not None
        assert task.status.state == TaskState.COMPLETED
        assert task.kind == "task"

    def test_json_rpc_request(self):
        """Test JSONRPCRequest model."""
        from src.messenger import JSONRPCRequest

        request = JSONRPCRequest(method="message/send", params={"test": "value"})
        assert request.jsonrpc == "2.0"
        assert request.method == "message/send"
        assert request.params["test"] == "value"
        assert request.id is not None

    def test_json_rpc_response(self):
        """Test JSONRPCResponse model."""
        from src.messenger import JSONRPCResponse

        response = JSONRPCResponse(result={"data": "test"}, id="123")
        assert response.jsonrpc == "2.0"
        assert response.result["data"] == "test"
        assert response.id == "123"
        assert response.error is None


class TestMessengerFactories:
    """Test message factory functions."""

    def test_create_text_message(self):
        """Test create_text_message function."""
        from src.messenger import create_text_message, MessageRole

        msg = create_text_message("Test message", role=MessageRole.AGENT)
        assert msg.role == MessageRole.AGENT
        assert len(msg.parts) == 1
        assert msg.parts[0]["kind"] == "text"
        assert msg.parts[0]["text"] == "Test message"

    def test_create_task_response(self):
        """Test create_task_response function."""
        from src.messenger import create_task_response, A2ATask, TaskStatus, TaskState

        task = A2ATask(status=TaskStatus(state=TaskState.COMPLETED))
        response = create_task_response(task, "req-123")
        assert response.id == "req-123"
        assert response.result is not None
        assert response.result["status"]["state"] == "completed"

    def test_create_error_response(self):
        """Test create_error_response function."""
        from src.messenger import create_error_response

        response = create_error_response(-32600, "Invalid request", "req-456")
        assert response.id == "req-456"
        assert response.error is not None
        assert response.error["code"] == -32600
        assert response.error["message"] == "Invalid request"

    def test_create_shopper_agent_card(self):
        """Test create_shopper_agent_card function."""
        from src.messenger import create_shopper_agent_card

        card = create_shopper_agent_card("http://localhost:8001")
        assert card.name == "WebShop+ Shopper Agent"
        assert card.url == "http://localhost:8001/a2a"
        assert card.protocolVersion == "0.3.0"
        assert len(card.skills) == 1
        assert card.skills[0].id == "shopping"


class TestMessengerParsing:
    """Test message parsing utilities."""

    def test_get_text_from_message(self):
        """Test get_text_from_message function."""
        from src.messenger import get_text_from_message

        message = {
            "parts": [
                {"kind": "text", "text": "First line"},
                {"kind": "text", "text": "Second line"},
            ]
        }
        text = get_text_from_message(message)
        assert "First line" in text
        assert "Second line" in text

    def test_get_text_from_message_empty(self):
        """Test get_text_from_message with empty message."""
        from src.messenger import get_text_from_message

        assert get_text_from_message({}) == ""
        assert get_text_from_message({"parts": []}) == ""

    def test_extract_task_instruction_with_prefix(self):
        """Test extract_task_instruction with TASK: prefix."""
        from src.messenger import extract_task_instruction

        message = {"parts": [{"kind": "text", "text": "TASK: Find running shoes under $100"}]}
        instruction = extract_task_instruction(message)
        assert instruction == "Find running shoes under $100"

    def test_extract_task_instruction_without_prefix(self):
        """Test extract_task_instruction without TASK: prefix."""
        from src.messenger import extract_task_instruction

        message = {"parts": [{"kind": "text", "text": "Buy a laptop with 16GB RAM"}]}
        instruction = extract_task_instruction(message)
        assert instruction == "Buy a laptop with 16GB RAM"

    def test_extract_observation_with_prefix(self):
        """Test extract_observation with OBSERVATION: prefix."""
        from src.messenger import extract_observation

        message = {"parts": [{"kind": "text", "text": "OBSERVATION: Found 10 products"}]}
        observation = extract_observation(message)
        assert observation == "Found 10 products"

    def test_extract_observation_webshop_keywords(self):
        """Test extract_observation recognizes WebShop output."""
        from src.messenger import extract_observation

        message = {"parts": [{"kind": "text", "text": "Search results: 5 products found. Price: $50"}]}
        observation = extract_observation(message)
        assert "products found" in observation.lower()

    def test_format_action_response_valid(self):
        """Test format_action_response with valid action."""
        from src.messenger import format_action_response

        assert format_action_response("search[running shoes]") == "search[running shoes]"
        assert format_action_response("click[B07XYZ123]") == "click[B07XYZ123]"

    def test_format_action_response_natural_language(self):
        """Test format_action_response with natural language."""
        from src.messenger import format_action_response

        # These should be converted or passed through
        result = format_action_response("search for running shoes")
        assert "search[" in result or "running shoes" in result


# =============================================================================
# Agent Tests
# =============================================================================


class TestAgentModels:
    """Test agent data models."""

    def test_agent_state_enum(self):
        """Test AgentState enum values."""
        from src.agent import AgentState

        assert AgentState.IDLE == "idle"
        assert AgentState.SEARCHING == "searching"
        assert AgentState.PURCHASING == "purchasing"

    def test_task_requirements_model(self):
        """Test TaskRequirements model."""
        from src.agent import TaskRequirements

        req = TaskRequirements(
            product_type="running shoes",
            budget=100.0,
            preferences=["comfortable", "lightweight"],
            constraints=["no leather"],
        )
        assert req.product_type == "running shoes"
        assert req.budget == 100.0
        assert len(req.preferences) == 2
        assert len(req.constraints) == 1

    def test_product_info_model(self):
        """Test ProductInfo model."""
        from src.agent import ProductInfo

        product = ProductInfo(
            asin="B07XYZ123",
            name="Nike Running Shoes",
            price=89.99,
            options=["Size: 10", "Color: Black"],
        )
        assert product.asin == "B07XYZ123"
        assert product.price == 89.99

    def test_session_context_model(self):
        """Test SessionContext model."""
        from src.agent import SessionContext, AgentState

        context = SessionContext(task_id="task-123")
        assert context.task_id == "task-123"
        assert context.state == AgentState.IDLE
        assert len(context.action_history) == 0


class TestShopperAgent:
    """Test ShopperAgent functionality."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock = MagicMock()
        mock.complete.return_value = """PRODUCT_TYPE: running shoes
BUDGET: 100
PREFERENCES: comfortable, lightweight
CONSTRAINTS: none
COMPARISON_REQUIRED: no
SEARCH_QUERY: running shoes comfortable"""
        return mock

    @pytest.fixture
    def agent(self, mock_llm_client):
        """Create a ShopperAgent with mock LLM."""
        from src.agent import ShopperAgent

        agent = ShopperAgent()
        agent.llm_client = mock_llm_client
        return agent

    def test_agent_reset(self, agent):
        """Test agent reset."""
        from src.agent import AgentState

        agent.reset(task_id="test-task")
        assert agent.context.task_id == "test-task"
        assert agent.context.state == AgentState.IDLE
        assert len(agent.context.action_history) == 0

    def test_process_task_instruction(self, agent):
        """Test processing task instruction."""
        agent.reset()
        action = agent.process_task_instruction("Find running shoes under $100")

        assert action.startswith("search[")
        assert agent.context.state.value == "searching"
        assert len(agent.context.action_history) == 1

    def test_process_task_instruction_parses_requirements(self, agent):
        """Test that task instruction parsing extracts requirements."""
        agent.reset()
        agent.process_task_instruction("Find running shoes under $100")

        req = agent.context.requirements
        assert req is not None
        assert "running shoes" in req.product_type.lower()

    def test_process_observation_search_results(self, agent):
        """Test processing search results observation."""
        from src.agent import AgentState

        # First process a task instruction
        agent.reset()
        agent.process_task_instruction("Find running shoes")
        agent.context.state = AgentState.SEARCHING

        # Mock LLM for product selection
        agent.llm_client.complete.return_value = "B07XYZ123"

        # Process search results
        observation = """Search results: 3 products found
1. B07XYZ123 - Nike Running Shoes - $89.99
2. B07ABC456 - Adidas Runner - $79.99
3. B07DEF789 - New Balance Sneakers - $95.00"""

        action = agent.process_observation(observation)

        assert "click[" in action
        assert agent.context.state.value in ["browsing", "viewing_product"]

    def test_process_observation_product_page(self, agent):
        """Test processing product page observation."""
        from src.agent import AgentState, TaskRequirements

        agent.reset()
        agent.context.state = AgentState.VIEWING_PRODUCT
        agent.context.requirements = TaskRequirements(
            product_type="running shoes",
            budget=100.0,
            raw_instruction="Find running shoes under $100",
        )

        # Mock LLM for product evaluation
        agent.llm_client.complete.return_value = """MEETS_BUDGET: yes
HAS_PREFERENCES: yes
VIOLATES_CONSTRAINTS: no
SHOULD_BUY: yes
REASON: Product meets all requirements"""

        observation = """Product page: Nike Air Zoom
Price: $89.99
Options: Size [8, 9, 10, 11], Color [Black, White]
Description: Comfortable running shoes with air cushioning"""

        action = agent.process_observation(observation)

        # Should either select option or buy
        assert "click[" in action

    def test_process_error(self, agent):
        """Test processing error notice."""
        agent.reset()
        agent.context.search_attempts = 0

        action = agent.process_error("Product not found")

        assert "search[" in action or "click[" in action
        assert agent.context.search_attempts > 0

    def test_extract_requirements_simple(self, agent):
        """Test simple requirements extraction (fallback)."""
        req = agent._extract_requirements_simple("Buy a laptop with 16GB RAM under $500")

        assert req.budget == 500.0
        assert "laptop" in req.product_type.lower() or "buy" in req.product_type.lower()

    def test_generate_search_query(self, agent):
        """Test search query generation."""
        from src.agent import TaskRequirements

        agent.context.requirements = TaskRequirements(
            product_type="wireless headphones",
            preferences=["noise cancelling"],
        )

        query = agent._generate_search_query()
        assert "wireless headphones" in query.lower()

    def test_detect_observation_type_search_results(self, agent):
        """Test detection of search results observation."""
        obs = "Search results: 10 products found"
        assert agent._detect_observation_type(obs) == "search_results"

    def test_detect_observation_type_product_page(self, agent):
        """Test detection of product page observation."""
        obs = "Product page\nPrice: $49.99\nOptions: Size, Color"
        assert agent._detect_observation_type(obs) == "product_page"

    def test_detect_observation_type_purchase_complete(self, agent):
        """Test detection of purchase complete observation."""
        obs = "Thank you for your order! Order placed successfully."
        assert agent._detect_observation_type(obs) == "purchase_complete"

    def test_parse_products_from_observation(self, agent):
        """Test parsing products from search results."""
        observation = """Search results:
B07XYZ123 - Nike Shoes - $89.99
B07ABC456 - Adidas Shoes - $79.99"""

        products = agent._parse_products_from_observation(observation)

        assert len(products) >= 2
        assert any(p.asin == "B07XYZ123" for p in products)
        assert any(p.price == 89.99 for p in products)

    def test_parse_product_info(self, agent):
        """Test parsing product info from product page."""
        observation = """Nike Air Zoom Running Shoes
B07XYZ123
Price: $89.99
Options: Size [8, 9, 10], Color [Black, White]
Great for running and training."""

        product = agent._parse_product_info(observation)

        assert product.asin == "B07XYZ123"
        assert product.price == 89.99
        assert "Nike" in product.name or "Nike" in product.description

    def test_needs_option_selection(self, agent):
        """Test detection of option selection need."""
        obs_with_options = "Select size: [Small, Medium, Large]"
        obs_without = "Product page with price $50"

        assert agent._needs_option_selection(obs_with_options) is True
        assert agent._needs_option_selection(obs_without) is False


class TestShopperAgentIntegration:
    """Integration tests for ShopperAgent."""

    @pytest.fixture
    def agent_with_mock_llm(self):
        """Create agent with comprehensive mock."""
        from src.agent import ShopperAgent

        agent = ShopperAgent()

        # Configure mock responses based on input
        def mock_complete(messages, **kwargs):
            content = messages[-1]["content"] if messages else ""

            if "PRODUCT_TYPE:" in content or "Parse the following" in content:
                return """PRODUCT_TYPE: running shoes
BUDGET: 100
PREFERENCES: comfortable
CONSTRAINTS: none
COMPARISON_REQUIRED: no
SEARCH_QUERY: running shoes"""

            elif "select the best product" in content.lower():
                return "B07XYZ123"

            elif "evaluate" in content.lower() or "should i buy" in content.lower():
                return """MEETS_BUDGET: yes
HAS_PREFERENCES: yes
VIOLATES_CONSTRAINTS: no
SHOULD_BUY: yes
REASON: Good product"""

            elif "next action" in content.lower():
                return "click[buy now]"

            return "search[running shoes]"

        agent.llm_client = MagicMock()
        agent.llm_client.complete = mock_complete
        return agent

    def test_full_shopping_flow(self, agent_with_mock_llm):
        """Test a complete shopping flow."""
        agent = agent_with_mock_llm

        # Step 1: Process task
        action1 = agent.process_task_instruction("Find running shoes under $100")
        assert action1.startswith("search[")

        # Step 2: Process search results
        search_results = """Search results: 3 products found
B07XYZ123 - Nike Running Shoes - $89.99
B07ABC456 - Adidas Shoes - $79.99"""

        action2 = agent.process_observation(search_results)
        assert "click[" in action2

        # Step 3: Process product page
        product_page = """Product page: Nike Running Shoes
B07XYZ123
Price: $89.99
Description: Great running shoes"""

        action3 = agent.process_observation(product_page)
        assert "click[" in action3  # Should click buy or an option


# =============================================================================
# Server Tests
# =============================================================================


class TestServerEndpoints:
    """Test server endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.server import app

        return TestClient(app)

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "webshop-plus-purple"

    def test_agent_card_endpoint(self, client):
        """Test agent card endpoint."""
        response = client.get("/.well-known/agent-card.json")
        assert response.status_code == 200

        card = response.json()
        assert card["name"] == "WebShop+ Shopper Agent"
        assert "skills" in card
        assert len(card["skills"]) > 0

    def test_a2a_message_send(self, client):
        """Test A2A message/send method."""
        with patch("src.server._get_or_create_agent") as mock_get_agent:
            # Create a mock agent
            mock_agent = MagicMock()
            mock_agent.context.action_history = []
            mock_agent.process_task_instruction.return_value = "search[test product]"
            mock_get_agent.return_value = mock_agent

            request_body = {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "TASK: Find running shoes"}],
                    }
                },
                "id": "test-123",
            }

            response = client.post("/a2a", json=request_body)
            assert response.status_code == 200

            data = response.json()
            assert data["jsonrpc"] == "2.0"
            assert data["id"] == "test-123"
            assert "result" in data

    def test_a2a_invalid_method(self, client):
        """Test A2A with invalid method."""
        request_body = {
            "jsonrpc": "2.0",
            "method": "invalid/method",
            "params": {},
            "id": "test-456",
        }

        response = client.post("/a2a", json=request_body)
        assert response.status_code == 400

        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32601

    def test_a2a_parse_error(self, client):
        """Test A2A with invalid JSON."""
        response = client.post(
            "/a2a",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400

        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32700


class TestServerSessionManagement:
    """Test server session management."""

    def test_get_or_create_agent_new_session(self):
        """Test creating a new session."""
        from src.server import _get_or_create_agent, state

        # Clear existing sessions
        state.sessions.clear()

        agent = _get_or_create_agent("context-123", "task-456")
        assert "context-123" in state.sessions
        assert agent.context.task_id == "task-456"

    def test_get_or_create_agent_existing_session(self):
        """Test retrieving existing session."""
        from src.server import _get_or_create_agent, state

        # Create initial session
        agent1 = _get_or_create_agent("context-789", "task-001")

        # Get same session
        agent2 = _get_or_create_agent("context-789", "task-002")

        # Should be the same agent instance
        assert agent1 is agent2

    def test_process_message_task_instruction(self):
        """Test processing task instruction message."""
        from src.server import _process_message
        from src.agent import ShopperAgent

        agent = ShopperAgent()
        agent.llm_client = MagicMock()
        agent.llm_client.complete.return_value = """PRODUCT_TYPE: shoes
BUDGET: none
PREFERENCES: none
CONSTRAINTS: none
COMPARISON_REQUIRED: no
SEARCH_QUERY: shoes"""

        action = _process_message(agent, "TASK: Find shoes", {})
        assert "search[" in action

    def test_process_message_observation(self):
        """Test processing observation message."""
        from src.server import _process_message
        from src.agent import ShopperAgent, AgentState

        agent = ShopperAgent()
        agent.llm_client = MagicMock()
        agent.llm_client.complete.return_value = "B07XYZ123"

        # Set up agent state
        agent.context.state = AgentState.SEARCHING
        agent.context.action_history = ["search[shoes]"]

        action = _process_message(
            agent, "Search results: B07XYZ123 - Shoes - $50", {"type": "observation"}
        )
        assert "click[" in action or "search[" in action


# =============================================================================
# LLM Client Tests (with mocking)
# =============================================================================


class TestLLMClient:
    """Test LLM client (with mocked LiteLLM)."""

    def test_llm_client_initialization(self):
        """Test LLMClient initialization."""
        from src.llm_client import LLMClient

        client = LLMClient()
        assert client.config.model == "ollama/qwen3-coder:30b"
        assert client.config.temperature == 0.7

    def test_llm_client_custom_config(self):
        """Test LLMClient with custom config."""
        from src.llm_client import LLMClient

        client = LLMClient(
            model="test-model",
            temperature=0.5,
            max_tokens=1024,
        )
        assert client.config.model == "test-model"
        assert client.config.temperature == 0.5
        assert client.config.max_tokens == 1024

    @patch("src.llm_client.litellm.completion")
    def test_llm_complete(self, mock_completion):
        """Test LLMClient.complete method."""
        from src.llm_client import LLMClient

        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_completion.return_value = mock_response

        client = LLMClient()
        result = client.complete([{"role": "user", "content": "Hello"}])

        assert result == "Test response"
        mock_completion.assert_called_once()

    @patch("src.llm_client.litellm.completion")
    def test_llm_complete_with_response(self, mock_completion):
        """Test LLMClient.complete_with_response method."""
        from src.llm_client import LLMClient

        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Test response"),
                finish_reason="stop",
            )
        ]
        mock_response.model = "test-model"
        mock_response.usage = MagicMock()
        mock_completion.return_value = mock_response

        client = LLMClient()
        result = client.complete_with_response([{"role": "user", "content": "Hello"}])

        assert result.content == "Test response"
        assert result.model == "test-model"
        assert result.finish_reason == "stop"

    @patch("src.llm_client.litellm.completion")
    def test_llm_complete_with_reasoning(self, mock_completion):
        """Test LLMClient.complete_with_reasoning method."""
        from src.llm_client import LLMClient

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Step 1... Step 2... Conclusion"))]
        mock_completion.return_value = mock_response

        client = LLMClient()
        result = client.complete_with_reasoning([{"role": "user", "content": "Analyze this"}])

        assert "Step 1" in result or result == "Step 1... Step 2... Conclusion"

    @patch("src.llm_client.litellm.completion")
    def test_llm_evaluate_with_rubric(self, mock_completion):
        """Test LLMClient.evaluate_with_rubric method."""
        from src.llm_client import LLMClient

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="SCORE: 8/10\nEXPLANATION: Good product selection"
                )
            )
        ]
        mock_completion.return_value = mock_response

        client = LLMClient()
        score, explanation = client.evaluate_with_rubric(
            content="Agent selected Product A",
            rubric="Did the agent make a good choice?",
        )

        assert score == 8
        assert "Good product selection" in explanation


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
