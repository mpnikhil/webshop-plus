"""
Tests for WebShop+ Purple Agent.

This module contains comprehensive tests for:
- Messenger utilities (SDK-based A2A protocol)
- ShopperAgent (shopping logic)
- LLM Client
"""

import pytest
from unittest.mock import MagicMock, patch


# =============================================================================
# Messenger Tests (SDK-based utilities)
# =============================================================================


class TestMessengerUtilities:
    """Test SDK-based messenger utilities."""

    def test_create_message(self):
        """Test create_message function creates valid Message."""
        from a2a.types import Role
        from src.messenger import create_message

        msg = create_message(Role.agent, "search[shoes]")
        assert msg.role == Role.agent
        assert len(msg.parts) == 1
        assert msg.message_id is not None

    def test_create_message_with_context(self):
        """Test create_message with context ID."""
        from a2a.types import Role
        from src.messenger import create_message

        msg = create_message(Role.user, "Hello", context_id="ctx-123")
        assert msg.context_id == "ctx-123"

    def test_get_message_text(self):
        """Test get_message_text extracts text from SDK Message."""
        from a2a.types import Message, Role, TextPart
        from src.messenger import get_message_text

        msg = Message(
            messageId="123",
            role=Role.user,
            parts=[TextPart(text="Hello world")],
        )
        text = get_message_text(msg)
        assert text == "Hello world"

    def test_get_message_text_multiple_parts(self):
        """Test get_message_text with multiple text parts."""
        from a2a.types import Message, Role, TextPart
        from src.messenger import get_message_text

        msg = Message(
            messageId="123",
            role=Role.user,
            parts=[TextPart(text="Line 1"), TextPart(text="Line 2")],
        )
        text = get_message_text(msg)
        assert "Line 1" in text
        assert "Line 2" in text

    def test_merge_parts(self):
        """Test merge_parts combines text parts."""
        from a2a.types import TextPart
        from src.messenger import merge_parts

        parts = [TextPart(text="Part 1"), TextPart(text="Part 2")]
        # Note: merge_parts expects SDK Part objects
        # For simplicity, we test the function works with direct TextPart objects
        result = merge_parts(parts)
        assert "Part 1" in result
        assert "Part 2" in result

    def test_get_text_from_dict_message(self):
        """Test get_text_from_dict_message function."""
        from src.messenger import get_text_from_dict_message

        message = {
            "parts": [
                {"kind": "text", "text": "First line"},
                {"kind": "text", "text": "Second line"},
            ]
        }
        text = get_text_from_dict_message(message)
        assert "First line" in text
        assert "Second line" in text

    def test_get_text_from_dict_message_empty(self):
        """Test get_text_from_dict_message with empty message."""
        from src.messenger import get_text_from_dict_message

        assert get_text_from_dict_message({}) == ""
        assert get_text_from_dict_message({"parts": []}) == ""


class TestMessengerParsing:
    """Test message parsing utilities."""

    def test_extract_task_instruction_with_prefix(self):
        """Test extract_task_instruction with TASK: prefix."""
        from src.messenger import extract_task_instruction

        instruction = extract_task_instruction("TASK: Find running shoes under $100")
        assert instruction == "Find running shoes under $100"

    def test_extract_task_instruction_without_prefix(self):
        """Test extract_task_instruction without TASK: prefix."""
        from src.messenger import extract_task_instruction

        instruction = extract_task_instruction("Buy a laptop with 16GB RAM")
        assert instruction == "Buy a laptop with 16GB RAM"

    def test_extract_task_instruction_empty(self):
        """Test extract_task_instruction with empty string."""
        from src.messenger import extract_task_instruction

        assert extract_task_instruction("") is None
        assert extract_task_instruction(None) is None

    def test_extract_observation_with_prefix(self):
        """Test extract_observation with OBSERVATION: prefix."""
        from src.messenger import extract_observation

        observation = extract_observation("OBSERVATION: Found 10 products")
        assert observation == "Found 10 products"

    def test_extract_observation_webshop_keywords(self):
        """Test extract_observation recognizes WebShop output."""
        from src.messenger import extract_observation

        observation = extract_observation("Search results: 5 products found. Price: $50")
        assert "products found" in observation.lower()

    def test_extract_observation_empty(self):
        """Test extract_observation with empty string."""
        from src.messenger import extract_observation

        assert extract_observation("") is None
        assert extract_observation(None) is None

    def test_format_action_response_valid(self):
        """Test format_action_response with valid action."""
        from src.messenger import format_action_response

        assert format_action_response("search[running shoes]") == "search[running shoes]"
        assert format_action_response("click[B07XYZ123]") == "click[B07XYZ123]"

    def test_format_action_response_natural_language(self):
        """Test format_action_response with natural language."""
        from src.messenger import format_action_response

        result = format_action_response("search for running shoes")
        assert "search[" in result


class TestMessengerClass:
    """Test Messenger class functionality."""

    def test_messenger_initialization(self):
        """Test Messenger initializes with empty context."""
        from src.messenger import Messenger

        messenger = Messenger()
        assert messenger._context_ids == {}

    def test_messenger_reset(self):
        """Test Messenger reset clears context IDs."""
        from src.messenger import Messenger

        messenger = Messenger()
        messenger._context_ids["http://example.com/a2a"] = "ctx-123"
        messenger.reset()
        assert messenger._context_ids == {}

    def test_messenger_get_context_id(self):
        """Test Messenger get_context_id returns correct ID."""
        from src.messenger import Messenger

        messenger = Messenger()
        messenger._context_ids["http://example.com/a2a"] = "ctx-123"
        assert messenger.get_context_id("http://example.com/a2a") == "ctx-123"
        assert messenger.get_context_id("http://other.com/a2a") is None


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
# LLM Client Tests (with mocking)
# =============================================================================


class TestLLMClient:
    """Test LLM client (with mocked LiteLLM)."""

    @patch.dict("os.environ", {}, clear=True)
    def test_llm_client_initialization(self):
        """Test LLMClient initialization with default values."""
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
