"""
Tests for the LLM client module.

These tests include:
- Unit tests with mocked LiteLLM responses
- Integration tests with LM Studio (skipped if not available)
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.llm_client import LLMClient, LLMResponse, get_default_client
import src.llm_client as llm_client_module


class TestLLMResponse:
    """Tests for LLMResponse model."""

    def test_response_creation(self):
        """Test creating an LLM response."""
        response = LLMResponse(
            content="Hello!",
            model="openai/qwen3-coder-30b-a3b-instruct-mlx",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            finish_reason="stop",
        )
        assert response.content == "Hello!"
        assert response.model == "openai/qwen3-coder-30b-a3b-instruct-mlx"
        assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5}
        assert response.finish_reason == "stop"

    def test_response_defaults(self):
        """Test response with minimal fields."""
        # Fix: LLMResponse requires content and model as per its definition
        response = LLMResponse(content="Test", model="test-model")
        assert response.usage == {}
        assert response.finish_reason is None


# =============================================================================
# Client Initialization Tests
# ============================================================================


class TestLLMClientInit:
    """Tests for LLMClient initialization."""

    def test_default_initialization(self):
        """Test client with default configuration."""
        client = LLMClient()
        assert client.model is not None
        assert client.config.temperature == 0.7
        assert client.config.timeout == 120.0

    def test_initialization_with_env_vars(self):
        """Test client uses environment variables."""
        with patch.dict(os.environ, {
            "LLM_MODEL": "nebius/Qwen/Qwen3-32B",
            "LLM_API_KEY": "test-api-key",
            "LLM_API_BASE": "https://custom.api.com"
        }):
            client = LLMClient()
            assert client.model == "nebius/Qwen/Qwen3-32B"
            assert client.config.api_key == "test-api-key"
            assert client.config.api_base == "https://custom.api.com"

    def test_initialization_with_explicit_params(self):
        """Test explicit params override defaults."""
        client = LLMClient(
            model="explicit-model",
            api_key="explicit-key",
            temperature=0.5,
            max_tokens=1000,
        )
        assert client.model == "explicit-model"
        assert client.config.api_key == "explicit-key"
        assert client.config.temperature == 0.5
        assert client.config.max_tokens == 1000

    def test_model_property(self):
        """Test model property returns configured model."""
        client = LLMClient(model="test-model")
        assert client.model == "test-model"


# =============================================================================
# Completion Tests (Mocked)
# ============================================================================


class TestLLMClientComplete:
    """Tests for LLMClient.complete() method."""

    def _create_mock_response(self, content: str, model: str = "test-model"):
        """Create a mock LiteLLM response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = model
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 5}
        return mock_response

    @patch("src.llm_client.litellm.completion")
    def test_simple_completion(self, mock_completion):
        """Test basic completion call."""
        mock_completion.return_value = self._create_mock_response("Hello!")

        client = LLMClient(model="test-model")
        result = client.complete([{"role": "user", "content": "Hi"}])

        assert result == "Hello!"
        mock_completion.assert_called_once()

    @patch("src.llm_client.litellm.completion")
    def test_completion_strips_thinking(self, mock_completion):
        """Test that thinking tags are stripped."""
        mock_completion.return_value = self._create_mock_response("<think>Hmmm...</think>  Final Answer")

        client = LLMClient()
        result = client.complete([{"role": "user", "content": "Test"}])

        assert result.strip() == "Final Answer"

    @patch("src.llm_client.litellm.completion")
    def test_completion_appends_think_prompt(self, mock_completion):
        """Test that /think is appended to prompt."""
        mock_completion.return_value = self._create_mock_response("Response")

        client = LLMClient()
        client.complete([{"role": "user", "content": "Original"}])

        call_kwargs = mock_completion.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["content"] == "Original\n\n/think"


class TestLLMClientCompleteWithResponse:
    """Tests for LLMClient.complete_with_response() method."""

    @patch("src.llm_client.litellm.completion")
    def test_complete_with_response(self, mock_completion):
        """Test getting full response with metadata."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test content"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "openai/qwen3-coder-30b-a3b-instruct-mlx"
        mock_response.usage = {"prompt_tokens": 15, "completion_tokens": 8}
        mock_completion.return_value = mock_response

        client = LLMClient()
        response = client.complete_with_response([{"role": "user", "content": "Test"}])

        assert isinstance(response, LLMResponse)
        assert response.content == "Test content"
        assert response.model == "openai/qwen3-coder-30b-a3b-instruct-mlx"
        assert response.finish_reason == "stop"


# =============================================================================
# Evaluation & Reasoning Tests (Mocked)
# ============================================================================


class TestLLMClientEvaluation:
    """Tests for specialized completion methods."""

    @patch("src.llm_client.litellm.completion")
    def test_reasoning_adds_system_prompt(self, mock_completion):
        """Test that reasoning mode adds step-by-step system prompt."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Step 1: ... Step 2: ... Conclusion: ..."
        mock_completion.return_value = mock_response

        client = LLMClient()
        client.complete_with_reasoning([{"role": "user", "content": "Compare A and B"}])

        call_kwargs = mock_completion.call_args[1]
        messages = call_kwargs["messages"]

        # First message should be the reasoning system prompt
        assert messages[0]["role"] == "system"
        assert "expert analyst" in messages[0]["content"].lower()

    @patch("src.llm_client.litellm.completion")
    def test_evaluate_parses_score(self, mock_completion):
        """Test that evaluation correctly parses score from response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "SCORE: 8\nEXPLANATION: Good choice."
        mock_completion.return_value = mock_response

        client = LLMClient()
        score, explanation = client.evaluate_with_rubric(
            content="Agent chose Product A",
            rubric="Did the agent make a good choice?",
        )

        assert score == 8
        assert "Good choice" in explanation

    @patch("src.llm_client.litellm.completion")
    def test_evaluate_handles_score_with_max(self, mock_completion):
        """Test parsing score in format '7/10'."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "SCORE: 7/10\nEXPLANATION: Reasonable."
        mock_completion.return_value = mock_response

        client = LLMClient()
        score, _ = client.evaluate_with_rubric(
            content="Test",
            rubric="Test rubric",
            max_score=10,
        )

        assert score == 7


# =============================================================================
# Factory Function Tests
# ============================================================================


class TestGetDefaultClient:
    """Tests for get_default_client() factory function."""

    def test_get_default_client_returns_client(self):
        """Test factory returns a configured client."""
        client = get_default_client()
        assert isinstance(client, LLMClient)

    def test_get_default_client_uses_env(self):
        """Test factory uses environment variables."""
        with patch.dict(os.environ, {"LLM_MODEL": "test-model"}):
            client = get_default_client()
            assert client.model == "test-model"
