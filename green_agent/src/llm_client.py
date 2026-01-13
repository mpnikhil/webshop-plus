"""
LLM Client for WebShop+ using LiteLLM.

This module provides a provider-agnostic interface to LLMs using LiteLLM.
It supports local LM Studio models and cloud providers like Nebius.

Configuration:
    Local development: openai/model_name (via LM Studio)
    Production: nebius/Qwen/Qwen3-32B
"""

import os
from typing import Optional

import litellm
from pydantic import BaseModel


class LLMConfig(BaseModel):
    """Configuration for the LLM client."""

    model: str = "openai/qwen3-coder-30b-a3b-instruct-mlx"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: float = 120.0


class LLMResponse(BaseModel):
    """Response from an LLM completion request."""

    content: str
    model: str
    usage: dict = {}
    finish_reason: Optional[str] = None


class LLMClient:
    """
    Provider-agnostic LLM client using LiteLLM.

    Supports LM Studio (local) and Nebius (cloud) with automatic configuration
    based on model prefix.

    Example:
        >>> client = LLMClient()  # Uses default from env or openai/model_name
        >>> response = client.complete([{"role": "user", "content": "Hello!"}])
        >>> print(response)

        >>> client = LLMClient(model="nebius/Qwen/Qwen3-32B", api_key="...")
        >>> response = client.complete_with_reasoning([...])
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ) -> None:
        """
        Initialize the LLM client.

        Args:
            model: Model identifier (e.g., "openai/model_name" for LM Studio).
                   Defaults to LLM_MODEL env var or "openai/qwen3-coder-30b-a3b-instruct-mlx".
            api_key: API key for cloud providers. Defaults to LLM_API_KEY env var.
            api_base: Custom API base URL. Usually auto-detected from model prefix.
            temperature: Sampling temperature (0.0-1.0). Default 0.7.
            max_tokens: Maximum tokens to generate. Default 2048.
            timeout: Request timeout in seconds. Default 120.
        """
        self.config = LLMConfig(
            model=model or os.getenv("LLM_MODEL", "openai/qwen3-coder-30b-a3b-instruct-mlx"),
            api_key=api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            api_base=api_base or os.getenv("LLM_API_BASE") or os.getenv("OPENAI_API_BASE"),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        # Configure LiteLLM settings
        litellm.drop_params = True  # Drop unsupported params gracefully
        litellm.set_verbose = False  # Reduce noise

    @property
    def model(self) -> str:
        """Return the configured model identifier."""
        return self.config.model

    def _get_completion_kwargs(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Build kwargs for litellm.completion call."""
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "timeout": self.config.timeout,
        }

        # Add API key if provided (for cloud providers)
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key

        # Add custom API base if provided
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base

        return kwargs

    def complete(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a completion from the LLM.

        Args:
            messages: List of message dicts with "role" and "content" keys.
                      Roles: "system", "user", "assistant".
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            The generated text content.

        Raises:
            litellm.exceptions.APIError: If the API call fails.

        Example:
            >>> response = client.complete([
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "What is 2+2?"}
            ... ])
            >>> print(response)  # "4"
        """
        kwargs = self._get_completion_kwargs(messages, temperature, max_tokens)
        response = litellm.completion(**kwargs)

        return response.choices[0].message.content

    def complete_with_response(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a completion and return full response metadata.

        Args:
            messages: List of message dicts.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            LLMResponse with content, model info, and usage stats.
        """
        kwargs = self._get_completion_kwargs(messages, temperature, max_tokens)
        response = litellm.completion(**kwargs)

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content,
            model=response.model,
            usage=dict(response.usage) if response.usage else {},
            finish_reason=choice.finish_reason,
        )

    def complete_with_reasoning(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a completion with chain-of-thought reasoning.

        Prepends a system message encouraging step-by-step thinking.
        Useful for complex evaluation tasks like comparative reasoning.

        Args:
            messages: List of message dicts.
            temperature: Override default temperature. Defaults to 0.3 for reasoning.
            max_tokens: Override default max tokens. Defaults to 4096 for reasoning.

        Returns:
            The generated reasoning and conclusion.

        Example:
            >>> response = client.complete_with_reasoning([
            ...     {"role": "user", "content": "Compare these two products: ..."}
            ... ])
        """
        reasoning_system = {
            "role": "system",
            "content": (
                "You are an expert analyst. Think step by step through the problem. "
                "First, identify the key factors to consider. "
                "Then, analyze each factor systematically. "
                "Finally, provide your conclusion with clear justification."
            ),
        }

        # Insert reasoning system prompt at the beginning
        augmented_messages = [reasoning_system] + [
            msg for msg in messages if msg.get("role") != "system"
        ]

        # Add any existing system messages after our reasoning prompt
        for msg in messages:
            if msg.get("role") == "system":
                augmented_messages.insert(1, msg)

        # Use lower temperature for more focused reasoning
        temp = temperature if temperature is not None else 0.3
        tokens = max_tokens if max_tokens is not None else 4096

        return self.complete(augmented_messages, temperature=temp, max_tokens=tokens)

    def evaluate_with_rubric(
        self,
        content: str,
        rubric: str,
        max_score: int = 10,
    ) -> tuple[int, str]:
        """
        Evaluate content against a rubric using LLM-as-judge.

        Args:
            content: The content to evaluate.
            rubric: The evaluation criteria/rubric.
            max_score: Maximum possible score. Default 10.

        Returns:
            Tuple of (score, explanation).

        Example:
            >>> score, explanation = client.evaluate_with_rubric(
            ...     content="Agent chose Product A over B",
            ...     rubric="Did the agent choose the best value for money?",
            ... )
        """
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an expert evaluator. Score the following content from 0 to {max_score} "
                    f"based on the rubric provided. Respond in the format:\n"
                    f"SCORE: <number>\nEXPLANATION: <reasoning>"
                ),
            },
            {
                "role": "user",
                "content": f"RUBRIC:\n{rubric}\n\nCONTENT TO EVALUATE:\n{content}",
            },
        ]

        response = self.complete(messages, temperature=0.2, max_tokens=1024)

        # Parse score from response
        score = 0
        explanation = response

        lines = response.strip().split("\n")
        for i, line in enumerate(lines):
            if line.upper().startswith("SCORE:"):
                try:
                    score_str = line.split(":", 1)[1].strip()
                    # Handle formats like "8/10" or just "8"
                    score = int(score_str.split("/")[0].strip())
                    score = max(0, min(score, max_score))  # Clamp to valid range
                except (ValueError, IndexError):
                    pass
            elif line.upper().startswith("EXPLANATION:"):
                explanation = "\n".join(lines[i:]).split(":", 1)[1].strip()
                break

        return score, explanation


def get_default_client() -> LLMClient:
    """
    Create an LLM client with default configuration from environment.

    Returns:
        LLMClient configured from LLM_MODEL and LLM_API_KEY env vars.
    """
    return LLMClient()
