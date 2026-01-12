"""
Tests for MCP URI and task data extraction in the Executor.

Stage 8 of the AAA (A2A + MCP Agentification) implementation.

These tests verify that the purple agent can correctly parse kickoff messages
from the green agent containing MCP URIs and task data.
"""

import json
import uuid

import pytest
from a2a.types import Message, Part, Role, TextPart

from src.executor import Executor


@pytest.fixture
def executor():
    """Create a fresh Executor instance for each test."""
    return Executor()


def create_test_message(content: str) -> Message:
    """Create a test Message with the given text content.

    Args:
        content: The text content for the message.

    Returns:
        An A2A Message with the content as a TextPart.
    """
    return Message(
        messageId=str(uuid.uuid4()),
        role=Role.user,
        parts=[TextPart(text=content)],
    )


def create_kickoff_payload(
    goal: str = "Find running shoes",
    budget: float = 50.0,
    constraints: list[str] | None = None,
    mcp_uri: str | None = None,
) -> str:
    """Create a JSON kickoff payload.

    Args:
        goal: The shopping task goal.
        budget: Maximum spending allowed.
        constraints: List of constraints.
        mcp_uri: Optional MCP server URI.

    Returns:
        JSON string of the kickoff payload.
    """
    payload: dict = {
        "goal": goal,
        "budget": budget,
        "constraints": constraints or [],
    }
    if mcp_uri:
        payload["resources"] = [
            {
                "type": "mcp",
                "uri": mcp_uri,
                "description": "WebShop MCP server",
            }
        ]
    return json.dumps(payload)


class TestExtractMcpUri:
    """Tests for _extract_mcp_uri method."""

    def test_extract_mcp_uri_found(self, executor):
        """MCP URI is successfully extracted from kickoff message."""
        mcp_uri = "http://localhost:8000/mcp/session-abc123"
        payload = create_kickoff_payload(mcp_uri=mcp_uri)
        message = create_test_message(payload)

        result = executor._extract_mcp_uri(message)

        assert result == mcp_uri

    def test_extract_mcp_uri_not_found_no_resources(self, executor):
        """Returns None when resources array is missing."""
        payload = create_kickoff_payload()  # No MCP URI
        message = create_test_message(payload)

        result = executor._extract_mcp_uri(message)

        assert result is None

    def test_extract_mcp_uri_not_found_wrong_type(self, executor):
        """Returns None when resource type is not 'mcp'."""
        payload = json.dumps({
            "goal": "Find shoes",
            "resources": [
                {"type": "other", "uri": "http://example.com"}
            ]
        })
        message = create_test_message(payload)

        result = executor._extract_mcp_uri(message)

        assert result is None

    def test_extract_mcp_uri_multiple_resources(self, executor):
        """Extracts MCP URI from multiple resources."""
        payload = json.dumps({
            "goal": "Find shoes",
            "resources": [
                {"type": "http", "uri": "http://other.com"},
                {"type": "mcp", "uri": "http://localhost:8000/mcp/session-123"},
                {"type": "file", "uri": "file://local"},
            ]
        })
        message = create_test_message(payload)

        result = executor._extract_mcp_uri(message)

        assert result == "http://localhost:8000/mcp/session-123"

    def test_extract_mcp_uri_empty_message(self, executor):
        """Returns None for empty message content."""
        message = create_test_message("")

        result = executor._extract_mcp_uri(message)

        assert result is None


class TestExtractTaskData:
    """Tests for _extract_task_data method."""

    def test_extract_task_data_full(self, executor):
        """Successfully extracts all task data fields."""
        payload = create_kickoff_payload(
            goal="Find waterproof running shoes",
            budget=75.50,
            constraints=["waterproof", "size 10", "black color"],
        )
        message = create_test_message(payload)

        result = executor._extract_task_data(message)

        assert result == {
            "goal": "Find waterproof running shoes",
            "budget": 75.50,
            "constraints": ["waterproof", "size 10", "black color"],
        }

    def test_extract_task_data_minimal(self, executor):
        """Uses default values when budget/constraints missing."""
        payload = json.dumps({"goal": "Find shoes"})
        message = create_test_message(payload)

        result = executor._extract_task_data(message)

        assert result == {
            "goal": "Find shoes",
            "budget": 100.0,  # Default
            "constraints": [],  # Default
        }

    def test_extract_task_data_no_goal(self, executor):
        """Returns empty dict when goal is missing."""
        payload = json.dumps({"budget": 50.0})
        message = create_test_message(payload)

        result = executor._extract_task_data(message)

        assert result == {}

    def test_extract_task_data_empty_message(self, executor):
        """Returns empty dict for empty message content."""
        message = create_test_message("")

        result = executor._extract_task_data(message)

        assert result == {}

    def test_extract_task_data_budget_as_string(self, executor):
        """Converts string budget to float."""
        payload = json.dumps({
            "goal": "Find shoes",
            "budget": "45.99",
        })
        message = create_test_message(payload)

        result = executor._extract_task_data(message)

        assert result["budget"] == 45.99


class TestMalformedJson:
    """Tests for handling malformed JSON gracefully."""

    def test_mcp_uri_malformed_json(self, executor):
        """Returns None for malformed JSON without crashing."""
        message = create_test_message("{ not valid json }")

        result = executor._extract_mcp_uri(message)

        assert result is None

    def test_task_data_malformed_json(self, executor):
        """Returns empty dict for malformed JSON without crashing."""
        message = create_test_message("{ incomplete json")

        result = executor._extract_task_data(message)

        assert result == {}

    def test_mcp_uri_plain_text(self, executor):
        """Returns None for plain text (not JSON)."""
        message = create_test_message("Find running shoes under $50")

        result = executor._extract_mcp_uri(message)

        assert result is None

    def test_task_data_plain_text(self, executor):
        """Returns empty dict for plain text (not JSON)."""
        message = create_test_message("Find running shoes under $50")

        result = executor._extract_task_data(message)

        assert result == {}

    def test_mcp_uri_null_values(self, executor):
        """Handles null values in resources gracefully."""
        payload = json.dumps({
            "goal": "Find shoes",
            "resources": [None, {"type": "mcp", "uri": None}]
        })
        message = create_test_message(payload)

        result = executor._extract_mcp_uri(message)

        assert result is None

    def test_mcp_uri_wrong_structure(self, executor):
        """Handles resources with wrong structure gracefully."""
        payload = json.dumps({
            "goal": "Find shoes",
            "resources": "not-a-list"
        })
        message = create_test_message(payload)

        result = executor._extract_mcp_uri(message)

        assert result is None
