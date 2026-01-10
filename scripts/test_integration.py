#!/usr/bin/env python3
"""
Quick integration test for WebShop+ agents.

Tests that the green agent can communicate with the purple agent
and complete a simple task flow.

This test uses subprocess isolation to avoid module import conflicts.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


async def test_green_agent_basic():
    """Test green agent basic functionality."""
    cmd = [
        "uv", "run", "python", "-c", """
import json
from fastapi.testclient import TestClient
from src.server import app

client = TestClient(app)

# Test agent card
response = client.get("/.well-known/agent-card.json")
assert response.status_code == 200, f"Agent card failed: {response.status_code}"
card = response.json()
assert "name" in card, "Missing name in agent card"
print(f"  Green agent card: {card.get('name')}")

# Test health
response = client.get("/health")
assert response.status_code == 200, f"Health check failed: {response.status_code}"
print("  Green agent health: OK")

# Test A2A message (simple)
request = {
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "Hello"}],
        }
    },
    "id": "test-1",
}
response = client.post("/a2a", json=request)
assert response.status_code == 200, f"A2A message failed: {response.status_code}"
result = response.json()
assert "result" in result or "error" not in result, f"A2A error: {result}"
print("  Green agent A2A: OK")
print("SUCCESS")
"""
    ]

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT / "green_agent"),
        capture_output=True,
        text=True,
        timeout=60,
        env={**subprocess.os.environ, "PYTHONPATH": str(PROJECT_ROOT / "green_agent")},
    )

    if "SUCCESS" in result.stdout:
        for line in result.stdout.strip().split("\n"):
            if line.strip().startswith("  "):
                print(line)
        return True
    else:
        print(f"  FAILED: {result.stderr or result.stdout}")
        return False


async def test_purple_agent_basic():
    """Test purple agent basic functionality."""
    cmd = [
        "uv", "run", "python", "-c", """
import json
from fastapi.testclient import TestClient
from src.server import app

client = TestClient(app)

# Test agent card
response = client.get("/.well-known/agent-card.json")
assert response.status_code == 200, f"Agent card failed: {response.status_code}"
card = response.json()
assert "name" in card, "Missing name in agent card"
print(f"  Purple agent card: {card.get('name')}")

# Test health
response = client.get("/health")
assert response.status_code == 200, f"Health check failed: {response.status_code}"
print("  Purple agent health: OK")

# Test A2A message with task instruction
request = {
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
        "message": {
            "role": "user",
            "parts": [
                {"kind": "text", "text": "TASK: Find running shoes under $100"}
            ],
        },
        "metadata": {"type": "task_instruction"},
    },
    "id": "test-1",
}
response = client.post("/a2a", json=request)
assert response.status_code == 200, f"A2A message failed: {response.status_code}"
result = response.json()

# Check that we got an action back
assert "result" in result, f"Missing result: {result}"
task = result.get("result", {})
history = task.get("history", [])

# Find the agent's response
agent_response = None
for msg in history:
    if msg.get("role") == "agent":
        for part in msg.get("parts", []):
            if part.get("kind") == "text":
                agent_response = part.get("text", "")
                break

assert agent_response, f"No agent response found in: {history}"
print(f"  Purple agent action: {agent_response[:50]}...")

# Verify it's a search action
assert "search[" in agent_response.lower(), f"Expected search action, got: {agent_response}"
print("  Purple agent A2A: OK")
print("SUCCESS")
"""
    ]

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT / "purple_agent"),
        capture_output=True,
        text=True,
        timeout=60,
        env={**subprocess.os.environ, "PYTHONPATH": str(PROJECT_ROOT / "purple_agent")},
    )

    if "SUCCESS" in result.stdout:
        for line in result.stdout.strip().split("\n"):
            if line.strip().startswith("  "):
                print(line)
        return True
    else:
        print(f"  FAILED: {result.stderr or result.stdout}")
        return False


async def test_executor_communication():
    """Test that the executor can communicate with purple agent."""
    cmd = [
        "uv", "run", "python", "-c", """
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.executor import Executor, ExecutorConfig

async def test():
    # Create a mock A2A client that simulates purple agent response
    mock_response = MagicMock()
    mock_response.error = None
    mock_response.result = {
        "id": "task-1",
        "status": {"state": "completed"},
        "history": [
            {
                "role": "agent",
                "parts": [{"kind": "text", "text": "search[running shoes]"}],
            }
        ],
    }

    config = ExecutorConfig(timeout=10.0, action_timeout=5.0)

    with patch("src.executor.A2AClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.send_message = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = mock_client

        async with Executor(config=config) as executor:
            result = await executor.send_task_instruction(
                endpoint="http://localhost:8001/a2a",
                instruction="Find running shoes under $100",
                task_id="test-task",
                context_id="test-context",
            )

            assert result.action is not None, f"No action returned: {result}"
            print(f"  Executor result: {result.action}")

    print("  Executor communication: OK")
    print("SUCCESS")
    return True

asyncio.run(test())
"""
    ]

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT / "green_agent"),
        capture_output=True,
        text=True,
        timeout=60,
        env={**subprocess.os.environ, "PYTHONPATH": str(PROJECT_ROOT / "green_agent")},
    )

    if "SUCCESS" in result.stdout:
        for line in result.stdout.strip().split("\n"):
            if line.strip().startswith("  "):
                print(line)
        return True
    else:
        print(f"  FAILED: {result.stderr or result.stdout}")
        return False


async def main():
    """Run all integration tests."""
    print("\nWebShop+ Integration Tests")
    print("=" * 50)

    tests = [
        ("Green Agent Basic", test_green_agent_basic),
        ("Purple Agent Basic", test_purple_agent_basic),
        ("Executor Communication", test_executor_communication),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n{name}:")
        try:
            result = await test_func()
            if result:
                passed += 1
                print(f"  PASSED")
            else:
                failed += 1
                print(f"  FAILED")
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
