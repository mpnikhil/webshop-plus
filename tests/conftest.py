"""
Shared pytest fixtures for WebShop+ integration tests.

This module provides fixtures for testing the complete A2A + MCP flow
between green and purple agents.
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Generator

import pytest
import httpx

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
GREEN_AGENT_DIR = PROJECT_ROOT / "green_agent"
PURPLE_AGENT_DIR = PROJECT_ROOT / "purple_agent"


# =============================================================================
# Server Process Fixtures
# =============================================================================


class ServerProcess:
    """Wrapper for managing a server subprocess."""

    def __init__(
        self,
        name: str,
        cwd: Path,
        host: str,
        port: int,
        startup_timeout: float = 10.0,
    ):
        self.name = name
        self.cwd = cwd
        self.host = host
        self.port = port
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen | None = None
        self._base_url = f"http://{host}:{port}"

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def a2a_url(self) -> str:
        return f"{self._base_url}/a2a"

    @property
    def agent_card_url(self) -> str:
        return f"{self._base_url}/.well-known/agent-card.json"

    def start(self) -> None:
        """Start the server process."""
        cmd = [
            "uv", "run", "python", "src/server.py",
            "--host", self.host,
            "--port", str(self.port),
            "--log-level", "WARNING",
        ]

        self.process = subprocess.Popen(
            cmd,
            cwd=str(self.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**subprocess.os.environ, "PYTHONPATH": str(self.cwd)},
        )

        # Wait for server to be ready
        self._wait_for_ready()

    def _wait_for_ready(self) -> None:
        """Wait for server to respond to health check."""
        import httpx

        start_time = time.time()
        while time.time() - start_time < self.startup_timeout:
            try:
                response = httpx.get(f"{self._base_url}/health", timeout=1.0)
                if response.status_code == 200:
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            time.sleep(0.1)

        # If we get here, server didn't start
        if self.process:
            self.process.terminate()
            stdout, stderr = self.process.communicate(timeout=5)
            raise RuntimeError(
                f"{self.name} server failed to start:\n"
                f"stdout: {stdout.decode()}\n"
                f"stderr: {stderr.decode()}"
            )

    def stop(self) -> None:
        """Stop the server process."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


@pytest.fixture
def green_server() -> Generator[ServerProcess, None, None]:
    """Start green agent server for integration testing."""
    server = ServerProcess(
        name="Green Agent",
        cwd=GREEN_AGENT_DIR,
        host="127.0.0.1",
        port=18000,  # Use non-standard port to avoid conflicts
    )
    server.start()
    yield server
    server.stop()


@pytest.fixture
def purple_server() -> Generator[ServerProcess, None, None]:
    """Start purple agent server for integration testing."""
    server = ServerProcess(
        name="Purple Agent",
        cwd=PURPLE_AGENT_DIR,
        host="127.0.0.1",
        port=18001,  # Use non-standard port to avoid conflicts
    )
    server.start()
    yield server
    server.stop()


# =============================================================================
# Test Client Fixtures
# =============================================================================


@pytest.fixture
def green_app():
    """Create green agent app for testing."""
    import importlib.util

    # Load green agent's server module directly
    spec = importlib.util.spec_from_file_location(
        "green_server",
        str(GREEN_AGENT_DIR / "src" / "server.py"),
    )
    green_module = importlib.util.module_from_spec(spec)

    # Add green_agent to path temporarily for imports within the module
    sys.path.insert(0, str(GREEN_AGENT_DIR))
    try:
        spec.loader.exec_module(green_module)
        return green_module.create_app(
            card_url="http://localhost:18000",
            host="localhost",
            port=18000,
        )
    finally:
        sys.path.remove(str(GREEN_AGENT_DIR))


@pytest.fixture
def purple_app():
    """Create purple agent app for testing."""
    import importlib.util

    # Load purple agent's server module directly
    spec = importlib.util.spec_from_file_location(
        "purple_server",
        str(PURPLE_AGENT_DIR / "src" / "server.py"),
    )
    purple_module = importlib.util.module_from_spec(spec)

    # Add purple_agent to path temporarily for imports within the module
    sys.path.insert(0, str(PURPLE_AGENT_DIR))
    try:
        spec.loader.exec_module(purple_module)
        return purple_module.create_app(card_url="http://localhost:18001")
    finally:
        sys.path.remove(str(PURPLE_AGENT_DIR))


# =============================================================================
# A2A Request Helpers
# =============================================================================


def make_a2a_request(
    method: str,
    params: dict[str, Any],
    request_id: str = "test-1",
) -> dict[str, Any]:
    """Create a JSON-RPC A2A request."""
    return {
        "jsonrpc": "2.0",
        "method": method,
        "id": request_id,
        "params": params,
    }


def make_message_send_request(
    text: str,
    metadata: dict[str, Any] | None = None,
    message_id: str | None = None,
) -> dict[str, Any]:
    """Create a message/send A2A request."""
    return make_a2a_request(
        method="message/send",
        params={
            "message": {
                "messageId": message_id or f"msg-{time.time()}",
                "role": "user",
                "parts": [{"kind": "text", "text": text}],
            },
            "metadata": metadata or {},
        },
    )


def make_mcp_kickoff_message(
    goal: str,
    budget: float,
    constraints: list[str],
    mcp_uri: str,
) -> dict[str, Any]:
    """Create an MCP kickoff message payload."""
    return {
        "goal": goal,
        "budget": budget,
        "constraints": constraints,
        "resources": [
            {"type": "mcp", "uri": mcp_uri}
        ],
    }


# =============================================================================
# Async Fixtures
# =============================================================================


@pytest.fixture
async def async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an async HTTP client for testing."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client


# =============================================================================
# Sample Task Data
# =============================================================================


@pytest.fixture
def sample_budget_task() -> dict[str, Any]:
    """Sample budget-constrained shopping task."""
    return {
        "goal": "Find running shoes under $50",
        "budget": 50.0,
        "constraints": ["waterproof", "size 10"],
    }


@pytest.fixture
def sample_mcp_session_id() -> str:
    """Sample MCP session ID."""
    return "test-session-12345"


@pytest.fixture
def sample_mcp_uri(sample_mcp_session_id: str) -> str:
    """Sample MCP URI."""
    return f"http://localhost:18000/mcp/{sample_mcp_session_id}"
