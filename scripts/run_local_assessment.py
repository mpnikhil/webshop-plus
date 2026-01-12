#!/usr/bin/env python3
"""
Local Assessment Runner for WebShop+

This script runs an end-to-end assessment using both the green agent (evaluator)
and purple agent (shopper) locally. It:
1. Starts both agent servers
2. Waits for them to be healthy
3. Sends an assessment request to the green agent
4. Monitors progress and displays results

Usage:
    python scripts/run_local_assessment.py --tasks 3
    python scripts/run_local_assessment.py --task-types budget_constrained negative_constraint
    python scripts/run_local_assessment.py --full
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx


# Configuration
GREEN_AGENT_PORT = 8000
PURPLE_AGENT_PORT = 8001
GREEN_AGENT_URL = f"http://localhost:{GREEN_AGENT_PORT}"
PURPLE_AGENT_URL = f"http://localhost:{PURPLE_AGENT_PORT}"
PROJECT_ROOT = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run WebShop+ local assessment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=3,
        help="Number of tasks to run",
    )
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=["all"],
        help="Task types to include (budget_constrained, preference_memory, "
        "negative_constraint, comparative_reasoning, error_recovery, all)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full assessment (all 80 tasks)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per task in seconds",
    )
    parser.add_argument(
        "--green-port",
        type=int,
        default=GREEN_AGENT_PORT,
        help="Port for green agent",
    )
    parser.add_argument(
        "--purple-port",
        type=int,
        default=PURPLE_AGENT_PORT,
        help="Port for purple agent",
    )
    parser.add_argument(
        "--no-start-agents",
        action="store_true",
        help="Don't start agents (assume they're already running)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    return parser.parse_args()


class AgentProcess:
    """Manages an agent server process."""

    def __init__(
        self,
        name: str,
        directory: Path,
        port: int,
        health_url: str,
    ):
        self.name = name
        self.directory = directory
        self.port = port
        self.health_url = health_url
        self.process: Optional[subprocess.Popen] = None

    def start(self, verbose: bool = False) -> bool:
        """Start the agent server."""
        print(f"Starting {self.name} on port {self.port}...")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.directory)

        cmd = [
            "uv",
            "run",
            "python",
            "src/server.py",
            "--port",
            str(self.port),
            "--log-level",
            "DEBUG" if verbose else "INFO",
        ]

        self.process = subprocess.Popen(
            cmd,
            cwd=str(self.directory),
            env=env,
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.STDOUT if not verbose else None,
        )

        return True

    async def wait_for_health(self, timeout: float = 30.0) -> bool:
        """Wait for the agent to be healthy."""
        start_time = time.time()
        async with httpx.AsyncClient() as client:
            while time.time() - start_time < timeout:
                try:
                    response = await client.get(
                        self.health_url,
                        timeout=5.0,
                    )
                    if response.status_code == 200:
                        print(f"  {self.name} is healthy")
                        return True
                except Exception:
                    pass
                await asyncio.sleep(0.5)
        return False

    def stop(self) -> None:
        """Stop the agent server."""
        if self.process:
            print(f"Stopping {self.name}...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None


async def run_assessment(
    green_url: str,
    purple_url: str,
    num_tasks: int,
    task_types: list[str],
    timeout_per_task: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run an assessment by sending a request to the green agent.

    Args:
        green_url: Green agent base URL
        purple_url: Purple agent base URL
        num_tasks: Number of tasks to run
        task_types: Task types to include
        timeout_per_task: Timeout per task in seconds
        verbose: Enable verbose output

    Returns:
        Assessment results dictionary
    """
    # Build the A2A request
    request_body = {
        "jsonrpc": "2.0",
        "method": "message/stream",
        "params": {
            "message": {
                "messageId": "assessment-msg-1",
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": f"Run WebShop+ assessment with {num_tasks} tasks",
                    }
                ],
            },
            "metadata": {
                "participants": {
                    "shopper": f"{purple_url}/a2a",
                },
                "config": {
                    "categories": task_types,  # Note: green agent expects "categories", not "task_types"
                    "num_tasks": num_tasks,
                    "timeout_per_task": timeout_per_task,
                },
            },
        },
        "id": "assessment-1",
    }

    print(f"\nSending assessment request to green agent...")
    print(f"  Tasks: {num_tasks}")
    print(f"  Types: {', '.join(task_types)}")
    print(f"  Shopper: {purple_url}/a2a")

    results = None
    task_count = 0
    last_status = ""

    # Use streaming to get real-time updates
    timeout = httpx.Timeout(timeout_per_task * num_tasks + 120, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream(
                "POST",
                f"{green_url}/a2a",
                json=request_body,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    return {
                        "error": f"HTTP {response.status_code}: {error_text.decode()}",
                        "success": False,
                    }

                # Process SSE stream
                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk

                    # Parse SSE events
                    while "\n\n" in buffer:
                        event, buffer = buffer.split("\n\n", 1)

                        for line in event.split("\n"):
                            if line.startswith("data: "):
                                data_str = line[6:]
                                try:
                                    data = json.loads(data_str)

                                    # Extract status and progress
                                    result = data.get("result", {})
                                    status = result.get("status", {})
                                    state = status.get("state", "")
                                    message = status.get("message", "")

                                    if message and message != last_status:
                                        if verbose or "Task" in message:
                                            print(f"  [{state}] {message}")
                                        last_status = message

                                        # Count tasks
                                        if "Task" in message and "/" in message:
                                            import re

                                            match = re.search(
                                                r"Task (\d+)/(\d+)", message
                                            )
                                            if match:
                                                task_count = int(match.group(1))

                                    # Check for artifacts (results)
                                    artifacts = result.get("artifacts", [])
                                    for artifact in artifacts:
                                        if artifact.get("name") == "assessment_results":
                                            for part in artifact.get("parts", []):
                                                if part.get("kind") == "text":
                                                    try:
                                                        results = json.loads(
                                                            part.get("text", "{}")
                                                        )
                                                    except json.JSONDecodeError:
                                                        pass

                                    # Check for completion
                                    if result.get("final") and state in (
                                        "completed",
                                        "failed",
                                        "canceled",
                                    ):
                                        if results:
                                            return results
                                        return {
                                            "state": state,
                                            "message": message,
                                            "tasks_completed": task_count,
                                        }

                                except json.JSONDecodeError:
                                    if verbose:
                                        print(f"  [warn] Could not parse: {data_str[:100]}")

        except httpx.TimeoutException:
            return {
                "error": "Assessment timed out",
                "success": False,
                "tasks_completed": task_count,
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False,
                "tasks_completed": task_count,
            }

    # If we got here without results, return what we have
    if results:
        return results
    return {
        "error": "No results received",
        "success": False,
        "tasks_completed": task_count,
    }


def print_results(results: dict[str, Any]) -> None:
    """Print assessment results in a formatted way."""
    print("\n" + "=" * 60)
    print("ASSESSMENT RESULTS")
    print("=" * 60)

    if "error" in results:
        print(f"\nError: {results['error']}")
        if "tasks_completed" in results:
            print(f"Tasks completed before error: {results['tasks_completed']}")
        return

    # Print aggregate results
    aggregate = results.get("aggregate", {})
    if aggregate:
        print(f"\nOverall Performance:")
        print(f"  Total Tasks:      {aggregate.get('total_tasks', 0)}")
        print(f"  Successful Tasks: {aggregate.get('successful_tasks', 0)}")
        print(f"  Average Score:    {aggregate.get('average_score', 0):.2%}")
        print(f"  Average Time:     {aggregate.get('average_time', 0):.1f}s")

        # Print by task type
        by_type = aggregate.get("by_task_type", {})
        if by_type:
            print("\nBy Task Type:")
            for task_type, stats in by_type.items():
                print(f"  {task_type}:")
                print(f"    Count:        {stats.get('count', 0)}")
                print(f"    Avg Score:    {stats.get('avg_score', 0):.2%}")
                print(f"    Success Rate: {stats.get('success_rate', 0):.2%}")

    # Print individual results (first 10)
    individual_results = results.get("results", [])
    if individual_results:
        print(f"\nIndividual Results (first 10):")
        for i, result in enumerate(individual_results[:10]):
            task_id = result.get("task_id", "unknown")
            task_type = result.get("task_type", "unknown")
            score = result.get("overall_score", 0)
            success = result.get("success", False)
            status = "PASS" if success else "FAIL"
            print(f"  {i+1}. [{status}] {task_type} ({task_id[:20]}): {score:.2%}")

    print("\n" + "=" * 60)


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Adjust ports if specified
    green_url = f"http://localhost:{args.green_port}"
    purple_url = f"http://localhost:{args.purple_port}"

    # Adjust task count
    num_tasks = 80 if args.full else args.tasks

    # Setup agent processes
    agents: list[AgentProcess] = []

    if not args.no_start_agents:
        green_agent = AgentProcess(
            name="Green Agent (Evaluator)",
            directory=PROJECT_ROOT / "green_agent",
            port=args.green_port,
            health_url=f"{green_url}/health",
        )
        purple_agent = AgentProcess(
            name="Purple Agent (Shopper)",
            directory=PROJECT_ROOT / "purple_agent",
            port=args.purple_port,
            health_url=f"{purple_url}/health",
        )
        agents = [green_agent, purple_agent]

        # Handle cleanup on interrupt
        def cleanup(signum, frame):
            print("\n\nInterrupted! Cleaning up...")
            for agent in agents:
                agent.stop()
            sys.exit(1)

        signal.signal(signal.SIGINT, cleanup)
        signal.signal(signal.SIGTERM, cleanup)

        # Start agents
        for agent in agents:
            if not agent.start(verbose=args.verbose):
                print(f"Failed to start {agent.name}")
                return 1

        # Wait for health
        print("\nWaiting for agents to be ready...")
        for agent in agents:
            if not await agent.wait_for_health():
                print(f"  {agent.name} failed to become healthy")
                for a in agents:
                    a.stop()
                return 1

    try:
        # Verify agents are accessible
        print("\nVerifying agent connectivity...")
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                green_resp = await client.get(f"{green_url}/.well-known/agent-card.json")
                purple_resp = await client.get(
                    f"{purple_url}/.well-known/agent-card.json"
                )

                if green_resp.status_code == 200 and purple_resp.status_code == 200:
                    print("  Both agents are accessible")
                else:
                    print(f"  Green: {green_resp.status_code}, Purple: {purple_resp.status_code}")
                    return 1
            except Exception as e:
                print(f"  Connection error: {e}")
                return 1

        # Run the assessment
        print("\n" + "=" * 60)
        print("STARTING ASSESSMENT")
        print("=" * 60)

        results = await run_assessment(
            green_url=green_url,
            purple_url=purple_url,
            num_tasks=num_tasks,
            task_types=args.task_types,
            timeout_per_task=args.timeout,
            verbose=args.verbose,
        )

        # Print results
        print_results(results)

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {output_path}")

        # Return success if we have results
        if "error" not in results:
            aggregate = results.get("aggregate", {})
            success_rate = (
                aggregate.get("successful_tasks", 0) / aggregate.get("total_tasks", 1)
                if aggregate.get("total_tasks", 0) > 0
                else 0
            )
            return 0 if success_rate > 0 else 1
        return 1

    finally:
        # Cleanup
        for agent in agents:
            agent.stop()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
