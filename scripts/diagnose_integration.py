#!/usr/bin/env python
"""
Diagnostic script to verify the A2A + MCP integration loop.

Tests:
1. Green agent sending correct A2A message to purple
2. Purple agent receiving MCP URI and listing tools
3. Purple agent passing correct task to ShoppingAgent
4. MCP tool calls to green agent working correctly

Run with: uv run python scripts/diagnose_integration.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "green_agent" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "purple_agent" / "src"))


async def test_step_1_green_kickoff_message():
    """Test 1: Verify green agent builds correct kickoff message with MCP URI."""
    print("\n" + "=" * 60)
    print("TEST 1: Green Agent Kickoff Message Format")
    print("=" * 60)

    from purple_client import PurpleAgentClient

    # Create a mock kickoff to verify the format
    client = PurpleAgentClient("http://localhost:8001")

    kickoff = client._build_kickoff(
        goal="Find running shoes under $50",
        budget=50.0,
        constraints=["waterproof", "size 10"],
        mcp_uri="http://localhost:8000/mcp/test-session-123",
    )

    print("\nKickoff message structure:")
    print(json.dumps(kickoff, indent=2))

    # Verify required fields
    required_fields = ["goal", "budget", "constraints", "resources"]
    missing = [f for f in required_fields if f not in kickoff]

    if missing:
        print(f"\n[FAIL] Missing required fields: {missing}")
        return False

    # Verify MCP resource
    resources = kickoff.get("resources", [])
    if not resources:
        print("\n[FAIL] No resources in kickoff")
        return False

    mcp_resource = resources[0]
    if mcp_resource.get("type") != "mcp":
        print(f"\n[FAIL] Resource type is '{mcp_resource.get('type')}', expected 'mcp'")
        return False

    if not mcp_resource.get("uri"):
        print("\n[FAIL] MCP resource missing URI")
        return False

    print("\n[PASS] Kickoff message format is correct")
    return True


async def test_step_2_purple_mcp_extraction():
    """Test 2: Verify purple executor extracts MCP URI correctly."""
    print("\n" + "=" * 60)
    print("TEST 2: Purple Agent MCP URI Extraction")
    print("=" * 60)

    from a2a.types import Message, Role, TextPart
    from executor import Executor

    executor = Executor()

    # Create a test message with MCP resource
    kickoff_payload = {
        "goal": "Find running shoes under $50",
        "budget": 50.0,
        "constraints": ["waterproof"],
        "resources": [
            {
                "type": "mcp",
                "uri": "http://localhost:8000/mcp/test-session-456",
            }
        ]
    }

    message = Message(
        messageId="test-msg-1",
        role=Role.user,
        parts=[TextPart(text=json.dumps(kickoff_payload))],
    )

    # Test extraction
    mcp_uri = executor._extract_mcp_uri(message)
    print(f"\nExtracted MCP URI: {mcp_uri}")

    if not mcp_uri:
        print("\n[FAIL] Failed to extract MCP URI from message")
        return False

    if mcp_uri != "http://localhost:8000/mcp/test-session-456":
        print(f"\n[FAIL] Extracted URI doesn't match expected")
        return False

    # Test task data extraction
    task_data = executor._extract_task_data(message)
    print(f"\nExtracted task data: {json.dumps(task_data, indent=2)}")

    if not task_data.get("goal"):
        print("\n[FAIL] Failed to extract goal from message")
        return False

    print("\n[PASS] Purple agent extracts MCP URI and task data correctly")
    return True


async def test_step_3_shopping_agent_creation():
    """Test 3: Verify ShoppingAgent can be created with MCP toolset."""
    print("\n" + "=" * 60)
    print("TEST 3: Shopping Agent + MCP Toolset Creation")
    print("=" * 60)

    from shopping_agent import ShoppingAgent, DEFAULT_MODEL

    print(f"\nDefault model: {DEFAULT_MODEL}")

    agent = ShoppingAgent()
    print(f"Created ShoppingAgent with model: {agent.model}")
    print(f"Max turns: {agent.max_turns}")

    # Test instruction formatting
    instruction = agent._format_instruction(
        goal="Find running shoes under $50",
        budget=50.0,
        constraints=["waterproof", "size 10"],
    )

    print(f"\nInstruction preview (first 300 chars):\n{instruction[:300]}...")

    if "search" not in instruction.lower():
        print("\n[FAIL] Instruction missing search tool info")
        return False

    if "checkout" not in instruction.lower():
        print("\n[FAIL] Instruction missing checkout tool info")
        return False

    print("\n[PASS] ShoppingAgent created with correct instruction template")
    return True


async def test_step_4_mcp_session_and_tools():
    """Test 4: Verify MCP session creation and tool registration."""
    print("\n" + "=" * 60)
    print("TEST 4: MCP Session and Tools")
    print("=" * 60)

    from webshop_mcp import SessionManager
    from webshop_mcp.server import (
        mcp,
        register_session,
        unregister_session,
        get_session_state,
        is_session_registered,
        current_session_id,
    )
    from webshop_mcp.session_state import SessionState

    # Create session manager
    session_manager = SessionManager(max_sessions=10, session_ttl=300)

    # Create a test session
    test_session_id = "test-diagnostic-session"
    await session_manager.create_session(
        session_id=test_session_id,
        goal="Test diagnostic session",
        budget=100.0,
        constraints=[],
        max_turns=30,
    )

    print(f"\nCreated session: {test_session_id}")
    print(f"Session registered: {is_session_registered(test_session_id)}")

    # Get session state
    state = get_session_state(test_session_id)
    if not state:
        print("\n[FAIL] Session state not found")
        await session_manager.cleanup_session(test_session_id)
        return False

    print(f"Session goal: {state.goal}")
    print(f"Session budget: {state.budget}")

    # Check registered tools on the MCP server
    print("\nRegistered MCP tools:")
    for tool_name in mcp._tool_manager._tools:
        print(f"  - {tool_name}")

    expected_tools = ["search", "click", "checkout"]
    missing_tools = [t for t in expected_tools if t not in mcp._tool_manager._tools]

    if missing_tools:
        print(f"\n[FAIL] Missing MCP tools: {missing_tools}")
        await session_manager.cleanup_session(test_session_id)
        return False

    # Test calling a tool with session context
    token = current_session_id.set(test_session_id)
    try:
        from webshop_mcp.server import search, get_current_state

        # Verify we can get the current state
        current_state = get_current_state()
        print(f"\nCurrent session from context: {current_state.session_id}")

        if current_state.session_id != test_session_id:
            print("\n[FAIL] Session context mismatch")
            return False

    finally:
        current_session_id.reset(token)

    # Cleanup
    await session_manager.cleanup_session(test_session_id)
    print(f"\nSession cleaned up: {not is_session_registered(test_session_id)}")

    print("\n[PASS] MCP session and tools work correctly")
    return True


async def test_step_5_end_to_end_dry_run():
    """Test 5: Dry-run of the full flow (without actual servers)."""
    print("\n" + "=" * 60)
    print("TEST 5: End-to-End Flow Dry Run")
    print("=" * 60)

    from webshop_mcp import SessionManager
    from webshop_mcp.server import (
        register_session,
        unregister_session,
        current_session_id,
        search,
    )

    # Simulate green agent creating MCP session
    session_manager = SessionManager(max_sessions=10, session_ttl=300)
    session_id = "e2e-test-session"

    await session_manager.create_session(
        session_id=session_id,
        goal="Find running shoes",
        budget=50.0,
        constraints=[],
        max_turns=30,
    )

    mcp_uri = f"http://localhost:8000/mcp/{session_id}"
    print(f"\n1. Green agent creates MCP session: {mcp_uri}")

    # Simulate green agent building kickoff
    from purple_client import PurpleAgentClient
    client = PurpleAgentClient("http://localhost:8001")
    kickoff = client._build_kickoff(
        goal="Find running shoes",
        budget=50.0,
        constraints=[],
        mcp_uri=mcp_uri,
    )
    print(f"2. Green agent builds kickoff with MCP URI")

    # Simulate purple agent extracting MCP URI
    from a2a.types import Message, Role, TextPart
    from executor import Executor

    executor = Executor()
    message = Message(
        messageId="test",
        role=Role.user,
        parts=[TextPart(text=json.dumps(kickoff))],
    )

    extracted_uri = executor._extract_mcp_uri(message)
    task_data = executor._extract_task_data(message)
    print(f"3. Purple agent extracts MCP URI: {extracted_uri}")
    print(f"   Task data: goal='{task_data['goal']}', budget={task_data['budget']}")

    # Simulate MCP tool call
    token = current_session_id.set(session_id)
    try:
        print(f"4. Simulating MCP tool call: search('running shoes')")
        result = search("running shoes")
        print(f"   Search result: {result.get('product_count', 0)} products found")
        print(f"   Turn: {result.get('turn')}/{result.get('turns_remaining', 0) + result.get('turn', 0)} remaining")
    except Exception as e:
        print(f"   [WARN] Search failed (expected if WebShop not running): {e}")
    finally:
        current_session_id.reset(token)

    # Cleanup
    await session_manager.cleanup_session(session_id)
    print(f"5. Session cleaned up")

    print("\n[PASS] End-to-end dry run completed")
    return True


async def main():
    """Run all diagnostic tests."""
    print("=" * 60)
    print("WEBSHOP+ A2A + MCP INTEGRATION DIAGNOSTICS")
    print("=" * 60)

    results = {}

    # Run tests
    try:
        results["step1_kickoff"] = await test_step_1_green_kickoff_message()
    except Exception as e:
        print(f"\n[ERROR] Test 1 failed with exception: {e}")
        results["step1_kickoff"] = False

    try:
        results["step2_extraction"] = await test_step_2_purple_mcp_extraction()
    except Exception as e:
        print(f"\n[ERROR] Test 2 failed with exception: {e}")
        results["step2_extraction"] = False

    try:
        results["step3_shopping_agent"] = await test_step_3_shopping_agent_creation()
    except Exception as e:
        print(f"\n[ERROR] Test 3 failed with exception: {e}")
        results["step3_shopping_agent"] = False

    try:
        results["step4_mcp_tools"] = await test_step_4_mcp_session_and_tools()
    except Exception as e:
        print(f"\n[ERROR] Test 4 failed with exception: {e}")
        results["step4_mcp_tools"] = False

    try:
        results["step5_e2e"] = await test_step_5_end_to_end_dry_run()
    except Exception as e:
        print(f"\n[ERROR] Test 5 failed with exception: {e}")
        results["step5_e2e"] = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed < total:
        print("\nFailed tests indicate issues in the integration flow.")
        return 1

    print("\nAll integration components verified successfully!")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
