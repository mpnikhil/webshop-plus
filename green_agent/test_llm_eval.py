
import os
import sys
from typing import Any

# Add src to path
sys.path.append(os.getcwd())

from src.models import (
    PreferenceMemoryTask, 
    SessionSequenceItem, 
    MemoryTest, 
    MemoryEvaluationCriteria, 
    TaskType,
    Difficulty
)
from src.evaluator import Evaluator
from src.webshop_mcp.session_state import SessionState as MCPSessionState

# Mock LLM Client
class MockLLMClient:
    def complete(self, messages, max_tokens=1024):
        # Simulate LLM saying YES
        return "YES, this item matches the preference."

def test_llm_evaluation():
    print("\n--- Testing LLM-as-a-Judge Evaluation ---")
    
    # 1. Create task
    task = PreferenceMemoryTask(
        task_id="test_mem_002",
        task_type=TaskType.PREFERENCE_MEMORY,
        instruction="Buy shirt.",
        difficulty=Difficulty.EASY,
        expected_actions=5,
        timeout_seconds=60,
        session_sequence=[
            SessionSequenceItem(session_id="s1", instruction="Want slim fit", establishes={"fit": "slim fit"})
        ],
        memory_test=MemoryTest(attribute_to_recall="fit", acceptable_values=["slim fit"]),
        evaluation_criteria=MemoryEvaluationCriteria()
    )

    # 2. Create session with correct item
    session = MCPSessionState(session_id="s1", goal="task", budget=100.0)
    session.completed = True
    session.cart.append({
        "product_id": "p1",
        "name": "Nice Shirt",
        "price": 20.0,
        "quantity": 1,
        "catalog_attributes": {"style": "slim fit"},
        "options": {}
    })

    # 3. Evaluate with Mock LLM
    print("Testing with Mock LLM (expecting 1.0)...")
    mock_llm = MockLLMClient()
    evaluator = Evaluator(llm_client=mock_llm)
    result = evaluator.evaluate(session, task)
    
    print(f"Result: {result.overall_score}")
    if result.overall_score == 1.0:
        print("SUCCESS: Mock LLM evaluation worked.")
    else:
        print("FAILURE: Mock LLM evaluation failed.")

    # 4. Evaluate with Fallback (No LLM)
    print("\nTesting with Fallback (expecting 1.0 via string match)...")
    evaluator_fallback = Evaluator(llm_client=None)
    result_fallback = evaluator_fallback.evaluate(session, task)
    
    print(f"Result: {result_fallback.overall_score}")
    if result_fallback.overall_score == 1.0:
        print("SUCCESS: Fallback evaluation worked.")
    else:
        print("FAILURE: Fallback evaluation failed.")

if __name__ == "__main__":
    test_llm_evaluation()
