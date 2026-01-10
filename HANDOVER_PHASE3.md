# Phase 3 Handover: Pydantic Models

**Completed**: January 9, 2026
**Next Phase**: Phase 4 - Task Generator

## What Was Built

### Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `green_agent/src/models.py` | All Pydantic data models | ~500 |
| `green_agent/tests/test_models.py` | Comprehensive model tests | ~550 |

### Model Categories

#### 1. Enums
- `TaskType` - 5 task types (budget_constrained, preference_memory, negative_constraint, comparative_reasoning, error_recovery)
- `Difficulty` - easy, medium, hard
- `OptimizationGoal` - maximize_quality, minimize_cost, balance

#### 2. Task Models (Base + 5 Types)

**BaseTask** - Common fields:
- `task_id`, `task_type`, `instruction`
- `difficulty`, `expected_actions`, `timeout_seconds`

**BudgetConstrainedTask** - Budget management tasks:
- `constraints.budget` - Maximum spend
- `constraints.required_items` - List of items to buy
- `constraints.optimization_goal` - Quality vs cost tradeoff
- `evaluation_criteria` - Weights for budget, completion, quality

**PreferenceMemoryTask** - Memory recall tasks:
- `session_sequence` - Prior sessions establishing preferences
- `memory_test.attribute_to_recall` - What to remember
- `memory_test.acceptable_values` - Valid recall values

**NegativeConstraintTask** - Constraint satisfaction tasks:
- `constraints.required_attributes` - Must have
- `constraints.forbidden_attributes` - Must NOT have
- `constraints.forbidden_terms` - Terms to avoid

**ComparativeReasoningTask** - Comparison tasks:
- `requirements.category`, `requirements.budget`
- `requirements.comparison_request` - What to compare
- `evaluation_criteria.minimum_options_explored` - Min products to view

**ErrorRecoveryTask** - Error correction tasks:
- `setup.cart_contents` - Initial (erroneous) cart
- `setup.error_description` - What's wrong
- `correct_state.expected_cart` - Target cart state

#### 3. State Models

**CartItem**
- `product_id`, `product_name`, `attributes`, `quantity`, `price`
- `total_price` property (price * quantity)

**CartState**
- `items` list with `add_item()`, `remove_item()`, `clear()` methods
- `total` and `item_count` computed properties
- Handles duplicate detection (same product_id + attributes)

**ActionRecord**
- `timestamp`, `action`, `observation`, `reward`

**PurchaseRecord**
- `product_id`, `product_name`, `attributes`, `price`, `purchased_at`

**SessionState**
- Full session tracking: `session_id`, `task_id`, `agent_id`
- `actions` list, `current_observation`, `cart`, `purchases`
- `preferences_established` for memory tasks
- Helper methods: `record_action()`, `complete()`, `to_summary()`

#### 4. Evaluation Models

**ScoringComponent**
- `name`, `weight`, `raw_value`, `normalized_score`, `explanation`

**EvaluationResult**
- `task_id`, `task_type`, `completed`, `success`
- `overall_score` (0-1), `metrics` dict
- `scoring_breakdown` list of ScoringComponents
- `add_component()` and `calculate_overall_score()` methods

#### 5. Memory Models

**SessionSummary**
- Condensed session info: `purchases`, `preferences`
- Used for agent memory across sessions

**AgentMemory**
- `agent_id`, `sessions` list
- Methods: `add_session()`, `get_sessions_by_type()`, `get_all_purchases()`, `get_all_preferences()`, `clear()`

#### 6. Assessment Models (A2A)

**AssessmentConfig**
- `task_types` (default: ["all"])
- `num_tasks` (default: 80)
- `timeout_per_task` (default: 300s)

**AssessmentRequest**
- `participants` - role -> endpoint URL mapping
- `config` - AssessmentConfig

**TaskUpdate**
- Progress updates during assessment
- `status`, `current_task`, `tasks_completed`, `tasks_total`, `progress`

**AssessmentResults**
- Final results with `calculate_aggregate()` method
- Computes scores by task type automatically

### Helper Functions

**`parse_task(data: dict) -> Task`**
- Auto-detects task type from `task_type` field
- Returns appropriate model instance
- Raises `ValueError` for unknown types

## Dependency Versions

No new dependencies added. Using existing:
```
pydantic>=2.0.0
```

Added `beautifulsoup4>=4.14.3` to pyproject.toml (missed in Phase 2).

## Verification Commands

```bash
# Run all tests (65 tests, all pass)
cd /Users/nikhilpujari/agentbeats/webshop-plus/green_agent
uv run python -m pytest tests/ -v

# Run only model tests (51 tests)
uv run python -m pytest tests/test_models.py -v

# Quick model verification
uv run python -c "
from src.models import parse_task, TaskType
import json
with open('data/tasks/budget_constrained.json') as f:
    data = json.load(f)
task = parse_task(data[0])
print(f'{task.task_id}: {task.task_type.value}')
print(f'Budget: \${task.constraints.budget}')
print(f'Items: {len(task.constraints.required_items)}')
"
```

## Test Results

```
tests/test_models.py - 51 passed
tests/test_webshop_wrapper.py - 14 passed
Total: 65 passed in 0.73s
```

Test coverage includes:
- All 5 task types creation and loading from JSON
- `parse_task()` function for all types + error handling
- CartState operations (add, remove, clear, duplicates)
- SessionState tracking and summary conversion
- EvaluationResult scoring calculations
- AgentMemory session management
- Loading all 80 tasks from JSON files
- Task type count verification (20+15+20+15+10 = 80)

## Phase 4 Objectives

1. Create `green_agent/src/task_generator.py` with:
   - `TaskGenerator.__init__(tasks_dir)` - Load and validate all tasks
   - `get_task(task_id: str) -> Task`
   - `get_tasks_by_type(task_type: str) -> List[Task]`
   - `get_all_tasks() -> List[Task]`
   - `get_random_task(task_type: Optional[str]) -> Task`

2. Create `green_agent/tests/test_task_generator.py`

### Key Notes for Phase 4

- Use `parse_task()` from models.py to convert JSON dicts
- Tasks are in `data/tasks/*.json` (5 files, 80 tasks total)
- TaskGenerator should cache tasks after loading
- Consider adding filtering by difficulty level

## Model Usage Examples

### Creating a Budget Task Programmatically
```python
from src.models import (
    BudgetConstrainedTask,
    BudgetConstraints,
    RequiredItem,
    OptimizationGoal,
)

task = BudgetConstrainedTask(
    task_id="custom_001",
    instruction="Buy office supplies",
    constraints=BudgetConstraints(
        budget=100.0,
        required_items=[
            RequiredItem(category="office", attributes={"type": "stapler"}),
            RequiredItem(category="office", attributes={"type": "pens"}, optional=True),
        ],
        optimization_goal=OptimizationGoal.BALANCE,
    ),
)
```

### Loading Task from JSON
```python
from src.models import parse_task
import json

with open("data/tasks/budget_constrained.json") as f:
    tasks_data = json.load(f)

task = parse_task(tasks_data[0])
print(task.constraints.budget)  # 50.0
```

### Using SessionState
```python
from src.models import SessionState, CartItem

session = SessionState(session_id="s1", task_id="t1", agent_id="agent1")
session.record_action("search[mouse]", "Found 10 products", 0.0)
session.cart.add_item(CartItem(
    product_id="P001",
    product_name="Wireless Mouse",
    price=29.99,
))
print(session.cart.total)  # 29.99
session.complete()
```

### Evaluating a Task
```python
from src.models import EvaluationResult, TaskType

result = EvaluationResult(
    task_id="budget_001",
    task_type=TaskType.BUDGET_CONSTRAINED,
    completed=True,
    success=True,
)
result.add_component("budget_compliance", 0.3, True, 1.0)
result.add_component("item_completion", 0.4, 0.75, 0.75)
result.add_component("quality", 0.3, 0.8, 0.8)
result.calculate_overall_score()
print(result.overall_score)  # 0.84
```

## Known Issues

None. All tests pass and models correctly load all 80 task files.
