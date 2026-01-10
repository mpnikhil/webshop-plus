# Phase 4 Handover: Task Generator

**Completed**: January 9, 2026
**Next Phase**: Phase 5 - State Manager

## What Was Built

### Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `green_agent/src/task_generator.py` | Task loading and management | ~190 |
| `green_agent/tests/test_task_generator.py` | Comprehensive tests | ~340 |

### TaskGenerator Class API

```python
from src.task_generator import TaskGenerator

tg = TaskGenerator()  # or TaskGenerator("/path/to/tasks")
```

#### Core Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `get_task(task_id)` | Get task by ID | `Task` |
| `get_tasks_by_type(task_type, difficulty?)` | Get tasks of a type | `List[Task]` |
| `get_all_tasks(difficulty?)` | Get all tasks | `List[Task]` |
| `get_random_task(task_type?, difficulty?)` | Get random task | `Task` |

#### Helper Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `get_task_count(task_type?, difficulty?)` | Count tasks | `int` |
| `get_task_ids(task_type?, difficulty?)` | Get task IDs | `List[str]` |
| `len(tg)` | Total task count | `int` |
| `task_id in tg` | Check if task exists | `bool` |

### Features

1. **Automatic Task Loading**
   - Loads all JSON files from `data/tasks/` directory
   - Validates each task using Pydantic models via `parse_task()`
   - Caches tasks in memory for fast access

2. **Flexible Filtering**
   - Filter by task type (enum or string)
   - Filter by difficulty (enum or string)
   - Combined filtering (type + difficulty)

3. **Error Handling**
   - `FileNotFoundError` - Directory or JSON files not found
   - `ValueError` - Invalid JSON structure, duplicate task IDs, invalid type/difficulty
   - `KeyError` - Task ID not found

4. **Type Hints**
   - Full type hints for all methods
   - Supports both `str` and `enum` for task type and difficulty parameters

### Usage Examples

```python
from src.task_generator import TaskGenerator
from src.models import TaskType, Difficulty

tg = TaskGenerator()

# Get specific task
task = tg.get_task("budget_001")
print(f"Task: {task.instruction}")
print(f"Budget: ${task.constraints.budget}")

# Get all budget tasks
budget_tasks = tg.get_tasks_by_type(TaskType.BUDGET_CONSTRAINED)
print(f"Found {len(budget_tasks)} budget tasks")

# Get easy budget tasks
easy_budget = tg.get_tasks_by_type("budget_constrained", difficulty="easy")

# Get random hard task
hard_task = tg.get_random_task(difficulty=Difficulty.HARD)

# Get count by type
print(f"Memory tasks: {tg.get_task_count(TaskType.PREFERENCE_MEMORY)}")

# Check if task exists
if "budget_001" in tg:
    print("Task exists!")
```

## Dependency Versions

No new dependencies added. Using existing:
```
pydantic>=2.0.0
```

## Verification Commands

```bash
# Run all tests (108 tests, all pass)
cd /Users/nikhilpujari/agentbeats/webshop-plus/green_agent
uv run python -m pytest tests/ -v

# Run only task generator tests (43 tests)
uv run python -m pytest tests/test_task_generator.py -v

# Quick verification
uv run python -c "from src.task_generator import TaskGenerator; tg = TaskGenerator(); print(f'{len(tg.get_all_tasks())} tasks')"
# Output: 80 tasks
```

## Test Results

```
tests/test_models.py - 51 passed
tests/test_task_generator.py - 43 passed
tests/test_webshop_wrapper.py - 14 passed
Total: 108 passed in 0.95s
```

Test coverage for TaskGenerator includes:
- Initialization (default path, custom path, error cases)
- `get_task()` - existing and non-existent tasks
- `get_tasks_by_type()` - all 5 types, string/enum input, difficulty filtering
- `get_all_tasks()` - with and without difficulty filter
- `get_random_task()` - type/difficulty filtering, distribution test
- `get_task_count()` - total and filtered counts
- `get_task_ids()` - all and filtered
- `__len__` and `__contains__` dunder methods
- Task validation - required fields, unique IDs
- Edge cases - custom directory, multiple JSON files

## Task Distribution

| Task Type | Count | Difficulties |
|-----------|-------|--------------|
| budget_constrained | 20 | easy, medium, hard |
| preference_memory | 15 | easy, medium, hard |
| negative_constraint | 20 | easy, medium, hard |
| comparative_reasoning | 15 | easy, medium, hard |
| error_recovery | 10 | easy, medium, hard |
| **Total** | **80** | |

## Phase 5 Objectives

1. Create `green_agent/src/state_manager.py` with:
   - `StateManager.__init__()`
   - `create_session(task_id, agent_id) -> SessionState`
   - `record_action(session_id, action, observation, reward)`
   - `get_session(session_id) -> SessionState`
   - `parse_cart_from_observation(observation) -> CartState`
   - `get_agent_memory(agent_id) -> AgentMemory`
   - `update_agent_memory(agent_id, summary)`
   - `inject_cart_state(cart) -> None`

2. Create `green_agent/tests/test_state_manager.py`

### Key Notes for Phase 5

- StateManager tracks multiple sessions in memory
- Parse cart state from WebShop HTML observations using BeautifulSoup
- AgentMemory persists across sessions for preference recall tasks
- `inject_cart_state()` is for error recovery tasks (pre-populate cart)

### Cart Parsing Hints

WebShop cart observations look like:
```html
<div class="cart">
  <div class="item">
    <span class="product-name">Product Name</span>
    <span class="price">$29.99</span>
    <span class="quantity">1</span>
  </div>
</div>
```

Use BeautifulSoup (already in dependencies) to parse.

## Known Issues

None. All tests pass.
