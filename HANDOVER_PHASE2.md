# Phase 2 Handover: WebShop Environment Wrapper

**Completed**: January 9, 2026
**Next Phase**: Phase 3 - Pydantic Models

## What Was Built

### Directory Structure
```
webshop-plus/
├── webshop/                      # Cloned Princeton WebShop repo
│   ├── data/
│   │   ├── items_shuffle_1000.json    (1000 products)
│   │   ├── items_ins_v2_1000.json     (product attributes)
│   │   └── items_human_ins.json       (human instructions)
│   └── web_agent_site/
│       └── templates/                  (HTML templates)
└── green_agent/
    ├── src/
    │   ├── webshop_wrapper.py         # Main wrapper class
    │   └── webshop_patched/           # Patched WebShop modules
    │       ├── __init__.py
    │       ├── engine.py              # BM25 search instead of Lucene
    │       ├── goal.py                # Simplified reward (no spacy)
    │       └── normalize.py           # Color/size normalization
    └── tests/
        └── test_webshop_wrapper.py    # 14 tests, all passing
```

### Key Files

**`green_agent/src/webshop_wrapper.py`**
- `WebShopWrapper` class with:
  - `__init__(mode="preview", observation_mode="text", num_products=None, human_goals=True)`
  - `reset(session=None, goal_idx=None) -> str` - Start new shopping session
  - `step(action: str) -> StepResult` - Execute action, return observation/reward/done
  - `get_available_actions() -> dict` - Get clickable elements
  - `get_goal() -> dict` - Get current session's goal
  - `get_instruction() -> str` - Get instruction text

**`green_agent/src/webshop_patched/`**
- Patched WebShop modules that don't require pyserini (Java/Lucene) or spacy
- Uses BM25 (rank_bm25) for product search instead
- Simpler reward calculation without spacy NLP

### Dependency Versions

```
flask>=3.1.2
gym>=0.26.2
numpy<2
thefuzz>=0.22.1
rank-bm25>=0.2.2
torch>=2.9.1
cleantext>=1.1.4
rich>=14.2.0
tqdm>=4.67.1
beautifulsoup4>=4.14.3
```

**Removed dependencies** (using patched modules instead):
- ~~pyserini~~ (requires Java 21, we have Java 17)
- ~~spacy~~ (simplified reward calculation)

### Verification Commands

```bash
# Run all tests (14 tests, all pass)
cd /Users/nikhilpujari/agentbeats/webshop-plus/green_agent
uv run python -m pytest tests/test_webshop_wrapper.py -v

# Quick verification
uv run python -c "
from src.webshop_wrapper import WebShopWrapper
w = WebShopWrapper(mode='preview')
obs = w.reset(goal_idx=0)
print(obs[:200])
"
```

## Implementation Notes

### Why Patched Modules?
The original WebShop uses:
1. **pyserini/Lucene** for product search - requires Java 21, we have Java 17
2. **spacy** for NLP-based reward calculation - heavy dependency

Our patched version:
1. Uses **BM25** (rank_bm25) for search - pure Python, no Java needed
2. Uses **simple string matching** for reward - thefuzz for fuzzy matching

### Search Quality
BM25 search is slightly less accurate than Lucene but sufficient for:
- Testing agent behavior
- Evaluating shopping task completion
- Development and debugging

### Observation Format
Text mode observations use `[SEP]` separators:
```
WebShop [SEP] Instruction: [SEP] find blue toothbrush... [SEP] Search
```

### Action Format
- Search: `search[keywords]`
- Click: `click[element text]`
- Examples: `search[blue shoes]`, `click[buy now]`, `click[next >]`

## Phase 3 Objectives

1. Create `green_agent/src/models.py` with Pydantic models:
   - Task models (base + 5 types: budget, memory, constraint, reasoning, recovery)
   - SessionState, CartState, CartItem
   - ActionRecord, PurchaseRecord
   - EvaluationResult, ScoringComponent
   - AgentMemory, SessionSummary

2. Create `green_agent/tests/test_models.py`

### Key Resources
- Task files in `green_agent/data/tasks/` (80 tasks across 5 types)
- Implementation spec: `./webshop-plus-implementation-spec.md`

## Known Issues

1. **BM25 vs Lucene**: Search results may differ slightly from original WebShop
2. **Product coverage**: Preview mode has 1000 products, 13 human goals
3. **Flask warnings**: Some template URL generation warnings (cosmetic, doesn't affect functionality)

## Test Results

```
tests/test_webshop_wrapper.py::TestWebShopWrapper::test_initialization PASSED
tests/test_webshop_wrapper.py::TestWebShopWrapper::test_reset_returns_observation PASSED
tests/test_webshop_wrapper.py::TestWebShopWrapper::test_reset_sets_instruction PASSED
tests/test_webshop_wrapper.py::TestWebShopWrapper::test_reset_sets_goal PASSED
tests/test_webshop_wrapper.py::TestWebShopWrapper::test_step_search_action PASSED
tests/test_webshop_wrapper.py::TestWebShopWrapper::test_step_invalid_action PASSED
tests/test_webshop_wrapper.py::TestWebShopWrapper::test_get_available_actions PASSED
tests/test_webshop_wrapper.py::TestWebShopWrapper::test_search_returns_products PASSED
tests/test_webshop_wrapper.py::TestWebShopWrapper::test_session_state_tracking PASSED
tests/test_webshop_wrapper.py::TestWebShopWrapper::test_multiple_sessions PASSED
tests/test_webshop_wrapper.py::TestWebShopWrapper::test_observation_mode_text PASSED
tests/test_webshop_wrapper.py::TestWebShopWrapper::test_step_without_reset_raises_error PASSED
tests/test_webshop_wrapper.py::TestWebShopWrapperIntegration::test_complete_shopping_flow PASSED
tests/test_webshop_wrapper.py::TestWebShopWrapperIntegration::test_back_to_search PASSED

14 passed in 0.60s
```
