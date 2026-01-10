# Phase 1 Handover: Repository Scaffolding

**Completed**: January 9, 2026
**Next Phase**: Phase 2 - WebShop Environment Wrapper

## What Was Built

### Directory Structure
```
webshop-plus/
├── .env.example              # Production (Nebius) config template
├── .env.local.example        # Local (Ollama) config template
├── .gitignore                # Python, IDE, env files excluded
├── README.md                 # Project overview
├── CLAUDE.md                 # Project instructions & session handoff
├── green_agent/
│   ├── pyproject.toml        # Dependencies: fastapi, pydantic, litellm, httpx, structlog, jsonschema
│   ├── uv.lock               # Locked dependencies
│   ├── src/
│   │   └── __init__.py
│   ├── tests/
│   │   └── __init__.py
│   └── data/tasks/           # 80 tasks already generated!
│       ├── budget_constrained.json     (20 tasks)
│       ├── preference_memory.json      (15 tasks)
│       ├── negative_constraint.json    (20 tasks)
│       ├── comparative_reasoning.json  (15 tasks)
│       └── error_recovery.json         (10 tasks)
└── purple_agent/
    ├── pyproject.toml        # Dependencies: fastapi, pydantic, litellm, httpx, structlog
    ├── uv.lock
    ├── src/
    │   └── __init__.py
    └── tests/
        └── __init__.py
```

### Dependency Versions Installed

**Green Agent (57 packages)**:
- fastapi==0.128.0
- pydantic==2.12.5
- litellm==1.80.13
- httpx==0.28.1
- structlog==25.5.0
- jsonschema==4.26.0
- uvicorn==0.40.0
- tiktoken==0.12.0
- tokenizers==0.22.2

**Purple Agent (57 packages)**:
- Same as above

**LLM Models (via LiteLLM)**:
- Local: `ollama/qwen3-coder:30b` (primary), `ollama/qwen3:32b` (fallback)
- Production: `nebius/Qwen/Qwen3-32B`

### Verification Commands
```bash
# Green agent
cd /Users/nikhilpujari/agentbeats/webshop-plus/green_agent
uv sync  # Already done, should be instant

# Purple agent
cd /Users/nikhilpujari/agentbeats/webshop-plus/purple_agent
uv sync  # Already done, should be instant
```

## Phase 2 Objectives

1. Clone Princeton WebShop into `webshop/` subdirectory
2. Create `green_agent/src/webshop_wrapper.py` with:
   - `WebShopWrapper.__init__(mode="preview")`
   - `WebShopWrapper.reset() -> str`
   - `WebShopWrapper.step(action: str) -> tuple`
3. Create basic test for WebShop wrapper

### Key Resources
- WebShop repo: https://github.com/princeton-nlp/WebShop
- Java 17 available at: `/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home/bin/java`
- Using preview mode (1,000 products) for faster iteration

### Expected Challenges
- WebShop has complex dependencies (spacy, sentence-transformers, Flask)
- May need to modify WebShop's environment to work as a library
- Lucene search engine requires Java

## Notes
- Tasks are already generated (80 total across 5 types)
- Environment templates ready for local (Ollama) and production (Nebius)
- Both agents have their venvs created with all dependencies
