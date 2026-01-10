# WebShop+

A stateful shopping benchmark for agentic AI, extending Princeton's WebShop with advanced evaluation dimensions.

## Overview

WebShop+ is a **green agent** (evaluator) for the AgentBeats platform that tests shopping agents on:

- **Budget Management**: Multi-item shopping within spending limits
- **Preference Memory**: Cross-session consistency and recall
- **Negative Constraints**: Avoiding forbidden attributes (allergies, restrictions)
- **Comparative Reasoning**: Exploring options and justifying choices
- **Error Recovery**: Fixing mistakes in existing cart state

## Project Structure

```
webshop-plus/
├── green_agent/          # Evaluator agent
│   ├── src/              # Source code
│   ├── tests/            # Unit tests
│   └── data/tasks/       # 80 task definitions
├── purple_agent/         # Baseline shopping agent
│   ├── src/              # Source code
│   └── tests/            # Unit tests
├── webshop/              # Princeton WebShop (submodule)
└── scenarios/            # AgentBeats scenarios
```

## Quick Start

### Prerequisites

- Python 3.10+
- Java 11+ (for WebShop's Lucene search)
- Ollama (for local LLM inference)

### Setup

```bash
# Clone and setup green agent
cd green_agent
uv sync

# Clone and setup purple agent
cd ../purple_agent
uv sync

# Copy environment config
cp .env.local.example .env.local
```

### Running Locally

```bash
# Start green agent
cd green_agent
uv run python src/server.py --host 0.0.0.0 --port 8000

# Start purple agent (in another terminal)
cd purple_agent
uv run python src/server.py --host 0.0.0.0 --port 8001
```

## Task Types

| Type | Count | Description |
|------|-------|-------------|
| Budget Constrained | 20 | Multi-item shopping within budget |
| Preference Memory | 15 | Cross-session consistency |
| Negative Constraint | 20 | Avoiding forbidden attributes |
| Comparative Reasoning | 15 | Comparing and justifying choices |
| Error Recovery | 10 | Fixing cart mistakes |

## License

MIT
