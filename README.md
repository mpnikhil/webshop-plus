# WebShop+

A stateful shopping benchmark for agentic AI, extending Princeton's WebShop with advanced evaluation dimensions.

## Overview

WebShop+ is a **green agent** (evaluator) for the AgentBeats platform that tests shopping agents on:

- **Budget Management**: Multi-item shopping within spending limits
- **Preference Memory**: Cross-session consistency and recall
- **Negative Constraints**: Avoiding forbidden attributes (allergies, restrictions)
- **Comparative Reasoning**: Exploring options and justifying choices
- **Error Recovery**: Fixing mistakes in existing cart state

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AgentBeats Platform                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Green Agent (Evaluator)                            │
│                              Port 8000                                       │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐  │
│  │ Task Generator │  │ State Manager │  │   Evaluator   │  │ A2A Server   │  │
│  └───────────────┘  └───────────────┘  └───────────────┘  └──────────────┘  │
│         │                   │                  │                  │          │
│         └───────────────────┴──────────────────┴──────────────────┘          │
│                                       │                                       │
│                              ┌────────┴────────┐                             │
│                              │  WebShop Env   │                              │
│                              │  (1000 items)   │                              │
│                              └─────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                               A2A Protocol
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Purple Agent (Shopper)                              │
│                              Port 8001                                       │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                    │
│  │  LLM Client   │  │ Action Parser │  │  A2A Server   │                    │
│  │  (LiteLLM)    │  │               │  │               │                    │
│  └───────────────┘  └───────────────┘  └───────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Docker (Recommended)

```bash
# Clone the repository
git clone git@github.com:mpnikhil/agentbeats.git
cd agentbeats/webshop-plus

# Configure environment
cp sample.env .env
# Edit .env with your LLM API key

# Start both agents
docker compose up -d

# Verify agents are running
curl http://localhost:8000/.well-known/agent-card.json
curl http://localhost:8001/.well-known/agent-card.json

# View logs
docker compose logs -f

# Stop agents
docker compose down
```

### Local Development

#### Prerequisites

- Python 3.10+
- Java 11+ (for WebShop's Lucene search)
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Ollama (for local LLM inference)

#### Setup

```bash
# Install Ollama and pull model
ollama pull qwen3-coder:30b

# Setup green agent
cd green_agent
uv sync
cp ../.env.local.example .env.local

# Setup purple agent
cd ../purple_agent
uv sync
```

#### Running Locally

```bash
# Terminal 1: Start green agent
cd green_agent
uv run python src/server.py --host 0.0.0.0 --port 8000

# Terminal 2: Start purple agent
cd purple_agent
uv run python src/server.py --host 0.0.0.0 --port 8001

# Terminal 3: Run assessment
cd webshop-plus
uv run python scripts/run_local_assessment.py --tasks 3 --verbose
```

## Project Structure

```
webshop-plus/
├── green_agent/              # Evaluator agent
│   ├── src/                  # Source code
│   │   ├── server.py         # FastAPI A2A server
│   │   ├── agent.py          # Orchestration logic
│   │   ├── evaluator.py      # Scoring engine
│   │   ├── state_manager.py  # Session & cart tracking
│   │   ├── task_generator.py # Task loading
│   │   ├── llm_client.py     # LiteLLM wrapper
│   │   ├── messenger.py      # A2A protocol utilities
│   │   ├── models.py         # Pydantic models
│   │   └── webshop_wrapper.py # WebShop environment
│   ├── tests/                # Unit tests (313 tests)
│   ├── data/tasks/           # 80 task definitions
│   └── Dockerfile
├── purple_agent/             # Baseline shopping agent
│   ├── src/                  # Source code
│   │   ├── server.py         # FastAPI A2A server
│   │   ├── agent.py          # Shopping logic
│   │   ├── llm_client.py     # LiteLLM wrapper
│   │   └── messenger.py      # A2A protocol utilities
│   ├── tests/                # Unit tests (53 tests)
│   └── Dockerfile
├── webshop/                  # Princeton WebShop (submodule)
│   └── data/                 # Product catalog (1000 items)
├── scenarios/                # AgentBeats scenarios
│   └── webshop_plus/
│       └── scenario.toml
├── scripts/
│   ├── run_local_assessment.py  # Local testing script
│   └── test_integration.py      # Integration tests
├── docker-compose.yml
├── sample.env
└── README.md
```

## Task Types

| Type | Count | Description | Scoring |
|------|-------|-------------|---------|
| Budget Constrained | 20 | Multi-item shopping within budget | Budget adherence + item selection |
| Preference Memory | 15 | Cross-session consistency | Preference recall accuracy |
| Negative Constraint | 20 | Avoiding forbidden attributes | Constraint violation penalty |
| Comparative Reasoning | 15 | Comparing and justifying choices | LLM-as-judge reasoning quality |
| Error Recovery | 10 | Fixing cart mistakes | Recovery efficiency |

## Assessment API

### Starting an Assessment

Send a message to the green agent's A2A endpoint:

```bash
curl -X POST http://localhost:8000/a2a \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "message/stream",
    "params": {
      "message": {
        "parts": [{"kind": "text", "text": "Start assessment"}]
      },
      "metadata": {
        "participants": {
          "shopper": "http://localhost:8001/a2a"
        },
        "config": {
          "num_tasks": 3,
          "task_types": ["budget_constrained", "negative_constraint"]
        }
      }
    }
  }'
```

### Assessment Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_tasks` | int | 10 | Number of tasks to run |
| `task_types` | list | all | Task types to include |
| `timeout_per_task` | int | 300 | Seconds per task |
| `max_actions_per_task` | int | 20 | Max actions before timeout |
| `random_seed` | int | null | Seed for reproducibility |

## LLM Configuration

WebShop+ uses [LiteLLM](https://github.com/BerriAI/litellm) for provider-agnostic LLM access:

| Environment | Model | Configuration |
|-------------|-------|---------------|
| Local (primary) | Qwen3-Coder 30B | `LLM_MODEL=ollama/qwen3-coder:30b` |
| Local (fallback) | Qwen3 32B | `LLM_MODEL=ollama/qwen3:32b` |
| Production | Qwen3 32B (Nebius) | `LLM_MODEL=nebius/Qwen/Qwen3-32B` |
| Alternative | OpenAI GPT-4o | `LLM_MODEL=openai/gpt-4o` |

## Docker Images

Pre-built images are available on GitHub Container Registry:

```bash
# Pull images
docker pull ghcr.io/mpnikhil/webshop-plus-green:latest
docker pull ghcr.io/mpnikhil/webshop-plus-purple:latest

# Run green agent
docker run -p 8000:8000 \
  -e LLM_MODEL=nebius/Qwen/Qwen3-32B \
  -e LLM_API_KEY=your-key \
  -v ./webshop/data:/app/webshop/data:ro \
  ghcr.io/mpnikhil/webshop-plus-green:latest

# Run purple agent
docker run -p 8001:8001 \
  -e LLM_MODEL=nebius/Qwen/Qwen3-32B \
  -e LLM_API_KEY=your-key \
  ghcr.io/mpnikhil/webshop-plus-purple:latest
```

## Development

### Running Tests

```bash
# Green agent tests
cd green_agent
uv run pytest tests/ -v

# Purple agent tests
cd purple_agent
uv run pytest tests/ -v

# Integration tests
cd webshop-plus
uv run python scripts/test_integration.py
```

### Building Docker Images

```bash
# Build both images
docker compose build

# Build specific image
docker build -t webshop-plus-green ./green_agent
docker build -t webshop-plus-purple ./purple_agent
```

## AgentBeats Integration

WebShop+ is designed for the AgentBeats platform. To register:

1. Deploy agents using Docker or to a cloud provider
2. Ensure agents are accessible via HTTPS
3. Register the green agent's agent card URL with AgentBeats
4. The platform will discover capabilities via `/.well-known/agent-card.json`

### Agent Cards

- **Green Agent**: `http://localhost:8000/.well-known/agent-card.json`
- **Purple Agent**: `http://localhost:8001/.well-known/agent-card.json`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `uv run pytest tests/ -v`
4. Submit a pull request

## License

MIT

## Acknowledgments

- [Princeton WebShop](https://webshop-pnlp.github.io/) - Original shopping environment
- [LiteLLM](https://github.com/BerriAI/litellm) - Provider-agnostic LLM SDK
- [AgentBeats](https://agentbeats.example.com) - Agent evaluation platform
