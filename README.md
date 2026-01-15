# WebShop+

A stateful shopping benchmark for agentic AI, extending Princeton's WebShop with advanced evaluation dimensions.

## Overview

WebShop+ is a **green agent** (evaluator) for the AgentBeats platform that tests shopping agents on:

- **Budget Management**: Multi-item shopping within spending limits
- **Preference Memory**: Cross-session consistency and recall
- **Negative Constraints**: Avoiding forbidden attributes (allergies, restrictions)
- **Comparative Reasoning**: Exploring options and justifying choices
- **Error Recovery**: Fixing mistakes in existing cart state

### Architecture Approach

WebShop+ uses a **hybrid A2A + MCP architecture**:
- **A2A Protocol**: Handles orchestration between green (evaluator) and purple (shopper) agents
- **MCP Protocol**: Provides tool execution layer (search, click, checkout)
- **Green Agent**: Hosts MCP server, monitors tool calls, evaluates performance
- **Purple Agent**: Uses ADK (Agent Development Kit) with `McpToolset` for automatic ReAct loops

This design separates concerns: A2A for task delegation and results, MCP for the actual shopping interactions.

## Architecture

WebShop+ uses a **hybrid A2A + MCP architecture** where A2A handles orchestration and MCP provides the tool execution layer:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Assessment Flow                                     │
│                                                                              │
│  1. Green receives assessment_request via A2A                               │
│  2. Green creates MCP session and sends kickoff to Purple                   │
│  3. Purple spawns ADK agent with MCP toolset                                │
│  4. ADK agent executes ReAct loop using MCP tools                          │
│  5. Green monitors MCP calls for evaluation                                 │
│  6. Purple sends completion message via A2A                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Green Agent (Evaluator) - Port 8000                     │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         A2A Server                                   │   │
│  │  • Receives assessment requests from AgentBeats                      │   │
│  │  • Sends task kickoffs with MCP URI to Purple                        │   │
│  │  • Receives completion messages from Purple                          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                       │                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    MCP Server (/mcp/{session_id})                    │   │
│  │  Tools: search(query), click(element_id), checkout()                │   │
│  │  Session State: cart, budget, turn_count, history                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                       │                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Evaluation Engine                               │   │
│  │  • Task Generator (80 tasks across 5 categories)                     │   │
│  │  • Evaluator (category-specific scoring logic)                       │   │
│  │  • WebShop Wrapper (1000-product catalog)                            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                            A2A (orchestration) + MCP (tools)
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Purple Agent (Shopper) - Port 8001                      │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         A2A Server                                   │   │
│  │  • Receives task kickoffs with MCP URI from Green                    │   │
│  │  • Sends completion messages back to Green                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                       │                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       ADK Shopping Agent                             │   │
│  │  • McpToolset (connects to Green's MCP server)                       │   │
│  │  • ReAct loop (automatic action/observation cycle)                   │   │
│  │  • LiteLLM (provider-agnostic LLM access)                            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Protocol Breakdown

- **A2A Protocol**: Used for orchestration between green and purple agents
  - Green sends task kickoffs with task data (goal, budget, constraints) and MCP URI in `resources` array
  - Purple sends completion messages back when task is done
  - Follows AgentBeats standard for agent-to-agent communication

- **MCP Protocol**: Used for tool execution within a task
  - Green hosts session-scoped MCP server at `/mcp/{session_id}`
  - Purple's ADK agent connects to MCP server via `McpToolset`
  - Tools: `search`, `click`, `checkout`
  - All tool calls monitored by green for evaluation

### Purple Agent Design (ADK Integration)

The purple agent uses Anthropic's Agent Development Kit (ADK) to handle shopping tasks:

1. **A2A Message Handling**: Purple receives kickoff message from green via A2A
2. **MCP URI Extraction**: Extracts MCP server URI from `message.resources[0].uri`
3. **ADK Agent Spawning**: Creates a `ShoppingAgent` instance with:
   - `McpToolset`: Automatically connects to green's MCP server
   - Task instructions from A2A message (goal, budget, constraints)
4. **ReAct Loop**: ADK runs automatic action/observation cycles:
   - Agent thinks about next step
   - Calls MCP tools (search/click/checkout)
   - Observes results
   - Repeats until task complete
5. **Completion**: Purple sends completion message back to green via A2A

**Key Benefit**: Purple agent requires no manual action parsing or tool call handling - ADK handles the entire ReAct loop automatically using the MCP toolset.

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
- Ollama (for green agent LLM evaluation)
- LM Studio (for purple agent ADK - runs on port 1234)

#### Setup

```bash
# 1. Install Ollama and pull model (for green agent)
ollama pull qwen3-coder:30b

# 2. Setup LM Studio (for purple agent)
# Download and start LM Studio: https://lmstudio.ai/
# Load model: qwen3-coder-30b-a3b-instruct-mlx
# Start server on port 1234 (default)

# 3. Setup green agent
cd green_agent
uv sync
cp ../.env.local.example .env.local

# 4. Setup purple agent
cd ../purple_agent
uv sync
cp ../.env.local.example .env.local
# Edit .env.local:
#   ADK_MODEL=openai/qwen3-coder-30b-a3b-instruct-mlx
#   OPENAI_API_BASE=http://localhost:1234/v1
```

#### Running Locally

```bash
# Prerequisites: Ensure Ollama and LM Studio are running
ollama serve                          # Green agent LLM
# LM Studio should be running on port 1234 with qwen3-coder-30b loaded

# Terminal 1: Start green agent (hosts MCP server + A2A endpoint)
cd green_agent
uv run python src/server.py --host 0.0.0.0 --port 8000

# Terminal 2: Start purple agent (ADK with A2A endpoint)
cd purple_agent
uv run python src/server.py --host 0.0.0.0 --port 8001

# Terminal 3: Run assessment (tests A2A + MCP integration)
cd webshop-plus
uv run python scripts/run_local_assessment.py --tasks 3 --verbose
```

**What Happens During Assessment:**
1. Green receives assessment request
2. Green creates MCP session at `/mcp/{session_id}`
3. Green sends A2A kickoff to purple with MCP URI
4. Purple spawns ADK agent with McpToolset pointing to MCP URI
5. ADK runs ReAct loop calling MCP tools (search/click/checkout)
6. Green monitors MCP calls for evaluation
7. Purple sends A2A completion message
8. Green calculates final score

## Project Structure

```
webshop-plus/
├── green_agent/              # Evaluator agent
│   ├── src/                  # Source code
│   │   ├── server.py         # FastAPI A2A + MCP server
│   │   ├── a2a_executor.py   # A2A executor (orchestration)
│   │   ├── agent.py          # Task management
│   │   ├── evaluator.py      # Scoring engine (5 categories)
│   │   ├── state_manager.py  # Session & cart tracking
│   │   ├── task_generator.py # Task loading (80 tasks)
│   │   ├── llm_client.py     # LiteLLM wrapper
│   │   ├── messenger.py      # A2A protocol utilities
│   │   ├── models.py         # 30+ Pydantic models
│   │   ├── webshop_wrapper.py # WebShop environment
│   │   └── webshop_mcp/      # MCP server
│   │       ├── server.py     # FastMCP tools (search/click/checkout)
│   │       ├── session_state.py # Per-session state
│   │       └── session_manager.py # Session lifecycle
│   ├── tests/                # Unit tests (503 tests)
│   ├── data/tasks/           # 80 task definitions (16 per category)
│   └── Dockerfile
├── purple_agent/             # Shopping agent (ADK-based)
│   ├── src/                  # Source code
│   │   ├── server.py         # FastAPI A2A server
│   │   ├── executor.py       # Routes MCP tasks to ADK agent
│   │   ├── shopping_agent.py # ADK agent with McpToolset
│   │   ├── llm_client.py     # LiteLLM wrapper
│   │   └── messenger.py      # A2A protocol utilities
│   ├── tests/                # Unit tests (142 tests)
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

### A2A Contract

WebShop+ uses A2A for orchestration and MCP for tool execution. The green agent sends task kickoffs to the purple agent with the following contract:

**Message from Green to Purple (Task Kickoff):**

```json
{
  "jsonrpc": "2.0",
  "id": "task-{task_id}",
  "method": "message/stream",
  "params": {
    "message": {
      "parts": [
        {
          "kind": "text",
          "text": "Task: {goal}\n\nBudget: ${budget}\nConstraints: {constraints}\n\nAvailable actions:\n- search(query): Search for products\n- click(element_id): Navigate to product, select options, add to cart\n- checkout(): Complete purchase (terminal action)"
        }
      ],
      "resources": [
        {
          "type": "mcp_server",
          "uri": "http://green-host:8000/mcp/{session_id}",
          "description": "WebShop MCP server for product search and purchase"
        }
      ]
    }
  }
}
```

**Key Fields:**
- `message.parts[0].text`: Task description including goal, budget, and constraints
- `message.resources[0].uri`: **CRITICAL:** This is the MCP server endpoint (session-scoped). The Purple Agent **must** connect to this URI to execute tools (`search`, `click`, `checkout`).
- `message.resources[0].type`: Always `"mcp_server"` for tool execution.

**Message from Purple to Green (Completion):**

The Purple Agent should send a completion message when it is done with the task (usually after calling the `checkout` tool or deciding to stop). This message aids in evaluation but the primary evaluation data comes from the MCP tool traces.

```json
{
  "jsonrpc": "2.0",
  "id": "completion-{task_id}",
  "method": "message/stream",
  "params": {
    "message": {
      "parts": [
        {
          "kind": "text",
          "text": "Task completed. Final cart summary: {cart_summary}. Reasoning: {reasoning_for_decisions}"
        }
      ]
    }
  }
}
```

*   **`text`**: Should include a summary of what was purchased and the reasoning behind decisions. The Green Agent's "LLM-as-a-Judge" uses this reasoning to evaluate the "Comparative Reasoning" dimension.

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

### Green Agent (Evaluator)
Used for LLM-as-judge evaluation prompts (comparative reasoning, preference memory analysis):

| Environment | Model | Configuration |
|-------------|-------|---------------|
| Local | Qwen3-Coder 30B | `LLM_MODEL=ollama/qwen3-coder:30b` |
| Production | Qwen3 32B (Nebius) | `LLM_MODEL=nebius/Qwen/Qwen3-32B` |

### Purple Agent (Shopper ADK)
Used by ADK for shopping decisions in ReAct loop:

| Environment | Model | Configuration |
|-------------|-------|---------------|
| Local (recommended) | Qwen3-Coder 30B via LM Studio | `ADK_MODEL=openai/qwen3-coder-30b-a3b-instruct-mlx`<br>`OPENAI_API_BASE=http://localhost:1234/v1` |
| Production | Qwen3 32B (Nebius) | `ADK_MODEL=nebius/Qwen/Qwen3-32B` |
| Alternative | OpenAI GPT-4o | `ADK_MODEL=openai/gpt-4o` |

**Note**: Purple agent's ADK requires LM Studio running locally on port 1234 for local development.

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
