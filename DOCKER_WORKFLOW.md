# WebShop+ Docker Workflow

This document describes the complete workflow for building, testing, and deploying WebShop+ Docker containers.

## Overview

WebShop+ consists of two Docker images:
- **Green Agent** (`ghcr.io/mpnikhil/webshop-plus-green`): Evaluator agent with WebShop catalog and MCP tools
- **Purple Agent** (`ghcr.io/mpnikhil/webshop-plus-purple`): Shopping agent with ADK and LiteLLM

Both images are built from the `webshop-plus/` directory and pushed to GitHub Container Registry (ghcr.io).

---

## Prerequisites

### 1. Docker Installation
```bash
# Verify Docker is installed
docker --version
```

### 2. GitHub Container Registry Authentication (One-time setup)

Create a GitHub Personal Access Token:
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token"
3. Select scopes: `write:packages` and `read:packages`
4. Generate and copy the token

Authenticate Docker with ghcr.io:
```bash
# Set your token as an environment variable
export GITHUB_TOKEN=ghp_your_token_here

# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u mpnikhil --password-stdin
```

---

## Build Script

Use the provided `build_and_push.sh` script for all Docker operations:

### Build Locally (No Push)
```bash
cd /Users/nikhilpujari/agentbeats/webshop-plus

# Build both images with :latest tag
./build_and_push.sh
```

This builds:
- `ghcr.io/mpnikhil/webshop-plus-green:latest`
- `ghcr.io/mpnikhil/webshop-plus-purple:latest`

### Build and Push
```bash
# Build and push to ghcr.io
./build_and_push.sh --push
```

### Build with Version Tag
```bash
# Build with specific version (also tags as :latest)
./build_and_push.sh --tag v1.3.0

# Build and push with version tag
./build_and_push.sh --tag v1.3.0 --push
```

This creates both `:v1.3.0` and `:latest` tags.

---

## Manual Build Commands

If you need to build manually without the script:

### Green Agent
```bash
cd /Users/nikhilpujari/agentbeats/webshop-plus

docker build \
  -t ghcr.io/mpnikhil/webshop-plus-green:latest \
  -f green_agent/Dockerfile \
  .
```

**Build Context**: The `.` at the end means the build context is `webshop-plus/`
- Includes: `green_agent/`, `webshop/` (catalog data)
- Dockerfile paths are relative to this directory

### Purple Agent
```bash
cd /Users/nikhilpujari/agentbeats/webshop-plus

docker build \
  -t ghcr.io/mpnikhil/webshop-plus-purple:latest \
  -f purple_agent/Dockerfile \
  .
```

**Important**: Both Dockerfiles use paths like `COPY green_agent/src/` or `COPY purple_agent/src/` because the build context is `webshop-plus/`.

### Force Rebuild (No Cache)
```bash
docker build --no-cache \
  -t ghcr.io/mpnikhil/webshop-plus-green:latest \
  -f green_agent/Dockerfile \
  .
```

---

## Testing Workflow

### 1. Build Images Locally
```bash
cd /Users/nikhilpujari/agentbeats/webshop-plus
./build_and_push.sh
```

### 2. Generate Docker Compose Configuration
```bash
cd /Users/nikhilpujari/agentbeats/agentbeats-leaderboard-template

# Generate from scenario file
python generate_compose.py --scenario ../webshop-plus/scenario_quick.toml
```

This creates:
- `docker-compose.yml` - Container orchestration
- `a2a-scenario.toml` - Assessment configuration

### 3. Manual Fixes (Required)

⚠️ **IMPORTANT**: After running `generate_compose.py`, you MUST manually edit `docker-compose.yml`:

#### Fix 1: Remove Platform Constraints (3 places)

Remove these lines that cause "no matching manifest" errors on ARM Macs:

```yaml
# Remove from green-agent service (around line 63):
    platform: linux/amd64

# Remove from shopper service (around line 97):
    platform: linux/amd64

# Remove from agentbeats-client service (around line 80):
    platform: linux/amd64
```

#### Fix 2: Add --advertise-host Flag to Green Agent

Update the green-agent command to include the `--advertise-host` flag:

**Before** (line 7):
```yaml
command: ["--host", "0.0.0.0", "--port", "9009", "--card-url", "http://green-agent:9009"]
```

**After**:
```yaml
command: ["--host", "0.0.0.0", "--port", "9009", "--card-url", "http://green-agent:9009", "--advertise-host", "green-agent"]
```

**Why**: The `--advertise-host` flag tells the green agent to advertise itself using the Docker service name instead of the container's internal hostname, which is required for proper A2A communication.

**Note**: These manual steps are temporary. The `generate_compose.py` script will be updated to include these fixes automatically in the future.

### 4. Configure Environment
```bash
# Create .env file with your API key
echo "OPENAI_API_KEY=your_nebius_api_key_here" > .env
```

### 5. Run Local Test
```bash
# Clean up any old containers
docker compose down

# Start assessment
docker compose up

# Or run in background
docker compose up -d
```

**Key Point**: Docker uses your **local images first** before pulling from the registry. So even though `docker-compose.yml` references `ghcr.io/mpnikhil/...`, it will use your locally built images.

### 6. Monitor Progress
```bash
# Follow all logs
docker compose logs -f

# Follow specific service
docker compose logs -f agentbeats-client  # Assessment progress
docker compose logs -f shopper            # Shopping actions
docker compose logs -f green-agent        # Evaluation logs

# Filter for key events
docker compose logs -f agentbeats-client | grep -E "task_id|Status:|Assessment complete"
```

### 7. Check Results
```bash
# View aggregate results
cat output/results.json | jq '.results[0].aggregate'

# View by task type
cat output/results.json | jq '.results[0].aggregate.by_task_type'

# View individual tasks
cat output/results.json | jq '.results[0].results[] | {task_id, task_type, success, overall_score}'
```

---

## Pushing to Registry

### Push Latest
```bash
cd /Users/nikhilpujari/agentbeats/webshop-plus

# Build and push
./build_and_push.sh --push
```

### Push Specific Version
```bash
# Build with version tag and push
./build_and_push.sh --tag v1.3.0 --push
```

This creates:
- `ghcr.io/mpnikhil/webshop-plus-green:v1.3.0`
- `ghcr.io/mpnikhil/webshop-plus-green:latest`
- `ghcr.io/mpnikhil/webshop-plus-purple:v1.3.0`
- `ghcr.io/mpnikhil/webshop-plus-purple:latest`

### Manual Push
```bash
docker push ghcr.io/mpnikhil/webshop-plus-green:latest
docker push ghcr.io/mpnikhil/webshop-plus-purple:latest
```

---

## Complete Development Cycle

```bash
# 1. Make code changes
cd /Users/nikhilpujari/agentbeats/webshop-plus
# Edit files in green_agent/src/ or purple_agent/src/

# 2. Rebuild affected container
./build_and_push.sh

# 3. Test locally
cd ../agentbeats-leaderboard-template
python generate_compose.py --scenario ../webshop-plus/scenario_quick.toml
# (Remove platform constraints from docker-compose.yml)
docker compose down && docker compose up

# 4. Check results
cat output/results.json | jq '.results[0].aggregate'

# 5. If tests pass, push to registry
cd ../webshop-plus
./build_and_push.sh --push

# 6. Commit changes
git add .
git commit -m "feat: description of changes"
git push
```

---

## Dockerfile Architecture

### Green Agent Dockerfile

**Location**: `green_agent/Dockerfile`

**Key Features**:
- Base: `python:3.11-slim`
- Java 21: Required for WebShop's Lucene search
- Build Context: `webshop-plus/` directory
- Includes:
  - `green_agent/src/` - Agent code
  - `green_agent/data/` - Task definitions
  - `webshop/` - Product catalog (1000 items)

**COPY Paths** (relative to build context):
```dockerfile
COPY green_agent/pyproject.toml ./
COPY green_agent/src/ ./src/
COPY green_agent/data/ ./data/
COPY webshop/ ./webshop/
```

### Purple Agent Dockerfile

**Location**: `purple_agent/Dockerfile`

**Key Features**:
- Base: `python:3.11-slim`
- Minimal dependencies (no Java needed)
- Build Context: `webshop-plus/` directory
- Includes:
  - `purple_agent/src/` - Agent code with ADK

**COPY Paths** (relative to build context):
```dockerfile
COPY purple_agent/pyproject.toml ./
COPY purple_agent/src/ ./src/
```

---

## Troubleshooting

### Build Fails: "COPY src/ ./src/: no such file"

**Problem**: Dockerfile paths are incorrect for the build context.

**Solution**: Ensure COPY commands use paths relative to `webshop-plus/`:
- ✅ `COPY green_agent/src/ ./src/`
- ❌ `COPY src/ ./src/`

### Platform Errors on ARM Macs

**Problem**: `no matching manifest for linux/arm64/v8`

**Solution**: Remove `platform: linux/amd64` from `docker-compose.yml`

### Docker Uses Old Image

**Problem**: Changes not reflected in container.

**Solution**: Force rebuild without cache:
```bash
docker build --no-cache -t ghcr.io/mpnikhil/webshop-plus-green:latest \
  -f green_agent/Dockerfile .
```

### Authentication Failed

**Problem**: `denied: permission_denied`

**Solution**: Re-authenticate with GitHub:
```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u mpnikhil --password-stdin
```

### Check Image Source (Local vs Remote)

```bash
# List local images
docker images | grep webshop-plus

# Inspect image to see where it came from
docker inspect ghcr.io/mpnikhil/webshop-plus-green:latest | grep -A 5 "RepoDigests"
```

### Force Pull Remote Image

```bash
# Remove local image
docker rmi ghcr.io/mpnikhil/webshop-plus-green:latest

# Next docker compose up will pull from registry
docker compose up
```

---

## Image Details

### Green Agent Image
- **Size**: ~2GB (includes Java + WebShop catalog)
- **Port**: 8000 (configurable via ENV)
- **Health Check**: `http://localhost:8000/.well-known/agent-card.json`
- **Environment Variables**:
  - `OPENAI_API_KEY` - LLM API key
  - `OPENAI_API_BASE` - API endpoint (default: Nebius)
  - `LLM_MODEL` - Model name (e.g., `openai/Qwen/Qwen3-32B`)
  - `WEBSHOP_DIR` - Path to WebShop catalog (default: `/app/webshop`)

### Purple Agent Image
- **Size**: ~1GB (minimal Python + ADK)
- **Port**: 8001 (configurable via ENV)
- **Health Check**: `http://localhost:8001/.well-known/agent-card.json`
- **Environment Variables**:
  - `OPENAI_API_KEY` - LLM API key
  - `OPENAI_API_BASE` - API endpoint
  - `LLM_MODEL` - Model name

---

## CI/CD Integration

### GitHub Actions Workflow (Example)

```yaml
name: Build and Push Docker Images

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          submodules: recursive

      - name: Login to GitHub Container Registry
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        run: |
          cd webshop-plus
          ./build_and_push.sh --push
```

---

## Quick Reference

### Essential Commands

```bash
# Build locally
./build_and_push.sh

# Build and push
./build_and_push.sh --push

# Build with version
./build_and_push.sh --tag v1.0.0 --push

# Test locally
cd ../agentbeats-leaderboard-template
python generate_compose.py --scenario ../webshop-plus/scenario_quick.toml
docker compose up

# View results
cat output/results.json | jq '.results[0].aggregate'

# Cleanup
docker compose down
```

### Important Paths

- **Build Script**: `/Users/nikhilpujari/agentbeats/webshop-plus/build_and_push.sh`
- **Green Dockerfile**: `/Users/nikhilpujari/agentbeats/webshop-plus/green_agent/Dockerfile`
- **Purple Dockerfile**: `/Users/nikhilpujari/agentbeats/webshop-plus/purple_agent/Dockerfile`
- **Scenario File**: `/Users/nikhilpujari/agentbeats/webshop-plus/scenario_quick.toml`
- **Test Directory**: `/Users/nikhilpujari/agentbeats/agentbeats-leaderboard-template/`
- **Results Output**: `/Users/nikhilpujari/agentbeats/agentbeats-leaderboard-template/output/results.json`

---

## Summary

**Local Development**:
1. Build with `./build_and_push.sh`
2. Test with `docker compose up`
3. Docker uses local images automatically

**Publishing**:
1. Test passes → `./build_and_push.sh --push`
2. Registry images available for others

**Key Insight**: Docker prefers local images over remote ones, so you can safely test with the same image names before pushing to the registry.
