# WebShop+ Purple Agent Dockerfile
# Baseline shopping agent for the WebShop+ benchmark
#
# Build: docker build -t webshop-plus-purple .
# Run: docker run -p 8001:8001 -e LLM_MODEL=... -e LLM_API_KEY=... webshop-plus-purple

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY purple_agent/pyproject.toml ./

# Create virtual environment and install dependencies
RUN uv venv .venv && \
    . .venv/bin/activate && \
    uv sync --no-dev

# Copy source code
COPY purple_agent/src/ ./src/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV PORT=8001
ENV HOST=0.0.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Run the server
# Use exec form for proper signal handling
ENTRYPOINT [".venv/bin/python", "src/server.py"]
CMD ["--host", "0.0.0.0", "--port", "8001"]
