# WebShop+ Green Agent Dockerfile
# Evaluator agent for the WebShop+ benchmark
#
# Build: docker build -t webshop-plus-green .
# Run: docker run -p 8000:8000 -e LLM_MODEL=... -e LLM_API_KEY=... webshop-plus-green

FROM python:3.11-slim

# Install system dependencies
# - Java 21 for WebShop's Lucene search (Java 17+ required)
# - curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jre-headless \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Java is in PATH after installation, no need to set JAVA_HOME explicitly
# The package manager handles architecture-specific paths

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml ./

# Create virtual environment and install dependencies
RUN uv venv .venv && \
    . .venv/bin/activate && \
    uv sync --no-dev

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# Copy WebShop data (mounted volume can override this)
# Note: WebShop data should be provided via volume mount for production
# COPY ../webshop/data ./webshop/data

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV PORT=8000
ENV HOST=0.0.0.0
ENV ADVERTISE_HOST=green

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Run the server
# Use exec form for proper signal handling
# advertise-host=green tells purple agent to connect via Docker service name
CMD [".venv/bin/python", "src/server.py", "--host", "0.0.0.0", "--port", "8000", "--advertise-host", "green"]
