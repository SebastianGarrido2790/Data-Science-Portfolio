# -----------------------------------------------------------------------------
# THE BRAWN: Deterministic Model Serving (FastAPI FTI Engine)
# -----------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Copy pyproject.toml and lockfile
COPY pyproject.toml uv.lock ./

# Install dependencies strictly from the lockfile without installing the project itself
RUN uv sync --frozen --no-dev --no-install-project

# Copy the actual application code
COPY src /app/src
COPY artifacts /app/artifacts
COPY config /app/config

# Install the project
RUN uv sync --frozen --no-dev

# Expose port
EXPOSE 8000

# Start the FastAPI server
CMD ["uv", "run", "uvicorn", "src.api.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
