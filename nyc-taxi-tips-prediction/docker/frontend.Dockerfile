# -----------------------------------------------------------------------------
# THE BRAIN: Agentic Chat UI (LangGraph + Gemini-2.5-flash)
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
COPY reports /app/reports

# Install the project
RUN uv sync --frozen --no-dev

# Expose port
EXPOSE 8501

# Start the Streamlit application
CMD ["uv", "run", "streamlit", "run", "src/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
