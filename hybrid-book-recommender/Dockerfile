# Use official Python 3.11 slim image
FROM python:3.11-slim

# Install uv directly from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Install system dependencies (e.g., for build tools if needed by dependencies)
# RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy project metadata files needed by hatchling to resolve the package.
# README.md must be included because pyproject.toml declares `readme = "README.md"`;
# hatchling validates this field during `uv sync --frozen` (editable install).
# Copying these together before `COPY . .` preserves optimal layer caching.
COPY pyproject.toml uv.lock README.md ./

# Install dependencies into the system or a virtual environment
# --frozen ensures we use exact versions from uv.lock
# --system installs into the system python environment (simplifies path, but uv default is venv)
# We will stick to uv's default venv which is robust
RUN uv sync --frozen

# Copy the rest of the application
COPY . .

# Expose the Gradio port
EXPOSE 7860

# Run the application
# uv run will automatically use the environment created by uv sync
CMD ["uv", "run", "python", "-m", "src.app.main"]
