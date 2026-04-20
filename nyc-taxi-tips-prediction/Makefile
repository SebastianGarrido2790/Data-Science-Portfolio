# Makefile for NYC Taxi Tips Prediction
# Usage: make <target>
# Requires: uv (https://docs.astral.sh/uv), Docker

.PHONY: help install lint format typecheck test pipeline serve docker clean

## ─────────────────────────────────────────────────────
##  Help
## ─────────────────────────────────────────────────────

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

## ─────────────────────────────────────────────────────
##  Environment
## ─────────────────────────────────────────────────────

install:  ## Install all dependencies (including dev)
	uv sync --all-extras

## ─────────────────────────────────────────────────────
##  Code Quality
## ─────────────────────────────────────────────────────

lint:  ## Lint and auto-fix using Ruff
	uv run ruff check --fix .

format:  ## Format all code using Ruff formatter
	uv run ruff format .

typecheck:  ## Run strict type checking with Pyright
	uv run pyright src/

## ─────────────────────────────────────────────────────
##  Testing
## ─────────────────────────────────────────────────────

test:  ## Run full test suite with coverage reporting
	uv run pytest --cov=src --cov-report=term-missing tests/

test-ci:  ## Run tests with coverage gate (used in CI)
	uv run pytest --cov=src --cov-fail-under=65 --cov-report=term-missing tests/

## ─────────────────────────────────────────────────────
##  ML Pipeline
## ─────────────────────────────────────────────────────

pipeline:  ## Reproduce the full DVC FTI pipeline
	uv run dvc repro

pipeline-dag:  ## Visualize the DVC pipeline DAG
	uv run dvc dag

## ─────────────────────────────────────────────────────
##  Serving
## ─────────────────────────────────────────────────────

serve:  ## Start FastAPI inference server (port 8000)
	uv run uvicorn src.api.predict_api:app --reload --port 8000

## ─────────────────────────────────────────────────────
##  Docker
## ─────────────────────────────────────────────────────

docker:  ## Build and start all services via Docker Compose
	docker compose up --build

docker-down:  ## Stop all Docker services
	docker compose down

## ─────────────────────────────────────────────────────
##  MLflow
## ─────────────────────────────────────────────────────

mlflow:  ## Open MLflow tracking UI locally
	uv run mlflow ui

## ─────────────────────────────────────────────────────
##  Cleanup
## ─────────────────────────────────────────────────────

clean:  ## Remove Python caches and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
	find . -type f -name "*.pyc" -delete; \
	rm -rf .coverage htmlcov/ dist/ build/ *.egg-info
