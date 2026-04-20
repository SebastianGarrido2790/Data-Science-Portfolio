# Developer Experience (DX) Report

## 1. Executive Summary
The *Hybrid Book Recommender* is designed with a "Dev-First" philosophy, prioritizing automation, type safety, and local environment reproducibility. This report details the tools and configurations that ensure a seamless developer workflow from onboarding to production deployment.

## 2. Environment Management (`uv`)
We leverage **`uv`** as our primary package manager and task orchestrator.
- **Speed**: Dependency resolution is significantly faster than traditional `pip`.
- **Reproducibility**: The `uv.lock` file ensures that every developer and CI/CD agent uses the exact same versions of packages.
- **Workflow Isolation**: The project uses a dedicated virtual environment managed automatically by `uv sync`.

## 3. Automation Layer (`Makefile`)
The `Makefile` serves as the centralized interface for all frequent operations, abstracting complex commands into intuitive aliases.

| Command | Action | Purpose |
| :--- | :--- | :--- |
| `./launch_recommender.bat` | `uv sync` + `python -m ...` | **Primary Entry Point (Windows)**: One-click setup/launch. |
| `./validate_recommender.bat` | Pyright, Ruff, Pytest, DVC | **Full System Health Check**: Verifies architecture integrity. |
| `make install` | `uv sync --extra dev` | Complete environment setup including dev tools. |
| `make lint` | `uv run ruff check` | Static code analysis and linting. |
| `make format` | `uv run ruff format` | Auto-formatting following `py311` standards. |
| `make typecheck`| `uv run pyright src/`| Validates strict type hint coverage. |
| `make test` | `uv run pytest tests/`| Executes the full test suite with coverage report. |
| `make serve` | `python -m src.app.main`| Launches the Gradio web application. |
| `make mlflow` | `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`| Launches the MLflow UI. |
| `.\launch_recommender.bat` | `launch_recommender.bat` | One-click launch for the App (Windows). |
| `.\validate_recommender.bat` | `validate_recommender.bat` | Full system integrity check (Windows). |
| `make pipeline` | `uv run dvc repro` | Executes the end-to-end MLOps DAG. |

## 4. Quality Guardrails (`pre-commit`)
We use the **`pre-commit`** framework to enforce code quality locally *before* code reaches the repository.
- **Ruff**: Integrated for lightning-fast linting and formatting.
- **Pyright**: Integrated as a local hook to prevent type violations from being committed.
- **Benefit**: This significantly reduces CI/CD failure rates by catching common errors at the source.

## 5. Configuration Strategy (`.env.example`)
To prevent "leaking" sensitive keys while ensuring easy onboarding, we maintain a strictly versioned `.env.example` file.
- **Context Separation**: Clearly defines where API keys (e.g., `GOOGLE_API_KEY`) and environmental overrides (`ENV=local`) should be placed.
- **Onboarding Hygiene**: New developers simply `cp .env.example .env` and fill in their credentials.

## 6. Project Architecture Standards
The project adheres to the **"Python-Development" Standard**:
- **Strict Typing**: 100% coverage with `pyright`.
- **Package Orientation**: No root-level scripts; everything executes as a modular package (`src`).
- **Separation of Concerns**: Decouples UI (`src.app`) from Core Models (`src.models`) and Ingestion (`src.components`).

---
*Created for the Advanced Agentic Coding Portfolio.*
