help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies with uv (including dev tools)"
	@echo "  make lint       - Run ruff check and ruff format --check"
	@echo "  make format     - Run ruff format to auto-fix code style"
	@echo "  make typecheck  - Run pyright for static type checking"
	@echo "  make test       - Run pytest with coverage"
	@echo "  make pipeline   - Run the DVC pipeline"
	@echo "  make serve      - Run the web application"
	@echo "  make mlflow     - Launch MLflow UI (SQLite)"
	@echo "  make docker     - Build the Docker image"
	@echo "  make clean      - Remove build artifacts, cache, and logs"

install:
	uv sync --extra dev

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run pyright src/

test:
	uv run pytest tests/ --cov=src --cov-fail-under=25 --cov-report=term-missing -v

pipeline:
	uv run dvc repro

serve:
	uv run python -m src.app.main

mlflow:
	@powershell -Command "$$env:MLFLOW_TRACKING_URI='sqlite:///mlflow.db'; uv run mlflow ui" || (set MLFLOW_TRACKING_URI=sqlite:///mlflow.db && uv run mlflow ui)

docker:
	docker build -t hybrid-recommender .

clean:
	cmd /c if exist artifacts rmdir /s /q artifacts
	cmd /c if exist logs rmdir /s /q logs
	cmd /c if exist mlruns rmdir /s /q mlruns
	cmd /c for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s/q "%%d"
	cmd /c for /d /r . %%d in (.pytest_cache) do @if exist "%%d" rd /s/q "%%d"
	cmd /c for /d /r . %%d in (.ruff_cache) do @if exist "%%d" rd /s/q "%%d"
