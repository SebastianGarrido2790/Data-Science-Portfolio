# Deployment Report

*Last updated: 2026-03-27 | System version: v1.3*

## 1. Executive Summary
This document details the successful execution of **Deployment**, transforming the *Hybrid Book Recommender* from a local development project into a production-ready, containerized application with automated testing and Continuous Integration/Continuous Deployment (CI/CD) pipelines.

The deployment strategy focuses on **Reliability**, **Portability**, and **Automation**. We have established a robust testing framework to catch regressions, a Docker container for consistent execution across environments, and a GitHub Actions workflow to automate the build verification process.

As of **v1.3**, the system now includes an **Agentic Layer** (`src/agent/`) powered by `pydantic-ai` and Google Gemini Flash. This introduces additional runtime and secret management considerations documented in §7.

## 2. Containerization Strategy (Docker)
The application has been packaged into a Docker container to guarantee **"Build Once, Run Anywhere"** portability.

### **Dockerfile Architecture**
*   **Base Image:** `python:3.11-slim` (Minimizes attack surface and image size).
*   **Package Management:** **`uv`** (Astral) installed directly from its official image (`ghcr.io/astral-sh/uv:latest`).
*   **Layer Caching:** `uv.lock` and `pyproject.toml` are copied and synced *before* the source code.
*   **Runtime Security:** The application runs as a standard process, listening on port `7860`.
*   **Command:** The container executes using `uv run python -m src.app.main` for full package support.

### **Docker Ignore**
A rigorous `.dockerignore` file ensures that local development artifacts (virtual environments, logs, caches, large datasets) are **excluded** from the build context, keeping the image lean and secure.

## 3. CI/CD Pipeline (GitHub Actions)
An automated pipeline has been established to guard the `main` branch against broken code.

### **Workflow: `.github/workflows/main.yaml`**
*   **Trigger:** Pushes and Pull Requests to the `main` branch.
*   **Jobs:**

#### **Job 1: `test_and_build`**

| Step | Command | Purpose |
|:---|:---|:---|
| Checkout | `actions/checkout@v4` | Fetch source code |
| Install uv | `astral-sh/setup-uv@v5` | Fast package manager with caching |
| Setup Python | `uv python install 3.11` | Pin runtime version |
| Install deps | `uv sync --extra dev` | Installs all deps including dev tools (`ruff`, `pyright`, `pytest`) |
| **Lint & Format** | `ruff format --check` + `ruff check` | Enforces code style and import hygiene across `src/` and `tests/` |
| **Type Check** | `uv run pyright src/` | Validates 100% type hint coverage (strict mode) |
| **Unit Tests** | `uv run pytest tests/ --cov=src --cov-fail-under=25 -v` | Runs full test suite; build **fails** if coverage drops below 25% |
| Docker Login | `docker/login-action@v3` | GHCR auth via `secrets.GITHUB_TOKEN` (skipped on PRs) |
| Docker Metadata | `docker/metadata-action@v5` | Generates `latest` tag + OCI labels |
| **Build & Push** | `docker/build-push-action@v5` | Builds image; pushes to GHCR **only** on merge to `main` |

> [!IMPORTANT]
> The `ruff`, `pyright`, and `pytest --cov` steps are **hard quality gates** — any failure blocks the Docker build and push. The coverage threshold (`--cov-fail-under=25`) will rise as the test suite matures.

#### **Job 2: `deploy`**

*   **Trigger:** Runs only after `test_and_build` succeeds on the `main` branch.
*   **Target:** EC2 instance via SSH (`appleboy/ssh-action@v1.0.3`).
*   **Required Secrets:** `EC2_HOST`, `EC2_USERNAME`, `EC2_SSH_KEY`.
*   **Deployment Script:**
    1. Login to GHCR on the remote host.
    2. Pull the latest image (`ghcr.io/<repo>:latest`).
    3. Stop and remove the old `hybrid-recommender` container.
    4. Run the new container with `--restart always` on port `7860`.
    5. Prune dangling images.
*   **Graceful Degradation:** If `EC2_HOST` is not set, the step is skipped and a warning is emitted — the build still succeeds, enabling portfolio CI/CD demonstration without a live server.

#### **Security & Authorization**
*   **GHCR Login:** Authenticates using `secrets.GITHUB_TOKEN` (automatic repository secret).
*   **Permissions:** `packages: write` verified for the job.

This pipeline is an **insurance policy**. It guarantees that:

1. The code works on a clean machine (not just "on my machine").
2. Dependencies are locked and secure (`uv.lock`).
3. Code formatting and type-safety are enforced before merging.
4. The core logic (Recommend + Agent) is tested with coverage enforcement.
5. The application can be packaged and deployed from a container.

## 4. Verification & Results
*   **Local Test Pass:** All unit tests (including 7 agent schema/tool tests) gathered and passed successfully in `< 1s`.
*   **Container Runtime:** The container `hybrid-recommender` successfully builds and serves the Gradio application on `http://localhost:7860`.
*   **Network Binding:** The application listening interface was updated to `0.0.0.0` to ensure accessibility from outside the container.
*   **GitHub Actions:** The CI/CD pipeline successfully executed all jobs — lint, type-check, tests, Docker build, and push — without failures.

## 5. Conclusion
The deployment successfully bridges the gap between "Science" and "Engineering". The *Hybrid Book Recommender* is no longer just a script; it is a tested, portable, and automatable software product ready for cloud deployment (e.g., AWS EC2, Hugging Face Spaces).

With the addition of the **Agentic Layer (v1.3)**, the system is now also a conversational AI product — and the deployment pipeline has been hardened to validate the agent's Pydantic schemas and tool contracts as part of the standard CI gate.

---

## 6. Appendix: Operational Commands

### **CI/CD Workflow (Local)**
To simulate the GitHub Actions pipeline locally:

```bash
# 1. Install Dependencies (Exact versions from uv.lock, including dev tools)
uv sync --extra dev

# 2. Lint & Format Check
uv run ruff format --check src/ tests/
uv run ruff check src/ tests/

# 3. Type Check
uv run pyright src/

# 4. Run Unit Tests with Coverage
uv run pytest tests/ --cov=src --cov-fail-under=25 --cov-report=term-missing -v

# 5. Build Docker Image
docker build . -t hybrid-recommender:latest
```

### **Docker Operations**

### **Build & Run**
```bash
# Build the image
docker build -t hybrid-recommender .

# Run the container with the required API key (detached mode, port 7860)
# GOOGLE_API_KEY is required for the Agentic Layer (src/agent/)
docker run -d \
  -p 7860:7860 \
  -e GOOGLE_API_KEY="your_api_key_here" \
  --name hybrid-app \
  hybrid-recommender
```

### **Management & Debugging**
```bash
# View running containers
docker ps

# View logs (add -f to follow real-time)
docker logs -f hybrid-app

# Stop the container
docker stop hybrid-app

# Remove the container
docker rm hybrid-app

# Access the container shell (for deep debugging)
docker exec -it hybrid-app /bin/bash
```

### **Modular Execution**
Running the app without Docker:
```bash
# Launch server via python modular command
uv run python -m src.app.main

# Or via Makefile (Recommended)
make serve
```

### **Cleanup**
```bash
# Remove the image
docker rmi hybrid-recommender

# Prune unused Docker objects (use with caution)
docker system prune
```
---

## 7. Agentic Layer Deployment Considerations (v1.3+)

The Agentic Layer introduces external API dependency on **Google Gemini** via the `GOOGLE_API_KEY`. This must be managed carefully across local, CI, and production environments.

### 7.1 Secret Management

| Environment | Method | Notes |
|:---|:---|:---|
| **Local dev** | `.env` file (via `python-dotenv`) or shell export | Never commit to version control |
| **Docker run** | `-e GOOGLE_API_KEY=...` flag | See §6 Build & Run command |
| **GitHub Actions** | `secrets.GOOGLE_API_KEY` repository secret | Used for live agent smoke tests (future) |
| **EC2 Production** | EC2 instance environment variable or AWS Secrets Manager | Injected via `docker run -e` in the deploy script |

> [!CAUTION]
> The agent will silently fall back to the error `AgentResponse` if `GOOGLE_API_KEY` is missing or invalid at runtime. Always verify the key is present before running the container in a new environment.

> [!TIP]
> Make sure you have added the `GOOGLE_API_KEY` to your **GitHub Settings > Secrets and variables > Actions** if you haven't already. This is essential for the AI Assistant's conversational capabilities.

To add the **`GOOGLE_API_KEY`** to your GitHub repository, follow these steps:

#### **1. Get your API Key**
If you don't have one yet, go to [Google AI Studio (aistudio.google.com)](https://aistudio.google.com/app/apikey) and create a new API key for **Gemini 1.5 Flash**.

#### **2. Add the Secret to GitHub**
1.  Go to your repository on GitHub: `SebastianGarrido2790/hybrid-book-recommender`.
2.  Click on the **Settings** tab at the top.
3.  In the left sidebar, click on **Secrets and variables** > **Actions**.
4.  Click the **New repository secret** button.
5.  Set the **Name** to: `GOOGLE_API_KEY`
6.  Paste your API key into the **Secret** field.
7.  Click **Add secret**.

#### **3. Why this is necessary**
Now that you've added the secret, GitHub Actions will automatically inject it into your Docker container during deployment. This enables:
-   **Conversation Reasoning**: The `pydantic-ai` agent will be able to process user messages.
-   **Tool Selection**: The agent can choose whether to call `search_books` or filter by tone.
-   **Contextual Synthesis**: The agent can explain *why* it's recommending specific books based on the user's mood.

> [!IMPORTANT]
> Once the secret is added, the next time the `deploy` job runs, your container on EC2 (or the simulation) will have full access to the Gemini model. You can trigger a new run by making a small change or clicking **Re-run all jobs** in your last successful Action.


### 7.2 Agent Model Configuration

The active model is configured via `config/params.yaml`:

```yaml
agent:
  model_name: gemini-flash-latest   # pydantic-ai resolves to google-gla:gemini-flash-latest
  temperature: 0.7
  max_results_per_search: 5
```

> [!NOTE]
> `temperature` is read from `params.yaml` but **not yet wired** into the `pydantic-ai` `GoogleModel` provider (pydantic-ai>=0.2 configures this via `ModelSettings`). This is tracked as a planned improvement in §9 of the [Agentic Layer Architecture Report](../architecture/agentic_layer.md).

### 7.3 Dependency Init & Resilience

`create_agent_dependencies()` uses a **lazy singleton** pattern — the `HybridRecommender` (ChromaDB + CSV) is initialized once on first request and reused across all chat turns. Two failure modes are handled gracefully:

1. **Init failure** (e.g., ChromaDB artifacts not built) → Chat tab shows a user-friendly message directing to the Search tab. Full traceback logged via `CustomException`.
2. **Agent execution failure** (e.g., API rate limit, malformed JSON from LLM) → `chat()` catches the exception and returns a safe fallback `AgentResponse`. Gradio never crashes.

### 7.4 Pre-Deployment Checklist

Before deploying a new image, verify the following:

- [ ] ChromaDB vector store is built (`dvc repro` pipeline completed).
- [ ] `data/processed/toned_books.csv` is present in the image or mounted as a volume.
- [ ] `GOOGLE_API_KEY` is set in the deployment environment.
- [ ] All 7 agent unit tests pass locally (`uv run pytest tests/unit/test_agent.py -v`).
- [ ] `uv run pyright src/` returns zero errors.
