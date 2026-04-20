<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white" alt="Python">
   <img src="https://img.shields.io/badge/managed_by-uv-purple?logo=python" alt="uv">
   <img src="https://img.shields.io/badge/DVC-Data_Version_Control-blueviolet?logo=dvc" alt="DVC">
   <img src="https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2?logo=mlflow&logoColor=white" alt="MLflow">
   <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
   <img src="https://img.shields.io/badge/LangGraph-Agentic_Orchestration-orange" alt="LangGraph">
   <img src="https://img.shields.io/badge/Google_Gemini-2.5_Flash-4285F4?logo=google&logoColor=white" alt="Gemini">
   <img src="https://img.shields.io/badge/FastAPI-Microservice-009688?logo=fastapi&logoColor=white" alt="FastAPI">
   <img src="https://img.shields.io/badge/Docker-Containerization-2496ED?logo=docker&logoColor=white" alt="Docker">
   <img src="https://img.shields.io/badge/GitHub_Actions-CI/CD-2088FF?logo=github-actions&logoColor=white" alt="GitHub Actions">

  <h1>🚕 NYC Taxi Tip Prediction System 🚕</h1>
  <p><strong>A Production-Grade Agentic MLOps System implementing the strict FTI (Feature, Training, Inference) Pattern</strong></p>
</div>

---

## 📖 Executive Summary
The **NYC Taxi Tip Prediction System** is a robust Machine Learning solution designed to predict tip amounts for NYC Yellow Taxi rides. By systematically analyzing over **5 million recorded trips**, the system identifies the key drivers of tipping behavior, translating raw data into actionable revenue optimization insights for Fleet Managers.

Moving away from exploratory "notebook thinking," this project embodies the **Agentic Data Scientist** mindset by strictly decoupling data engineering from model development using the modular **FTI (Feature, Training, Inference)** architecture.

---

## 🏗️ Architecture: The FTI Pattern
To ensure massive scalability, strict reproducibility, and readiness for batch production workloads, the system is architected into three distinct, independently operational pipelines:

<div align="center">
  <img src="reports/figures/fti_𝗽attern.png" alt="FTI Pipeline Architecture" width="700">
</div>
<br>

1. 🔵 **Feature Pipeline (Data Engineering):** Transforms raw, messy taxi logs into a pristine Feature Store. It manages missing values, prunes anomalies (e.g., negative fares, impossible distances), and intelligently engineers cyclical temporal features (Hour/Day/Month) via chunked or native Polars processing to securely handle massive datasets out-of-core.
2. 🟠 **Training Pipeline (Model Development):** Operates on the engineered features to evaluate and identify the best predictive model. It actively utilizes **Temporal Splitting** (preventing disastrous look-ahead bias), benchmarks multiple algorithms simultaneously (XGBoost, Random Forest, Ridge, etc.), and cleanly registers the ultimate Champion Model using **MLflow**.
3. 🟢 **Inference Pipeline (Model Serving):** A standalone batch prediction engine that applies the frozen Champion Model to fresh, unseen holdout data. Deployed natively as a **FastAPI** microservice, decoupling compute from rendering.
4. 🌐 **Agentic Orchestration (Brain vs. Brawn):**
    *   **The Brain (`src/agents/`):** A probabilistic reasoning layer powered by **LangGraph** and **Gemini-2.5-flash**. It parses natural language, manages conversation state, and orchestrates the ReAct pattern to decide when to invoke tools.
    *   **The Brawn (`src/tools/`):** A deterministic execution layer that provides a hardened Python interface to the **FastAPI** model serving API, ensuring strict Pydantic validation and "Agentic Healing" for reliable ML inferences.

---

## 🔀 Pipeline Orchestration (DVC)

The entire MLOps lifecycle is governed seamlessly by **Data Version Control (DVC)** as a Directed Acyclic Graph (DAG). Data artifacts and intermediate states are heavily versioned without bloating Git, meaning the predictive process is 100% auditable and reproducible.

*   `stage_01_data_ingestion`: Joins massive trip data parquets with spatial lookup tables.
*   `stage_02_data_validation`: Checks structural data quality (schema enforcement).
*   `stage_03_data_transformation`: High-performance imputation and filtering heuristics.
*   `stage_04_feature_engineering`: Temporal splitting and cyclical feature extraction converting times to `sin`/`cos` vectors across `train`, `val`, and `test` clusters.
*   `stage_05_model_trainer`: Model searching, multi-metric candidate ranking, and Joblib Champion extraction.
*   `stage_06_model_evaluation`: Scoring the actualized champion model against the test set and logging all telemetry insights.

---

## ️ Interactive Dashboard

Bridging the gap between the complex backend and business stakeholders, a native **Streamlit Interactive Application** serves insights directly, consuming pre-computed artifacts.

<div align="center">
  <h3>📊 Dashboard & Evaluation</h3>
  <p>Dedicated to high-level analytics, model transparency, and offline batch prediction reviews.</p>
  <img src="reports/figures/nyc_tips_dashboard_evaluation_01.png" alt="Dashboard Evaluation - Model Metrics" width="800">
  <br><br>
  <img src="reports/figures/nyc_tips_dashboard_evaluation_02.png" alt="Dashboard Evaluation - Feature Importance" width="800">
  <br><br>
  <img src="reports/figures/nyc_tips_dashboard_evaluation_03.png" alt="Dashboard Evaluation - Inferences Ledger" width="800">
  <br><br>
  <h3>🤖 Agentic Taxi Analyst</h3>
  <p>Chat naturally with the Agentic Analyst — describe your ride and get an ML-powered tip prediction, or ask any NYC taxi question. Powered by the ReAct pattern and Gemini-2.5-flash.</p>
  <img src="reports/figures/nyc_tips_interactive_prediction.png" alt="Agentic Chat UI - Natural Language Prediction" width="800">
</div>
<br>

* **Performance Validation:** Easily review Test MAE, Test MSE, R², and the internal custom weights utilized during automated model selection.
* **Global Explanation:** Explore global visual Feature Importance charts dynamically extracted directly from the champion model to identify what precisely drives high tipping behavior.
* **Live Inferences:** Users can manage rides natively in a rich interactive table (`st.data_editor`), insert new simulated samples, and seamlessly run them through the loaded model artifact for instant real-time inference forecasting.

---

## 🧪 Testing & Multi-Point Validation

This system employs a rigorous multi-layered validation strategy to ensure it remains production-ready and free of regressions.

### 🔬 Automated Testing Suite
The codebase relies on `pytest` for deterministic logic verification:
*   **Component Logic**: 100% coverage of critical cleaning rules, temporal split strategies, and feature engineering math.
*   **Fast Execution**: Replaces massive physical disk reads with instantaneous in-memory `Polars` test fixtures.
*   **Edge-Case Traps**: Extensive evaluation of boundaries (e.g., zero distance handling, negative fare prevention, agent tool hallucinations).

### ✅ Production Quality Gates
Beyond standard testing, we enforce a **Multi-Point System Validation** script (`validate_system.bat`) that must pass before any production deployment. It evaluates the four pillars of system health:

1.  **Pillar 1: Static Code Quality**: Enforces strict `pyright` type checking and `ruff` linting/formatting.
2.  **Pillar 2: Functional Logic & Coverage**: Executes the full `pytest` suite and enforces a **65% mandatory coverage gate**.
3.  **Pillar 3: Pipeline Synchronization**: Verifies that the `DVC` DAG and all data artifacts are perfectly synchronized.
4.  **Pillar 4: API Service Health**: Performs a runtime diagnostic against the versioned `/v1/health` endpoint.

---

## 📊 Project Analysis & Reports

The project maintains a comprehensive suite of generated documentation and analytical reports acting as the "Master Source of Truth."

*   **[Codebase Review & Production Readiness (v1.0)](reports/docs/evaluations/codebase_review_v1.0.md)**: A deep-dive audit assessing the system's alignment with elite MLOps standards, identifying strengths (FTI pattern, Brain vs. Brawn) and hardening requirements.
*   **[System Validation Report](reports/docs/evaluations/system_validation_report.md)**: The latest verified pass status of all production quality gates.
*   **[Test Suite Report](reports/docs/evaluations/test_suite_report.md)**: Detailed breakdown of current test coverage (67.82%) and logical verification steps.

Detailed architecture diagrams, decision logs, and runbooks can be found in the `reports/docs/` directory.

---

## 🛠️ Technology Stack

| Category | Tool | Justification |
| :--- | :--- | :--- |
| **Package Manager** | **uv** | Blazing fast deterministic dependency resolution in purely clean Python environments. |
| **Orchestration** | **DVC** | Connects DAG pipeline stages and versions massive immutable parquet artifacts securely. |
| **Data Engine** | **Polars / Pandas** | High-performance feature manipulation and mutation avoiding standard OOM complications. |
| **Experiment Tracking**| **MLflow** | Unobtrusive offline hyperparameter logging and local Model Registry governance. |
| **Testing** | **Pytest** | Component-level behavior and logic evaluations. |
| **Agentic Framework** | **LangGraph** | Orchestrates stateful, multi-turn reasoning and the ReAct pattern for tool-use. |
| **Orchestrator Model**| **Gemini-2.5-flash** | Low-latency, high-reliability model for natural language parsing and parameter extraction. |
| **Microservice** | **FastAPI** | High-performance, asynchronous REST API serving machine learning inferences reliably. |
| **Containerization**| **Docker** | Encapsulates the frontend and backend services into isolated, reproducible execution environments. |
| **CI/CD** | **GitHub Actions**| Automates deterministic unit testing and Agentic code quality (Ruff) gating on every push. |

---

## � Setup & Execution

### 1. Installation Environment
This project rigorously depends on modern `uv` dependency management for lightning-fast setup.
```bash
# Clone the repository
git clone https://github.com/SebastianGarrido2790/nyc-taxi-tips-prediction.git

cd nyc-taxi-tips-prediction

# Sync packages and dependencies entirely using uv
uv sync --all-extras
```

### 2. DVC Pipeline Execution
Trigger the reproduction of the entire DAG. DVC automatically detects what code or data components have changed and runs only the minimum required compute:
```bash
# Initialize and pull remote data assets (if remote storage is configured)
uv run dvc pull

# Execute end-to-end Pipeline Lifecycle seamlessly
uv run dvc repro
```

### 3. System Validation
Before launching the services, verify the system's health across all 4 pillars. This is mandatory for production-grade assurance:
```bash
.\validate_system.bat
```

### 4. Environment Configuration
Create a `.env` file in the root directory to define the MLflow tracking URI and API configurations. Example `.env.example`:
```env
# Define the environment (local/staging/production)
ENV=local

# FTI ARCHITECTURE PATHS
# Frontend API reference. For local development, use localhost.
# In Docker, `docker-compose.yml` automatically overrides this to `http://backend:8000`.
API_URL=http://localhost:8000

# AGENTIC LAYER — LangGraph + Google Gemini
# Required to power the "Agentic Taxi Analyst" chat UI on Page 2.
# Get a free key at: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=AIza...

# MLFLOW TRACKING CONFIGURATION
# Set to remote host URL, network-based tracking (http://127.0.0.1:5000), or local file-based:
MLFLOW_TRACKING_URI=file:./mlruns
```

### 5. Launching the App (Local Dev)
Interact natively with the trained artifacts using the Streamlit visual dashboard.

**Option A - Automated Launch (Windows)**:
You can rapidly spin up both the FastAPI backend and Streamlit frontend using the provided batch script:
```cmd
.\launch_app.bat
```

**Option B - Manual Launch**:
Run the backend and frontend simultaneously in separate terminals:
```bash
# Start the FastAPI backend
uv run uvicorn src.api.predict_api:app --reload --port 8000

# Start the Streamlit frontend (in a new terminal)
uv run streamlit run src/app/main.py
```

### 6. Production Deployment (Docker Compose)
Launch the entire system inside isolated, network-meshed containers:
```bash
# Build and spin up the services
docker-compose up --build -d

# The Streamlit dashboard is now live at http://localhost:8501
# The FastAPI Swagger docs are live at http://localhost:8000/docs
```

<br>
<div align="center">
  <i>Developed applying the <b>MLOps Stack Principles</b> for Professional Agentic Data Science</i>
</div>
