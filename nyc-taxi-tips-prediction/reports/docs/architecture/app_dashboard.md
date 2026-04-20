# Phase 5: NYC Taxi Tips Predictor - App Architecture & Functionality

## 1. Executive Summary

The `src/app/` modular package serves as the interactive frontend (Phase 5) of the NYC Taxi Tip Prediction System. Operating strictly as a UI layer, it bridges the complex machine learning pipelines (Feature, Training, Inference - FTI) and business stakeholders (like Fleet Managers). The app allows users to visually explore the model's performance, understand what drives tip amounts, and simulate real-time rides, delegating all heavy machine learning computations to a robust FastAPI microservice backend.

## 2. Architecture & The FTI Pattern Integration

Following strict **Agentic MLOps Rules**, the system perfectly executes the **FTI (Feature, Training, Inference)** architecture by decoupling the Model Serving pipeline completely from the frontend UI interface.

### The FastAPI Serving Layer (`src/api/predict_api.py`)
Instead of running prediction computations on the fly in the frontend, the Streamlit app acts as a client to an independent FastAPI microservice.
*   **API Versioning:** All endpoints are strictly versioned under the `/v1/` prefix (e.g., `/v1/predict`, `/v1/health`) to allow for future structural iterations without breaking existing client consumption modes.
*   **Model Registry & Lifespan:** The FastAPI server utilizes `lifespan` event handlers to securely load the trained champion model (`model.joblib`) from `artifacts/model_trainer/` into a global memory dictionary precisely once upon startup. This guarantees rapid, sub-millisecond inference times on subsequent API calls without costly repetitive I/O operations.
*   **Strict Pydantic Contracts:** All communication between the Streamlit UI and the FastAPI backend is validated via strict Pydantic schemas (`PredictRequest`, `PredictResponse` in `src/entity/api_entity.py`), enforcing rigid type safety before any code touches the ML model.
*   **Training-Serving Parity:** Incoming requests explicitely pass through the identical `src/utils/feature_utils.py:encode_cyclical` function used during the offline DVC feature engineering phase, structurally preventing the model from predicting on skewed temporal properties.
*   **Custom Exception Handling:** The backend raises standard HTTP exceptions combined with comprehensive exception chaining (`raise from e`) when the model registry fails, preventing silent failures and displaying clean stack traces.
*   **Modular Layout:** The frontend logic is no longer a monolith. It is structured under `src/app/` following **Clean Architecture** principles:
    *   `main.py`: The entry orchestrator and navigation system.
    *   `styles.py`: Centralized CSS injection and aesthetic theme management.
    *   `data_loaders.py`: Optimized I/O handlers for artifact consumption.
    *   `pages/`: Individual specialized modules for the Dashboard and AI Chat Analyst.

### Artifact Consumption
To ensure the dashboard loads quickly and interactively, it utilizes Streamlit's caching mechanisms (`@st.cache_data`) to parse pre-computed outputs from the orchestrated DVC pipelines:
*   **Model Evaluation Metrics:** Loads `artifacts/model_evaluation/metrics.json` to extract performance markers like Test MAE, MSE, and R².
*   **Batch Predictions:** Loads `artifacts/predictions/inference_results.csv` to display the offline batch inference outputs.
*   **Training Parameters:** Loads `config/params.yaml` to extract the metric weights used during automated champion selection.

## 3. Orchestrator Launchpad (`launch_app.bat`)
To ensure smooth boots of this decoupled architecture, the system relies on a central windows batch orchestrator (`launch_app.bat`), or standard `make` commands (`make serve` & `uv run streamlit run src/app/main.py`).
Mimicking enterprise Agentic System tooling, this script guarantees reliable startups via sequential checks:
1.  **Dependency Synchronization:** Runs `uv sync` to ensure local environments match the `pyproject.toml` lockfile exactly.
2.  **API Background Execution:** Spins up the Uvicorn/FastAPI server (`src.api.predict_api:app --reload`) directly executing logic independently maintaining visual cleanliness.
3.  **Frontend Execution:** Safely sleeps for a warmup period before starting the Streamlit interface on `localhost:8501`.

## 4. User Interface & Core Functionality

The dashboard features a persistent sidebar for navigation and is divided into two primary pages.

### Page 1: Dashboard & Evaluation
Dedicated to high-level analytics, model transparency, and offline batch prediction reviews.

*   **Test MAE (Mean Absolute Error):** Represents the average absolute difference between the predicted tip and the actual tip in dollars. A lower value indicates higher precision and reliability in the model's dollar-amount predictions.
*   **Test R² (Variance Explained):** Indicates the proportion of the variance in the tip amount that is predictable from the trip characteristics. A value closer to 1.0 (or > 0.90) signifies an excellently performing model covering edge cases effectively.
*   **Test MSE (Mean Squared Error):** Represents the average of the squares of the errors. It heavily penalizes larger errors, ensuring the model deals effectively with anomalies and outliers, minimizing extreme deviations.
*   **Champion Selection Weights:** An informative banner displaying the exact metric weights (e.g., MAE, MSE, R2) used by the automated backend pipeline to evaluate, rank, and select this specific champion model from all candidate models.

*   **Model Version Context:** The UI continuously pings the FastAPI `/v1/health` endpoint to confirm server availability and dynamically extract and display the name of the active ML model loaded in the registry (e.g., *serving (Ridge model) predictions*).
*   **Dynamic Sliders & Interactive Sampling:** To maintain high performance without crashing the browser, users are equipped with interactive sliders mapped to dataframe sampling logic:
    *   **Feature Importance Chart:** Users dial the slider to configure the exact number of top drivers they wish to investigate (e.g., Top 3 to Top 20 features). This visual requests data via a dedicated `/v1/feature-importance` FastAPI endpoint.
    *   **Distribution Histogram:** Users can adjust the sample size (up to 10,000) of the latest batch inferences rendered in the distribution plot.
    *   **Inferences Ledger:** A slider adjusts the random sample generated from the offline dataset for granular visual review. The ledger seamlessly maps Vendor IDs 1 and 2 to human-readable strings ("Creative Mobile Technologies", "VeriFone Inc.").
*   **Test Metrics & Weights:** Read-only metric cards visually report Test MAE, MSE, and R², alongside the specific weight distribution that configured the algorithm.

### Page 2: Interactive Prediction
This page has been upgraded from a static form to an **Agentic Natural Language Chat UI**, transforming the interaction from manual data entry to a sophisticated conversational experience.

*   **The Agentic Taxi Analyst (The Brain):**
    *   **Orchestration Engine:** Built using **LangGraph** to manage the "Reason-Act" (ReAct) loop, allowing the system to think, call tools, and respond dynamically.
    *   **LLM Core:** Powered by **Google `gemini-2.5-flash`**, providing high-speed natural language understanding and structured intent extraction.
    *   **Contextual Memory:** The agent maintains a persistent message history, enabling it to remember trip details across multiple turns and respond to follow-up prompts without requesting redundant information.

*   **Intelligent Interaction Logic:**
    *   **Frictionless Prediction:** Adhering to "Fast-Action UX" principles, the analyst only requires two critical fields to proceed: **Trip Distance** and **Total Fare** (excluding tip).
    *   **Sensible Defaults:** To ensure immediate value, the agent automatically fills non-essential parameters (e.g., passenger count, rate code, time of day) with sensible defaults if they are not provided by the user.
    *   **Refinement Opportunities:** After providing a prediction based on assumptions, the agent transparently informs the user of the defaults used and proactively lists optional features (like Airport Fees or Congestion Surcharges) that can be supplied for a more surgical prediction.

*   **Hardened Tool Integration (The Brawn):**
    *   **The `predict_taxi_tip` Tool:** The agent delegates the actual machine learning inference to a deterministic Python tool. This tool enforces strict **Pydantic validation** on extracted parameters before they ever reach the ML model.
    *   **Agentic Healing:** If the FastAPI backend is offline or returns an error, the tool provides rich, descriptive error context. The "Brain" interprets these errors to provide helpful troubleshooting advice to the user rather than raw code tracebacks.

*   **Modern Chat Experience:**
    *   **Responsive UI:** Leverages Streamlit's native `st.chat_message` and `st.chat_input` components for a premium, mobile-friendly interface.
    *   **Rich Formatting:** Automatically handles currency escaping (e.g., `\$`) and markdown styling to ensure predictions and explanations are visually clear and professional.

## 5. Design & Aesthetics

Following strict guidelines for high-quality, professional tooling, the UI employs **custom CSS** to inject a modern, dark-mode aesthetic.

*   Uses a tailored `#0E1117` base with `#FAFAFA` text for high contrast.
*   Incorporates subtle accent colors like gold (`#FFD700`) for headers and buttons, alongside green (`#00FF7F`) for positive metrics, bypassing Streamlit's default generic look to deliver a premium user experience.
