## High-level architectural plan to build this **NYC Taxi Tip Prediction System**.

This is not a data science experiment; it is a software engineering project that happens to involve machine learning. We will move away from "notebook thinking" where state is hidden in memory, to "pipeline thinking" where state is explicitly managed in artifacts.

---

### 1. What exactly am I going to build?

We are building a **Batch Prediction System** orchestrated by DVC (Data Version Control). Unlike a monolithic script, this system is composed of three decoupled pipelines that communicate via file artifacts.

* **Pipeline A (The Feeder):** A robust ETL process that transforms raw NYC taxi logs into a "Feature Store" (parquet files in `data/processed`).
* **Pipeline B (The Brain):** A training workflow that consumes the Feature Store to produce a versioned XGBoost artifact, logged in a Model Registry.
* **Pipeline C (The Worker):** A batch inference job that wakes up, pulls the latest "Brain" and fresh data, calculates expected tips, and saves them for downstream analytics.
* **Pipeline D (The Application):** A Streamlit dashboard to visually present the model data, feature importance, and final predictions to end-users.

### 2. What is the expected result?

The final deliverable is a repository where a single command (e.g., `dvc repro`) triggers the entire lifecycle.

* **Quantifiable Output:** An XGBoost model optimized for MSE (Mean Squared Error), with a clear baseline comparison (MAE).
* **Business Insight:** A ranked list of "Feature Importance" explaining *why* a tip was high or low (e.g., "Trip duration impacts tips 3x more than distance").
* **Operational Output:** A table of predictions stored in `artifacts/predictions/` mimicking a data warehouse load.
* **Reproducibility:** A guarantee that anyone cloning the repo and running `uv sync` can reproduce your exact results, bit-for-bit.

### 3. What steps do I need to take to achieve this result?

We will execute this in four architectural phases, mapping to your `src/` folder structure.

#### **Phase 1: Environment & Orchestration Setup**

* Initialize the project using **UV** for strict dependency management (`pyproject.toml`).
* Configure **DVC** to track the `data/` and `models/` directories, ensuring large files are not committed to Git.
* Set up the `src/config/configuration.py` manager to read from `config.yaml`, centralizing paths and constants.

#### **Phase 2: The Feature Pipeline (`src/features/`)**

* **Ingestion:** Implement chunked reading (or use Polars) to handle the 5M rows without OOM (Out of Memory) errors.
* **Cleaning Strategy (As defined):**
* *Imputation:* Fill `Airport_fee`/`congestion_surcharge` with 0; `passenger_count` with 1; `RatecodeID` with 99.
* *Filtering:* Drop refunds (negative amounts). Enforce `0.5 < trip_distance < 100` and `total_amount < 1000`.


* **Transformation:** Convert `tpep_pickup_datetime` into cyclic features (Hour of Day, Day of Week) and calculated features (Trip Duration).
* **Output:** Save the result as highly compressed Parquet files in `data/processed`.

#### **Phase 3: The Training Pipeline (`src/models/train_model.py`)**

* **Temporal Splitting:** **Crucial Step.** Random splitting is forbidden here. We will split by time (e.g., Train: Jan-May, Test: June). This prevents **Look-ahead Bias** (training on future rides to predict past ones).
* **Baseline:** Calculate the Mean Absolute Error (MAE) of a "naive" model (predicting the average tip) to set the floor.
* **Training:** Train the **XGBoost Regressor**.
* **Tuning:** Use `params.yaml` to control hyperparameters (learning rate, max_depth) without changing code.
* **Registry:** Save the trained model binary (`model.pkl` or `.json`) to `models/` and log metrics (MSE, R2) to MLflow.

#### **Phase 4: The Inference Pipeline (`src/models/predict_model.py`)**

* Simulate a "batch run" by loading a holdout set of data.
* Load the trained artifact from Phase 3.
* Generate predictions and merge them with `VendorID` for business context.
* Save the final dataset to `artifacts/inference_results.csv`.

#### **Phase 5: The Interactive Dashboard (`app.py`)**

* **Visualization:** Build a Streamlit application to visually explore the model data and evaluation metrics.
* **Interactive Predictions:** Connect the dashboard to the MLflow Model Registry to serve and display the predictions in a user-friendly format, optimizing the MLOps system for production readiness.

---

### 4. What could go wrong along the way?

* **Data Leakage (The Silent Killer):** If we accidentally normalize features using statistics (mean/std) from the *entire* dataset instead of just the *training* set, the model will cheat. We must fit scalers only on training data.
* **Memory Exhaustion:** 5 million rows with 19 columns can spike memory usage. If Pandas fails, we may need to switch to **Polars** or process data in chunks.
* **Concept Drift:** Taxi tipping behavior changes over time (e.g., inflation, holidays). A model trained on January might fail in July. Our temporal split helps detect this, but in production, this requires monitoring.
* **Feature-Target Skew:** Since `tip_amount` cannot be negative and is likely right-skewed (many small tips, few huge ones), the model might struggle. We may need to predict the *log* of the tip amount or use a specific objective function in XGBoost.

### 5. What tools should I use to develop this project?

We will stick to a "Boring Technology" stack—reliable, industry-standard tools.

| Category | Tool | Why? |
| --- | --- | --- |
| **Package Manager** | **UV** | Extremely fast, replaces pip/poetry, and ensures deterministic environments. |
| **Orchestration** | **DVC (Data Version Control)** | Connects the stages (Feature -> Train -> Inference) and versions the large data files. |
| **Data Processing** | **Pandas** (or Polars) | The standard for tabular data manipulation. Polars is a backup if speed/memory becomes an issue. |
| **Modeling** | **XGBoost** | The state-of-the-art for tabular regression tasks. Fast and interpretable. |
| **Experiment Tracking** | **MLflow** | To log parameters, metrics, and artifacts. It allows us to compare "Experiment A" vs "Experiment B". |
| **Configuration** | **Hydra** or **PyYAML** | To decouple code from configuration (as seen in your `config.yaml`). |
