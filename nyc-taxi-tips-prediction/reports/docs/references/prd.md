## Product Requirements Document (PRD)

**Project Name:** NYC Yellow Taxi Tip Prediction System
**Version:** 1.0
**Status:** Draft
**Date:** 2026-02-14
**Author:** MLOps practitioner

---

### 1. Executive Summary

The goal of this project is to build a production-grade Machine Learning system that predicts tip amounts for NYC Yellow Taxi rides. Unlike standard "hello world" projects, this system handles a real-world dataset of ~5 million rows using the FTI (Feature, Training, Inference) pattern. The system will decouple data engineering from model development to ensure scalability, reproducibility, and maintainability.

### 2. Problem Statement

Taxi drivers and fleet operators lack actionable insights into the factors that drive higher tips. While ride volume is high, the variability in tipping behavior (influenced by distance, time of day, passenger count, etc.) makes revenue prediction difficult.

**Business Question:** "What specific factors drive people to tip more?"

### 3. Objectives & Success Metrics

* **Primary Objective:** Develop an XGBoost regression model to predict `tip_amount` with high accuracy.
* **Secondary Objective:** Implement a robust MLOps pipeline using DVC and UV, culminating in a Streamlit dashboard to visualize model data and predictions interactively.
* **Key Performance Indicators (KPIs):**
    * **Model Accuracy:** RMSE (Root Mean Squared Error) lower than the baseline MAE (Mean Absolute Error).
    * **Pipeline Latency:** End-to-end execution (Ingestion to Inference) optimized for batch processing.
    * **Interpretability:** Deliver a "Feature Importance" report ranking the drivers of tipping behavior.

---

### 4. Functional Requirements

#### 4.1 Feature Pipeline (Data Engineering)

* **Input:** Raw NYC Yellow Taxi data (CSV/Parquet) from the `data/raw/` directory.
* **Processing Logic:**
* **Ingestion:** Efficiently load the ~5M row dataset handling memory constraints.
    * **Cleaning Rules**:
        * **Imputation:**
            * `Airport_fee` & `congestion_surcharge`: Fill `NaN` with `0`.
            * `passenger_count`: Fill `NaN` with `1`.
            * `RatecodeID`: Fill `NaN` with `99` (Unknown).
        * **Drop:** Remove the `store_and_fwd_flag` column.
        * **Filtering:**
            * Drop rows with negative amounts (refunds).
            * Keep `trip_distance` between `0.5` and `100` miles.
            * Cap `total_amount` at `$1,000` to remove outliers.
    * **Feature Engineering:** Convert `tpep_pickup_datetime` and `tpep_dropoff_datetime` into numerical features (Month, Day, Hour, Duration).
* **Output:** Saved processed features in `data/processed/` (e.g., Parquet format) and registered in the Feature Store.

#### 4.2 Training Pipeline (Model Development)

* **Input:** Processed data from the Feature Store.
* **Splitting Strategy:** Implement **Temporal Splitting** (e.g., Train on Jan-May, Test on June) to strictly prevent look-ahead bias and data leakage.
* **Modeling:**
    * **Algorithm:** XGBoost Regressor.
    * **Baseline:** Compute Mean Absolute Error (MAE) of a naive model (mean prediction) to establish a floor.
    * **Optimization:** Hyperparameter tuning to minimize Mean Squared Error (MSE).
* **Output:**
    * Trained model artifact saved to `models/` (Model Registry).
    * Evaluation metrics (MSE, R2) logged.
    * Feature Importance plot saved to `reports/figures/`.

#### 4.3 Inference Pipeline (Model Serving)
* **Input:**
    * New batch of raw data (e.g., "last hour" simulation).
    * Latest trained model artifact from the Model Registry.
* **Process:**
    * Load model and data.
    * Generate `tip_amount` predictions.
* **Output:** Save predictions to `artifacts/predictions/` (simulating a Data Warehouse load) for downstream analysis.

#### 4.4 Visualization Pipeline (Dashboard)
* **Input:** Evaluation metrics, feature importance data, and generated predictions.
* **Process:** Serve an interactive Streamlit application to visualize model data and prediction results, bridging the gap between the MLOps backend and business users.
* **Output:** A user-friendly web interface accessible to stakeholders.

---

### 5. Technical Architecture & Stack

#### 5.1 Technology Stack

* **Language:** Python 3.10+
* **Package Management:** **UV** (for deterministic dependency resolution).
* **Orchestration:** **DVC** (Data Version Control) for pipeline management and artifact tracking.
* **Data Processing:** **Pandas** (primary) or **Polars** (fallback for memory efficiency).
* **Modeling:** **XGBoost**.
* **Configuration:** `config.yaml` and `params.yaml` managed via `src/config/configuration.py`.

#### 5.2 System Design Pattern

The system adheres to the **FTI Pattern**:

1. **Feature Pipeline:** `src/features/build_features.py` -> Transforms Raw to Processed.
2. **Training Pipeline:** `src/models/train_model.py` -> Consumes Processed, produces Artifacts.
3. **Inference Pipeline:** `src/models/predict_model.py` -> Consumes Artifacts & New Data, produces Predictions.

---

### 6. Non-Functional Requirements

* **Reproducibility:** A user must be able to clone the repo, run `uv sync`, and execute `dvc repro` to regenerate the exact same model and results.
* **Scalability:** The data ingestion step must handle 5M+ rows without crashing (OOM).
* **Maintainability:** Code must be modular (`src/components`, `src/pipeline`) and documented with "Why, not What" comments.
* **Folder Structure:** strict adherence to the provided `folder_structure.md` layout.

---

### 7. Risk Management

* **Memory Constraints:** High risk with 5M rows. *Mitigation:* Use chunking or Polars if Pandas fails.
* **Data Drift:** Temporal nature of taxi rides suggests patterns change. *Mitigation:* Strict temporal splitting during validation.
* **Skewed Target:** Tips are non-negative and right-skewed. *Mitigation:* Log-transformation of the target variable may be required if model performance stalls.

---

### 8. Implementation Roadmap

1. **Phase 1: Setup:** Initialize Git, UV, DVC, and directory structure.
2. **Phase 2: Data Engineering:** Implement `build_features.py` with cleaning rules.
3. **Phase 3: Model Training:** Implement `train_model.py` with XGBoost and Temporal Split.
4. **Phase 4: Inference & Orchestration:** Implement `predict_model.py` and wire everything with `dvc.yaml`.
5. **Phase 5: Visualization:** Implement a Streamlit dashboard (`app.py`) to visualize the model data and predictions natively.
