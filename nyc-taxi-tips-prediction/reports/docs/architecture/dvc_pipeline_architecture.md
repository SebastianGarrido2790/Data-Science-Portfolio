# DVC Pipeline Architecture Report

## 1. Overview
The **NYC Taxi Tips Prediction** pipeline is orchestrated by Data Version Control (DVC).
It defines a Directed Acyclic Graph (DAG) ensuring that:
1.  **Reproducibility**: Every artifact is tied to a specific Git commit and code version.
2.  **Efficiency**: Stages are re-run `ONLY` if their dependencies (code or data) change.
3.  **Traceability**: The lineage of every model is fully auditable.

## 2. Pipeline DAG
The pipeline consists of 6 primary stages:

```mermaid
flowchart TD
    subgraph Raw_Data [External Data]
        A[Raw Text Data]
        B[Taxi Zones CSV]
    end

    subgraph Stage_01 [1. Data Ingestion]
        A & B --> C((Run Ingestion))
        C --> D[Enriched Trip Data Parquet]
    end

    subgraph Stage_02 [2. Data Validation]
        D --> E((Run Validation))
        E --> F[Validation Status Report]
    end

    subgraph Stage_03 [3. Data Transformation]
        D --> G((Run Transformation))
        G --> H[Cleaned Trip Data Parquet]
    end

    subgraph Stage_04 [4. Feature Engineering]
        H --> I((Run Feature Eng))
        I --> J[Train Set]
        I --> K[Validation Set]
        I --> L[Test Set]
    end

    subgraph Stage_05 [5. Model Trainer]
        J & K --> M((Run Model Training))
        M --> N[Champion Model.joblib]
    end

    subgraph Stage_06 [6. Model Evaluation]
        L & N --> O((Run Model Evaluation))
        O --> P[Inference Results CSV]
        O --> Q[Metrics JSON]
    end

    style C fill:#f9f,stroke:#333
    style E fill:#f9f,stroke:#333
    style G fill:#f9f,stroke:#333
    style I fill:#f9f,stroke:#333
    style M fill:#f9f,stroke:#333
    style O fill:#f9f,stroke:#333
```

## 3. Detailed Stage Breakdown

### 3.1 Data Ingestion (`stage_01_data_ingestion`)
*   **Input**: Raw Distilled Trip Records (`.txt`), Taxi Zones (`.csv`).
*   **Logic**: Joins trip data with zone names, handles basic schema enforcement.
*   **Output**: `artifacts/data_ingestion/enriched_trip_data.parquet`.
*   **Dependency**: `src/components/data_ingestion.py`.

### 3.2 Data Validation (`stage_02_data_validation`)
*   **Input**: Enriched Data (`.parquet`).
*   **Logic**: Validates schema against `config/schema.yaml`. Checks for critical column existence and verifies the **Target Column** (`tip_amount`) is present for offline stages.
*   **Output**: `artifacts/data_validation/status.txt`.
*   **Dependency**: `src/components/data_validation.py`.
*   **Note**: This stage is a "Gatekeeper". If it fails, the pipeline halts.

### 3.3 Data Transformation (`stage_03_data_transformation`)
*   **Input**: Enriched Data (`.parquet`), `config/params.yaml`.
*   **Logic**:
    *   **Imputation**: Fills nulls in `airport_fee`, `congestion_surcharge`, `passenger_count`.
    *   **Filtering**: Prunes negative fares, invalid distances, and outliers based on thresholds defined dynamically in `params.yaml` (DataCleaning section).
*   **Output**: `artifacts/data_transformation/cleaned_trip_data.parquet`.
*   **Dependency**: `src/components/data_transformation.py`.

### 3.4 Feature Engineering (`stage_04_feature_engineering`)
*   **Input**: Cleaned Data (`.parquet`), `config/params.yaml`.
*   **Logic**:
    *   **Cyclical Encoding**: Transforms Hour/Day/Month into Sin/Cos pairs using the canonical `src/utils/feature_utils.py:encode_cyclical` function to prevent training-serving skew.
    *   **Temporal Splitting**: Splits data chronologically into Train/Val/Test sets based on boundaries defined in `params.yaml` (FeatureEngineering section).
*   **Outputs**:
    *   `artifacts/feature_engineering/train.parquet`
    *   `artifacts/feature_engineering/val.parquet`
    *   `artifacts/feature_engineering/test.parquet`
*   **Dependency**: `src/components/feature_engineering.py`, `src/utils/feature_utils.py`.

### 3.5 Model Trainer (`stage_05_model_trainer`)
*   **Input**: Training and Validation Set (`.parquet`), `config/schema.yaml`, `config/params.yaml`.
*   **Logic**:
    *   **Configurable Target**: Reads the target column name dynamically from `schema.yaml`, eliminating hardcoded logic.
    *   **Subsampling**: Fast local training toggle.
    *   **Model Benchmarking**: Trains multiple candidates (XGBoost, RandomForest, Ridge, etc.).
    *   **Selection**: Multi-metric weighted scoring on Validation Set.
*   **Outputs**:
    *   `artifacts/model_trainer/model.joblib` (Best Model)
*   **Tracking**: Integrated with **MLflow** for experiment tracking and model registration.
*   **Dependency**: `src/components/model_trainer.py`.

### 3.6 Model Evaluation (`stage_06_model_evaluation`)
*   **Input**: Test Set (`.parquet`), Champion Model (`.joblib`), `config/config.yaml`, `config/schema.yaml`.
*   **Logic**:
    *   **Evaluation**: Calculates MAE, MSE, and R² against the hold-out test set (dynamically targeting the column from `schema.yaml`).
    *   **Inference Simulation**: Uses a dedicated `PredictModelConfig` to ingest incoming trip records and persist predictions.
*   **Outputs**:
    *   `artifacts/predictions/inference_results.csv`
    *   `artifacts/model_evaluation/metrics.json`
*   **Tracking**: Test metrics logged directly to the MLflow experiment.
*   **Dependency**: `src/components/model_evaluation.py`, `src/components/predict_model.py`.

## 4. Execution
To reproduce the entire pipeline (or only what changed):
```bash
uv run dvc repro
```

To visualize the DAG in terminal:
```bash
uv run dvc dag
```
