# Test Suite Report (Data Pipeline)

## 1. Overview
The testing strategy for the NYC Taxi Tips Prediction pipeline is built on the principles set for this project: Test logic, not libraries.
We employ a robust unit testing suite using `pytest` to verify the deterministic behavior of our data cleaning and feature engineering components.

## 2. Test Suite Structure

The testing directory mirrors the source code structure for intuitive navigation:

```
tests/
├── conftest.py          # Global Shared Fixtures (Sample Data)
└── unit/                # Component-level Logic Tests
    ├── test_data_ingestion.py
    ├── test_data_transformation.py
    ├── test_feature_engineering.py
    ├── test_model_evaluation.py
    ├── test_model_trainer.py
    ├── test_predict_model.py
    ├── test_api.py
    └── test_taxi_analyst_agent.py
```

## 3. Testing Architecture

**NOTE**: We won't test Data Validation (`src/components/data_validation.py`) logic. Why? It is mostly a wrapper around standard Polars functionality.

    - The Logic: It checks if columns exist (if col in df.columns).
    - The Risk: Low. The risk is mostly in configuring the schema.yaml incorrectly, not in the Python code that reads it.
    - Recommendation: This is better covered by an integration test (or just running the pipeline) rather than a unit test, as you'd mostly be testing that polars.columns works, which we know it does.

### 3.1 Fixtures (`tests/conftest.py`)
To ensure tests are fast and isolated, we use `polars` DataFrame fixtures instead of reading from disk.

*   `sample_trip_data`: A small DataFrame containing specific "trap" rows designed to trigger cleaning rules (e.g., negative fares, zero distance, null fees).
*   `cleaned_trip_data`: A DataFrame spanning all 12 months of 2023 to verify temporal splitting logic.

### 3.2 Unit Tests (`tests/unit/`)

#### 3.2.1 Data Ingestion Tests (`tests/unit/test_data_ingestion.py`)
This suite validates the joining logic that enriches trip data with taxi zones.

| Test Case | Description | Goal |
| :--- | :--- | :--- |
| `test_merge_taxi_zones` | Mock trip data joined with mock zone data. | Ensure location enrichment works correctly (Left Join). |

#### 3.2.2 Data Transformation Tests (`tests/unit/test_data_transformation.py`)
This suite validates the "Pruning" system.

| Test Case | Description | Goal |
| :--- | :--- | :--- |
| `test_initial_imputation` | Checks if `airport_fee`, `congestion_surcharge` are filled with 0, and `passenger_count` defaults to 1. | Verify data completeness. |
| `test_colum_dropping` | Checks if `store_and_fwd_flag` is removed. | Verify schema reduction. |
| `test_filtering_refunds` | Checks if rows with negative `total_amount` are dropped. | Prevent "Refund Layout" (garbage in). |
| `test_filtering_outliers_distance` | Checks if trips < 0.5 miles or > 100 miles are removed. | Remove physical impossibilities. |
| `test_filtering_outliers_amount` | Checks if fares > $1000 are removed. | Remove data entry errors. |
| `test_data_consistency` | Verifies the output is a valid Polars DataFrame and rows were actually dropped. | Sanity check. |

#### 3.2.3 Feature Engineering Tests (`tests/unit/test_feature_engineering.py`)
This suite validates the "Brain" preparation logic.

| Test Case | Description | Goal |
| :--- | :--- | :--- |
| `test_feature_engineering_transforms` | Checks if Sine/Cosine features are created for Hour, Day, Month and are within range [-1, 1]. | Verify cyclical encoding math. |
| `test_temporal_splitting` | Injects 12-month data and verifies correct assignment to Train (Jan-Aug), Val (Sept-Oct), Test (Nov-Dec). | **CRITICAL**: Prevent Data Leakage. |
| `test_feature_columns_consistency` | Checks if allexpected feature columns exist in the output. | ensure downstream model has all inputs. |

#### 3.2.4 Model Trainer Tests (`tests/unit/test_model_trainer.py`)
This suite validates the model training, prediction, and evaluation logic.

| Test Case | Description | Goal |
| :--- | :--- | :--- |
| `test_model_training` | Verifies that a model can be trained without errors and returns a valid model object. | Ensure training process is stable. |
| `test_model_prediction` | Checks if the trained model can make predictions on new data and outputs valid tip percentages. | Validate prediction functionality. |
| `test_model_evaluation` | Ensures evaluation metrics (e.g., RMSE) are calculated correctly and within expected bounds. | Verify model performance assessment. |
| `test_model_persistence` | Tests if the model can be saved and loaded correctly, maintaining its state and prediction capability. | Ensure model deployability. |

#### 3.2.5 Model Evaluation Tests (`tests/unit/test_model_evaluation.py`)
This suite validates the test-set evaluation and MLflow logging logic on unseen data.

| Test Case | Description | Goal |
| :--- | :--- | :--- |
| `test_model_evaluation_initialization` | Verifies correct property extraction from the `ModelEvaluationConfig`. | Ensure correct file paths and MLflow URIs. |
| `test_evaluate_workflow` | Mocks the MLflow API, Joblib, and Pandas to assert that predictions correctly filter out `tip_amount`, produce valid shapes, calculate metrics, save local JSON files, and log to MLflow successfully. | Isolate the evaluation logic from the physical IO boundary to cleanly guarantee workflow adherence. |

#### 3.2.6 Predict Model Tests (`tests/unit/test_predict_model.py`)
This suite validates the batch inference behavior mirroring production systems.

| Test Case | Description | Goal |
| :--- | :--- | :--- |
| `test_predict_model_initialization` | Checks configuration bindings. | Validate object state. |
| `test_predict_model_inference_with_target` | Ingests data that unintentionally contains `tip_amount` and confirms it is dropped before inference. Validates `VendorID` passes through properly. | Check robustness against unexpected evaluation metadata. |
| `test_predict_model_inference_production_data` | Simulates a pure production string without `tip_amount` or `VendorID`, asserting predictions successfully output to CSV. | Ensure proper functioning of the batch prediction component in real-world FTI environment. |

#### 3.2.7 API Tests (`tests/unit/test_api.py`)
This suite validates the FastAPI Serving Layer (Inference API).

| Test Case | Description | Goal |
| :--- | :--- | :--- |
| `test_health_check_endpoint` | Probes `/v1/health` to verify `200 OK` and active status. | Ensure reliable automated uptime monitoring. |
| `test_predict_endpoint_validation` | Sends empty arrays and incomplete Pydantic models to `/v1/predict`. | Verify system forcefully truncates malformed requests with `422 Unprocessable Entity` before models calculate on garbage data. |
| `test_predict_feature_importance_no_model` | Exercises the `/v1/feature-importance` endpoint directly to observe its resilience when lifespans fail/skip. | Ensure system survives unhandled model load crashes. |
| `test_predict_endpoint_success` | Injects valid ride characteristics payload over HTTP into FastAPI handler. | Validate 100% End-to-End serialization from HTTP -> JSON -> Pandas -> ML -> JSON. |

#### 3.2.8 Agent Tools Tests (`tests/unit/test_taxi_analyst_agent.py`)
This suite validates the strict deterministic boundaries of the new `TaxiPredictionTool` abstraction to prevent silent LLM failures.

| Test Case | Description | Goal |
| :--- | :--- | :--- |
| `test_pydantic_schema_validation_success` | Validates that acceptable tool arguments execute successfully. | Ensure strict structural adherence to API contracts. |
| `test_pydantic_schema_validation_failure` | Injects LLM hallucinations (e.g., negative distances, invalid hours). | Force aggressive failure via Pydantic `ValidationError` *before* hitting network layer. |
| `test_tool_predict_success` | Mocks a positive HTTP response from the backend. | Verify data serialization and retrieval works flawlessly for Agents. |
| `test_tool_predict_timeout` | Simulates a backend that hangs infinitely. | Ensure the tool deliberately throws a domain-specific `PredictionToolError` instead of crashing. |
| `test_tool_predict_http_error` | Simulates HTTP 500 errors from the backend. | Intercept standard library errors and inject domain metadata for Agentic healing logic. |
| `test_tool_predict_empty_list` | Passes an empty list to the predict method. | Force fast-fail logic to save compute resources. |

## 4. Execution
To run the full suite:
```bash
uv run pytest tests/
```

## 5. Test Coverage
*   **Target Coverage**: 65% (Mandatory Gate)
*   **Current Coverage**: **73.64%**
*   **Exclusions**: UI logic in `src/app/` is excluded from the coverage gate to focus on the pipeline's FTI (Feature, Training, Inference) integrity.
*   **Components Covered**: `DataIngestion`, `DataTransformation`, `FeatureEngineering`, `ModelTrainer`, `PredictAPI`, `Agents`, `Tools`.
*   **Logic Covered**: 100% of critical logic (cleaning rules, split strategy, weighted model choice, model training/evaluation, and API routing).
*   **Integration**: Partial coverage via endpoint testing and agent-tool mock integration.

## 6. Output
```bash
uv run pytest tests/ --cov=src --cov-report=term-missing
============================= test session starts =============================
platform win32 -- Python 3.11.13, pytest-9.0.2, pluggy-1.6.0
rootdir: C:\Users\sebas\Desktop\nyc-taxi-tips-prediction
configfile: pyproject.toml
plugins: anyio-4.12.1, hydra-core-1.3.2, langsmith-0.7.9, cov-7.0.0
collected 33 items

tests/unit/test_agent_tools.py ......                                    [ 18%]
tests/unit/test_api.py ....                                              [ 30%]
tests/unit/test_data_ingestion.py .                                      [ 33%]
tests/unit/test_data_transformation.py ......                            [ 51%]
tests/unit/test_feature_engineering.py ...                               [ 60%]
tests/unit/test_model_evaluation.py ..                                   [ 66%]
tests/unit/test_model_trainer.py ....                                    [ 78%]
tests/unit/test_predict_model.py ...                                     [ 87%]
tests/unit/test_taxi_analyst_agent.py ....                               [100%]

Name                                    Stmts   Miss  Cover
-----------------------------------------------------------
TOTAL                                     770    203    74%
Required test coverage of 65% reached. Total coverage: 73.64%
======================= 33 passed, 2 warnings in 10.35s =======================
```
