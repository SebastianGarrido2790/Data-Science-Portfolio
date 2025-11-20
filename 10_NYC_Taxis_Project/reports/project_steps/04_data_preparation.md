### Step 4: Data Preparation & Feature Engineering

This step transforms raw data into a clean, enriched dataset for the batch-scoring system’s three pipelines: Feature Pipeline, Training Pipeline, and Inference Pipeline. The process ensures reliability, scalability, maintainability, and adaptability, using Dask for large-scale processing and a Feature Store for consistency.

#### 1. Data Transformation
- **Validation and Cleaning**:
  - Validate TLC data: Filter for 2019 (confirmed via `tpep_pickup_datetime`), remove negative `trip_distance` and `fare_amount` (already applied).
  - Weather data: Impute `amount of precipitation` with median (0.1); handle `cloud cover` missing values with mode.
  - Zone data: Fill missing `Borough`, `Zone`, `service_zone` with "Unknown".
  - Check data types: Ensure `tpep_pickup_datetime` is datetime, convert `weather` timestamps to datetime.
- **Normalization/Scaling**: Apply StandardScaler to `trip_distance`, `fare_amount` for consistency.
- **Categorical Encoding**: Use target encoding for `PULocationID`, `DOLocationID` based on ride counts.
- **Pipeline**: Implement in `src/features/feature_engineering.py` using Scikit-learn Pipeline.

#### 2. Feature Engineering
- **Domain-Driven Features**:
  - Temporal: Extract hour, day, month; create lagged ride counts (1, 2, 3 hours) per zone.
  - Weather: Include `temperature`, `humidity`, `wind speed`, `amount of precipitation`.
  - Zone Characteristics: Merge `Borough`, `service_zone` from zone data.
- **Time-Series Aggregation**:
  - Aggregate TLC data into hourly counts per `PULocationID` using Dask groupby.
  - Target: Number of rides in the next hour per zone.
  - Features: Lagged counts, weather at pickup time, zone info.
- **Transformation to (Features, Target)**:
  - Create supervised dataset: Each row as (zone, timestamp) with features (lagged counts, weather) and target (next hour rides).
  - Store in `data/processed/features_2019.parquet`.
- **Feature Store**: Use MLflow to version features, ensuring reusability.

#### 3. Train/Test Splitting
- **Strategy**: Time-based split: Jan-Feb 2019 for training, Mar 2019 for testing.
- **Validation**: Reserve 10% of training for validation; prevent leakage by ensuring no future data in features.
- **Implementation**: Split in `src/features/feature_engineering.py` using Pandas.

#### 4. Pipeline Development
- **Feature Pipeline (`src/models/batch-scoring_system/01_feature_pipeline.py`)**:
  - Sub-step 1: Load raw data (TLC, weather, zones) with Dask.
  - Sub-step 2: Clean and validate data (missing values, outliers).
  - Sub-step 3: Engineer features (temporal, weather, zone).
  - Sub-step 4: Aggregate into time-series, define (features, target).
  - Sub-step 5: Split into train/test, save to Feature Store.
  - Output: `data/processed/train_features.parquet`, `data/processed/test_features.parquet`.
- **Training Pipeline (`src/models/batch-scoring_system/02_training_pipeline.py`)**: Placeholder for Step 5.
- **Inference Pipeline (`src/models/batch-scoring_system/03_inference_pipeline.py`)**: Placeholder for Step 5.

#### 5. Governance and Documentation
- **Documentation**: Detail transformations, feature logic, and splitting in `references/feature_engineering.md`.
- **Stakeholder Review**: Present cleaned dataset and features to stakeholders for validation.

#### Outcome
A prepared dataset in `data/processed`, integrated into a scalable Feature Pipeline, ready for modeling. The process is documented and aligned with business needs, with features stored for production use. Proceed to implementation upon instruction.