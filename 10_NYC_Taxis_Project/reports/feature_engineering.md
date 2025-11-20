# Feature Engineering Documentation

## Project Overview
This document details the feature engineering process for the NYC Taxis Project, aimed at predicting yellow taxi demand in Manhattan. The process aligns with the CRISP-DM methodology, ensuring reliability, scalability, maintainability, and adaptability.

## Data Transformations
- **Source Data**: 
  - TLC Trip Data (yellow_tripdata_2019-01 to -03, ~22M rows)
  - Weather Data (nyc_weather.csv, 2936 rows)
  - Zone Data (taxi_zone_lookup.csv, 265 rows)
- **Cleaning**:
  - TLC: Filtered for 2019, removed trips with `trip_distance <= 0` or `fare_amount <= 0`, dropped rows with missing `passenger_count`.
  - Weather: Converted `date and time` to datetime, imputed `amount of precipitation` with median (0.1), filled `cloud cover` with mode.
  - Zone: Filled missing `Borough`, `Zone`, and `service_zone` with "Unknown".
- **Scaling**: Applied `StandardScaler` to `trip_distance` and `fare_amount` using Dask `map_partitions` for memory efficiency.

## Feature Engineering
- **Temporal Features**:
  - `hour_of_day`: Extracted from `hour` (0-23).
  - `day_of_week`: Extracted from `hour` (0-6, Monday=0).
  - `month`: Extracted from `hour` (1-3 for 2019).
- **Lagged Features**:
  - `lag_1`, `lag_2`, `lag_3`: Previous 1, 2, and 3-hour ride counts per `PULocationID`, filled with 0 for initial periods.
- **Weather Features**:
  - `temperature`, `humidity`, `wind speed`, `amount of precipitation`: Merged from weather data, imputed with median values.
- **Zone Characteristics**:
  - `Borough`, `service_zone`: Merged from zone data, filled with "Unknown" if missing.
- **Target**: `target` defined as the next hour's ride count per `PULocationID`.
- **Scaling**: Applied `StandardScaler` to `ride_count`, `lag_1`, `lag_2`, `lag_3` post-aggregation.

## Data Splitting
- **Strategy**: Time-based split using `month` column.
  - Training: Jan-Feb 2019 (months 1-2).
  - Testing: Mar 2019 (month 3).
- **Validation**: 10% of training data reserved for validation, preserving time order with `shuffle=False`.
- **Output**: Saved as `data/processed/train_features.parquet`, `data/processed/test_features.parquet`.

## Governance
- **Data Integrity**: Ensured no leakage by maintaining chronological order in splits.
- **Versioning**: Features logged with MLflow, tracked at `http://localhost:5000`.
- **Compliance**: Anonymization and imputation align with GDPR/CCPA requirements.

## Validation
- Stakeholders to review dataset integrity and feature relevance.
- Date of Review: Planned for June 18, 2025, 10:00 AM -04.

## Contact
- Author: Sebastian Garrido (sebastiangarrido279@gmail.com)
- Date Created: June 17, 2025


