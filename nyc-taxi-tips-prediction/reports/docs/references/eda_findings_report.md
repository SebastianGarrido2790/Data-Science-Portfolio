# Exploratory Data Analysis (EDA) Findings Report

**Project**: NYC Yellow Taxi Tip Prediction
**Context**: FTI (Feature, Training, Inference) MLOps Architecture

This report details the findings from the initial Exploratory Data Analysis (EDA) conducted on the raw `Distilled_2023_Yellow_Taxi_Trip_Data.txt` dataset. These insights directly dictate the deterministic cleaning logic that will be implemented in the Data Engineering (Feature Store ingestion) pipeline.

---

## 1. Data Health & Missing Value Analysis

The raw dataset contains 5,000,000 rows. A systematic check for missing values revealed that several key columns exhibit a missing rate of exactly **3.41% (170,476 rows)**:
- `passenger_count`
- `RatecodeID`
- `store_and_fwd_flag`
- `congestion_surcharge`
- `airport_fee`

### Imputation Strategies Defined:
To avoid discarding 3.4% of the dataset, we will implement the following imputation logic:
*   **Financial Surcharges** (`airport_fee`, `congestion_surcharge`): Assume nulls represent "no charge" and fill with `0.0`.
*   **Capacity** (`passenger_count`): Since a trip occurred, assume at least 1 rider and fill with `1.0`.
*   **Categorical Codes** (`RatecodeID`): Map missing codes to `99.0` (which corresponds to "Unknown" in the data dictionary).

### `store_and_fwd_flag` Irrelevance:
Analysis showed that `96.00%` of this column is 'N', while only `0.58%` is 'Y'. Grouping by this flag showed negligible variance in average tips (`$3.55` vs `$3.05`).
*   **Action**: Drop `store_and_fwd_flag` during ingestion as it adds no predictive value.

---

## 2. Univariate Analysis (Anomaly Detection)

The dataset contains entry errors and extreme outliers that must be filtered out prior to training to ensure model stability.

### Financial Anomalies (`total_amount`):
*   **Refunds/Chargebacks**: `48,941` rows (`~0.98%`) have strictly negative values. **Action**: DROP.
*   **Zero Values / Blank Trips**: `777` rows (`~0.015%`) have a total cost of $0. **Action**: DROP.
*   **Below Minimum Fare Threshold**: Taking into account base fare ($3.00), MTA tax ($0.50), and improvement surcharge ($0.30), a minimal valid trip is roughly $3.80. There are `1,583` rows below $3.70. **Action**: DROP.
*   **Extreme Outliers**: The maximum observed value is $2,100.00, while the 99.9th percentile is only $181.39. **Action**: Cap or drop strictly `> $1,000` to prevent outlier warping.

### Distance Anomalies (`trip_distance`):
*   **Static/Parking Trips**: `100,567` trips (`~2.01%`) traveled exactly `0` miles. **Action**: DROP.
*   **Impossible Distances**: The maximum distance recorded is an impossible 345,729.44 miles, while the 99.9th percentile sits at 30.08 miles. **Action**: Drop trips exceeding a logical upper limit of `100` miles.

---

## 3. Temporal Analysis & Feature Extraction

The raw `tpep_pickup_datetime` and `tpep_dropoff_datetime` strings must be cast to `datetime` objects.

### Derived Features:
*   **Duration**: A `trip_duration_mins` feature was derived by subtracting pickup from drop-off time.
*   **Duration Anomalies**: `2,023` trips resulted in negative or zero duration. **Action**: DROP.
*   **Granularity**: Extracting `pickup_month`, `pickup_day_of_week`, and `pickup_hour` successfully maps out the temporal variation in tipping behavior. For instance, tips visibly fluctuate based on the hour of the day.

---

## 4. Multivariate Analysis (Drivers of Tipping)

After applying the baseline cleaning rules defined above, a highly clean subset retaining **97.08% (4,853,877 rows)** of the original data was generated.

### Correlation & Feature Importance:
*   `trip_distance` shows a looser correlation with the tip compared to financial columns, as distance is already "baked into" the various fare charges via toll and time variations.
*   A proxy `XGBRegressor` was used to determine raw feature impacts on `tip_amount`.

### 🚨 CRITICAL MLOps DISCOVERY: Data Leakage Warning 🚨
The feature importance proxy demonstrated that using `total_amount` as a predictive feature constitutes **severe data leakage**. Because `total_amount` mathematically *includes* the `tip_amount` in most standard scenarios, passing it to the model gives away the answer.

**Architectural Prevention (Data Layer Action)**:
To preserve the integrity of the FTI pattern, the Feature Pipeline must enforce one of the following proxy billing features instead:
1.  Derive `total_amount_before_tip` = `total_amount` - `tip_amount`.
2.  Aggregate manually: `fare_amount` + `tolls_amount` + `surcharges`.

---

## 5. Summary of Automated Ingestion Rules

Based on the EDA, the Data Engineering (Feature Pipeline) must programmatically execute:

1. **Impute** `airport_fee` and `congestion_surcharge` -> `0.0`.
2. **Impute** `passenger_count` -> `1.0`.
3. **Impute** `RatecodeID` -> `99.0`.
4. **Drop Column** `store_and_fwd_flag`.
5. **Drop Rows** where `total_amount < 3.70`.
6. **Drop Rows** where `total_amount > 1000`.
7. **Drop Rows** where `trip_distance <= 0`.
8. **Drop Rows** where `trip_distance > 100`.
9. **Cast** string dates to explicit `datetime`.
10. **Derive** `trip_duration_mins`.
11. **Drop Rows** where `trip_duration_mins <= 0`.
12. **Derive Temporal Features**: `pickup_month`, `pickup_day_of_week`, `pickup_hour`.
13. **Derive Non-leaking Target Proxy**: `total_amount_before_tip` (`total_amount` - `tip_amount`).
