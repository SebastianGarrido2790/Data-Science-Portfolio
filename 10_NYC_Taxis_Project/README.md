# NYC Taxis Demand Prediction Project

## Overview
This project develops a predictive model to forecast yellow taxi demand in Manhattan, utilizing historical trip data from the NYC Taxi and Limousine Commission (TLC), weather data, and zone information. The solution leverages machine learning techniques, including XGBoost and Prophet, to provide actionable insights for urban planning and transportation optimization.

## Objectives
- Build a robust model to predict hourly taxi ride counts by zone.
- Implement a scalable deployment pipeline with monitoring and retraining capabilities.
- Ensure data privacy and security compliance.
- Provide a user-friendly dashboard for performance monitoring.

## Project Structure
```
├── LICENSE
├── README.md          <- This file
├── .env               <- Environmental variable for APIs
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
├── mlartifacts
├── mlruns
├── models             <- Trained and serialized models
│   ├── prophet_model.pkl
│   ├── scaler.joblib
│   └── xgboost_model.xgb
├── notebooks          <- Jupyter notebooks for exploration
├── references         <- Data dictionaries and manuals
├── reports            <- Generated analysis and figures
│   ├── figures        <- Generated graphics
│   ├── project_steps  <- Step-by-step project documentation
│   ├── deployment.md  <- Detailed deployment implementation
│   └── other reports
├── requirements.txt   <- Environment requirements
├── src                <- Source code
│   ├── data           <- Data handling scripts
│   ├── models         <- Model training and inference scripts
│   └── visualization  <- Visualization scripts
├── pyproject.toml
├── uv.lock
```

## Methodology
1. **Data Understanding**: Analyze TLC trip data, weather, and zone lookup files.
2. **Data Preparation**: Clean data, engineer features (e.g., hourly counts, lags), and split into train/test sets.
3. **Modeling**: Train XGBoost and Prophet models with hyperparameter tuning via Optuna.
4. **Evaluation**: Assess model performance using RMSE and MAE metrics.
5. **Deployment**: Package models, integrate with APIs, and deploy using Docker and Kubernetes.
6. **Monitoring**: Implement Prometheus and Grafana for real-time metrics tracking.
7. **Security**: Apply encryption and access controls.

## Technical Architecture
- **Data Pipeline**: Dask for parallel data processing, MLflow for feature store and model tracking.
- **Modeling**: XGBoost for regression, Prophet for time-series forecasting.
- **Deployment**: Docker containers, Kubernetes for scalability, Flask for API serving.
- **Monitoring**: Prometheus for metrics, Grafana for visualization, Streamlit for dashboard.
- **CI/CD**: GitHub Actions for automated testing and deployment.
- **Security**: Fernet encryption, OAuth 2.0 authentication.

## Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/SebastianGarrido2790/nyc-taxis-project.git
   cd nyc-taxis-project
   ```
2. Install UV and dependencies:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync --frozen
   ```
3. Set up environment variables in `.env` (e.g., MLflow URI).
4. Install Docker and Kubernetes (for deployment).
5. Run MLflow server locally:
   ```bash
   mlflow server --host 0.0.0.0 --port 5000
   ```

## Usage Examples
- **Run Feature Pipeline**:
  ```bash
  uv run python src/models/batch-scoring_system/01_feature_pipeline.py
  ```
- **Train Models**:
  ```bash
  uv run python src/models/batch-scoring_system/02_training_pipeline.py
  ```
- **Generate Predictions**:
  ```bash
  uv run python src/models/batch-scoring_system/03_inference_pipeline.py
  ```
- **Launch Dashboard**:
  ```bash
  uv run streamlit run src/models/dashboard.py
  ```

## Key Results
- XGBoost RMSE: ~0.279 (training), online MAE: ~0.4985 (test set).
- Prophet RMSE: Comparable to XGBoost, suitable for time-series trends.
- Dashboard visualizes MAE trends, predictions vs. actuals, and Manhattan zone demand heatmap.

## Maintenance
- **Retraining**: Schedule monthly retraining via the monitoring script.
- **Monitoring**: Check Prometheus metrics at `http://localhost:8000` and Grafana dashboard.
- **Updates**: Pull latest changes and run `uv sync` for dependency updates.

## Troubleshooting
- **ModuleNotFoundError**: Ensure all dependencies are installed with `uv sync`.
- **KeyError**: Verify data columns match expected schema; regenerate parquet files if needed.
- **Deployment Issues**: Check Docker/Kubernetes status; review logs in `reports/deployment.md`.

## License
This project is licensed under the terms of the [LICENSE](./LICENSE.txt) file. Ensure you comply with the licensing agreements when using or modifying the code.

## Additional Resources
The deployment implementation is well detailed with code snippets in `reports/deployment.md`, covering:
- Step 1: Preparation for Deployment
- Step 2: Packaging & Versioning
- Step 3: Integration and Scalability
- Step 4: CI/CD Integration and Testing
- Step 5: Monitoring, Maintenance, and Retraining
- Step 6: Security

Refer to this file to continue with the project or troubleshoot deployment-specific issues.