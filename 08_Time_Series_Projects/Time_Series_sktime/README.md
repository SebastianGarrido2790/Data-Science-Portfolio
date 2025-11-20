# Sales Forecasting Project

## Overview
This project implements a time series forecasting solution for sales data using multiple machine learning models, including Prophet, Sktime, and XGBoost. The primary dataset is derived from `sales_for_fc.csv` for aggregated sales forecasting with Prophet, and `subcategory_*.csv` files for subcategory-level predictions with Sktime and XGBoost. The project includes exploratory data analysis (EDA) notebooks, data ingestion and processing scripts, and a modular codebase to support model training, validation, and forecasting. The focus is on providing reliable, scalable, and maintainable predictions for business decision-making.

## Objectives
- Develop accurate time series forecasts for total sales and subcategories.
- Evaluate model performance using metrics such as MAE, RMSE, MAPE, and SMAPE.
- Provide visualizations to interpret trends and forecast uncertainty.
- Ensure a reusable and extensible framework for future enhancements.

## Project Structure
```
sales-forecasting
│
├── LICENSE
├── README.md           <- This file.
├── data
│   ├── external        <- Data from third-party sources.
│   ├── interim         <- Intermediate transformed data.
│   ├── processed       <- Final datasets for modeling.
│   │   ├── subcategories
│   │   │    └── subcategory_*.csv
│   │   └── sales_for_fc.csv
│   └── raw             <- Original, immutable data dump.
│
├── logs
│
├── models              <- Trained models, predictions, and summaries.
│   ├── prophet
│   ├── sktime
│   └── xgboost
│
├── notebooks           <- Jupyter notebooks for EDA.
│   ├── 1.0-eda-total-sales.ipynb   <- EDA for total sales with decomposition.
│   └── 2.0-subcategory-decomposition-and-analysis.ipynb
│
├── references          <- Data dictionaries and manuals.
│
├── reports             <- Generated analysis files.
│   ├── figures         <- Graphics for reporting.
│   │   ├── prophet
│   │   ├── sktime
│   │   ├── subcategories
│   │   ├── trends
│   │   ├── xgboost
│   │   ├── sales_components.png
│   │   └── sales_forecast.png
│   ├── prophet_overview.md
│   ├── subcategories.md
│   ├── subcategory_notebook.md
│   └── tune_forecaster.md
│
├── pyproject.toml
├── uv.lock
├── .gitignore
│
├── src                 <- Source code.
│   ├── __init__.py     <- Makes src a Python module.
│   ├── data            <- Data ingestion and processing scripts.
│   │   ├── data_ingestor.py
│   │   └── make_dataset.py
│   ├── features        <- Feature engineering scripts.
│   │   └── build_features.py
│   ├── models          <- Model training and prediction scripts.
│   │   ├── prophet
│   │   │    ├── backtesting.py
│   │   │    ├── final_forecasting.py
│   │   │    └── train_model.py
│   │   ├── predict_with_sktime.py
│   │   └── predict_with_xgboost.py
│   ├── utils           <- Utility scripts.
│   │   ├── helper.py   <- Functions to support workflow.
│   │   └── plot_settings.py
```

## Methodology
1. **Data Ingestion**: Uses `data_ingestor.py` to handle CSV, JSON, Parquet, and ZIP files, extracting and validating raw data into `data/raw`.
2. **Data Processing**: `make_dataset.py` cleans data, aggregates sales by subcategory, adds time-based features, removes outliers, and saves processed datasets to `data/processed`.
3. **Exploratory Data Analysis**: Notebooks (e.g., `1.0-eda-total-sales.ipynb`) decompose time series into trend, seasonality, and residuals.
4. **Model Training**: `train_model.py` (Prophet) tunes hyperparameters and saves them to `models/prophet/params.pkl`.
5. **Model Validation**: `backtesting.py` (Prophet) evaluates performance with metrics saved to `models/prophet/metrics.json`.
6. **Forecasting**: `final_forecasting.py` (Prophet) generates 30-step forecasts and plots, saved to `models/prophet/forecasts.csv` and `reports/figures/prophet`.
7. **Subcategory Forecasting**: `predict_with_sktime.py` and `predict_with_xgboost.py` use `subcategory_*.csv` for detailed predictions.

## Technical Architecture Description
- **Data Flow**: Raw data (`data/raw`) is ingested via `data_ingestor.py`, processed by `make_dataset.py`, and stored as `sales_for_fc.csv` (Prophet) or `subcategory_*.csv` (Sktime/XGBoost) in `data/processed`. Interim data is saved in `data/interim`.
- **Modeling**: Prophet scripts (`train_model.py`, `backtesting.py`, `final_forecasting.py`) operate on aggregated sales data. Sktime and XGBoost scripts use subcategory data. Models are stored in `models/`.
- **Utilities**: `helper.py` provides functions like `load_sales_data`, `compute_metrics`, and `save_forecasts` to support workflow consistency.
- **Visualization**: Plots are generated in `reports/figures` using `matplotlib`, with settings from `plot_settings.py`.
- **Logging**: All scripts log to `logs/` for debugging and tracking.

## Installation Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SebastianGarrido2790/sales-forecasting.git
   cd sales-forecasting
   ```
2. **Set Up Virtual Environment**:
   ```bash
   uv init
   uv sync
   ```
3. **Install Dependencies**:
   Ensure `pyproject.toml` includes required packages (e.g., `prophet`, `sktime`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `pyyaml`). Install with:
   ```bash
   uv pip install -r requirements.txt
   ```
4. **Verify Installation**:
   ```bash
   python -c "import prophet; print(prophet.__version__)"
   python -c "import sktime; print(sktime.__version__)"
   python -c "import xgboost; print(xgboost.__version__)"
   ```

## Usage Examples
### Run Data Ingestion and Processing
```bash
uv run src/data/data_ingestor.py
uv run src/data/make_dataset.py
```
- Ingests `train.csv.zip` and processes it into `data/processed/subcategories/subcategory_*.csv`.

### Train Prophet Model
```bash
uv run src/models/prophet/train_model.py
```
- Trains the model and saves parameters to `models/prophet/params.pkl`.

### Backtest Prophet Model
```bash
uv run src/models/prophet/backtesting.py
```
- Validates the model and saves metrics to `models/prophet/metrics.json`.

### Generate Prophet Forecast
```bash
uv run src/models/prophet/final_forecasting.py
```
- Produces a 30-step forecast and plot in `reports/figures/prophet/`.

### Run Sktime or XGBoost Predictions
```bash
uv run src/models/predict_with_sktime.py
uv run src/models/predict_with_xgboost.py
```
- Uses `subcategory_*.csv` for subcategory forecasts.

### Explore EDA Notebooks
- Open `notebooks/1.0-eda-total-sales.ipynb` or `2.0-subcategory-decomposition-and-analysis.ipynb` in Jupyter Notebook to analyze trends and seasonality.

## Key Results
- **Prophet Metrics** (from `models/prophet/metrics.json`):
  - MAE: 1811.86, RMSE: 2222.57, MAPE: 1147.20%, SMAPE: 77.60% (indicates room for improvement).
- **Prophet Forecast** (from `models/prophet/forecasts.csv`):
  - 30-day forecast for `sales` (e.g., day 1259: 2077.43, CI: -1922.57 to 6019.99).
- **Visualization**: `reports/figures/prophet/prophet_forecast_sales.png` shows historical sales and a 30-step forecast with a 95% confidence interval.

## Maintenance
- **Version Control**: Use Git to track changes; update `pyproject.toml` and `uv.lock` with dependency versions.
- **Data Updates**: Regularly ingest new data with `data_ingestor.py` and reprocess with `make_dataset.py`.
- **Model Retraining**: Periodically run `train_model.py` with updated data to refine parameters.
- **Documentation**: Update `README.md`, `references/`, and `reports/` with new findings or changes.

## Troubleshooting
- **Installation Errors**: Ensure all dependencies are installed; use `uv pip install --force-reinstall <package>` if issues persist.
- **Data Ingestion Failures**: Check `logs/data_ingestion.log` for file path or format errors; verify `train.csv.zip` exists.
- **Model Errors**: Review `logs/prophet.log` for stack traces; ensure `sales_for_fc.csv` and `params.pkl` are valid.
- **Performance Issues**: Increase memory or reduce dataset size; adjust `horizon` in forecasting scripts.
- **Common Fixes**:
  - Missing Dates: Preprocess `sales_for_fc.csv` with `load_sales_data` to fill gaps.
  - Wide Confidence Intervals: Refine hyperparameters in `train_model.py`.

## Contributing
- Fork the repository, create a feature branch, and submit pull requests.
- Follow PEP 8 style guidelines and add unit tests in a future `tests/` directory.
- Report issues or suggestions via GitHub Issues.

## License
This project is licensed under the terms of the [LICENSE](./LICENSE.txt) file. Ensure you comply with the licensing agreements when using or modifying the code.

---

### Notes
- `predict_with_sktime.py` and `predict_with_xgboost.py` utilize `subcategory_*.csv` for subcategory-level forecasting.
- `train_model.py`, `backtesting.py`, and `final_forecasting.py` (Prophet) use `sales_for_fc.csv` for aggregated sales forecasting.
- `helper.py` is a critical utility script containing functions (`load_sales_data`, `compute_metrics`, `save_forecasts`, etc.) to support the workflow across all scripts.