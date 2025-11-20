# Customer Churn Prediction Project

## Overview
This project develops a machine learning solution to predict customer churn for a telecom company using structured customer data and unstructured ticket notes. By integrating text summarization, embeddings, and an XGBoost model, the system identifies at-risk customers to enable proactive retention strategies. The pipeline includes data preprocessing, feature engineering, model training, evaluation, and deployment preparation. It supports both Hugging Face and OpenAI as options for text summarization and embedding generation.

## Objectives
- Predict customer churn using demographic data (`age`, `tenure`, `spend_rate`, `plan_type`) and ticket note embeddings.
- Handle class imbalance (41 non-churn vs. 19 churn in training) using techniques like `scale_pos_weight`.
- Achieve a ROC AUC score above 0.75 and improve recall for the churn class (class `1`).
- Prepare the model for deployment by saving critical components (model, scaler, feature names).
- Provide actionable insights through visualizations (e.g., feature importance, ROC curve).

## Project Structure
```
├── LICENSE
├── README.md          <- Project overview and usage instructions.
├── data
│   ├── external       <- Third-party data.
│   ├── interim        <- Intermediate data (e.g., summaries).
│   ├── processed      <- Final datasets with embeddings.
│   └── raw            <- Original, immutable data.
├── models             <- Trained models and metadata.
│   ├── feature_names.txt
│   ├── label_encoder.pkl
│   ├── metadata.json
│   ├── scaler.pkl
│   ├── xgb_churn_model.json
│   └── xgb_churn_model.pkl
├── references         <- Data dictionaries and supplementary materials.
├── reports            <- Analysis reports and figures.
│   ├── figures        <- Generated visualizations.
│   │   ├── feature_importance.png
│   │   └── roc_curve.png
│   ├── adaptability_enhancement.md  <- Adaptability Enhancement Plan.
│   ├── model_deployment.md          <- Deployment preparation details.
│   ├── model_performance.md         <- Model performance analysis.
│   └── scalability_enhancement.md   <- Scalability Enhancement Plan.
├── .env
├── .gitignore
├── pyproject.toml
├── uv.lock
└── src                <- Source code.
    ├── __init__.py
    ├── config
    │   ├── factories.py      <- Factories for embeddings and summarization (Hugging Face and OpenAI).
    │   └── settings.py       <- Environment settings.
    ├── data
    │   └── make_dataset.py   <- Data preprocessing.
    ├── features
    │   └── build_features.py <- Feature engineering.
    ├── models
    │   ├── data_processing.py    <- Data loading and preparation.
    │   ├── train_model.py        <- Model training and evaluation.
    │   ├── predict_model.py      <- Model saving and inference.
    │   └── main.py               <- Full training pipeline.
    └── visualization
        └── visualize.py      <- Visualization scripts.
```

## Methodology
1. **Data Collection**:
   - Load raw customer data (`data/raw`) with features like `age`, `tenure`, `spend_rate`, `plan_type`, and `ticket_notes`.
2. **Preprocessing**:
   - Validate and clean data (`make_dataset.py`).
   - Summarize `ticket_notes` using `sshleifer/distilbart-cnn-12-6` (Hugging Face) or OpenAI's API.
   - Generate embeddings with `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face) or OpenAI's embedding API (e.g., `text-embedding-ada-002`).
   - Save processed data to `data/processed`.
3. **Feature Engineering**:
   - Encode categorical features (`plan_type`).
   - Scale numeric features (`age`, `tenure`, `spend_rate`) using `StandardScaler`.
   - Combine scaled features with embeddings (`build_features.py`).
4. **Model Training**:
   - Split data into train, validation, and test sets (80-20 split with stratification).
   - Train an XGBoost model with `scale_pos_weight=2.16` to address class imbalance.
   - Apply early stopping to prevent overfitting (`main.py`).
5. **Evaluation**:
   - Evaluate on test set using precision, recall, F1-score, and ROC AUC.
   - Generate visualizations (ROC curve, feature importance) in `reports/figures`.
6. **Deployment Preparation**:
   - Save model, scaler, and feature names to `models/` (see `reports/model_deployment.md`).
   - Perform batch inference on new data.

## Technical Architecture
- **Data Processing**: Pandas for data handling, `StandardScaler` for feature scaling.
- **NLP**: 
  - Hugging Face: `distilbart-cnn-12-6` for summarization, `all-MiniLM-L6-v2` for embeddings (384 dimensions).
  - OpenAI: API-based summarization and embeddings (e.g., `text-embedding-ada-002`, typically 1536 dimensions).
- **Modeling**: XGBoost (`XGBClassifier`) with early stopping and `scale_pos_weight` for imbalance.
- **Visualization**: Matplotlib for ROC curves and feature importance plots.
- **Deployment Prep**: Joblib for saving/loading model and scaler, text files for feature names.

## Installation Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SebastianGarrido2790/customer-churn-prediction.git
   cd customer-churn-prediction
   ```
2. **Set Up Virtual Environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   uv pip install -r pyproject.toml
   ```
4. **Set Environment Variables**:
   - Create a `.env` file with necessary configurations.
   - For Hugging Face (default):
     ```bash
     echo "DATA_PATH=data/raw" > .env
     ```
   - For OpenAI (optional):
     ```bash
     echo "DATA_PATH=data/raw" > .env
     echo "OPENAI_API_KEY=your_openai_api_key" >> .env
     ```
   - Ensure the API key is valid and sourced from your OpenAI account.

## Usage Examples
1. **Run the Full Pipeline with Hugging Face (Default)**:
   - Executes data processing, training, evaluation, and batch inference.
   ```bash
   python src/models/main.py
   ```
   - Outputs: Classification report, ROC AUC score, visualizations in `reports/figures`, batch predictions.

2. **Run the Full Pipeline with OpenAI**:
   - Set `OPENAI_API_KEY` in `.env` and configure `factories.py` to use OpenAI.
   ```bash
   python src/models/main.py --provider openai
   ```
   - Note: Adjust `feature_names.txt` to accommodate OpenAI's 1536-dimensional embeddings if used.

3. **Generate Visualizations**:
   ```bash
   python src/visualization/visualize.py
   ```
   - Outputs: Feature importance and ROC curve plots in `reports/figures`.

4. **Inspect Model Performance**:
   - Review `reports/model_performance.md` for detailed analysis and recommendations.

## Key Results
- **Model Performance** (see `reports/model_performance.md`):
  - ROC AUC: 0.7857 (improved from 0.7679).
  - Classification Report (Test Set):
    ```
    precision    recall  f1-score   support
    0       0.75      0.86      0.80        14
    1       0.50      0.33      0.40         6
    accuracy                           0.70        20
    ```
  - Batch Inference: Predicted `[1, 0]` for new data, indicating improved churn detection.
- **Class Imbalance Handling**: Applied `scale_pos_weight=2.16`, but recall for class `1` dropped to 0.33 (needs further tuning).
- **Deployment Readiness**: Model, scaler, and feature names saved for inference (see `reports/model_deployment.md`).

## Maintenance
- **Model Updates**:
  - Retrain the model periodically with new data by running `main.py`.
  - Update `scale_pos_weight` based on new class imbalance ratios.
- **Monitoring**:
  - Monitor prediction logs for drift in input data distributions.
  - Revisit `reports/model_performance.md` for ongoing improvement recommendations.
- **Enhancement Plans**:
  - **Adaptability**: See `reports/adaptability_enhancement.md` for strategies to adapt the model to new data sources or providers (e.g., OpenAI).
  - **Scalability**: See `reports/scalability_enhancement.md` for scaling the pipeline for larger datasets or API calls.

## Troubleshooting
- **Pipeline Fails to Run**:
  - Ensure all dependencies are installed (`uv pip install -r pyproject.toml`).
  - Check `.env` for correct paths and configurations.
- **Model Prediction Errors**:
  - Verify that `feature_names.txt` matches the training data structure (e.g., 384 for Hugging Face, 1536 for OpenAI).
  - Ensure the scaler and model files in `models/` are not corrupted.
- **Low Recall for Churn**:
  - Adjust `scale_pos_weight` or apply SMOTE as suggested in `reports/model_performance.md`.
  - Lower the prediction threshold (e.g., from 0.5 to 0.3) in `train_model.py`.
- **OpenAI Integration Issues**:
  - Confirm `OPENAI_API_KEY` is valid in `.env`.
  - Check API rate limits or network connectivity.
  - Ensure `factories.py` is configured to switch providers (e.g., via command-line argument).
- **Performance Issues**:
  - Use GPU support for embeddings and summarization if runtime is a bottleneck.
  - Optimize data processing by batching in `make_dataset.py`.

## License
This project is licensed under the terms of the [LICENSE](./LICENSE.txt) file. Ensure you comply with the licensing agreements when using or modifying the code.

## References
- `reports/model_deployment.md`: Details on saving model components for deployment.
- `reports/model_performance.md`: Analysis of model performance and recommendations.
- `references/`: Data dictionaries and supplementary materials.

## Acknowledgements

This project integrates state-of-the-art AI methods for text summarization and embeddings, leveraging libraries from Hugging Face, OpenAI, and the broader Python ecosystem for data science and machine learning.

## Contact

For any questions or support, please open an issue in the repository or contact the project maintainer at [sebastiangarrido2790@gmail.com].
