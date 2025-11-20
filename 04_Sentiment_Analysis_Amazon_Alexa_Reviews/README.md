# Sentiment Analysis on Amazon Alexa Reviews

## Overview

This project performs sentiment analysis on Amazon Alexa reviews using multiple machine learning models, including a fine-tuned DistilBERT, Random Forest, and XGBoost. The goal is to classify reviews as "positive" or "negative" (with an additional "neutral" class in some models), providing insights into customer sentiment. The project includes a Flask web application for real-time predictions, a standalone script for batch predictions, and a Jupyter notebook leveraging RAPIDS for GPU-accelerated processing. Results are visualized through various figures, such as precision-recall curves and confusion matrices.

## About the Data

This dataset consists of a nearly 3000 Amazon customer reviews (input text), star ratings, date of review, variant and feedback of various amazon Alexa products like Alexa Echo, Echo dots, Alexa Firesticks etc. for learning how to train Machine for sentiment analysis.

**Source**: Extracted from Amazon's website

**Link**: https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews

## Objectives

- **Analyze Sentiment**: Classify Amazon Alexa reviews into positive, negative, and neutral sentiments using DistilBERT, Random Forest, and XGBoost models.
- **Provide Real-Time Predictions**: Enable users to input reviews via a web interface and receive immediate sentiment predictions.
- **Support Batch Processing**: Allow batch predictions on multiple reviews via a command-line script, with options to save results to CSV.
- **Leverage GPU Acceleration**: Use RAPIDS (via cuML and cuDF) in a Jupyter notebook for faster data processing and model training.
- **Achieve High Accuracy**: Fine-tune models to achieve high accuracy, with configurable thresholds for classification.
- **Visualize Results**: Generate figures to analyze model performance, including precision-recall curves and confusion matrices.

## Project Structure

```
04_Sentiment_Analysis_Amazon_Alexa_Reviews/
├── LICENSE                        # Project license
├── README.md                      # Project documentation (this file)
├── .venv/                         # Virtual environment
├── pyproject.toml                 # Project dependencies
├── uv.lock                        # Dependency lock file
├── data/                          # Datasets
│   ├── external/                  # Data from third-party sources
│   ├── interim/                   # Intermediate data (e.g., train.csv, test.csv)
│   ├── processed/                 # Processed data (e.g., train_features.csv, test_features.csv)
│   └── raw/                       # Original data dump
├── models/                        # Trained models
│   ├── distilbert_model/          # Fine-tuned DistilBERT models
│   │   ├── distilbert_3class_model/
│   │   └── distilbert_binary_model/
│   ├── rf_model.pkl               # Random Forest model
│   ├── vectorizer.pkl             # CountVectorizer for RF/XGBoost
│   └── xgb_model.pkl              # XGBoost model
├── notebooks/                     # Jupyter notebooks
│   └── distlbert.ipynb            # Notebook with RAPIDS (cuML, cuDF) for DistilBERT training
├── references/                    # Data dictionaries, manuals, etc.
├── reports/                       # Generated analysis and figures
│   ├── figures/                   # Generated figures
│   │   ├── distilbert/            # DistilBERT figures (e.g., training loss plots)
│   │   ├── random_forest/         # Random Forest figures (e.g., rf_pr_curve.png, rf_confusion_matrix.png)
│   │   └── xgboost/               # XGBoost figures (e.g., xgb_pr_curve.png, xgb_confusion_matrix.png)
│   ├── model_performance.md       # Model performance summary
│   └── distilbert_notes.md        # Notes on DistilBERT features
├── src/                           # Source code
│   ├── __init__.py                # Makes src a Python module
│   ├── data/                      # Data preprocessing scripts
│   │   └── make_dataset.py
│   ├── features/                  # Feature engineering scripts
│   │   └── build_features.py
│   ├── models/                    # Model training and prediction scripts
│   │   ├── predict_model.py       # Standalone prediction script (DistilBERT)
│   │   ├── train_distilbert.py    # DistilBERT training script
│   │   └── train_model.py         # Random Forest and XGBoost training script
│   └── visualization/             # Visualization scripts
│       └── visualize.py
├── static/                        # Static files for Flask app
│   └── index.html                 # Web interface for Flask app
├── api.py                         # Flask web app for real-time predictions
├── sentiment_predictions.csv      # Output CSV from batch predictions
└── .gitignore                     # Git ignore file
```

## Methodology

1. **Data Collection**:
   - Amazon Alexa reviews were collected and stored in `data/raw/`.
   - The dataset includes review text, feedback labels (positive/negative/neutral), and metadata (e.g., review length).

2. **Data Preprocessing**:
   - Processed using `src/data/make_dataset.py` to clean text (e.g., remove special characters, handle missing values).
   - Split into train/test sets (`data/interim/train.csv`, `data/interim/test.csv`).
   - Engineered features (e.g., review length, VADER sentiment scores) using `src/features/build_features.py` (`data/processed/train_features.csv`, `data/processed/test_features.csv`).

3. **Feature Engineering**:
   - For DistilBERT: Tokenized reviews using `DistilBertTokenizer` (max length: 128 tokens).
   - For Random Forest/XGBoost: Used `CountVectorizer` (max features: 5000) for text vectorization, combined with engineered features (e.g., `review_length`, `vader_compound`).

4. **Model Training**:
   - **DistilBERT**: Fine-tuned `DistilBertForSequenceClassification` for binary (positive/negative) and 3-class (positive/negative/neutral) classification using `src/models/train_distilbert.py`. A Jupyter notebook (`notebooks/distlbert.ipynb`) leverages RAPIDS with `%load_ext cuml.accel` and cuDF for GPU-accelerated data processing.
   - **Random Forest**: Trained using `src/models/train_model.py` with Optuna hyperparameter tuning, SMOTE for class imbalance, and class weights.
   - **XGBoost**: Trained using `src/models/train_model.py` with Optuna hyperparameter tuning, SMOTE, and scale_pos_weight for class imbalance.
   - Models saved to `models/` (e.g., `distilbert_binary_model`, `rf_model.pkl`, `xgb_model.pkl`).

5. **Prediction**:
   - **Web App**: `api.py` uses Flask to serve a web interface (`index.html`) for real-time predictions with DistilBERT.
   - **Batch Script**: `src/models/predict_model.py` supports command-line predictions with DistilBERT, with options for file input, custom thresholds, and CSV output.

6. **Visualization**:
   - Generated figures for model evaluation:
     - **DistilBERT**: Training loss curves, validation accuracy plots (in `notebooks/distlbert.ipynb`, saved to `reports/figures/distilbert/`).
     - **Random Forest**: Precision-recall curve (`rf_pr_curve.png`) and confusion matrix (`rf_confusion_matrix.png`) in `reports/figures/random_forest/`.
     - **XGBoost**: Precision-recall curve (`xgb_pr_curve.png`) and confusion matrix (`xgb_confusion_matrix.png`) in `reports/figures/xgboost/`.

## Technical Architecture

- **Models**:
  - **DistilBERT**: Fine-tuned `DistilBertForSequenceClassification` from Hugging Face `transformers`.
  - **Random Forest**: `RandomForestClassifier` from scikit-learn, tuned with Optuna.
  - **XGBoost**: `XGBClassifier` from XGBoost, tuned with Optuna.
- **Framework**:
  - **Flask**: Lightweight web framework for real-time prediction app (`api.py`).
  - **PyTorch**: Used for DistilBERT inference.
  - **RAPIDS**: cuML and cuDF for GPU-accelerated processing in `distlbert.ipynb`.
- **Dependencies**: Managed via `pyproject.toml` using `uv` (or `pip`).
- **Input Processing**:
  - DistilBERT: Reviews tokenized with `DistilBertTokenizer` (max length: 128).
  - Random Forest/XGBoost: Text vectorized with `CountVectorizer`, combined with engineered features.
  - Inference on CPU/GPU (auto-detected for DistilBERT, GPU used in RAPIDS notebook).
- **Output**:
  - Web app displays sentiment, probability, and threshold.
  - Batch script outputs to console and CSV.
  - Figures saved to `reports/figures/` for model evaluation.

### Model Files
The DistilBERT model files (`model.safetensors`) were removed from the repository due to GitHub's file size limit. Download them from the following links and place them in the respective directories:
- `models/distilbert_model/distilbert_3class_model/model.safetensors`: [Google Drive Link](https://drive.google.com/drive/folders/1ah736tetlyMlKcnynhPq0AfyUSgXGay9)
- `models/distilbert_model/distilbert_binary_model/model.safetensors`: [Google Drive Link](https://drive.google.com/drive/folders/1ah736tetlyMlKcnynhPq0AfyUSgXGay9)

Update the `MODEL_PATH` variable in `predict_model.py` and `api.py` to point to the local location (e.g., `C:/Users/Models/`).

## Installation Instructions

### Prerequisites
- Python 3.11.9
- `uv` (or `pip`) for dependency management
- CUDA-enabled GPU (required for RAPIDS in `distlbert.ipynb`, optional for others)
- RAPIDS (for `distlbert.ipynb`): Requires CUDA 11.2+, compatible NVIDIA GPU, and RAPIDS installation

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SebastianGarrido2790/04_Sentiment_Analysis_Amazon_Alexa_Reviews.git
   cd 04_Sentiment_Analysis_Amazon_Alexa_Reviews
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Install Dependencies**:
   Using `uv`:
   ```bash
   uv sync
   ```
   Or using `pip`:
   ```bash
   pip install -r requirements.txt
   ```
   (Note: Generate `requirements.txt` from `pyproject.toml` using `uv export > requirements.txt`.)

4. **Install RAPIDS (for `distlbert.ipynb`)**:
   - Follow the RAPIDS installation guide: [RAPIDS Installation](https://rapids.ai/start.html).
   - Example for Conda (if not using `uv`):
     ```bash
     conda create -n rapids-env -c rapidsai -c nvidia -c conda-forge rapids=23.10 python=3.11 cudatoolkit=11.8
     conda activate rapids-env
     ```

5. **Verify Model Files**:
   Ensure the following directories/files exist:
   - `models/distilbert_model/distilbert_binary_model/` (contains `config.json`, `model.safetensors`, etc.)
   - `models/rf_model.pkl`
   - `models/xgb_model.pkl`
   - `models/vectorizer.pkl`

## Usage Examples

### 1. Real-Time Predictions with Flask (DistilBERT)
Run the Flask web app:
```bash
python api.py
```
- Open `http://127.0.0.1:8000` in your browser.
- Enter a review (e.g., "I love this product!") and click "Predict".
- View the sentiment, probability, and threshold on the page.

![Sentiment Analysis API](reports/figures/Sentiment_Analysis_DistilBERT.png)

### 2. Batch Predictions with `predict_model.py` (DistilBERT)
#### Example 1: Using Command-Line Arguments
```bash
python src/models/predict_model.py --texts "I love this product!" "This is terrible." --threshold 0.5 --output sentiment_predictions.csv
```
Output:
```
Text: I love this product!
Sentiment: positive
Probability (Positive): 0.9709
Threshold: 0.5000
--------------------------------------------------
Text: This is terrible.
Sentiment: negative
Probability (Positive): 0.1384
Threshold: 0.5000
--------------------------------------------------
Predictions appended to sentiment_predictions.csv
```

#### Example 2: Using a File Input
Create a file `reviews.txt` with:
```
I like this product!
This is not the best.
```
Run:
```bash
python src/models/predict_model.py --file reviews.txt --threshold 0.95 --output sentiment_predictions.csv
```
The predictions will be appended to `sentiment_predictions.csv` without repeating the header.

### 3. GPU-Accelerated Training with `distlbert.ipynb`
- Open the notebook:
  ```bash
  jupyter notebook notebooks/distlbert.ipynb
  ```
- Ensure RAPIDS is installed and activated.
- Run the first cell to load the cuML accelerator:
  ```python
  %load_ext cuml.accel
  ```
- Use cuDF for data loading and preprocessing (e.g., `cudf.read_csv` instead of `pandas.read_csv`).
- Follow the notebook to preprocess data, train DistilBERT, and generate figures (e.g., training loss curves).

## Key Results

- **Model Performance**:
  - **DistilBERT**: High accuracy with a threshold of 0.95 for positive sentiment (binary classification).
  - **Random Forest**: Tuned for negative class recall with precision >= 0.3; generates PR curves and confusion matrices.
  - **XGBoost**: Optimized for negative class recall; similar figures generated.
- **Sample Predictions (DistilBERT)**:
  - Input: "I love this product!" → Sentiment: positive, Probability: 0.9709, Threshold: 0.95
  - Input: "This is terrible." → Sentiment: negative, Probability: 0.1384, Threshold: 0.5
- **Figures Generated**:
  - **DistilBERT**: Training loss curves, validation accuracy plots (in `reports/figures/distilbert/` via `distlbert.ipynb`).
  - **Random Forest**: `rf_pr_curve.png` (precision-recall curve for negative class), `rf_confusion_matrix.png` (confusion matrix).
  - **XGBoost**: `xgb_pr_curve.png` (precision-recall curve for negative class), `xgb_confusion_matrix.png` (confusion matrix).
- **Scalability**:
  - Batch prediction script supports up to 100 reviews per run (configurable via `MAX_BATCH_SIZE`).
  - RAPIDS in `distlbert.ipynb` accelerates data processing and training on GPU.

## Maintenance

- **Model Updates**:
  - Retrain DistilBERT with new data using `src/models/train_distilbert.py` or `distlbert.ipynb`.
  - Retrain Random Forest/XGBoost with `src/models/train_model.py`.
  - Update model files in `models/`.
- **Dependency Updates**:
  - Update `pyproject.toml` and run `uv sync` or `pip install -r requirements.txt`.
  - Check RAPIDS compatibility for `distlbert.ipynb` with `conda list rapids`.
- **Code Maintenance**:
  - Follow PEP 8 style guidelines.
  - Add unit tests in `src/tests/` to ensure functionality.

## Troubleshooting

### Model Loading Issues
- **Error**: `HFValidationError: Repo id must use alphanumeric chars...`
  - Ensure `MODEL_PATH` in `predict_model.py` and `api.py` uses forward slashes (e.g., `C:/path/to/model`).
  - Verify the directory contains all required files (`config.json`, `model.safetensors`, etc.).

### Flask App Not Starting
- **Error**: `ModuleNotFoundError: No module named 'flask'`
  - Ensure Flask is installed (`uv pip list | findstr flask` or `pip list | findstr flask`).
  - Run `uv sync` or `pip install -r requirements.txt`.

### Prediction Errors
- **Error**: `ValueError: Batch size exceeds limit of 100 texts`
  - Reduce the number of reviews in the input file or increase `MAX_BATCH_SIZE` in `predict_model.py`.

### CSV Output Issues
- **Issue**: Repeated headers in CSV file
  - Ensure `save_predictions` in `predict_model.py` checks `os.path.isfile(output_file)` before writing the header.

### RAPIDS Issues
- **Error**: `ModuleNotFoundError: No module named 'cudf'`
  - Ensure RAPIDS is installed and the environment is activated (`conda activate rapids-env`).
  - Verify CUDA version compatibility (`nvcc --version`).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Hugging Face for the `transformers` library.
- RAPIDS team for cuML and cuDF.
- Flask team for the lightweight web framework.
- scikit-learn and XGBoost teams for Random Forest and XGBoost implementations.
- Amazon Alexa review dataset providers (if applicable).
