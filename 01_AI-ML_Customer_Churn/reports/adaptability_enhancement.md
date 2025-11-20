## **Enhancing Adaptability in the Churn Prediction Project**

To improve adaptability, we will enable the pipeline to handle evolving data patterns, such as changing customer behavior or ticket note styles, which is critical for production ML systems (constantly shifting data). Key enhancements include implementing periodic retraining and adding basic data drift detection to trigger model updates when necessary.

---

### **Adaptability Enhancement Plan**
1. **Periodic Retraining**:
   - Add a mechanism in `src/models/main.py` to schedule retraining based on a time interval or new data availability, using a simple timestamp check.
2. **Data Drift Detection**:
   - Implement a basic drift detection method in `src/data/make_dataset.py` or a new script to compare statistical properties (e.g., mean, variance) of new data against the training data, triggering a retraining flag.
3. **Dynamic Model Adjustment**:
   - Update `src/models/train_model.py` to incorporate new data incrementally or retrain with updated datasets when drift is detected.

---

### **Implementation**

#### **1. Periodic Retraining in `src/models/train_model.py`**
We will add a timestamp-based check to determine if retraining is needed, configurable via `config.yml`.

```python
import sys
import os
import logging
import datetime
import matplotlib.pyplot as plt
import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Import modular scripts
from src.data.make_dataset import make_dataset
from src.models.data_processing import load_data, generate_summaries, generate_embeddings, prepare_features
from src.models.train_model import split_data, train_model, evaluate_model, interpret_model
from src.models.predict_model import save_model_and_metadata, load_model_and_assets, predict_new_data
from src.config import EmbeddingFactory, SummaryFactory

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Script execution started at %s", datetime.datetime.now().strftime("%Y-%m-d %H:%M:%S"))

# Define Paths Concisely
PATHS = {
    "config": os.path.join(project_root, "config.yml"),
    "raw_data": os.path.join(project_root, "data", "raw", "customer_churn.csv"),
    "cleaned_data": os.path.join(project_root, "data", "processed", "cleaned_customer_churn.csv"),
    "summaries": os.path.join(project_root, "data", "interim", "customer_churn_summary.csv"),
    "embeddings": os.path.join(project_root, "data", "processed", "customer_churn_summary_embeddings.csv"),
    "models": os.path.join(project_root, "models"),
    "roc_fig": os.path.join(project_root, "reports", "figures", "roc_curve.png"),
    "feat_fig": os.path.join(project_root, "reports", "figures", "feature_importance.png"),
    "last_train_timestamp": os.path.join(project_root, "models", "last_train_timestamp.txt"),
}

for path in ["summaries", "embeddings", "models", "roc_fig", "feat_fig"]:
    os.makedirs(os.path.dirname(PATHS[path]), exist_ok=True)

# Main execution
def should_retrain(config, last_train_path):
    """Check if retraining is needed based on time interval."""
    try:
        if os.path.exists(last_train_path):
            with open(last_train_path, "r") as f:
                last_train_time = float(f.read())
            retrain_interval = config.get("retrain_interval_days", 7) * 86400  # Default 7 days in seconds
            if time.time() - last_train_time > retrain_interval:
                logger.info("Retraining triggered due to time interval.")
                return True
        return False
    except Exception as e:
        logger.error(f"Error checking retrain condition: {e}")
        return True  # Force retrain on error to ensure model stays current

if __name__ == "__main__":
    try:
        # Load config
        with open(PATHS["config"], "r") as file:
            config = yaml.safe_load(file)

        # Run data validation
        make_dataset(PATHS["raw_data"], PATHS["config"], PATHS["cleaned_data"])

        # Initialize factories
        provider = config["embedding"]["embedding_model"]
        llm = EmbeddingFactory(provider)
        summary_provider = config["summary"]["summary_provider"]
        summary_llm = SummaryFactory(summary_provider)

        # Check if retraining is needed
        if should_retrain(config, PATHS["last_train_timestamp"]):
            # Data Processing
            df, _ = load_data(PATHS["config"], PATHS["cleaned_data"])
            df = generate_summaries(df, summary_llm, PATHS["summaries"], config["data"].get("batch_size", 10), config["data"].get("num_threads", 4))
            df = generate_embeddings(df, llm, PATHS["embeddings"], config["data"].get("num_threads", 4))
            X_df, y = prepare_features(df, PATHS["models"])

            # Model Training and Evaluation
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_df, y, config["data"], config["model"])
            if config["model"]["scale_features"]:
                scaler = StandardScaler()
                X_train[["age", "tenure", "spend_rate"]] = scaler.fit_transform(X_train[["age", "tenure", "spend_rate"]])
                X_val[["age", "tenure", "spend_rate"]] = scaler.transform(X_val[["age", "tenure", "spend_rate"]])
                X_test[["age", "tenure", "spend_rate"]] = scaler.transform(X_test[["age", "tenure", "spend_rate"]])
                joblib.dump(scaler, os.path.join(PATHS["models"], "scaler.pkl"))
            model = train_model(X_train, X_val, y_train, y_val, config["model"])
            evaluate_model(model, X_test, y_test)
            interpret_model(model, X_train)

            # Model Deployment
            save_model_and_metadata(model, X_train, config, PATHS["models"], config["model"].get("save_format", "json"))
            # Update last train timestamp
            with open(PATHS["last_train_timestamp"], "w") as f:
                f.write(str(time.time()))

        # Load model and assets for inference
        model, scaler, feature_names = load_model_and_assets(PATHS["models"])

        # Batch inference example
        new_data_records = [
            {"ticket_notes": "Customer reported a billing issue that was resolved quickly.", "age": 50, "tenure": 10, "spend_rate": 100, "plan_type_encoded": 1},
            {"ticket_notes": "Customer asked about discounts, no major dissatisfaction.", "age": 30, "tenure": 7, "spend_rate": 85, "plan_type_encoded": 0}
        ]
        predictions = predict_new_data(new_data_records, summary_llm, llm, scaler, feature_names, PATHS["models"])
        logger.info(f"Batch predictions for new data: {predictions}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
```

**Changes**:
- Added `should_retrain` function to check if the time since the last training exceeds a configurable interval (default 7 days).
- Updated `PATHS` to include `last_train_timestamp` to store the last training time.
- Added conditional retraining logic, updating the timestamp after successful training.

---

#### **2. Data Drift Detection in `src/data/make_dataset.py`**
We will add a drift detection function to compare new data statistics with training data statistics, setting a flag for retraining.

```python
import os
import logging
import pandas as pd
import numpy as np
import yaml

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from config.yml."""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise

def validate_data(df, expected_columns):
    """Validate the dataset for missing values, data types, and integrity."""
    try:
        # Check for missing columns
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values detected:\n{missing_values[missing_values > 0]}")
            df = df.dropna()
            logger.info("Dropped rows with missing values.")

        # Validate data types and constraints
        if not pd.to_numeric(df["age"], errors="coerce").notnull().all():
            logger.error("Invalid 'age' values: must be numeric.")
            raise ValueError("Invalid 'age' values: must be numeric.")
        if (df["age"] < 0).any():
            logger.error("Invalid 'age' values: must be non-negative.")
            raise ValueError("Invalid 'age' values: must be non-negative.")

        if not pd.to_numeric(df["tenure"], errors="coerce").notnull().all():
            logger.error("Invalid 'tenure' values: must be numeric.")
            raise ValueError("Invalid 'tenure' values: must be numeric.")
        if (df["tenure"] < 0).any():
            logger.error("Invalid 'tenure' values: must be non-negative.")
            raise ValueError("Invalid 'tenure' values: must be non-negative.")

        if not pd.to_numeric(df["spend_rate"], errors="coerce").notnull().all():
            logger.error("Invalid 'spend_rate' values: must be numeric.")
            raise ValueError("Invalid 'spend_rate' values: must be numeric.")
        if (df["spend_rate"] < 0).any():
            logger.error("Invalid 'spend_rate' values: must be non-negative.")
            raise ValueError("Invalid 'spend_rate' values: must be non-negative.")

        if not df["plan_type"].apply(lambda x: isinstance(x, str)).all():
            logger.error("Invalid 'plan_type' values: must be strings.")
            raise ValueError("Invalid 'plan_type' values: must be strings.")

        if not df["churn"].isin([0, 1]).all():
            logger.error("Invalid 'churn' values: must be 0 or 1.")
            raise ValueError("Invalid 'churn' values: must be 0 or 1.")

        if not df["ticket_notes"].apply(lambda x: isinstance(x, str)).all():
            logger.error("Invalid 'ticket_notes' values: must be strings.")
            raise ValueError("Invalid 'ticket_notes' values: must be strings.")

        logger.info("Data validation passed successfully.")
        return df
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise

def detect_data_drift(new_data, training_stats_path):
    """Detect drift by comparing new data statistics with training data."""
    try:
        if os.path.exists(training_stats_path):
            with open(training_stats_path, "r") as f:
                training_stats = yaml.safe_load(f)
        else:
            logger.warning("No training stats found, skipping drift detection.")
            return False

        new_stats = {
            "age_mean": new_data["age"].mean(),
            "age_std": new_data["age"].std(),
            "tenure_mean": new_data["tenure"].mean(),
            "tenure_std": new_data["tenure"].std(),
            "spend_rate_mean": new_data["spend_rate"].mean(),
            "spend_rate_std": new_data["spend_rate"].std(),
        }

        drift_detected = False
        threshold = 0.1  # 10% change threshold
        for key in training_stats:
            if abs(new_stats[key] - training_stats[key]) / training_stats[key] > threshold:
                logger.warning(f"Drift detected in {key}: {new_stats[key]} vs {training_stats[key]}")
                drift_detected = True

        if drift_detected:
            logger.info("Significant data drift detected, retraining recommended.")
        else:
            logger.info("No significant data drift detected.")
        return drift_detected
    except Exception as e:
        logger.error(f"Error detecting data drift: {e}")
        return False

def make_dataset(raw_data_path, config_path, output_path, training_stats_path, chunk_size=1000):
    """Load, validate, and save the dataset in chunks for scalability, with drift detection."""
    try:
        # Load config
        config = load_config(config_path)
        logger.info("Configuration loaded successfully.")

        # Define expected columns
        expected_columns = ["customer_id", "age", "tenure", "spend_rate", "plan_type", "churn", "ticket_notes"]

        # Process data in chunks
        first_chunk = True
        all_chunks = []
        for chunk in pd.read_csv(raw_data_path, chunksize=chunk_size):
            logger.info(f"Processing chunk with {len(chunk)} rows...")
            validated_chunk = validate_data(chunk, expected_columns)
            all_chunks.append(validated_chunk)

        df = pd.concat(all_chunks, ignore_index=True)

        # Detect data drift
        drift_detected = detect_data_drift(df[["age", "tenure", "spend_rate"]], training_stats_path)
        if drift_detected:
            config["data"]["retrain_needed"] = True
            with open(config_path, "w") as file:
                yaml.dump(config, file)

        # Save training stats if not present
        if not os.path.exists(training_stats_path):
            stats = {
                "age_mean": df["age"].mean(),
                "age_std": df["age"].std(),
                "tenure_mean": df["tenure"].mean(),
                "tenure_std": df["tenure"].std(),
                "spend_rate_mean": df["spend_rate"].mean(),
                "spend_rate_std": df["spend_rate"].std(),
            }
            with open(training_stats_path, "w") as f:
                yaml.dump(stats, f)
            logger.info("Initial training statistics saved.")

        # Save cleaned dataset
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Cleaned dataset saved to {output_path}")
        return df
    except Exception as e:
        logger.error(f"Error in make_dataset: {e}")
        raise

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    raw_data_path = os.path.join(project_root, "data", "raw", "customer_churn.csv")
    config_path = os.path.join(project_root, "config.yml")
    output_path = os.path.join(project_root, "data", "processed", "cleaned_customer_churn.csv")
    training_stats_path = os.path.join(project_root, "data", "processed", "training_stats.yml")
    make_dataset(raw_data_path, config_path, output_path, training_stats_path)
```

**Changes**:
- Added `detect_data_drift` to compare new data statistics with stored training statistics.
- Updated `make_dataset` to call `detect_data_drift` and update `config.yml` with a `retrain_needed` flag if drift is detected.
- Added `training_stats_path` to store initial statistics and compare against new data.

---

#### **3. Dynamic Model Adjustment in `src/models/model_training.py`**
We will modify `train_model` to handle retraining based on the `retrain_needed` flag.

```python
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_data(X_df, y, data_config, model_config):
    try:
        stratify = y if data_config["stratify"] else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_df, y, test_size=data_config["test_size"], stratify=stratify, random_state=model_config["random_state"]
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.25, stratify=y_train_full if data_config["stratify"] else None, random_state=model_config["random_state"]
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    except Exception as e:
        raise Exception(f"Error splitting data: {e}")

def train_model(X_train, X_val, y_train, y_val, model_config, retrain=False):
    try:
        model_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": model_config["learning_rate"],
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": model_config["random_state"],
        }
        model = XGBClassifier(**model_params)
        eval_set = [(X_val, y_val)]

        if retrain:
            logger.info("Performing full retraining due to data drift or time interval.")
            model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=model_config["early_stopping_rounds"], verbose=True)
        else:
            logger.info("Performing incremental training.")
            model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=model_config["early_stopping_rounds"], verbose=True, xgb_model=model)  # Incremental update
        return model
    except TypeError as e:
        logger.error(f"Early stopping failed: {e}")
        logger.info("Falling back to training without early stopping.")
        model = XGBClassifier(**model_params)
        model.fit(X_train, y_train, verbose=True)
        return model
    except Exception as e:
        raise Exception(f"Error training XGBoost model: {e}")

def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC AUC Score: {roc_auc:.4f}")

        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, color="blue", label=f"ROC AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig("reports/figures/roc_curve.png")
        plt.show()
        return roc_auc
    except Exception as e:
        raise Exception(f"Error evaluating model: {e}")

def interpret_model(model, X_train):
    try:
        booster = model.get_booster()
        plt.figure(figsize=(12, 6))
        xgb.plot_importance(booster)
        plt.savefig("reports/figures/feature_importance.png")
        plt.tight_layout()
        plt.show()

        importances = model.feature_importances_
        feature_names = X_train.columns
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)
        return importance_df.head(10)
    except Exception as e:
        raise Exception(f"Error interpreting model: {e}")
```

**Changes**:
- Added `retrain` parameter to `train_model` to switch between full retraining and incremental training based on drift or time conditions.

---

### **`config.yml` Update**
Ensure `config.yml` includes adaptability settings:
```yaml
data:
  batch_size: 10
  num_threads: 4
  stratify: true
  test_size: 0.2
  retrain_interval_days: 7
model:
  scale_features: true
  learning_rate: 0.1
  early_stopping_rounds: 10
  random_state: 42
  save_format: json
embedding:
  embedding_model: huggingface
summary:
  summary_provider: huggingface
```

---

### **Validation**
- Run the pipeline:
  ```bash
  python src/models/main.py
  ```
- Check logs for retraining messages (e.g., "Retraining triggered due to time interval") or drift detection (e.g., "Significant data drift detected").
- Simulate drift by modifying `customer_churn.csv` (e.g., increasing `age` or `spend_rate` values significantly) and rerun to trigger retraining.

---

### **Adaptability Benefits**
1. **Periodic Retraining**: Ensures the model adapts to new data over time.
2. **Data Drift Detection**: Identifies shifts in data distribution, prompting retraining when needed.
3. **Dynamic Adjustment**: Supports incremental or full retraining based on conditions, maintaining model relevance.
