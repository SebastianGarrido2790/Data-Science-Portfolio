## **Enhancing Scalability in the Churn Prediction Project**

To improve scalability, we will focus on optimizing the pipeline to handle larger datasets and higher inference demands, aligning with production requirements (e.g., fast inference, low latency). Key areas include parallelizing summarization and embedding generation, optimizing data processing, and preparing the model for high-throughput inference.

---

### **Scalability Enhancement Plan**
1. **Parallelize Summarization and Embedding**:
   - Use `multiprocessing` or `concurrent.futures` to parallelize the summarization and embedding steps in `src/models/data_processing.py`, which currently process batches sequentially.
2. **Optimize Data Processing**:
   - Modify `src/data/make_dataset.py` to handle large datasets efficiently using chunking with `pandas`.
3. **Enable High-Throughput Inference**:
   - Update `src/models/predict_model.py` to support batch inference for multiple customer records, reducing latency for production use.

---

### **Implementation**

#### **1. Parallelize Summarization and Embedding in `src/models/data_processing.py`**
We will use `concurrent.futures.ThreadPoolExecutor` to parallelize the summarization and embedding tasks, leveraging multiple threads for faster processing.

```python
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import ast
from src.config import EmbeddingFactory, SummaryFactory
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(config_path, cleaned_data_path):
    try:
        with open(config_path, "r") as file:
            config = {"data": {"batch_size": 10, "num_threads": 4}, "model": {"scale_features": True}}
        df = pd.read_csv(cleaned_data_path)
        return df, config
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Cleaned data or config file not found: {e}")
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"Cleaned data file is empty: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error loading dataset: {e}")

def summarize_batch(batch, summary_llm):
    """Helper function to summarize a batch of ticket notes."""
    return [summary_llm.summarize(text) for text in batch]

def generate_summaries(df, summary_llm, summaries_path, batch_size=10, num_threads=4):
    try:
        if os.path.exists(summaries_path):
            logger.info("Loading existing summaries...")
            df = pd.read_csv(summaries_path)
        else:
            logger.info("Generating ticket summaries in parallel (this may take time)...")
            summaries = []
            batches = [df["ticket_notes"][i : i + batch_size].tolist() for i in range(0, len(df), batch_size)]
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                batch_summaries = list(executor.map(lambda batch: summarize_batch(batch, summary_llm), batches))
            for batch in batch_summaries:
                summaries.extend(batch)
            df["ticket_summary"] = summaries
            df.to_csv(summaries_path, index=False)
            logger.info(f"Summaries saved to {summaries_path}")
        return df
    except Exception as e:
        logger.error(f"Error generating or loading summaries: {e}")
        raise

def get_embedding(text, llm):
    """Helper function to generate embedding for a single text."""
    return llm.get_embeddings(text)

def generate_embeddings(df, llm, embeddings_path, num_threads=4):
    try:
        if os.path.exists(embeddings_path):
            logger.info("Loading existing embeddings...")
            df = pd.read_csv(embeddings_path)
            df["summary_embedding"] = df["summary_embedding"].apply(ast.literal_eval)
        else:
            logger.info("Generating embeddings in parallel (this may take time)...")
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                embeddings = list(executor.map(lambda text: get_embedding(text, llm), df["ticket_summary"]))
            df["summary_embedding"] = embeddings
            df = df[df["summary_embedding"].notnull()]
            df.to_csv(embeddings_path, index=False)
            logger.info(f"Embeddings saved to {embeddings_path}")
        return df
    except Exception as e:
        logger.error(f"Error generating or loading embeddings: {e}")
        raise

def prepare_features(df, models_path):
    try:
        le = LabelEncoder()
        df["plan_type_encoded"] = le.fit_transform(df["plan_type"])
        joblib.dump(le, os.path.join(models_path, "label_encoder.pkl"))

        embeddings_df = pd.DataFrame(df["summary_embedding"].tolist(), index=df.index)
        X_df = pd.concat([df[["age", "tenure", "spend_rate", "plan_type_encoded"]], embeddings_df], axis=1)
        X_df.columns = X_df.columns.astype(str)
        y = df["churn"].values

        with open(os.path.join(models_path, "feature_names.txt"), "w") as f:
            for feature in X_df.columns:
                f.write(f"{feature}\n")
        return X_df, y
    except Exception as e:
        raise Exception(f"Error preparing features: {e}")
```

**Changes**:
- Added `num_threads` parameter (default 4) to control parallelization, configurable via `config.yml`.
- Used `ThreadPoolExecutor` to parallelize `summarize_batch` and `get_embedding` tasks.
- Updated `config` in `load_data` to include `num_threads` for demonstration (in a real scenario, this should come from `config.yml`).

---

#### **2. Optimize Data Processing in `src/data/make_dataset.py`**
We will modify `make_dataset.py` to use chunking for large datasets, processing the CSV file in smaller chunks to reduce memory usage.

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
            df = df.dropna()  # Drop rows with missing values
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

def make_dataset(raw_data_path, config_path, output_path, chunk_size=1000):
    """Load, validate, and save the dataset in chunks for scalability."""
    try:
        # Load config
        config = load_config(config_path)
        logger.info("Configuration loaded successfully.")

        # Define expected columns
        expected_columns = ["customer_id", "age", "tenure", "spend_rate", "plan_type", "churn", "ticket_notes"]

        # Process data in chunks
        first_chunk = True
        for chunk in pd.read_csv(raw_data_path, chunksize=chunk_size):
            logger.info(f"Processing chunk with {len(chunk)} rows...")
            validated_chunk = validate_data(chunk, expected_columns)
            
            # Save to file (append mode after first chunk)
            mode = "w" if first_chunk else "a"
            header = first_chunk
            validated_chunk.to_csv(output_path, mode=mode, header=header, index=False)
            first_chunk = False

        logger.info(f"Cleaned dataset saved to {output_path}")
        return pd.read_csv(output_path)  # Return the full cleaned dataset
    except Exception as e:
        logger.error(f"Error in make_dataset: {e}")
        raise

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    raw_data_path = os.path.join(project_root, "data", "raw", "customer_churn.csv")
    config_path = os.path.join(project_root, "config.yml")
    output_path = os.path.join(project_root, "data", "processed", "cleaned_customer_churn.csv")
    make_dataset(raw_data_path, config_path, output_path)
```

**Changes**:
- Added `chunk_size` parameter (default 1000) to process the CSV file in chunks.
- Used `pd.read_csv(chunksize=...)` to read the file incrementally, validating and saving each chunk.
- Appends chunks to the output file to avoid memory overload.

---

#### **3. Enable High-Throughput Inference in `src/models/predict_model.py`**
We will update `predict_new_data` to support batch inference, allowing multiple customer records to be processed simultaneously.

```python
import os
import json
import pandas as pd
from xgboost import XGBClassifier
import joblib
from src.config.factories import EmbeddingFactory, SummaryFactory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_model_and_metadata(model, X_train, config, models_path, save_format):
    try:
        if save_format not in ["json", "pickle", "both"]:
            save_format = "json"
        if save_format in ["json", "both"]:
            model.save_model(os.path.join(models_path, "xgb_churn_model.json"))
        if save_format in ["pickle", "both"]:
            joblib.dump(model, os.path.join(models_path, "xgb_churn_model.pkl"))

        metadata = {
            "model_save_format": save_format,
            "feature_names": list(X_train.columns),
            "model_params": model.get_params(),
            "config": config,
            "embedding_dim": len(X_train.columns) - 4,
        }
        with open(os.path.join(models_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        return True
    except Exception as e:
        raise Exception(f"Error saving model and metadata: {e}")

def load_model_and_assets(models_path):
    try:
        model = XGBClassifier()
        model.load_model(os.path.join(models_path, "xgb_churn_model.json"))
        scaler = joblib.load(os.path.join(models_path, "scaler.pkl"))
        with open(os.path.join(models_path, "feature_names.txt"), "r") as f:
            feature_names = [line.strip() for line in f]
        return model, scaler, feature_names
    except Exception as e:
        raise Exception(f"Error loading model and assets: {e}")

def predict_new_data(new_data_records, summary_llm, llm, scaler, feature_names, models_path):
    """
    Predict churn for a batch of new data records.
    
    Args:
        new_data_records (list): List of dicts with keys 'ticket_notes', 'age', 'tenure', 'spend_rate', 'plan_type_encoded'.
        summary_llm: Summary LLM instance.
        llm: Embedding LLM instance.
        scaler: Scaler instance.
        feature_names: List of feature names.
        models_path: Path to models directory.
    
    Returns:
        list: Predicted churn values.
    """
    try:
        # Generate summaries and embeddings for all records
        summaries = [summary_llm.summarize(record["ticket_notes"]) for record in new_data_records]
        embeddings = [llm.get_embeddings(summary) for summary in summaries]

        # Prepare new data DataFrame
        new_data = pd.DataFrame([
            {
                "age": record["age"],
                "tenure": record["tenure"],
                "spend_rate": record["spend_rate"],
                "plan_type_encoded": record["plan_type_encoded"],
            }
            for record in new_data_records
        ])

        # Scale numeric features
        numeric_features = ["age", "tenure", "spend_rate"]
        new_data_scaled = new_data.copy()
        new_data_scaled[numeric_features] = scaler.transform(new_data[numeric_features])

        # Add embeddings as columns
        embedding_dim = len(embeddings[0])
        embedding_columns = [f"{i}" for i in range(embedding_dim)]
        embedding_df = pd.DataFrame(embeddings, columns=embedding_columns)
        new_data_final = pd.concat([new_data_scaled, embedding_df], axis=1)

        # Align with training feature names
        new_data_final = new_data_final[feature_names]

        # Load model and predict
        model = XGBClassifier()
        model.load_model(os.path.join(models_path, "xgb_churn_model.json"))
        predictions = modeltekst.predict(new_data_final)
        return predictions.tolist()
    except Exception as e:
        logger.error(f"Error during batch inference: {e}")
        raise
```

**Changes**:
- Updated `predict_new_data` to accept a list of records and process them in batch.
- Modified input processing to handle multiple records, performing summarization, embedding, and prediction in a single pass.

---

#### **4. Update `src/models/train_model.py` to Use the New Batch Inference**
Adjust the main script to call the updated `predict_new_data` with a batch of records.

```python
import sys
import os
import logging
import datetime
import matplotlib.pyplot as plt

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Import modular scripts
from src.data.make_dataset import make_dataset
from src.models.data_processing import load_data, generate_summaries, generate_embeddings, prepare_features
from src.models.model_training import split_data, train_model, evaluate_model, interpret_model
from src.models.model_deployment import save_model_and_metadata, load_model_and_assets, predict_new_data
from src.config.factories import EmbeddingFactory, SummaryFactory

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
}

for path in ["summaries", "embeddings", "models", "roc_fig", "feat_fig"]:
    os.makedirs(os.path.dirname(PATHS[path]), exist_ok=True)

# Main execution
if __name__ == "__main__":
    try:
        # Run data validation
        make_dataset(PATHS["raw_data"], PATHS["config"], PATHS["cleaned_data"])

        # Initialize factories
        provider = {"embedding_model": "huggingface"}  # Placeholder, replace with config load if needed
        llm = EmbeddingFactory(provider["embedding_model"])
        summary_provider = {"summary_provider": "huggingface"}  # Placeholder
        summary_llm = SummaryFactory(summary_provider["summary_provider"])

        # Data Processing
        df, config = load_data(PATHS["config"], PATHS["cleaned_data"])
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
- Updated the inference call to use a batch of records with the new `predict_new_data` function.
- Passed `num_threads` to `generate_summaries` and `generate_embeddings` for parallel processing.

---

### **Validation**
- Run the pipeline:
  ```bash
  python src/models/main.py
  ```
- Check logs for parallel processing messages (e.g., "Generating ticket summaries in parallel...").
- Verify that batch inference works by checking the predictions for multiple records.

---

### **Scalability Benefits**
1. **Parallel Processing**: Summarization and embedding tasks are now parallelized, reducing processing time for large datasets.
2. **Chunked Data Loading**: `make_dataset.py` processes data in chunks, minimizing memory usage for large files.
3. **Batch Inference**: `predict_new_data` supports multiple records, improving inference throughput and reducing latency in production.
