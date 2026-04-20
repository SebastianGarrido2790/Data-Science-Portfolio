import sys
import os
import logging
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modular scripts
from src.data.make_dataset import make_dataset
from src.models.data_processing import (
    load_data,
    generate_summaries,
    generate_embeddings,
    prepare_features,
)
from src.models.model_training import (
    split_data,
    train_model,
    evaluate_model,
    interpret_model,
)
from src.models.model_deployment import (
    save_model_and_metadata,
    load_model_and_assets,
    predict_new_data,
)
from src.config.factories import EmbeddingFactory, SummaryFactory

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(
    "Script execution started at %s",
    datetime.datetime.now().strftime("%Y-%m-d %H:%M:%S"),
)

# Define Paths Concisely
PATHS = {
    "raw_data": os.path.join(project_root, "data", "raw", "customer_churn.csv"),
    "cleaned_data": os.path.join(
        project_root, "data", "processed", "cleaned_customer_churn.csv"
    ),
    "summaries": os.path.join(
        project_root, "data", "interim", "customer_churn_summary.csv"
    ),
    "embeddings": os.path.join(
        project_root, "data", "processed", "customer_churn_summary_embeddings.csv"
    ),
    "models": os.path.join(project_root, "models"),
    "roc_fig": os.path.join(project_root, "reports", "figures", "roc_curve.png"),
    "feat_fig": os.path.join(
        project_root, "reports", "figures", "feature_importance.png"
    ),
}

for path in ["summaries", "embeddings", "models", "roc_fig", "feat_fig"]:
    os.makedirs(os.path.dirname(PATHS[path]), exist_ok=True)

# Hardcoded configuration
CONFIG = {
    "embedding": {"embedding_model": "huggingface"},
    "summary": {"summary_provider": "huggingface"},
    "data": {"batch_size": 10, "test_size": 0.2, "stratify": True},
    "model": {
        "scale_features": True,
        "early_stopping_rounds": 10,
        "learning_rate": 0.1,
        "random_state": 42,
        "save_format": "both",
    },
}

# Main execution
if __name__ == "__main__":
    try:
        # Run data validation
        make_dataset(PATHS["raw_data"], PATHS["cleaned_data"])

        # Initialize factories
        provider = CONFIG["embedding"]["embedding_model"]
        llm = EmbeddingFactory(provider)
        summary_provider = CONFIG["summary"]["summary_provider"]
        summary_llm = SummaryFactory(summary_provider)

        # Data Processing
        df = load_data(PATHS["cleaned_data"])
        df = generate_summaries(
            df, summary_llm, PATHS["summaries"], CONFIG["data"]["batch_size"]
        )
        df = generate_embeddings(df, llm, PATHS["embeddings"])
        X_df, y = prepare_features(df, PATHS["models"])

        # Model Training and Evaluation
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X_df, y, CONFIG["data"], CONFIG["model"]
        )
        if CONFIG["model"]["scale_features"]:
            scaler = StandardScaler()
            X_train[["age", "tenure", "spend_rate"]] = scaler.fit_transform(
                X_train[["age", "tenure", "spend_rate"]]
            )
            X_val[["age", "tenure", "spend_rate"]] = scaler.transform(
                X_val[["age", "tenure", "spend_rate"]]
            )
            X_test[["age", "tenure", "spend_rate"]] = scaler.transform(
                X_test[["age", "tenure", "spend_rate"]]
            )
            joblib.dump(scaler, os.path.join(PATHS["models"], "scaler.pkl"))
        model = train_model(X_train, X_val, y_train, y_val, CONFIG["model"])
        evaluate_model(model, X_test, y_test)
        interpret_model(model, X_train)

        # Model Deployment
        save_model_and_metadata(
            model, X_train, PATHS["models"], CONFIG["model"]["save_format"]
        )  # Adjusted call
        model, scaler, feature_names = load_model_and_assets(PATHS["models"])

        # Batch inference example
        new_data_records = [
            {
                "ticket_notes": "Customer reported a billing issue that was resolved quickly.",
                "age": 50,
                "tenure": 10,
                "spend_rate": 100,
                "plan_type_encoded": 1,
            },
            {
                "ticket_notes": "Customer asked about discounts, no major dissatisfaction.",
                "age": 30,
                "tenure": 7,
                "spend_rate": 85,
                "plan_type_encoded": 0,
            },
        ]
        predictions = predict_new_data(
            new_data_records, summary_llm, llm, scaler, feature_names, PATHS["models"]
        )
        logger.info(f"Batch predictions for new data: {predictions}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
