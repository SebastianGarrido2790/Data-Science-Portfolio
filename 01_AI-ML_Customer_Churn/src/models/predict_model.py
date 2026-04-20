import os
import json
import pandas as pd
from xgboost import XGBClassifier
import joblib
from src.config.factories import EmbeddingFactory, SummaryFactory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_model_and_metadata(model, X_train, models_path, save_format="both"):
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


def predict_new_data(
    new_data_records, summary_llm, llm, scaler, feature_names, models_path
):
    try:
        summaries = [
            summary_llm.summarize(record["ticket_notes"]) for record in new_data_records
        ]
        embeddings = llm.get_embeddings(summaries)  # Batch processing

        new_data = pd.DataFrame(
            [
                {
                    "age": record["age"],
                    "tenure": record["tenure"],
                    "spend_rate": record["spend_rate"],
                    "plan_type_encoded": record["plan_type_encoded"],
                }
                for record in new_data_records
            ]
        )

        numeric_features = ["age", "tenure", "spend_rate"]
        new_data_scaled = new_data.copy()
        new_data_scaled[numeric_features] = scaler.transform(new_data[numeric_features])

        embedding_dim = len(embeddings[0])
        embedding_columns = [str(i) for i in range(embedding_dim)]
        embedding_df = pd.DataFrame(embeddings, columns=embedding_columns)
        new_data_final = pd.concat([new_data_scaled, embedding_df], axis=1)

        logger.info(
            f"new_data_final columns before reindex: {new_data_final.columns.tolist()}"
        )
        logger.info(f"Expected feature_names: {feature_names}")

        # Reindex to match feature_names, filling missing columns with 0
        missing_cols = [
            col for col in feature_names if col not in new_data_final.columns
        ]
        for col in missing_cols:
            new_data_final[col] = 0
        new_data_final = new_data_final[feature_names]

        logger.info(
            f"new_data_final columns after reindex: {new_data_final.columns.tolist()}"
        )

        model = XGBClassifier()
        model.load_model(os.path.join(models_path, "xgb_churn_model.json"))
        predictions = model.predict(new_data_final)
        return predictions.tolist()
    except Exception as e:
        logger.error(f"Error during batch inference: {e}")
        raise
