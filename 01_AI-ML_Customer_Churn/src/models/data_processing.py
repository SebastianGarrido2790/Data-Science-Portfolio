import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import ast
from src.config.factories import EmbeddingFactory, SummaryFactory


def load_data(cleaned_data_path):
    try:
        df = pd.read_csv(cleaned_data_path)
        return df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Cleaned data file not found: {e}")
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"Cleaned data file is empty: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error loading dataset: {e}")


def generate_summaries(df, summary_llm, summaries_path, batch_size=10):
    try:
        if os.path.exists(summaries_path):
            df = pd.read_csv(summaries_path)
        else:
            summaries = []
            for i in range(0, len(df), batch_size):
                batch = df["ticket_notes"][i : i + batch_size].tolist()
                batch_summaries = [summary_llm.summarize(text) for text in batch]
                summaries.extend(batch_summaries)
            df["ticket_summary"] = summaries
            df.to_csv(summaries_path, index=False)
        return df
    except Exception as e:
        raise Exception(f"Error generating or loading summaries: {e}")


def generate_embeddings(df, llm, embeddings_path):
    try:
        if os.path.exists(embeddings_path):
            df = pd.read_csv(embeddings_path)
            df["summary_embedding"] = df["summary_embedding"].apply(ast.literal_eval)
        else:
            df["summary_embedding"] = df["ticket_summary"].apply(llm.get_embeddings)
            df = df[df["summary_embedding"].notnull()]
            df.to_csv(embeddings_path, index=False)
        return df
    except Exception as e:
        raise Exception(f"Error generating or loading embeddings: {e}")


def prepare_features(df, models_path):
    try:
        le = LabelEncoder()
        df["plan_type_encoded"] = le.fit_transform(df["plan_type"])
        joblib.dump(le, os.path.join(models_path, "label_encoder.pkl"))

        embeddings_df = pd.DataFrame(df["summary_embedding"].tolist(), index=df.index)
        X_df = pd.concat(
            [df[["age", "tenure", "spend_rate", "plan_type_encoded"]], embeddings_df],
            axis=1,
        )
        X_df.columns = X_df.columns.astype(str)
        y = df["churn"].values

        with open(os.path.join(models_path, "feature_names.txt"), "w") as f:
            for feature in X_df.columns:
                f.write(f"{feature}\n")
        return X_df, y
    except Exception as e:
        raise Exception(f"Error preparing features: {e}")
