import logging
import numpy as np
import pandas as pd
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
            X_df,
            y,
            test_size=data_config["test_size"],
            stratify=stratify,
            random_state=model_config["random_state"],
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.25,
            stratify=y_train_full if data_config["stratify"] else None,
            random_state=model_config["random_state"],
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    except Exception as e:
        raise Exception(f"Error splitting data: {e}")


def train_model(X_train, X_val, y_train, y_val, model_config):
    try:
        # Calculate scale_pos_weight to handle class imbalance
        neg_samples = sum(y_train == 0)
        pos_samples = sum(y_train == 1)
        scale_pos_weight = neg_samples / pos_samples if pos_samples > 0 else 1
        logger.info(
            f"Class imbalance: {neg_samples} non-churn, {pos_samples} churn. Setting scale_pos_weight to {scale_pos_weight:.2f}"
        )

        model_params = {
            "max_depth": 6,
            "learning_rate": model_config["learning_rate"],
            "eval_metric": "logloss",
            "random_state": model_config["random_state"],
            "scale_pos_weight": scale_pos_weight,
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        eval_list = [(dtrain, "train"), (dval, "eval")]
        evals_result = {}
        try:
            bst = xgb.train(
                model_params,
                dtrain,
                num_boost_round=100,
                evals=eval_list,
                early_stopping_rounds=model_config["early_stopping_rounds"],
                evals_result=evals_result,
                verbose_eval=True,
            )
        except TypeError as e:
            logger.error(f"Early stopping failed: {e}")
            logger.info("Falling back to training without early stopping.")
            bst = xgb.train(
                model_params,
                dtrain,
                num_boost_round=100,
                evals=eval_list,
                evals_result=evals_result,
                verbose_eval=True,
            )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    model = XGBClassifier(**model_params)
    model._Booster = bst
    return model


def evaluate_model(model, X_test, y_test):
    try:
        # Use the Booster directly for prediction to avoid n_classes_ issue
        booster = model.get_booster()
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = booster.predict(dtest)
        y_pred = (y_pred_proba >= 0.5).astype(
            int
        )  # Convert probabilities to binary predictions

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
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        ).sort_values("Importance", ascending=False)
        return importance_df.head(10)
    except Exception as e:
        raise Exception(f"Error interpreting model: {e}")
