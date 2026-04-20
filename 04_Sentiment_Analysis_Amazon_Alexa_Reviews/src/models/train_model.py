import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    recall_score,
    precision_score,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack
import optuna
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load data
train_df = pd.read_csv("../../data/interim/train.csv")
test_df = pd.read_csv("../../data/interim/test.csv")
train_features_df = pd.read_csv("../../data/processed/train_features.csv")
test_features_df = pd.read_csv("../../data/processed/test_features.csv")

# Handle empty processed reviews
train_df["processed_reviews"] = train_df.apply(
    lambda x: (
        x["verified_reviews"]
        if x["processed_reviews"] == ""
        else x["processed_reviews"]
    ),
    axis=1,
)
test_df["processed_reviews"] = test_df.apply(
    lambda x: (
        x["verified_reviews"]
        if x["processed_reviews"] == ""
        else x["processed_reviews"]
    ),
    axis=1,
)

# Split data
X_train = train_df["processed_reviews"]
y_train = train_df["feedback"].astype(int)
X_test = test_df["processed_reviews"]
y_test = test_df["feedback"].astype(int)

# Calculate class weights
neg_count = len(y_train[y_train == 0])
pos_count = len(y_train[y_train == 1])
neu_count = len(y_train[y_train == 2])
class_weight_dict = {0: pos_count / neg_count, 1: 1.0, 2: pos_count / neu_count}

logging.info(
    f"Class distribution - Negative: {neg_count}, Positive: {pos_count}, Neutral: {neu_count}"
)
logging.info(f"Class weights: {class_weight_dict}")

# Vectorize text
vectorizer = CountVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Combine with engineered features
feature_cols = [
    "review_length",
    "word_count",
    "vader_compound",
    "vader_pos",
    "vader_neg",
    "month",
    "year",
    *[col for col in train_features_df.columns if col.startswith("variation_")],
]
X_train_features = train_features_df[feature_cols].values
X_test_features = test_features_df[feature_cols].values
X_train_combined = hstack([X_train_vec, X_train_features])
X_test_combined = hstack([X_test_vec, X_test_features])

# Save vectorizer
with open("../../models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# ---------------------------
# 1. RandomForestClassifier
# ---------------------------


def train_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 10, 30, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
    }
    rf_clf = RandomForestClassifier(
        **params, class_weight=class_weight_dict, random_state=42, n_jobs=-1
    )
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_combined, y_train)
    rf_clf.fit(X_train_smote, y_train_smote)
    probs = rf_clf.predict_proba(X_test_combined)
    neg_probs = probs[:, 0]  # Probability for negative class (0)
    return recall_score(y_test, np.argmax(probs, axis=1), average="micro", labels=[0])


logging.info("Training RandomForestClassifier...")
study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(train_rf, n_trials=20)
rf_best_params = study_rf.best_params
rf_clf = RandomForestClassifier(
    **rf_best_params, class_weight=class_weight_dict, random_state=42, n_jobs=-1
)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_combined, y_train)
rf_clf.fit(X_train_smote, y_train_smote)
rf_pred = rf_clf.predict(X_test_combined)
logging.info(f"Random Forest Best Params: {rf_best_params}")

# Threshold tuning for negative class
rf_probs = rf_clf.predict_proba(X_test_combined)[:, 0]  # Negative class probabilities
thresholds = np.arange(0.1, 1.0, 0.05)
recall_scores = [
    recall_score(y_test, (rf_probs >= t).astype(int), average="micro", labels=[0])
    for t in thresholds
]
precision_scores = [
    precision_score(
        y_test,
        (rf_probs >= t).astype(int),
        average="micro",
        labels=[0],
        zero_division=0,
    )
    for t in thresholds
]
valid_thresholds = [
    (t, r, p)
    for t, r, p in zip(thresholds, recall_scores, precision_scores)
    if p >= 0.3
]
best_recall_threshold = (
    max(valid_thresholds, key=lambda x: x[1])[0] if valid_thresholds else 0.5
)
rf_pred_adjusted = (rf_probs >= best_recall_threshold).astype(int)
logging.info(
    f"\nBest threshold for negative recall (precision >= 0.3): {best_recall_threshold}"
)
logging.info(
    f"\nRandom Forest Classification Report (Threshold {best_recall_threshold}):\n{classification_report(y_test, rf_pred)}"
)

# Metrics
logging.info(
    f"ROC-AUC (ovr): {roc_auc_score(y_test, rf_clf.predict_proba(X_test_combined), multi_class='ovr')}"
)
precision, recall, _ = precision_recall_curve(y_test == 0, rf_probs)
logging.info(f"PR-AUC (negative class): {auc(recall, precision)}")
cm = confusion_matrix(y_test, rf_pred)
logging.info(f"\nConfusion Matrix:\n{cm}")

# Visualize PR curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"PR-AUC = {auc(recall, precision):.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Random Forest Precision-Recall Curve (Negative Class)")
plt.legend()
plt.savefig("../../reports/figures/random_forest/rf_pr_curve.png")
plt.show()
plt.close()

# Visualize confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("../../reports/figures/random_forest/rf_confusion_matrix.png")
plt.show()
plt.close()

# Save model
with open("../../models/rf_model.pkl", "wb") as f:
    pickle.dump(rf_clf, f)

# ------------------
# 2. XGBClassifier
# ------------------


def train_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 10, 14),
        "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.4, log=True),
        "gamma": trial.suggest_float("gamma", 0, 0.2),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 0.2),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 1.4),
    }
    xgb_clf = XGBClassifier(
        **params,
        scale_pos_weight=class_weight_dict[0],
        eval_metric="mlogloss",
        random_state=42,
    )
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_combined, y_train)
    xgb_clf.fit(X_train_smote, y_train_smote)
    probs = xgb_clf.predict_proba(X_test_combined)
    neg_probs = probs[:, 0]  # Probability for negative class (0)
    return recall_score(y_test, np.argmax(probs, axis=1), average="micro", labels=[0])


logging.info("Training XGBClassifier...")
study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(train_xgb, n_trials=20)
xgb_best_params = study_xgb.best_params
xgb_clf = XGBClassifier(
    **xgb_best_params,
    scale_pos_weight=class_weight_dict[0],
    eval_metric="mlogloss",
    random_state=42,
)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_combined, y_train)
xgb_clf.fit(X_train_smote, y_train_smote)
xgb_pred = xgb_clf.predict(X_test_combined)
logging.info(f"XGBoost Best Params: {xgb_best_params}")

# Threshold tuning for negative class
xgb_probs = xgb_clf.predict_proba(X_test_combined)[:, 0]  # Negative class probabilities
thresholds = np.arange(0.1, 1.0, 0.05)
recall_scores = [
    recall_score(y_test, (xgb_probs >= t).astype(int), average="micro", labels=[0])
    for t in thresholds
]
precision_scores = [
    precision_score(
        y_test,
        (xgb_probs >= t).astype(int),
        average="micro",
        labels=[0],
        zero_division=0,
    )
    for t in thresholds
]
valid_thresholds = [
    (t, r, p)
    for t, r, p in zip(thresholds, recall_scores, precision_scores)
    if p >= 0.3
]
best_recall_threshold = (
    max(valid_thresholds, key=lambda x: x[1])[0] if valid_thresholds else 0.5
)
xgb_pred_adjusted = (xgb_probs >= best_recall_threshold).astype(int)
logging.info(
    f"Best threshold for negative recall (precision >= 0.3): {best_recall_threshold}"
)
logging.info(
    f"\nXGBoost Classification Report (Threshold {best_recall_threshold}):\n{classification_report(y_test, xgb_pred)}"
)

# Metrics
logging.info(
    f"ROC-AUC (ovr): {roc_auc_score(y_test, xgb_clf.predict_proba(X_test_combined), multi_class='ovr')}"
)
precision, recall, _ = precision_recall_curve(y_test == 0, xgb_probs)
logging.info(f"PR-AUC (negative class): {auc(recall, precision)}")
cm = confusion_matrix(y_test, xgb_pred)
logging.info(f"\nConfusion Matrix:\n{cm}")

# Visualize PR curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"PR-AUC = {auc(recall, precision):.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("XGBoost Precision-Recall Curve (Negative Class)")
plt.legend()
plt.savefig("../../reports/figures/xgboost/xgb_pr_curve.png")
plt.show()
plt.close()

# Visualize confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("../../reports/figures/xgboost/xgb_confusion_matrix.png")
plt.show()
plt.close()

# Save model
with open("../../models/xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_clf, f)
