import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# Set MLflow experiment
mlflow.set_experiment("EmployeeAttritionExperiment")

# Load processed data
X_train = pd.read_csv("../../data/processed/X_train.csv")
X_test = pd.read_csv("../../data/processed/X_test.csv")
y_train = pd.read_csv("../../data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("../../data/processed/y_test.csv").values.ravel()

# Get feature names for logging
feature_names = X_train.columns.tolist()


# Define evaluation function with custom threshold
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, threshold=0.5):
    with mlflow.start_run(run_name=model_name):
        # Debug: Print data shape and feature names
        print(
            f"X_train shape: {X_train.shape}, Sample feature names: {feature_names[:5]}..."
        )

        # Fit model
        model.fit(X_train, y_train)

        # Predictions with custom threshold
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        # Metrics
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # Log metrics and parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("threshold", threshold)
        if "param_grid" in locals():
            mlflow.log_params(model.best_params_)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)

        # Log model with input example as numpy array
        input_example = X_test.iloc[:1].values
        if "XGB" in model_name:
            best_xgb_model = model.best_estimator_.named_steps["model"]
            mlflow.xgboost.log_model(
                best_xgb_model, f"model_{model_name}", input_example=input_example
            )
        else:
            mlflow.sklearn.log_model(
                model, f"model_{model_name}", input_example=input_example
            )

        # Register model
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model_{model_name}", model_name
        )

        # Save model locally
        joblib.dump(model, f"../../models/{model_name}.pkl")

        print(f"\n{model_name} Results (Threshold={threshold}):")
        print("Recall:", recall)
        print("F1-Score:", f1)
        print("AUC:", auc)
        print(classification_report(y_test, y_pred))


# SMOTE for class imbalance
smote = SMOTE(random_state=42, sampling_strategy=0.7)  # Adjusted for balance

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 1. Logistic Regression
lr_pipeline = ImbPipeline(
    [
        ("smote", SMOTE(random_state=42, sampling_strategy=0.7)),
        ("model", LogisticRegression(class_weight={0: 1, 1: 5}, random_state=42)),
    ]
)
param_grid = {"model__C": [0.1, 1, 10, 100]}
lr_grid = GridSearchCV(lr_pipeline, param_grid, cv=cv, scoring="recall", n_jobs=-1)
evaluate_model(
    lr_grid, X_train, X_test, y_train, y_test, "LogisticRegression", threshold=0.4
)

# 2. Random Forest
rf_pipeline = ImbPipeline(
    [
        ("smote", SMOTE(random_state=42, sampling_strategy=0.7)),
        ("model", RandomForestClassifier(class_weight={0: 1, 1: 5}, random_state=42)),
    ]
)
param_grid = {"model__n_estimators": [300, 400], "model__max_depth": [30, 40]}
rf_grid = GridSearchCV(rf_pipeline, param_grid, cv=cv, scoring="recall", n_jobs=-1)
evaluate_model(rf_grid, X_train, X_test, y_train, y_test, "RandomForest", threshold=0.4)

# 3. XGBoost
xgb_pipeline = ImbPipeline(
    [
        ("smote", SMOTE(random_state=42, sampling_strategy=0.7)),
        (
            "model",
            XGBClassifier(
                scale_pos_weight=len(y_train[y_train == 0])
                / len(y_train[y_train == 1])
                * 3.0,  # Increased weight
                random_state=42,
                n_estimators=400,
                max_depth=8,
            ),
        ),
    ]
)
param_grid = {
    "model__n_estimators": [400, 600],
    "model__max_depth": [8, 12],
    "model__learning_rate": [0.01, 0.1],
    "model__subsample": [0.7, 0.9],
    "model__colsample_bytree": [0.7, 0.9],
}
xgb_grid = GridSearchCV(xgb_pipeline, param_grid, cv=cv, scoring="recall", n_jobs=-1)
evaluate_model(
    xgb_grid, X_train, X_test, y_train, y_test, "XGBoost", threshold=0.3
)  # Lower threshold to boost recall

# Dictionary of model names and their corresponding GridSearchCV objects
models = {"LogisticRegression": lr_grid, "RandomForest": rf_grid, "XGBoost": xgb_grid}

# SHAP explanation for each best model
for name, grid in models.items():
    print(f"\nGenerating SHAP plot for {name}...")

    best_pipeline = grid.best_estimator_
    best_model = best_pipeline.named_steps["model"]

    # Create SHAP explainer depending on the model type
    if name == "LogisticRegression":
        masker = shap.maskers.Independent(X_test)
        explainer = shap.Explainer(best_model, masker)
        shap_values = explainer(X_test)
    else:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)

    # Generate SHAP summary bar plot
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f"SHAP Summary: {name}")
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/shap_summary_{name}.png")
    plt.show()
    plt.close()

# Fairness Metrics
y_pred = xgb_grid.predict(X_test)

# Demographic Parity for Gender (requires original X_test with Gender)
gender_col = pd.read_csv("../../data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")[
    "Gender"
].iloc[X_test.index]
dp_male = y_pred[gender_col == "Male"].mean()
dp_female = y_pred[gender_col == "Female"].mean()
print(f"Demographic Parity (Male): {dp_male:.2f}, (Female): {dp_female:.2f}")

# Equalized Odds (simplified, true positive rate)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
tpr = tp / (tp + fn)  # True Positive Rate (Recall)
print(f"Equalized Odds (TPR): {tpr:.2f}")
