# Modeling & Experimentation

## Process:
**Goal**: Define the machine learning problem type, loss functions, and KPIs.
- **Problem Type**: This is a **binary classification problem** (predict `Attrition`: 1=Yes, 0=No). Confirmed in Step 1; no temporal data (Step 2), so not a time-series task.
- **Loss Function**:
  - Use **weighted binary cross-entropy** to handle class imbalance, prioritizing the minority class (`Attrition: Yes`).
  - Focus on **recall** as the primary metric (business goal: ≥ 80% recall to catch most at-risk employees).
- **KPIs**:
  - **Technical**: Recall ≥80%, F1-score ≥0.75 (balances precision/recall), AUC-ROC for overall performance.
  - **Business**: Identify ≥70% of at-risk employees, enable cost savings (e.g., $100,000 annually via retention strategies).
- **Future Evolution**: If temporal data (e.g., employee exit dates) becomes available, we could shift to time-series forecasting (e.g., survival analysis). Currently, static classification is appropriate.

**Outcome**:
- Problem framed as binary classification with recall-focused evaluation.

## Experimentation
**Goal**: Track experiments systematically for reproducibility and comparison.
- **Tool**: MLflow for tracking (logs in `mlruns/`). It’s lightweight, integrates with scikit-learn, and fits your project setup. Experiments are visible in the MLflow UI under `EmployeeAttritionExperiment`.
```bash
mlflow ui --backend-store-uri file://C:/Users/.../12_Employee_Attrition/src/features/mlruns
```
- **Models (Latest Run)**:
  - **Logistic Regression (Threshold=0.4)**: Recall 0.8298, F1 0.3319, AUC 0.7750.
  - **Random Forest (Threshold=0.4)**: Recall 0.3404, F1 0.4267, AUC 0.7800.
  - **XGBoost (Threshold=0.3)**: Recall 0.6596, F1 0.4218, AUC 0.7741.
- **Tuning**: Grid search with 5-fold stratified CV, SMOTE for imbalance (sampling_strategy=0.7, class weights adjusted).

## Algorithm Selection and Experimentation
**Goal**: Evaluate multiple algorithms, tune hyperparameters, and select the best model.

### Model Choice Criteria:
- **Performance**: High recall (≥80%) and F1-score (≥0.75) for imbalanced data.
- **Interpretability**: Prioritize models with good explainability (e.g., logistic regression, decision trees) to provide HR with actionable insights.
- **Scalability**: Ensure the model can handle larger datasets (future scalability).
- **Cost**: Low computational cost for deployment (e.g., avoid overly complex models like deep learning for this dataset).

### Algorithms:
**1. Logistic Regression**: Interpretable, good baseline, handles imbalanced data with class weights.

**2. Random Forest**: Handles non-linear relationships, provides feature importance, but less interpretable.

**3. Gradient Boosting (XGBoost)**: High performance for imbalanced data, feature importance, but requires tuning.

### Process:
- **Cross-Validation**: Use 5-fold stratified cross-validation to account for class imbalance.
- **Hyperparameter Tuning**: Apply grid search for key parameters (e.g., class weights, regularization, learning rate, tree depth).
- **Class Imbalance**: Use `class_weight` (e.g., {0: 1, 1: 5}) and SMOTE for oversampling the minority class.
- **Threshold Adjustment**: Lowered decision thresholds (0.4 for LogisticRegression/RandomForest, 0.3 for XGBoost) to prioritize recall.
- **Time Series**: Not applicable (confirmed in Step 2).

## Interpretability
- **SHAP**:
  - **Logistic Regression**: Top features identified (e.g., `OverTime`, `SatisfactionScore`, etc.). Plot saved in `reports/figures/shap_logistic_regression.png`.
  - **Random Forest**: Top features identified (e.g., `DistanceFromHome`, `MonthlyIncome`, etc.). Plot saved in `reports/figures/shap_random_forest.png`.
  - **XGBoost**: Top features: `OverTime`, `SatisfactionScore`, `DistanceFromHome`. Plot saved in `reports/figures/shap_xgboost.png`.

## Fairness
- **Demographic Parity**: Male (0.08), Female (0.13) – slight disparity.
- **Equalized Odds**: TPR 0.38 (based on XGBoost), consistent across groups but lower than desired.

## Ethical Considerations
- No significant bias across `Gender`.
- Ensure HR uses predictions for support, not punishment.

## Shortlisted Models
- **Logistic Regression**: Achieves target recall (0.8298) but very low F1-score (0.3319), highly interpretable.
- **XGBoost**: Below target recall (0.6596) in the latest run, needs further tuning to reach ≥0.80.

## Next Steps
- **Further Tuning for XGBoost**:
  - **Dynamic Threshold**: Use precision-recall curve to find the optimal threshold for maximizing recall while improving F1-score.
  - **Adjust `scale_pos_weight`**: Increase to 4.0 to further prioritize the minority class.
  - **SMOTE Adjustment**: Test `sampling_strategy=0.5` to reduce noise from oversampling.
  - **Hyperparameter Tuning**: Focus on smaller `learning_rate` (e.g., 0.01) and higher `n_estimators` (e.g., 800) for better learning stability.
  - **Feature Selection**: Use SHAP insights to prioritize top features (`OverTime`, `SatisfactionScore`, `DistanceFromHome`) and reduce noise.
- **Reconcile Performance**: Continue investigating why XGBoost performance dropped from previous best (Recall 0.85, F1 0.78, AUC 0.89).
- **Finalize Model Choice**: Re-run with optimized parameters, then decide between XGBoost (performance) and Logistic Regression (interpretability) with HR.
- **Proceed to Step 6**: Move to Model Evaluation & Business Review once recall and F1-score targets are consistently met.