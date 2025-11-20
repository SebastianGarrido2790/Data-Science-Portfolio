## **Analysis of Pipeline Execution and Model Performance**

The pipeline executed successfully, completing all steps: data processing, training, evaluation, and batch inference. The `n_classes_` issue was resolved by using the `Booster` object for predictions, and the fallback for early stopping was reincorporated, though it wasn't triggered in this run since `xgb.train()` with `early_stopping_rounds` worked. The `scale_pos_weight` of 2.16 was applied to address the class imbalance (41 non-churn vs. 19 churn). Below is an analysis of the results and recommendations for further improvement.

---

### **Summary of Execution**
1. **Data Processing**:
   - Raw dataset loaded, validated, and saved to `data/processed/cleaned_customer_churn.csv`.
   - Summaries and embeddings generated using `sentence-transformers/all-MiniLM-L6-v2` and `sshleifer/distilbart-cnn-12-6`.

2. **Model Training**:
   - Class imbalance addressed with `scale_pos_weight = 2.16` (41 non-churn, 19 churn).
   - Early stopping with `xgb.train()` stopped after 12 iterations, as the validation log-loss increased (from 0.66178 at iteration 2 to 0.75266 at iteration 12), indicating overfitting prevention.

3. **Model Evaluation**:
   - Classification report:
     ```
     precision    recall  f1-score   support
     0       0.75      0.86      0.80        14
     1       0.50      0.33      0.40         6
     accuracy                           0.70        20
     macro avg       0.62      0.60      0.60        20
     weighted avg       0.68      0.70      0.68        20
     ```
   - ROC AUC Score: `0.7857`, an improvement over the previous 0.7679.
   - Recall for class `1` (churn) dropped from 0.50 to 0.33, despite `scale_pos_weight`, indicating the imbalance correction needs adjustment.

4. **Batch Inference**:
   - Successfully processed new data, with columns aligned correctly.
   - Predictions: `[1, 0]`, indicating one customer is predicted to churn, an improvement over the previous `[0, 0]` (better at detecting churn).

---

### **Observations**
- **Class Imbalance Impact**:
  - The `scale_pos_weight` of 2.16 aimed to improve recall for class `1`, but recall decreased from 0.50 to 0.33, suggesting the weight might be too low or the model is overfitting to the majority class despite early stopping.
  - Precision for class `1` remained at 0.50, and overall accuracy dropped from 0.75 to 0.70, reflecting the trade-off.

- **ROC AUC Improvement**:
  - The ROC AUC score improved from 0.7679 to 0.7857, indicating better overall discrimination between classes, likely due to `scale_pos_weight` and early stopping.

- **Early Stopping**:
  - Early stopping worked correctly, halting training after 12 iterations when validation log-loss stopped improving (best at iteration 2: 0.66178).

- **Inference**:
  - The prediction `[1, 0]` shows the model is now identifying some churn cases, which aligns with the goal of improving churn detection.

---

### **Recommendations for Improvement**

#### **1. Adjust `scale_pos_weight` or Use Alternative Imbalance Techniques**
- **Increase `scale_pos_weight`**:
  - The current `scale_pos_weight` of 2.16 may not be sufficient. Try increasing it to 3.0 or 4.0 to give more emphasis to the minority class (churn).
  - Update in `train_model`:
    ```python
    scale_pos_weight = 3.0  # Experiment with higher values
    ```
- **SMOTE (Synthetic Minority Oversampling Technique)**:
  - Oversample the minority class before training to balance the dataset.
  - Install `imbalanced-learn`: `uv pip install imbalanced-learn`.
  - Add to `train_model`:
    ```python
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=model_config["random_state"])
    X_train, y_train = smote.fit_resample(X_train, y_train)
    neg_samples = sum(y_train == 0)
    pos_samples = sum(y_train == 1)
    logger.info(f"After SMOTE: {neg_samples} non-churn, {pos_samples} churn.")
    ```

#### **2. Tune the Prediction Threshold**
- The default threshold of 0.5 for `y_pred` in `evaluate_model` may not be optimal for an imbalanced dataset. Lowering it (e.g., to 0.3) can increase recall for class `1`.
- Update `evaluate_model`:
  ```python
  threshold = 0.3  # Experiment with lower values
  y_pred = (y_pred_proba >= threshold).astype(int)
  ```

#### **3. Hyperparameter Tuning**
- The model stopped early (12 iterations), suggesting the learning rate or tree depth might be too aggressive. Tune `max_depth` and `learning_rate` to improve generalization.
- Example using grid search (add to `train_model.py`):
  ```python
  param_grid = {
      "max_depth": [3, 6, 9],
      "learning_rate": [0.01, 0.1, 0.3],
  }
  best_score = float("inf")
  best_params = {}
  for max_depth in param_grid["max_depth"]:
      for learning_rate in param_grid["learning_rate"]:
          model_params = {
              "max_depth": max_depth,
              "learning_rate": learning_rate,
              "eval_metric": "logloss",
              "random_state": model_config["random_state"],
              "scale_pos_weight": scale_pos_weight,
          }
          bst = xgb.train(model_params, dtrain, num_boost_round=100, evals=eval_list, early_stopping_rounds=10, verbose_eval=False)
          score = bst.best_score
          if score < best_score:
              best_score = score
              best_params = model_params
  logger.info(f"Best params: {best_params}")
  ```

#### **4. Monitor Training Time**
- The script started at 14:57:44, and the previous run (14:49:58) took about 7 minutes (given the current time is 15:00:00). If runtime becomes a bottleneck, consider GPU support for embeddings and summarization.

---

### **Validation**
- Run the script after adjustments:
  ```bash
  python src/models/main.py
  ```
- Check the logs for:
  - Improved recall for class `1` (target > 0.33).
  - ROC AUC score (target > 0.7857).
  - Batch predictions reflecting better churn detection.
- Verify the pipeline completes without errors.

---

### **Additional Notes**
- **Fallback for Early Stopping**: The fallback wasn’t triggered, but it’s in place for robustness across environments.
- **Performance Trade-off**: Increasing recall for class `1` may reduce precision for class `0`. Monitor the F1-score for balance.
- **Data Size**: The small dataset (60 training samples) limits performance. Collecting more data could help.
