### Model Comparison and Selection

Let’s analyze the results of the three models (Random Forest, XGBoost, and DistilBERT) to determine which one to use for predictions and API deployment. We'll also address the class imbalance and performance metrics to make an informed decision.

#### Class Distribution and Imbalance
- **Class Distribution**:
  - Negative (0): 205
  - Positive (1): 2193
  - Neutral (2): 121
- **Class Weights**:
  - Negative: 10.70
  - Positive: 1.0
  - Neutral: 18.12
- **Observation**: The dataset is heavily imbalanced, with the positive class dominating (2193 samples) compared to negative (205) and neutral (121). This imbalance impacts model performance, especially for the minority classes (negative and neutral).

#### Model Performance Metrics
1. **RandomForestClassifier (3-class)**:
   - **Best Parameters**: `{'n_estimators': 266, 'max_depth': 21, 'min_samples_split': 4, 'min_samples_leaf': 4, 'max_features': 'sqrt'}`
   - **Classification Report**:
     - Negative (0): Precision=0.48, Recall=0.49, F1=0.49
     - Positive (1): Precision=0.94, Recall=0.95, F1=0.95
     - Neutral (2): Precision=0.65, Recall=0.42, F1=0.51
     - Accuracy: 0.89
     - Macro Avg F1: 0.65
   - **ROC-AUC (ovr)**: 0.9347
   - **PR-AUC (negative)**: 0.5561
   - **Confusion Matrix**:
     ```
     [[ 25  24   2]
      [ 20 523   5]
      [  7  11  13]]
     ```
   - **Analysis**:
     - Struggles with the neutral class (recall=0.42), misclassifying many neutral samples as positive.
     - Negative class recall (0.49) is moderate but not ideal for identifying negative reviews.
     - High accuracy (0.89) is driven by the dominant positive class.

2. **XGBClassifier (3-class)**:
   - **Best Parameters**: `{'n_estimators': 397, 'max_depth': 13, 'learning_rate': 0.1836, 'gamma': 0.1089, 'reg_alpha': 0.0091, 'reg_lambda': 1.1327}`
   - **Classification Report**:
     - Negative (0): Precision=0.58, Recall=0.49, F1=0.53
     - Positive (1): Precision=0.94, Recall=0.96, F1=0.95
     - Neutral (2): Precision=0.62, Recall=0.52, F1=0.56
     - Accuracy: 0.90
     - Macro Avg F1: 0.68
   - **ROC-AUC (ovr)**: 0.9240
   - **PR-AUC (negative)**: 0.5232
   - **Confusion Matrix**:
     ```
     [[ 25  23   3]
      [ 16 525   7]
      [  2  13  16]]
     ```
   - **Analysis**:
     - Slightly better than Random Forest for the neutral class (recall=0.52) and negative class precision (0.58).
     - Negative class recall (0.49) is the same as Random Forest, indicating difficulty identifying negative reviews.
     - Accuracy (0.90) is marginally better, but macro F1 (0.68) shows improved balance across classes.

3. **DistilBERT (Binary: Positive vs. Negative)**:
   - **Training Details**:
     - 3-class model trained for 5 epochs, best validation accuracy=0.9190, recall_neg=0.9190.
     - Binary model fine-tuned for 5 epochs, best validation accuracy=0.9499, recall_neg=0.7843 (epoch 2).
   - **Classification Report**:
     - Negative (0): Precision=0.68, Recall=0.78, F1=0.73
     - Positive (1): Precision=0.98, Recall=0.97, F1=0.97
     - Accuracy: 0.95
     - Macro Avg F1: 0.85
   - **ROC-AUC**: 0.9760
   - **PR-AUC**: 0.9978
   - **Confusion Matrix**:
     ```
     [[ 40  11]
      [ 19 529]]
     ```
   - **Analysis**:
     - Focuses on binary classification (positive vs. negative), excluding the neutral class.
     - Significantly better recall for negative class (0.78) compared to Random Forest and XGBoost (0.49).
     - High accuracy (0.95) and macro F1 (0.85), showing strong performance across both classes.
     - Excellent ROC-AUC (0.9760) and PR-AUC (0.9978), indicating robust discriminative power.

#### Model Selection for Predictions and API
- **Goal**: The API likely needs to predict sentiment (positive vs. negative) for reviews, as this is a common use case for sentiment analysis.
- **DistilBERT** is the best choice for the following reasons:
  1. **Superior Performance**: DistilBERT’s binary model has the highest recall for negative reviews (0.78), which is critical for identifying negative feedback in an imbalanced dataset. It also achieves the highest macro F1 (0.85), ROC-AUC (0.9760), and PR-AUC (0.9978).
  2. **Binary Focus**: The binary classification setup (positive vs. negative) aligns with typical sentiment analysis needs, and DistilBERT handles this task better than the 3-class models.
  3. **Robustness**: DistilBERT leverages contextual embeddings, making it more adept at understanding nuanced text compared to Random Forest and XGBoost, which rely on TF-IDF or similar features.
  4. **API Suitability**: While DistilBERT is more computationally intensive, it can be deployed efficiently with frameworks like FastAPI and optimized for inference (e.g., using ONNX or quantization).

Random Forest and XGBoost, while faster for inference, struggle with the minority classes (negative and neutral), making them less suitable for this task.

---

### Model Performance and Improvement Suggestions

#### Model Performance Summary
- **RandomForestClassifier**:
  - Strengths: Fast inference, decent ROC-AUC (0.9347), good performance on the positive class.
  - Weaknesses: Poor recall for neutral (0.42) and negative (0.49) classes, indicating it struggles with minority classes.
- **XGBClassifier**:
  - Strengths: Slightly better than Random Forest for neutral (recall=0.52) and negative precision (0.58), good accuracy (0.90).
  - Weaknesses: Same negative recall (0.49) as Random Forest, lower ROC-AUC (0.9240) and PR-AUC (0.5232).
- **DistilBERT (Binary)**:
  - Strengths: Best performance overall—high negative recall (0.78), macro F1 (0.85), ROC-AUC (0.9760), and PR-AUC (0.9978). Excellent at identifying negative reviews.
  - Weaknesses: Computationally intensive, binary focus excludes neutral class.

#### Improvement Suggestions with Code Snippets

1. **Address Class Imbalance More Effectively**:
   - **Issue**: The dataset is heavily imbalanced, causing all models to favor the positive class.
   - **Suggestion**: Use SMOTE (Synthetic Minority Oversampling Technique) for Random Forest and XGBoost to balance the training data. For DistilBERT, consider adjusting the focal loss `alpha` dynamically based on class frequencies.
   - **Code Snippet for SMOTE (Random Forest/XGBoost)**:
     ```python
     from imblearn.over_sampling import SMOTE
     from sklearn.feature_extraction.text import TfidfVectorizer

     # Assuming X_train and y_train are your training data (text and labels)
     vectorizer = TfidfVectorizer(max_features=5000)
     X_train_tfidf = vectorizer.fit_transform(X_train)
     smote = SMOTE(random_state=42)
     X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)

     # Train RandomForest with balanced data
     from sklearn.ensemble import RandomForestClassifier
     rf = RandomForestClassifier(
         n_estimators=266,
         max_depth=21,
         min_samples_split=4,
         min_samples_leaf=4,
         max_features='sqrt',
         random_state=42
     )
     rf.fit(X_train_balanced, y_train_balanced)
     ```
   - **Code Snippet for Dynamic Focal Loss (DistilBERT)**:
     ```python
     class FocalLoss(torch.nn.Module):
         def __init__(self, class_weights, gamma=2.0, reduction='mean'):
             super(FocalLoss, self).__init__()
             self.class_weights = class_weights  # Dict with class weights
             self.gamma = gamma
             self.reduction = reduction

         def forward(self, inputs, targets):
             ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
             pt = torch.exp(-ce_loss)
             weights = torch.tensor([self.class_weights[t.item()] for t in targets], device=inputs.device)
             focal_loss = weights * (1 - pt) ** self.gamma * ce_loss
             return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

     # Update FocalTrainer
     class FocalTrainer(Trainer):
         def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
             labels = inputs.pop('labels').to(model.device)
             outputs = model(**{k: v.to(model.device) for k, v in inputs.items()})
             logits = outputs.logits
             loss_fct = FocalLoss(class_weights=class_weight_dict, gamma=2.0)
             loss = loss_fct(logits, labels)
             return (loss, outputs) if return_outputs else loss
     ```

2. **Optimize Hyperparameters Further**:
   - **Issue**: Random Forest and XGBoost trials show variability in recall for the negative class.
   - **Suggestion**: Expand the hyperparameter search space and prioritize negative recall in the optimization objective.
   - **Code Snippet for Enhanced Optuna Search (XGBoost)**:
     ```python
     import optuna
     from xgboost import XGBClassifier
     from sklearn.metrics import recall_score

     def objective(trial):
         params = {
             'n_estimators': trial.suggest_int('n_estimators', 100, 500),
             'max_depth': trial.suggest_int('max_depth', 8, 16),
             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.4, log=True),
             'gamma': trial.suggest_float('gamma', 0, 0.3),
             'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.2),
             'reg_lambda': trial.suggest_float('reg_lambda', 0.8, 1.5),
             'scale_pos_weight': class_weight_dict[0]  # Use class weights
         }
         model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')
         model.fit(X_train_tfidf, y_train)
         preds = model.predict(X_test_tfidf)
         neg_recall = recall_score(y_test, preds, labels=[0], average='micro')
         return neg_recall

     study = optuna.create_study(direction='maximize')
     study.optimize(objective, n_trials=30)
     ```

3. **Increase Training Epochs for DistilBERT**:
   - **Issue**: DistilBERT’s best negative recall (0.7843) was at epoch 2, but validation loss increased afterward, indicating potential overfitting.
   - **Suggestion**: Use early stopping and increase epochs to allow better convergence.
   - **Code Snippet for Early Stopping**:
     ```python
     binary_training_args = TrainingArguments(
         output_dir='./distilbert_binary_results',
         num_train_epochs=10,  # Increase epochs
         per_device_train_batch_size=8,
         per_device_eval_batch_size=8,
         gradient_accumulation_steps=2,
         warmup_steps=200,
         weight_decay=0.01,
         logging_dir='./distilbert_binary_logs',
         logging_steps=50,
         eval_strategy='epoch',
         save_strategy='epoch',
         load_best_model_at_end=True,
         metric_for_best_model='recall_neg',
         greater_is_better=True,
         report_to=[],
         learning_rate=2e-5,
         save_total_limit=2,  # Limit saved checkpoints
         early_stopping_patience=2,  # Stop if no improvement after 2 epochs
         early_stopping_threshold=0.01  # Minimum improvement to consider
     )
     ```

4. **Incorporate Neutral Class in DistilBERT**:
   - **Issue**: DistilBERT’s binary model excludes the neutral class, limiting its applicability.
   - **Suggestion**: Use the 3-class DistilBERT model (`distilbert_3class_model`) for prediction, then map neutral predictions to positive/negative based on a threshold.
   - **Code Snippet for 3-Class Prediction Mapping**:
     ```python
     def predict_with_mapping(review: str) -> dict:
         inputs = tokenizer(review, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
         inputs = {k: v.to(device) for k, v in inputs.items()}
         with torch.no_grad():
             outputs = model_3class(**inputs)
             probs = torch.softmax(outputs.logits, dim=1)
             pred_probs = probs[0].tolist()  # [neg, pos, neu]
             pred_label = torch.argmax(probs, dim=1).item()
         
         # Map neutral to positive/negative
         if pred_label == 2:  # Neutral
             pred_label = 1 if pred_probs[1] > pred_probs[0] else 0
         return {
             "sentiment": "positive" if pred_label == 1 else "negative",
             "confidence": pred_probs[pred_label]
         }
     ```

#### Final Recommendations
- **Model Choice**: Use DistilBERT’s binary model for the API due to its superior performance on negative recall and overall metrics.
- **Improvement Focus**:
  - Implement SMOTE for Random Forest/XGBoost to improve minority class performance.
  - Use dynamic focal loss and early stopping for DistilBERT to prevent overfitting and improve convergence.
  - Expand hyperparameter tuning for all models, prioritizing negative recall.
  - Consider integrating the neutral class into predictions for broader applicability.
