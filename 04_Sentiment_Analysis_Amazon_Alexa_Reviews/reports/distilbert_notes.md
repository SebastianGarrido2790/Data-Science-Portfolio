# DistilBERT Implementation Notes

## Why DistilBERT?

- **Contextual Understanding**: Unlike Random Forest and XGBoost, which rely on bag-of-words representations (via `CountVectorizer`), DistilBERT leverages transformer-based embeddings to capture semantic meaning and context. For example, it can understand negation (e.g., "not good") and complex relationships in text, making it ideal for sentiment analysis on Amazon Alexa reviews.
- **Performance Potential**: DistilBERT has the potential to improve negative recall while maintaining precision, especially on our imbalanced dataset (257 negative vs. 2893 positive samples in training). Fine-tuning with techniques like Focal Loss and data augmentation helps address this imbalance.
- **Efficiency**: DistilBERT is a lighter version of BERT, with 40% fewer parameters, making it more suitable for training on modest hardware (e.g., 8 GB RAM, CPU-only setup) while retaining strong performance.

## Why a Custom `SentimentDataset` Class?

The `SentimentDataset` class is critical for training our DistilBERT model on a supervised task (sentiment classification: 3-class initially, then binary positive vs. negative). It integrates with Hugging Face’s `Trainer` API and PyTorch’s `DataLoader`. Here’s why it’s necessary and how it’s implemented in `src/models/train_distilbert.py`:

### 1. Handles Batched Training Data
- **Dataset Size**: Our dataset consists of 2397 training samples (205 negative, 2193 positive, with 121 neutral samples for 3-class training) and 600 test samples. The `Trainer` processes these in batches (e.g., batch size 8, adjusted with gradient accumulation) for efficient optimization.
- **PyTorch Dataset Interface**: The `SentimentDataset` class implements `__len__` and `__getitem__`, enabling compatibility with `DataLoader`:
  - `__len__`: Returns the total number of samples (e.g., 2397 for training).
  - `__getitem__`: Returns a dictionary with `input_ids`, `attention_mask`, and `labels` for a given sample, formatted for DistilBERT.
- **Why PyTorch?**: PyTorch is preferred over TensorFlow due to:
  - Seamless integration with Hugging Face’s `Trainer` and `TrainingArguments`.
  - Easier customization of loss functions (e.g., `FocalLoss` in `FocalTrainer`).
  - Flexibility for transfer learning and fine-tuning.
  - In contrast, TensorFlow (e.g., `TFDistilBertForSequenceClassification`) is harder to customize for weighted loss or metrics like negative recall.

### 2. Pre-Tokenization for Efficiency
- **Change from Dynamic Tokenization**: Unlike the previous implementation, the updated `SentimentDataset` class now pre-tokenizes all text samples during initialization (`self.encodings = tokenizer(texts, ...)`). This ensures all data is tokenized once upfront, reducing overhead during training.
- **Memory Consideration**: While pre-tokenizing consumes more memory initially, it’s manageable for our dataset size (2397 samples) on an 8 GB RAM setup. The `max_length=128` with padding and truncation keeps memory usage reasonable.
- **Benefits**: Pre-tokenization avoids repeated tokenization during each epoch, speeding up training iterations, especially with the `Trainer` API.

### 3. Pairs Text with Labels
- **Supervised Learning**: Each review (e.g., "I love my Alexa") must be paired with a label (0 for negative, 1 for positive, or 2 for neutral in the 3-class model). The `SentimentDataset` class ensures this pairing by storing labels alongside tokenized inputs.
- **Output Format**: For each sample, `__getitem__` returns:
  ```python
  {
      "input_ids": torch.tensor([...]),  # Token IDs for the review
      "attention_mask": torch.tensor([...]),  # Attention mask for padding
      "labels": torch.tensor(label, dtype=torch.long)  # Sentiment label
  }
  ```
- **Trainer Compatibility**: This format is directly compatible with `DistilBertForSequenceClassification`, which expects `input_ids`, `attention_mask`, and `labels` for computing loss.

### 4. Simplified Preprocessing
- **Edge Cases Handled Externally**: The updated script handles edge cases (e.g., empty reviews) before creating the dataset, using `train_df.apply` to replace empty `processed_reviews` with `verified_reviews`. This simplifies the `SentimentDataset` class, as it no longer needs to handle NaN or empty strings.
- **Customization**: The `max_length=128`, `padding="max_length"`, and `truncation=True` ensure consistent input sizes for DistilBERT, optimizing for our dataset’s review lengths.

### How It Fits Together
1. **Initialization**: `SentimentDataset(X_train_aug.tolist(), y_train_aug.tolist(), tokenizer)` pre-tokenizes all 2397+ augmented reviews and stores the encodings and labels.
2. **Length**: `len(train_dataset)` returns the total sample count, informing the `Trainer` how many samples to process.
3. **getitem**: When `Trainer` requests a sample (e.g., `train_dataset[0]`), it retrieves the pre-tokenized `input_ids`, `attention_mask`, and `labels` for that sample.
4. **Batching**: The `Trainer` (via an internal `DataLoader`) batches samples (e.g., batch size 8, yielding `input_ids` of shape `[8, 128]`), which DistilBERT processes efficiently on CPU.

### Why This Design?
- **Performance**: Pre-tokenizing all samples reduces training overhead, as tokenization is a one-time cost.
- **Compatibility**: The class ensures seamless integration with `Trainer` and `DistilBertForSequenceClassification`.
- **Simplicity**: External preprocessing simplifies the dataset class, focusing it on data delivery rather than data cleaning.

## Additional Implementation Details

### Transfer Learning and Fine-Tuning
- **3-Class Model**: Initially, a 3-class model (positive, negative, neutral) is trained using `distilbert-base-uncased` with `num_labels=3`. This model is saved to `models/distilbert_3class_model/`.
- **Binary Fine-Tuning**: The 3-class model is then fine-tuned for binary classification (positive vs. negative) by:
  - Filtering out neutral samples (feedback != 2).
  - Re-labeling (0 for negative, 1 for positive).
  - Loading the 3-class model and training with `num_labels=2`.
  - Saved to `models/distilbert_binary_model/`.
- **Why Transfer Learning?**: Pre-training on a 3-class task helps the model learn general sentiment patterns before specializing in binary classification, improving performance on our imbalanced dataset.

### Data Augmentation
- **Negative Class Augmentation**: Negative reviews (256 samples) are augmented using `nlpaug`’s `SynonymAug` (augmentation probability 0.3) to address class imbalance. Augmented texts are concatenated with the original training data.
- **Impact**: Increases the number of negative samples, helping the model learn better representations for the minority class.

### Focal Loss
- **Custom Loss Function**: A `FocalLoss` class is implemented to focus on hard-to-classify examples (e.g., negative reviews) by down-weighting easy examples. Parameters:
  - `alpha`: Set to `class_weight_dict[0]` (ratio of positive to negative samples) to prioritize negative class.
  - `gamma=2.0`: Controls the focus on hard examples.
- **Integration**: A custom `FocalTrainer` class overrides the `compute_loss` method to use `FocalLoss` instead of the default cross-entropy loss.
- **Benefit**: Improves recall for the negative class, addressing the imbalance (257 negative vs. 2893 positive).

### Training Configuration
- **3-Class Training**:
  - Epochs: 3
  - Batch Size: 8 (effective batch size 16 with gradient accumulation steps=2)
  - Learning Rate: 2e-5
  - Warmup Steps: 500
  - Weight Decay: 0.01
  - Evaluation Strategy: Per epoch, with `metric_for_best_model="recall_neg"`
- **Binary Fine-Tuning**:
  - Epochs: 3
  - Batch Size: 8 (effective batch size 16 with gradient accumulation)
  - Learning Rate: 2e-5
  - Warmup Steps: 200
  - Weight Decay: 0.01
- **Hardware**: Forced to CPU (`use_cpu=True`) due to hardware constraints (no GPU available, 8 GB RAM).

### Evaluation and Visualization
- **Metrics**:
  - Accuracy, negative recall, and negative precision are computed using `compute_metrics`.
  - ROC-AUC and PR-AUC are calculated for the binary model.
- **Figures Generated**:
  - **Precision-Recall Curve**: `reports/figures/distilbert/distilbert_pr_curve.png` shows the PR curve for the positive class, with the PR-AUC value annotated.
  - **Confusion Matrix**: `reports/figures/distilbert/distilbert_confusion_matrix.png` visualizes the binary classification confusion matrix using a heatmap.

## Comparison with Random Forest and XGBoost
- **Feature Representation**:
  - DistilBERT uses contextual embeddings, capturing semantic relationships, while Random Forest and XGBoost rely on `CountVectorizer` (bag-of-words) combined with engineered features (e.g., `review_length`, `vader_compound`).
  - DistilBERT’s embeddings lead to better handling of negation and context, potentially improving recall for the negative class.
- **Training Approach**:
  - DistilBERT uses transfer learning and fine-tuning with a custom `FocalLoss` to address class imbalance.
  - Random Forest and XGBoost use Optuna for hyperparameter tuning, SMOTE for oversampling, and class weights/scale_pos_weight to handle imbalance.
- **Performance**:
  - DistilBERT may outperform on nuanced text understanding but requires more computational resources ( mitigated by CPU training and pre-tokenization).
  - Random Forest and XGBoost are faster to train and interpret but may struggle with semantic nuances.

## Notes on Hardware Constraints
- **CPU-Only Training**: Due to the absence of a GPU (`torch.cuda.is_available()=False`), training is forced to CPU. This increases training time but is manageable with a small batch size and gradient accumulation.
- **Memory Efficiency**: Pre-tokenization in `SentimentDataset` and a `max_length=128` keep memory usage within the 8 GB RAM limit. Augmentation is applied selectively to negative samples to avoid excessive memory overhead.

## Future Improvements
- **GPU Support**: If a GPU becomes available, remove `use_cpu=True` and leverage CUDA for faster training.
- **Advanced Augmentation**: Explore more augmentation techniques (e.g., back-translation) to further balance the dataset.
- **Hyperparameter Tuning**: Use Optuna or grid search to optimize `TrainingArguments` (e.g., learning rate, batch size).
- **Ensemble**: Combine DistilBERT with Random Forest/XGBoost predictions for potentially higher accuracy.