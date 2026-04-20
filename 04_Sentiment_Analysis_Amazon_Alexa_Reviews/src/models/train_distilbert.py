import pandas as pd
import numpy as np
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset
import nlpaug.augmenter.word as naw
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
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def download_nltk_resources():
    """Download required NLTK resources."""
    resources = ["averaged_perceptron_tagger_eng", "wordnet", "punkt", "punkt_tab"]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            logging.info(f"Successfully downloaded NLTK resource: {resource}")
        except Exception as e:
            logging.error(f"Failed to download NLTK resource {resource}: {str(e)}")
            raise


# Download NLTK resources
download_nltk_resources()

# Load data
train_df = pd.read_csv("../../data/interim/train.csv")
test_df = pd.read_csv("../../data/interim/test.csv")

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

# Calculate class weights
neg_count = len(train_df[train_df["feedback"] == 0])
pos_count = len(train_df[train_df["feedback"] == 1])
neu_count = len(train_df[train_df["feedback"] == 2])
class_weight_dict = {0: pos_count / neg_count, 1: 1.0, 2: pos_count / neu_count}

logging.info(
    f"Class distribution - Negative: {neg_count}, Positive: {pos_count}, Neutral: {neu_count}"
)
logging.info(f"Class weights: {class_weight_dict}")

# Augment negative reviews
aug = naw.SynonymAug(aug_p=0.3)
neg_texts = train_df[train_df["feedback"] == 0]["processed_reviews"].tolist()
neg_labels = train_df[train_df["feedback"] == 0]["feedback"].tolist()
augmented_texts = [aug.augment(text)[0] for text in neg_texts]
augmented_df = pd.DataFrame(
    {
        "processed_reviews": augmented_texts + train_df["processed_reviews"].tolist(),
        "feedback": neg_labels + train_df["feedback"].tolist(),
    }
)

# Prepare data for three-class transfer learning
X_train_aug = augmented_df["processed_reviews"]
y_train_aug = augmented_df["feedback"].astype(int)
X_test = test_df["processed_reviews"]
y_test = test_df["feedback"].astype(int)


# Custom Dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# Focal Loss
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# Custom Trainer with Focal Loss
class FocalTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels").to(model.device)
        outputs = model(**{k: v.to(model.device) for k, v in inputs.items()})
        logits = outputs.logits
        loss_fct = FocalLoss(alpha=class_weight_dict[0], gamma=2.0)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
device = "cpu"  # Force CPU for your hardware
logging.info(f"GPU Available: {torch.cuda.is_available()}")

# Transfer learning (three-class)
model_3class = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3
)
model_3class.to(device)
train_dataset_3class = SentimentDataset(
    X_train_aug.tolist(), y_train_aug.tolist(), tokenizer
)
test_dataset_3class = SentimentDataset(X_test.tolist(), y_test.tolist(), tokenizer)

training_args_3class = TrainingArguments(
    output_dir="./distilbert_3class_results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./distilbert_3class_logs",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="recall_neg",
    greater_is_better=True,
    use_cpu=True,
    learning_rate=2e-5,
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "recall_neg": recall_score(
            labels,
            preds,
            pos_label=0,
            average="micro" if len(np.unique(labels)) > 2 else "binary",
        ),
        "precision_neg": precision_score(
            labels,
            preds,
            pos_label=0,
            zero_division=0,
            average="micro" if len(np.unique(labels)) > 2 else "binary",
        ),
    }


trainer_3class = FocalTrainer(
    model=model_3class,
    args=training_args_3class,
    train_dataset=train_dataset_3class,
    eval_dataset=test_dataset_3class,
    compute_metrics=compute_metrics,
)
trainer_3class.train()
model_3class.save_pretrained("../../models/distilbert_3class_model")
tokenizer.save_pretrained("../../models/distilbert_3class_model")

# Fine-tune for binary classification
train_binary_df = train_df[train_df["feedback"] != 2].copy()
test_binary_df = test_df[test_df["feedback"] != 2].copy()
train_binary_df["feedback"] = train_binary_df["feedback"].apply(
    lambda x: 0 if x == 0 else 1
)
test_binary_df["feedback"] = test_binary_df["feedback"].apply(
    lambda x: 0 if x == 0 else 1
)

# Augment negative reviews for binary
neg_texts_binary = train_binary_df[train_binary_df["feedback"] == 0][
    "processed_reviews"
].tolist()
neg_labels_binary = train_binary_df[train_binary_df["feedback"] == 0][
    "feedback"
].tolist()
augmented_texts_binary = [aug.augment(text)[0] for text in neg_texts_binary]
augmented_binary_df = pd.DataFrame(
    {
        "processed_reviews": augmented_texts_binary
        + train_binary_df["processed_reviews"].tolist(),
        "feedback": neg_labels_binary + train_binary_df["feedback"].tolist(),
    }
)

# Binary dataset
train_binary_dataset = SentimentDataset(
    augmented_binary_df["processed_reviews"].tolist(),
    augmented_binary_df["feedback"].tolist(),
    tokenizer,
)
test_binary_dataset = SentimentDataset(
    test_binary_df["processed_reviews"].tolist(),
    test_binary_df["feedback"].tolist(),
    tokenizer,
)

# Initialize binary model
model_binary = DistilBertForSequenceClassification.from_pretrained(
    "../../models/distilbert_3class_model", num_labels=2
)
model_binary.to(device)

# Binary training arguments
binary_training_args = TrainingArguments(
    output_dir="./distilbert_binary_results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir="./distilbert_binary_logs",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="recall_neg",
    greater_is_better=True,
    use_cpu=True,
    learning_rate=2e-5,
)

# Binary trainer
binary_trainer = FocalTrainer(
    model=model_binary,
    args=binary_training_args,
    train_dataset=train_binary_dataset,
    eval_dataset=test_binary_dataset,
    compute_metrics=compute_metrics,
)
binary_trainer.train()
model_binary.save_pretrained("../../models/distilbert_binary_model")
tokenizer.save_pretrained("../../models/distilbert_binary_model")

# Evaluate binary model
predictions = binary_trainer.predict(test_binary_dataset)
probs = torch.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()
pred_labels = np.argmax(predictions.predictions, axis=1)

# Metrics
logging.info(
    f"\nBinary Classification Report:\n{classification_report(test_binary_df['feedback'], pred_labels)}"
)
logging.info(f"ROC-AUC: {roc_auc_score(test_binary_df['feedback'], probs)}")
precision, recall, _ = precision_recall_curve(
    test_binary_df["feedback"], probs, pos_label=1
)
logging.info(f"PR-AUC: {auc(recall, precision)}")
cm = confusion_matrix(test_binary_df["feedback"], pred_labels)
logging.info(f"\nConfusion Matrix:\n{cm}")

# Visualize PR curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"PR-AUC = {auc(recall, precision):.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("DistilBERT Precision-Recall Curve")
plt.legend()
plt.savefig("../../reports/figures/distilbert/distilbert_pr_curve.png")
plt.show()
plt.close()

# Visualize confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("DistilBERT Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("../../reports/figures/distilbert/distilbert_confusion_matrix.png")
plt.show()
plt.close()

# Save model
model_binary.save_pretrained("../../models/distilbert_binary_model")
tokenizer.save_pretrained("../../models/distilbert_binary_model")
