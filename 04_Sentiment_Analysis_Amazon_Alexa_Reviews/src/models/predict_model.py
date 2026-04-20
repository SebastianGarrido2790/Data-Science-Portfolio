import torch
import argparse
from typing import List, Dict, Union
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
import csv
import os

# Constants
MODEL_PATH = "C:\Users\Desktop\Models\distilbert_binary_model"
DEFAULT_THRESHOLD = 0.95
MAX_BATCH_SIZE = 100

# Resolve the model path to ensure it’s absolute
MODEL_PATH = os.path.abspath(MODEL_PATH)
print(f"Resolved MODEL_PATH: {MODEL_PATH}")

# Load model and tokenizer with local_files_only to avoid Hugging Face validation
try:
    print(f"Loading model and tokenizer from {MODEL_PATH}...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_PATH, local_files_only=True
    )
except Exception as e:
    raise RuntimeError(f"Failed to load model or tokenizer from {MODEL_PATH}: {str(e)}")

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)
model.eval()  # Set to evaluation mode for inference


def predict_sentiment(
    texts: Union[str, List[str]],
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizer,
    threshold: float = DEFAULT_THRESHOLD,
) -> List[Dict[str, Union[str, float]]]:
    """Predict sentiment for a single text or a list of texts using a trained model.

    Args:
        texts (str or list[str]): The input text(s) to classify. Can be a single string
            or a list of strings for batch prediction.
        model (transformers.PreTrainedModel): The trained DistilBERT model for prediction.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for text preprocessing.
        threshold (float, optional): Decision threshold for positive sentiment.
            Defaults to DEFAULT_THRESHOLD (0.95).

    Returns:
        list[dict]: A list of predictions, each containing:
            - 'text' (str): The input text.
            - 'sentiment' (str): Predicted sentiment ("positive" or "negative").
            - 'probability_positive' (float): Probability of the positive class.
            - 'threshold' (float): The threshold used for classification.

    Raises:
        ValueError: If inputs are invalid (e.g., empty list, non-string items).
    """
    # Normalize input to a list
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list):
        raise ValueError("Input 'texts' must be a string or a list of strings")
    if not texts:
        raise ValueError("Input 'texts' list cannot be empty")

    # Clean and validate texts
    cleaned_texts = [
        str(text) if isinstance(text, str) or pd.notna(text) else "" for text in texts
    ]

    # Check batch size
    if len(texts) > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size exceeds limit of {MAX_BATCH_SIZE} texts")

    # Tokenize the batch
    inputs = tokenizer(
        cleaned_texts,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prob_positive = probs[:, 1].tolist()  # Probabilities of positive class
    except Exception as e:
        raise RuntimeError(f"Inference failed: {str(e)}")

    # Generate predictions
    predictions = [
        {
            "text": text,
            "sentiment": "positive" if prob >= threshold else "negative",
            "probability_positive": prob,
            "threshold": threshold,
        }
        for text, prob in zip(cleaned_texts, prob_positive)
    ]

    return predictions


def save_predictions(predictions: List[Dict], output_file: str):
    """Save predictions to a CSV file, appending without repeating headers.

    Args:
        predictions (list[dict]): List of prediction dictionaries.
        output_file (str): Path to the output CSV file.
    """
    # Check if the file already exists to decide whether to write the header
    file_exists = os.path.isfile(output_file)

    with open(output_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["text", "sentiment", "probability_positive", "threshold"]
        )
        # Write header only if the file is new
        if not file_exists:
            writer.writeheader()
        writer.writerows(predictions)
    print(f"Predictions appended to {output_file}")


def main():
    """Command-line interface for sentiment prediction.

    Allows users to input texts via command-line arguments or a file,
    with options for custom threshold and output file.
    """
    parser = argparse.ArgumentParser(
        description="Predict sentiment using a trained DistilBERT model."
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        default=["This movie was great!", "This movie was terrible!"],
        help="Text(s) to classify (space-separated). Default: two example texts.",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a text file containing one review per line.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Threshold for positive sentiment (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save predictions as a CSV file (optional).",
    )
    args = parser.parse_args()

    # Load texts from file if provided, otherwise use command-line texts
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
        except Exception as e:
            raise ValueError(f"Failed to read input file: {str(e)}")
    else:
        texts = args.texts

    # Validate threshold
    if not 0 <= args.threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    # Predict sentiment
    try:
        predictions = predict_sentiment(
            texts, model, tokenizer, threshold=args.threshold
        )
        for pred in predictions:
            print(f"Text: {pred['text']}")
            print(f"Sentiment: {pred['sentiment']}")
            print(f"Probability (Positive): {pred['probability_positive']:.4f}")
            print(f"Threshold: {pred['threshold']:.4f}")
            print("-" * 50)

        # Save to file if output path is provided
        if args.output:
            save_predictions(predictions, args.output)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
