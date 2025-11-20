import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------
# Data Preprocessing
# --------------------


def download_nltk_resources():
    """
    Download required NLTK resources with error handling.
    """
    resources = ["stopwords", "wordnet", "punkt", "punkt_tab"]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            logging.info(f"Successfully downloaded NLTK resource: {resource}")
        except Exception as e:
            logging.error(f"Failed to download NLTK resource {resource}: {str(e)}")
            raise


def map_feedback(rating):
    """
    Map rating to feedback:
    - 1, 2: Negative (0)
    - 3: Neutral (2)
    - 4, 5: Positive (1)
    """
    if rating in [1, 2]:
        return 0  # Negative
    elif rating == 3:
        return 2  # Neutral
    elif rating in [4, 5]:
        return 1  # Positive
    return None


def preprocess_text(text):
    """
    Preprocess review text:
    - Expand contractions
    - Remove URLs, emojis, and special characters
    - Lowercase and tokenize
    - Handle negations
    - Lemmatize and remove stopwords
    """
    # Handle non-string or NaN inputs
    text = str(text) if pd.notna(text) else "no_review"
    if text == "":
        text = "no_review"

    # Expand contractions
    text = contractions.fix(text)

    # Remove URLs, emojis, and special characters
    text = re.sub(r"http\S+|www\S+|[\U0001F600-\U0001F64F]", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())

    # Tokenize
    tokens = word_tokenize(text)

    # Handle negations
    negation_words = {
        "not",
        "never",
        "no",
        "nobody",
        "none",
        "nothing",
        "nowhere",
        "neither",
        "nor",
    }
    processed_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] in negation_words and i + 1 < len(tokens):
            processed_tokens.append(f"{tokens[i]}_{tokens[i+1]}")
            i += 2
        else:
            processed_tokens.append(tokens[i])
            i += 1

    # Lemmatize and remove stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    tokens = [
        lemmatizer.lemmatize(token)
        for token in processed_tokens
        if token not in stop_words
    ]

    return " ".join(tokens) if tokens else "no_review"


def main():
    # Download NLTK resources
    download_nltk_resources()

    # Load raw data
    df = pd.read_csv("../../data/raw/amazon_alexa.tsv", delimiter="\t", quoting=3)

    # Check for missing values
    print("Missing values:\n", df.isnull().sum())
    print("\nNull records:\n", df[df["verified_reviews"].isna()])

    # Drop rows with null verified_reviews
    df.dropna(subset=["verified_reviews"], inplace=True)

    # Map feedback
    df["feedback"] = df["rating"].apply(map_feedback)

    # Verify feedback distribution
    print(
        "\nFeedback distribution (%):\n",
        round(df["feedback"].value_counts() / len(df) * 100, 2),
    )

    # Preprocess reviews
    df["processed_reviews"] = df["verified_reviews"].apply(preprocess_text)

    # Check for empty or NaN processed reviews
    nan_count = df["processed_reviews"].isna().sum()
    empty_count = (df["processed_reviews"] == "no_review").sum()
    print("\nNaN processed reviews:", nan_count)
    print("Placeholder (no_review) processed reviews:", empty_count)
    if nan_count > 0:
        logging.error(f"Found {nan_count} NaN processed_reviews")
        raise ValueError("NaN values in processed_reviews")

    # Train-test split (80/20, stratified by feedback)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["feedback"], random_state=42
    )

    # Save preprocessed data
    train_df.to_csv("../../data/interim/train.csv", index=False)
    test_df.to_csv("../../data/interim/test.csv", index=False)

    print("\nTrain set shape:", train_df.shape)
    print("Test set shape:", test_df.shape)
    print(
        "\nTrain feedback distribution (%):\n",
        round(train_df["feedback"].value_counts() / len(train_df) * 100, 2),
    )
    print(
        "\nTest feedback distribution (%):\n",
        round(test_df["feedback"].value_counts() / len(test_df) * 100, 2),
    )


if __name__ == "__main__":
    main()
