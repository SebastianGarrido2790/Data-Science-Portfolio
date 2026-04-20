import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def download_nltk_resources():
    """Download required NLTK resources."""
    try:
        nltk.download("vader_lexicon", quiet=True)
        logging.info("Successfully downloaded NLTK resource: vader_lexicon")
    except Exception as e:
        logging.error(f"Failed to download NLTK resource vader_lexicon: {str(e)}")
        raise


def extract_features(df):
    """Extract features from the dataframe."""
    # Review length and word count
    df["review_length"] = df["processed_reviews"].str.len()
    df["word_count"] = df["processed_reviews"].str.split().str.len()

    # VADER sentiment scores
    sia = SentimentIntensityAnalyzer()
    vader_scores = df["processed_reviews"].apply(sia.polarity_scores)
    df["vader_compound"] = vader_scores.apply(lambda x: x["compound"])
    df["vader_pos"] = vader_scores.apply(lambda x: x["pos"])
    df["vader_neg"] = vader_scores.apply(lambda x: x["neg"])

    # One-hot encode variation
    variation_dummies = pd.get_dummies(df["variation"], prefix="variation").astype(
        float
    )
    df = pd.concat([df, variation_dummies], axis=1)

    # Extract month and year from date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    return df


def main():
    # Download NLTK resources
    download_nltk_resources()

    # Load data
    train_df = pd.read_csv("../../data/interim/train.csv")
    test_df = pd.read_csv("../../data/interim/test.csv")

    # Extract features
    train_features = extract_features(train_df)
    test_features = extract_features(test_df)

    # Define feature columns
    feature_cols = [
        "review_length",
        "word_count",
        "vader_compound",
        "vader_pos",
        "vader_neg",
        "month",
        "year",
        *[col for col in train_features.columns if col.startswith("variation_")],
    ]

    # Save features
    train_features[feature_cols].to_csv(
        "../../data/processed/train_features.csv", index=False
    )
    test_features[feature_cols].to_csv(
        "../../data/processed/test_features.csv", index=False
    )

    print(f"Train features shape: {train_features[feature_cols].shape}")
    print(f"Test features shape: {test_features[feature_cols].shape}")
    print(f"\nNew features added: {feature_cols}")


if __name__ == "__main__":
    main()
