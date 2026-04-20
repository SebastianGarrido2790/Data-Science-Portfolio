"""
This module serves as the 'Worker' for the Tone Analysis Stage of the pipeline.
It handles the extraction of emotional tone from book descriptions using a pre-trained Transformer model.
"""

import re
import sys
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

from src.entity.config_entity import SchemaConfig, ToneAnalysisConfig
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ToneAnalysis:
    """
    Enhanced Component for Metadata Enrichment using Sentence-Level Tone Analysis.

    This class captures emotional nuance by splitting book descriptions into individual sentences,
    analyzing the emotional tone of each sentence using a Transformer model, and then
    aggregating these scores to determine the dominant tone of the book.
    """

    def __init__(self, config: ToneAnalysisConfig, schema: SchemaConfig):
        self.config = config
        self.schema = schema

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Splits text into sentences using a robust regex.
        """
        if not isinstance(text, str) or not text.strip():
            return []
        # Regex to split on . ! ? followed by space and capital letter or end of string
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > self.config.min_sentence_len]
        return sentences

    def initiate_tone_analysis(self) -> None:
        """
        Executes the sentence-level tone analysis process.

        Flow:
        1. Loads the enriched data.
        2. Initializes the Hugging Face emotion classification pipeline.
        3. Iterates through each book, splitting descriptions into sentences.
        4. Classifies each sentence to get emotion scores (e.g., joy, sadness, fear).
        5. Aggregates scores to calculate the average emotion profile for the book.
        6. Determines the dominant tone based on the highest non-neutral score.
        7. Saves the dataset with new columns: 'dominant_tone' and individual emotion scores.

        Raises:
            CustomException: If any error occurs during the analysis process.
        """
        try:
            logger.info(f"Loading data from {self.config.data_path}")
            df = pd.read_csv(self.config.data_path)

            cols = self.schema.columns
            enriched_cols = self.schema.enriched_columns

            logger.info(f"Initializing Emotion Classifier: {self.config.model_name}")

            # Device selection
            device = 0 if torch.cuda.is_available() else -1
            if device == 0:
                logger.info("GPU detected, using GPU for classification.")
            else:
                logger.info("No GPU detected, using CPU. This might be slow.")

            # Pipeline with top_k=None to get all 7 emotion scores
            classifier = pipeline(
                "text-classification",
                model=self.config.model_name,
                device=device,
                top_k=None,
            )

            book_dominant_tones = []
            emotion_scores_list = []

            logger.info("Processing books using Sentence-Level Analysis...")

            for _idx, row in tqdm(
                df.iterrows(), total=len(df), desc="Analyzing Books (Sentence-Level)"
            ):
                description = str(row.get(cols["description"], ""))
                sentences = self._split_into_sentences(description)

                if not sentences:
                    book_dominant_tones.append("neutral")
                    emotion_scores_list.append(dict.fromkeys(self.config.target_emotions, 0.0))
                    continue

                try:
                    # Truncate to configured max sentences
                    sentences = sentences[: self.config.max_sentences_per_book]
                    results = classifier(sentences)
                except Exception as e:
                    logger.warning(
                        f"Error classifying sentences for book '{row.get(cols['title'])}': {e}. Falling back to neutral."
                    )
                    book_dominant_tones.append("neutral")
                    emotion_scores_list.append(dict.fromkeys(self.config.target_emotions, 0.0))
                    continue

                emotion_sums = dict.fromkeys(self.config.target_emotions, 0.0)
                num_sentences = len(results)

                for sentence_result in results:
                    # sentence_result is a list of dicts: [{'label': 'joy', 'score': 0.9}, ...]
                    for emotion_score in sentence_result:
                        # Cast to Any to satisfy Pyright about dictionary indexing
                        es_any: Any = emotion_score
                        label = str(es_any["label"])
                        score = float(es_any["score"])
                        if label in emotion_sums:
                            emotion_sums[label] += score

                emotion_averages = {
                    label: score / num_sentences for label, score in emotion_sums.items()
                }
                emotion_scores_list.append(emotion_averages)

                non_neutral_averages = {
                    label: score for label, score in emotion_averages.items() if label != "neutral"
                }
                top_non_neutral = max(non_neutral_averages, key=lambda k: non_neutral_averages[k])
                top_non_neutral_score = non_neutral_averages[top_non_neutral]

                if top_non_neutral_score > self.config.detection_threshold:
                    book_dominant_tones.append(top_non_neutral)
                else:
                    book_dominant_tones.append("neutral")

            # Add columns to dataframe using schema mapping
            df[enriched_cols["dominant_tone"]] = book_dominant_tones
            scores_df = pd.DataFrame(emotion_scores_list)
            for col in scores_df.columns:
                df[col] = scores_df[col]

            logger.info(f"Saving refined toned data to {self.config.output_path}")
            df.to_csv(self.config.output_path, index=False)

            logger.info("✅ Sentence-level tone analysis completed successfully.")

        except Exception as e:
            raise CustomException(e, sys)
