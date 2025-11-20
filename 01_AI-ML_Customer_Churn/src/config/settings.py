from pydantic import Field
from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv
import os

# Load .env file if it exists (fallback for local development)
load_dotenv()


class LLMProviderSettings(BaseSettings):
    temperature: float = 0.8  # Increase for more diversity
    max_tokens: int | None = None
    max_retries: int = 3


# OpenAI settings for embeddings (optional)
class OpenAIEmbeddingSettings(LLMProviderSettings):
    api_key: str = os.getenv(
        "OPENAI_API_KEY", ""
    )  # Optional, defaults to empty if not set
    default_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")


# OpenAI settings for summaries (optional)
class OpenAISummarySettings(LLMProviderSettings):
    api_key: str = os.getenv(
        "OPENAI_API_KEY", ""
    )  # Optional, defaults to empty if not set
    default_model: str = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-3.5-turbo")


# Hugging Face settings for embeddings
class HuggingFaceEmbeddingSettings(LLMProviderSettings):
    default_model: str = os.getenv(
        "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )


# Hugging Face settings for summaries
class HuggingFaceSummarySettings(LLMProviderSettings):
    default_model: str = os.getenv(
        "HUGGINGFACE_SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6"
    )
    dynamic_length: bool = (
        os.getenv("HUGGINGFACE_SUMMARY_DYNAMIC_LENGTH", "true").lower() == "true"
    )


class Settings(BaseSettings):
    app_name: str = "Embedding and Summary Generator"
    openai_embeddings: OpenAIEmbeddingSettings = (
        OpenAIEmbeddingSettings()
    )  # Still instantiated but tolerant of missing key
    openai_summaries: OpenAISummarySettings = (
        OpenAISummarySettings()
    )  # Still instantiated but tolerant of missing key
    huggingface_embeddings: HuggingFaceEmbeddingSettings = (
        HuggingFaceEmbeddingSettings()
    )
    huggingface_summaries: HuggingFaceSummarySettings = HuggingFaceSummarySettings()


@lru_cache
def get_settings():
    return Settings()
