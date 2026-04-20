from typing import Any, List
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from src.config.settings import get_settings
from functools import lru_cache
import logging
from tenacity import retry, stop_after_attempt, wait_fixed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingFactory:
    _clients = {}

    def __init__(self, provider: str):
        self.provider = provider
        self.settings = getattr(get_settings(), f"{provider}_embeddings")
        self.client = self._get_client(self.provider)  # Pass the provider argument

    @classmethod
    @lru_cache
    def _get_client(cls, provider: str) -> Any:
        if provider not in cls._clients:
            if provider == "openai":
                if not get_settings().openai_embeddings.api_key:
                    logger.warning(
                        "OPENAI_API_KEY not found; skipping OpenAI initialization."
                    )
                    return None
                cls._clients[provider] = OpenAI(
                    api_key=get_settings().openai_embeddings.api_key
                )
            elif provider == "huggingface":
                cls._clients[provider] = SentenceTransformer(
                    get_settings().huggingface_embeddings.default_model
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        return cls._clients[provider]

    @retry(
        stop=stop_after_attempt(get_settings().huggingface_embeddings.max_retries),
        wait=wait_fixed(2),
    )
    def get_embeddings(self, texts: str | List[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        if self.provider == "openai":
            if self.client is None:
                raise ValueError("OpenAI not initialized due to missing API key.")
            try:
                response = self.client.embeddings.create(
                    input=texts, model=self.settings.default_model
                )
                return [data.embedding for data in response.data]
            except Exception as e:
                logger.error(f"OpenAI embedding error: {e}")
                raise
        elif self.provider == "huggingface":
            return self.client.encode(texts).tolist()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")


class SummaryFactory:
    _clients = {}

    def __init__(self, provider: str):
        self.provider = provider
        self.settings = getattr(get_settings(), f"{provider}_summaries")
        self.client = self._get_client(self.provider)  # Pass the provider argument
        logger.info(
            f"Initialized {provider} summary client with model {self.settings.default_model}"
        )

    @classmethod
    @lru_cache
    def _get_client(cls, provider: str) -> Any:
        if provider not in cls._clients:
            if provider == "openai":
                if not get_settings().openai_summaries.api_key:
                    logger.warning(
                        "OPENAI_API_KEY not found; skipping OpenAI initialization."
                    )
                    return None
                cls._clients[provider] = OpenAI(
                    api_key=get_settings().openai_summaries.api_key
                )
            elif provider == "huggingface":
                cls._clients[provider] = pipeline(
                    "summarization",
                    model=get_settings().huggingface_summaries.default_model,
                )
            else:
                raise ValueError(f"Unsupported summary provider: {provider}")
        return cls._clients[provider]

    @retry(
        stop=stop_after_attempt(get_settings().huggingface_summaries.max_retries),
        wait=wait_fixed(2),
    )
    def summarize(self, text: str) -> str:
        logger.info(f"Summarizing text with {self.provider} provider")
        if self.provider == "openai":
            if self.client is None:
                raise ValueError("OpenAI not initialized due to missing API key.")
            try:
                prompt = f"Summarize the following customer ticket focusing on the main complaint or request:\n\n{text}\n\nSummary:"
                response = self.client.chat.completions.create(
                    model=self.settings.default_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.settings.temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"OpenAI summarization error: {e}")
                raise
        elif self.provider == "huggingface":
            input_length = len(text.split())
            max_length = (
                min(
                    self.settings.max_tokens if self.settings.max_tokens else 20,
                    max(2, input_length // 5),
                )
                if get_settings().huggingface_summaries.dynamic_length
                else self.settings.max_tokens or 20
            )
            min_length = max(1, input_length // 6)
            temperature = (
                self.settings.temperature if self.settings.temperature > 0 else 1.0
            )
            if self.settings.temperature <= 0:
                logger.warning(
                    "Temperature <= 0 detected; defaulting to 1.0 for sampling."
                )
            summary = self.client(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
            )
            return summary[0]["summary_text"]
        else:
            raise ValueError(f"Unsupported summary provider: {self.provider}")
