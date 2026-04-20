"""
Configuration entities for the hybrid book recommender system.
These models enforce strict type safety and validation at startup to prevent
runtime attribute errors across pipeline stages.
All entities use Pydantic for validation, as mandated by the Agentic Architecture Standards.

Fail-Fast Validation: Enforced extra="forbid" across all models, ensuring the pipeline crashes
immediately at startup if a configuration file contains typos, extra keys, or invalid types,
preventing silent failures during multi-hour training/enrichment jobs.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SchemaConfig(BaseModel):
    """
    Configuration mapping for data contracts (Schema).
    """

    model_config = ConfigDict(extra="forbid")

    columns: dict[str, str] = Field(..., description="Map of logical -> physical column names")
    target_column: dict[str, str] = Field(..., description="Information about the target column")
    enriched_columns: dict[str, str] = Field(
        ..., description="Map of logical -> physical enriched columns"
    )
    types: dict[str, str] = Field(..., description="Dictionary of data types for validation")


class DataIngestionConfig(BaseModel):
    """
    Configuration for the Data Ingestion stage.
    """

    model_config = ConfigDict(extra="forbid")

    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


class DataValidationConfig(BaseModel):
    """
    Configuration for the Data Validation stage.
    """

    model_config = ConfigDict(extra="forbid")

    root_dir: Path
    unzip_data_dir: Path
    STATUS_FILE: Path
    cleaned_data_file: Path
    min_desc_len: int
    categories_min_len: int


class DataTransformationConfig(BaseModel):
    """
    Configuration for the Data Transformation stage.
    """

    model_config = ConfigDict(extra="forbid")

    root_dir: Path
    data_path: Path
    test_size: float
    val_size: float
    random_state: int
    tokenizer_name: str | None = None


class DataEnrichmentConfig(BaseModel):
    """
    Configuration for the Data Enrichment (Zero-Shot) stage.
    """

    model_config = ConfigDict(extra="forbid")

    root_dir: Path
    data_path: Path
    enriched_data_path: Path
    model_name: str
    candidate_labels: list[str]
    batch_size: int


class ToneAnalysisConfig(BaseModel):
    """
    Configuration for the Tone Analysis (Sentiment) stage.
    """

    model_config = ConfigDict(extra="forbid")

    root_dir: Path
    data_path: Path
    output_path: Path
    model_name: str
    target_emotions: list[str]
    batch_size: int
    min_sentence_len: int
    max_sentences_per_book: int
    detection_threshold: float


class ModelTrainerConfig(BaseModel):
    """
    Configuration for the Model Training (Vector DB) stage.
    """

    model_config = ConfigDict(extra="forbid")

    root_dir: Path
    data_path: Path
    db_path: Path
    model_name: str
    collection_name: str
    embedding_provider: str
    batch_size: int


class InferenceConfig(BaseModel):
    """
    Configuration for the Inference Stage (Hybrid Recommender).
    """

    model_config = ConfigDict(extra="forbid")

    model_name: str
    embedding_provider: str
    chroma_db_dir: Path
    data_path: Path
    collection_name: str
    top_k: int
    popularity_weight: float
    search_buffer_multiplier: int
    filtered_search_boost: int


class BatchPredictionConfig(BaseModel):
    """
    Configuration for the Batch Prediction stage.
    """

    model_config = ConfigDict(extra="forbid")

    root_dir: Path
    results_file: Path


class ModelEvaluationConfig(BaseModel):
    """
    Configuration for the Model Evaluation stage (MLflow tracking).
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    root_dir: Path
    data_path: Path
    model_path: Path
    all_params: dict[str, Any]
    mlflow_uri: str
    experiment_name: str


class AgentConfig(BaseModel):
    """
    Configuration for the Agentic Layer (pydantic-ai).
    """

    model_config = ConfigDict(extra="forbid")

    model_name: str
    temperature: float
    max_results_per_search: int
