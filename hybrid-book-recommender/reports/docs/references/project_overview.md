# Project Executive Summary — Hybrid Book Recommender

*Last updated: 2026-03-26 | System version: v1.3*

---

## 1. Project Goals

The primary objective of this project is to develop a **production-grade Hybrid Book Recommender System** that bridges the gap between traditional Machine Learning and modern Agentic AI Engineering. The project serves a dual purpose:

1. **Technical:** Solve the "Cold Start" and "Long-Tail" problems in recommendation engines by combining semantic content analysis with popularity-weighted scoring.
2. **Portfolio:** Demonstrate mastery of the full **MLOps (Machine Learning Operations)** lifecycle — from raw data ingestion through model deployment to an Agentic conversational interface.

Key outcomes include the implementation of a modular FTI pipeline architecture, strict data contracts via Pydantic, structured LLM output enforcement via `pydantic-ai`, and an Agentic storefront layer that illustrates the **Brain vs. Brawn** separation of concerns.

---

## 2. System Architecture (v1.3)

The system is composed of four layers, each with a clearly bounded responsibility:

```
┌────────────────────────────────────────────────────────────────┐
│  LAYER 4 — Storefront (Gradio UI + Agentic Layer)              │
│  src/app/ + src/agent/                                          │
│  ├── Search Tab: Direct semantic search with filters            │
│  └── AI Book Assistant: pydantic-ai ReAct agent (Gemini Flash)  │
├────────────────────────────────────────────────────────────────┤
│  LAYER 3 — Inference Engine                                      │
│  src/models/hybrid_recommender.py                               │
│  Hybrid scoring: Semantic distance + Popularity weight           │
│  Dynamic mode: General search OR Tone-probability sort           │
├────────────────────────────────────────────────────────────────┤
│  LAYER 2 — Model Training & Vector Index                         │
│  ChromaDB + Sentence-Transformers (HuggingFace)                 │
│  Embeddings stored in artifacts/model_trainer/vectordb/          │
├────────────────────────────────────────────────────────────────┤
│  LAYER 1 — Data Pipeline (FTI — Feature, Train, Inference)       │
│  8-stage DVC DAG: Ingestion → Validation → Transformation →      │
│  Enrichment → Tone Analysis → Training → Prediction → Evaluation │
└────────────────────────────────────────────────────────────────┘
```

### The Hybrid Scoring Algorithm

The `HybridRecommender` applies a **Dynamic Context-Aware Strategy**:

**Default Mode (General Search)**
$$Score_{final} = (1 - Distance_{cosine}) + \alpha \cdot \frac{Rating_{avg}}{5.0}$$

**Tone Mode (Emotional Targeting)**
When a user requests a specific mood (e.g., "Suspenseful"), the system:
1. Filters to books where $P(\text{target\_emotion}) > 0$
2. Ranks by $P(\text{target\_emotion})$ descending
3. Uses semantic similarity as a tie-breaker

---

## 3. Agentic Layer (v1.3 Addition)

The v1.3 release introduces an **Agentic Layer** that transforms the recommender into a conversational AI system:

- **Framework:** `pydantic-ai` with Google Gemini (`gemini-2.0-flash`)
- **Pattern:** Brain vs. Brawn — the Agent reasons, the `HybridRecommender` executes
- **Output:** Strictly validated `AgentResponse(BaseModel)` — never free text
- **Tools:** `search_books()`, `get_available_categories()`, `get_available_tones()`
- **Prompt:** Versioned system prompt v1.0.0 (`src/agent/prompts.py`)

See [Agentic Layer Architecture](../architecture/agentic_layer.md) for the full breakdown.

---

## 4. Design Requirements & Standards

The project is governed by four non-functional requirements:

| Requirement | Implementation |
|---|---|
| **Reliability** | Pydantic `extra="forbid"` on all config entities; `CustomException` wrapping; mocked unit tests |
| **Scalability** | ChromaDB vector search scales with corpus size; Docker-ready containerization |
| **Maintainability** | FTI separation of concerns; DVC-centric config via `params.yaml`; `ruff` + `pyright` CI gates |
| **Adaptability** | LLM provider swappable via `params.yaml` (`model_name`); MLflow experiment tracking |

---

## 5. Technology Stack

| Category | Technology |
|---|---|
| **Language & Management** | Python 3.11, `uv` |
| **Orchestration & Versioning** | DVC, Git, MLflow |
| **Agentic Framework** | `pydantic-ai` ≥ 0.2.0 |
| **LLM & Embeddings** | Google Gemini API, HuggingFace Transformers, Sentence-Transformers |
| **Vector Database** | ChromaDB (LangChain integration) |
| **Traditional ML** | Scikit-Learn |
| **Type Safety** | Pyright (standard mode), Pydantic v2 |
| **Code Quality** | Ruff (linting + formatting), `types-pyyaml` |
| **Testing** | Pytest, pytest-cov |
| **Interface** | Gradio (`Glass` theme) |
| **Deployment** | Docker, AWS EC2, GitHub Actions |

---

## 6. Configuration System

All system behavior is governed by a three-YAML configuration system managed by `ConfigurationManager`:

| File | Purpose |
|---|---|
| `config/config.yaml` | Immutable system paths and artifact directory structure |
| `config/params.yaml` | All tunable hyperparameters (model names, weights, thresholds, agent settings) |
| `config/schema.yaml` | Data contracts — maps logical column names to physical CSV fields |

Configuration is loaded into strictly-typed Pydantic `BaseModel` entities with `extra="forbid"`. This means any YAML key typo fails loudly at startup, not silently at runtime.

**Entity classes in `src/entity/config_entity.py`:**

```
DataIngestionConfig | DataValidationConfig | DataTransformationConfig
DataEnrichmentConfig | ToneAnalysisConfig | ModelTrainerConfig
InferenceConfig | BatchPredictionConfig | ModelEvaluationConfig
SchemaConfig | AgentConfig  ← (v1.3)
```

---

## 7. DVC Pipeline (8 Stages)

The data backbone is a **Directed Acyclic Graph (DAG)** reproducible via `uv run dvc repro`:

```
Stage 01: Ingestion          → artifacts/data_ingestion/books.csv
Stage 02: Validation         → artifacts/data_validation/clean_books.csv
Stage 03: Transformation     → artifacts/data_transformation/{train,val,test}.csv
Stage 04: Data Enrichment    → artifacts/data_enrichment/enriched_books.csv
Stage 05: Tone Analysis      → artifacts/tone_analysis/toned_books.csv
Stage 06: Model Training     → artifacts/model_trainer/vectordb/
Stage 07: Batch Prediction   → artifacts/prediction/results.txt
Stage 08: Model Evaluation   → artifacts/model_evaluation/metrics.json → MLflow
```

See [DVC Pipeline Report](../architecture/dvc_pipeline_report.md) for the full stage breakdown and caching strategy.

---

## 8. Project Structure

```
hybrid-book-recommender/
├── config/
│   ├── config.yaml          # System paths (immutable)
│   ├── params.yaml          # Tunable hyperparameters + agent config
│   └── schema.yaml          # Data contracts
│
├── src/
│   ├── agent/               # Agentic Layer (v1.3)
│   │   ├── schemas.py       # Pydantic structured outputs
│   │   ├── prompts.py       # Versioned system prompt
│   │   ├── tools.py         # Deterministic tools
│   │   └── agent.py         # Agent definition + chat()
│   ├── app/                 # Gradio UI
│   │   ├── main.py          # Blocks layout + event handlers
│   │   ├── styles.py        # HTML/CSS templates
│   │   └── data_loaders.py  # Engine init + helpers
│   ├── components/          # FTI pipeline workhorses
│   ├── config/              # ConfigurationManager
│   ├── entity/              # Config dataclasses/Pydantic models
│   ├── models/              # HybridRecommender + EmbeddingFactory
│   ├── pipeline/            # DVC stage orchestrators
│   └── utils/               # Logger, exceptions, helpers
│
├── tests/
│   ├── unit/
│   │   ├── test_recommender.py   # 2 tests (scoring math + filtering)
│   │   └── test_agent.py         # 7 tests (schemas + tools)
│   └── integration/              # 3 tests (accuracy + enrichment)
│
├── reports/docs/
│   ├── architecture/        # System architecture deep-dives
│   ├── evaluations/         # Test reports, codebase reviews
│   ├── references/          # Project overview (this file)
│   └── workflows/           # Implementation plan history
│
├── pyproject.toml           # Single source of truth for all tool config
├── dvc.yaml                 # DAG definition
├── validate_recommender.bat # 4-pillar health check
└── launch_recommender.bat   # App launcher
```

---

## 9. Expected Deliverables

By end of project:

| Deliverable | Status |
|---|---|
| Clean, modular Python package (uv managed) | ✅ Complete |
| 8-stage DVC pipeline with reproducible artifacts | ✅ Complete |
| Strict data contracts (Pydantic + schema.yaml) | ✅ Complete |
| ChromaDB vector index + HybridRecommender engine | ✅ Complete |
| Gradio Search UI (Glass theme) | ✅ Complete |
| MLflow experiment tracking | ✅ Complete |
| Agentic Layer (pydantic-ai + Gemini) | ✅ Complete (v1.3) |
| Docker containerization | 🔄 Planned |
| GitHub Actions CI/CD | 🔄 Planned |
| AWS EC2 deployment | 🔄 Planned |

---

## 10. Key Architectural Decisions

| Decision | Rationale |
|---|---|
| `uv` over `pip`/`poetry` | Deterministic lockfile, 10-100× faster resolution, aligns with Astral toolchain (`ruff`) |
| `pydantic-ai` over LangChain ReAct | Lighter abstraction, native Pydantic output validation, no graph overhead for single-agent use case |
| `gemini-2.0-flash` for agent | Cost-efficient for routing/reasoning; Gemini Pro reserved for future complex reasoning |
| HuggingFace for embeddings | Privacy-safe, no API key required for inference, deterministic outputs |
| Gradio over Streamlit | Native `gr.Chatbot` component for agentic tab; better ML model showcase alignment |
| `ConfigDict(extra="forbid")` everywhere | Fail-fast at startup, not at runtime — prevents silent config drift |

---

## 11. Project Analogy

Building this agentic hybrid recommender using MLOps practices is like running a **high-tech bookstore**:

- **The HybridRecommender** is the expert librarian — knows the entire catalogue, finds exact matches instantly.
- **The Agentic Layer** is the store manager — listens to what you want, asks clarifying questions, then directs the librarian on what to find.
- **The DVC Pipeline** is the supply chain — ensures every book in the catalogue was processed, tagged, and shelved correctly.
- **pydantic-ai schemas** are the order forms — every recommendation must be on the right form, or it doesn't leave the back office.
- **MLflow** is the inventory ledger — tracks exactly which books were recommended, under which parameters, and how they performed.