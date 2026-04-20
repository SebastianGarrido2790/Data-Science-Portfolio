# NYC Taxi Tip Prediction — Rules & Guardrails Runbook

**Project:** NYC Taxi Tip Prediction System
**Document Type:** Runbook · The Rules
**Version:** 1.0
**Date:** 2026-03-10
**Status:** Active — Authoritative Reference

---

## 1. Purpose & Scope

This runbook is the **single source of truth** for all constraints, prohibitions, coding standards, and operational guardrails governing the NYC Taxi Tip Prediction project. It applies to every human or AI agent contributing to the codebase and must be consulted before any architectural change, new feature implementation, or modification of existing pipelines.

This document does **not** describe how the system works (see `reports/docs/architecture/`) or why decisions were made (see `reports/docs/decisions/`). It describes the **boundaries within which all work must operate**.

---

## 2. Core Philosophy

> **"The Brain (Agent) directs; The Hands (Tools) execute."**

All design decisions in this system flow from this principle. LLM agents are probabilistic interpreters. Deterministic Python tools are the only entities permitted to compute, fetch, or transform data. Any violation of this separation is a **critical architectural defect**.

---

## 3. Python Code Standards

### 3.1 Typing

| Rule | Requirement | Enforcement |
| :--- | :--- | :--- |
| **Type Hints** | 100% coverage on all functions, methods, and class attributes | `pyright` (Standard Mode) |
| **py.typed** | Mandatory marker file in `src/` to signal PEP 561 compliance | Project Skeleton Rule |
| **Pydantic Models** | Every external input (API, tool call, config) must use a `BaseModel` | Code review |
| **No Untyped Dicts** | `dict` must never cross module or agent/tool boundaries | Linter (`ruff`) |

**✅ DO:**
```python
class TaxiRideInput(BaseModel):
    trip_distance: float = Field(..., gt=0, description="Total distance of the ride in miles.")
    total_amount: float = Field(..., gt=0, description="Total amount charged, excluding tip.")
```

**❌ DO NOT:**
```python
def get_prediction(data: dict): ...   # Naked dict — rejected at review
```

### 3.2 Linting & Formatting

All code must pass the following checks **before any commit**:

```bash
uv run ruff check . --output-format=github   # Pycodestyle, Pyflakes, isort, Pyupgrade
uv run ruff format --check .                  # Formatter check
```

Active `ruff` rule sets (from `pyproject.toml`):

| Code | Ruleset | Notes |
| :--- | :--- | :--- |
| `E`, `W` | pycodestyle | Style and whitespace |
| `F` | Pyflakes | Unused imports, undefined names |
| `I` | isort | Import ordering |
| `UP` | pyupgrade | Modern Python idioms |
| `RUF` | Ruff-specific | Specialized Ruff linting rules |

`E501` (line length) is exempt — handled by the formatter at 88 characters.

### 3.3 Docstrings

**Google-style docstrings are mandatory** on every public function and class.

```python
def predict_tips(self, rides: List[TaxiRideInput]) -> List[Dict[str, Any]]:
    """Calculates the expected tip amount for a batch of NYC Taxi rides.

    Args:
        rides: A list of TaxiRideInput objects detailing the ride parameters.

    Returns:
        A list of dictionary results containing the 'predicted_tip' in USD.

    Raises:
        PredictionToolError: If the backend is unreachable or returns validation errors.
    """
```

> **Why this matters:** LLM agents rely on docstrings to understand tool capabilities. A poorly documented tool leads to misuse that cannot be debugged in the Python layer.

### 3.4 Dependency Management

| Rule | Requirement |
| :--- | :--- |
| **Runtime** | Always use `uv` — never `pip install` directly |
| **Lockfile** | `uv.lock` must be committed with every dependency change |
| **Project Config** | All metadata, dependencies, and tool config live in `pyproject.toml` |
| **Dev extras** | Testing and linting tools must be declared under `[project.optional-dependencies] dev =` |

---

## 4. Agentic Architecture Guardrails

### 4.1 Strict Separation — Brain vs. Hands

| Allowed for Agents (Brain) | Prohibited for Agents |
| :--- | :--- |
| Reasoning, synthesis, classification | Arithmetic and complex calculations |
| Language generation and formatting | Raw data fetching from Parquet/CSV/DB |
| Tool orchestration and routing | ML inference execution |
| Business interpretation of results | Any direct `exec()` or `eval()` |

**Violation Class: CRITICAL.** If an LLM is performing math or retrieving data without a tool, the system is producing hallucinated outputs.

### 4.2 No Naked Prompts

System prompts are **forbidden** from being hardcoded inline anywhere other than configuration files or a dedicated `prompts.py` module.

| Requirement | Location |
| :--- | :--- |
| All system prompts | Agent's config or `prompts.py` module |
| Config parameters | `src/config/` mapping to Pydantic Settings |
| Business logic rules | A deterministic Tool or config file — **never the prompt directly generated inline** |

### 4.3 Structured Output Enforcement

- Agents that feed downstream code must output structured text with machine-parseable schemas (like Pydantic Function Calling defaults).
- Conversational output must synthesize backend predictions accurately. The LLM must not fabricate tips without calling the prediction tool first.

### 4.4 Tool Design Rules

Every agent tool must satisfy all of the following:

- [ ] Has a Pydantic `BaseModel` input schema (validates arguments before execution)
- [ ] Has a clear Google-style docstring (agent reads this to decide when to call the tool)
- [ ] Is **deterministic** — identical inputs must always produce identical outputs
- [ ] Guards against network failures gracefully explicitly returning descriptive error strings via custom exceptions (e.g. `PredictionToolError`) instead of crashing the LLM pipeline
- [ ] Is **stateless** — tools must not store or mutate global agent state themselves

### 4.5 Data Leakage Prevention

Agent tools must **never expose the ground truth label** (`tip_amount` or full raw rows including testing signals) when retrieving data to build context before prediction.
Information leakage invalidates all agent functionality.

### 4.6 Provider Hot-Swapping via Config — Not Code

The LLM provider and API keys are controlled exclusively via `.env` (e.g., `GOOGLE_API_KEY`, `DEFAULT_LLM_PROVIDER=gemini`). **No API keys or hardcoded provider details** are allowed inside the application code.

Changing a provider architecture mandates zero code changes and operates via environment variable shifts only.

---

## 5. MLOps Pipeline Rules

### 5.1 FTI Pipeline Independence

The three pipelines must be independently operable at all times:

| Pipeline | Entry Point | Can run without | Must NOT depend on |
| :--- | :--- | :--- | :--- |
| **Feature** | `dvc repro` stages 1–4 | Training & Inference | Model artifacts |
| **Training** | `dvc repro` stages 5–6 | Running Inference API | Live streaming data |
| **Inference** | `uvicorn src.api.predict_api:app` | Training scripts | Raw data logs & DVC DAG execution routines |

**❌ PROHIBITED:** Importing `src.pipeline.*` modules directly inside `src.api.*` (inference backend).

### 5.2 DVC — Versioned Artifacts

All ML artifacts produced by the pipeline must be systematically tracked by DVC:

- Raw ingested data (`stage_01`)
- Validation and structural metadata (`stage_02`)
- High-performance feature engineering/cyclical parsing (`stage_03`, `stage_04`)
- Fitted models & artifacts (`artifacts/model_trainer/`)
- Inference benchmarks (`stage_06`)

> **Rule:** Running `uv run dvc repro` must safely reproduce all artifacts utilizing locally cached dependencies without issues.

### 5.3 MLflow Experiment Tracking

Every training run must be seamlessly logged to MLflow including the champion Joblib models, feature arrays, algorithms variants (e.g., XGBoost, Random Forest), alongside key validation metrics (MAE, MSE, R²).

> **Rule:** All stochastic operations globally strictly mandate an explicitly deterministic seed (`random_state`) to assure rigid test reproducibility.

### 5.4 Data Contracts

Raw data entering the pipeline and APIs strictly follows explicit schema definitions:

| Field Example | Type | Validation Rule |
| :--- | :--- | :--- |
| `trip_distance` | `float` | Required, `> 0` |
| `total_amount` | `float` | Required, `> 0` |
| `passenger_count` | `int` | Required, bound `[1, 10]` |
| `ratecode_id` | `int` | Required |
| `hour`, `day`, `month`| `int` | Required temporal attributes for cyclic processing bounds |

Failure of incoming API data format triggers an immediate validation response block, avoiding bad inferences.

---

## 6. API Service Rules

### 6.1 Prediction API (`src/api/predict_api.py`)

The FastAPI service must always:
- Load the model artifact **once on startup** via the `lifespan` async context manager.
- Hold model registers in memory (`MODEL_REGISTRY["champion"]`) without reloading it sequentially inside endpoints.
- Provide a `/health` endpoint to empower readiness probes on production containers.
- Abstain fully from mixing internal training pipelines with REST routing execution architectures.

**Default inference endpoint:** `POST http://localhost:8000/predict`

### 6.2 API Response Contract

The `PredictResponse` schema is the **contract** between the ML Fastapi service and the Application/Agent layer:

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `predicted_tip` | `float` | The estimated tip amount in USD |
| `model_version` | `str` | The version/name of the model used for prediction |

### 6.3 Graceful Degradation

The tools interfacing with LLMs (such as `TaxiPredictionTool`) **must elegantly capture failures**. On HTTP timeouts, connection resets, or unavailable models, it raises domain-specific traps (`PredictionToolError`) yielding descriptive error strings returned gracefully, giving the governing LLM the capability to advise the user safely.

---

## 7. Testing Rules

### 7.1 Testing Pyramid

| Layer | Tool | Scope |
| :--- | :--- | :--- |
| **Unit** | `pytest` | Tools, data parsers, pipelines, explicit Pydantic schemas |
| **Integration** | `pytest` | Mock-enabled endpoint access mapping to FTI stages |
| **API/Routing** | `pytest` | `TestClient` validations encompassing `/predict` & `/health` |

### 7.2 CI Enforcement & Pre-commit Hooks

The project utilizes `.pre-commit-config.yaml` locally and workflows structurally ensuring compliance.
The GitHub Actions CI strictly enforces the following gates. Pushing to master forces the build process against:

```
Lint & Format (Ruff) ───► Strict Type Checking (Pyright) ───► Unit & Logic Tests
```

A PR **cannot be deployed** manually overriding failed Action runs.

### 7.3 Multi-Point System Validation
Before any production deployment or major architectural PR, the local multi-point validation script must be executed to verify all four pillars of health:

```bash
.\validate_system.bat
```

This script orchestrates:
1. **Static Analysis**: Pyright and Ruff checks.
2. **Logic & Coverage**: Pytest with a mandatory >65% coverage gate.
3. **Synchronization**: DVC artifact and lineage verification.
4. **Service Health**: Runtime check against versioned FastAPI endpoints.

### 7.4 Unit Test Requirements

Whenever testing explicit Data transformations or Inference endpoints, always emphasize **mocking external I/O integrations** (Polars mock sets, Requests timeouts, Joblib model loaders) via standard `unittest.mock`.
Focus on Testing internal Logic execution, not the network robustness of third parties sequentially.

---

## 8. Containerization Rules

### 8.1 Dockerfile Standards

All `Dockerfile` implementations fundamentally must:
- Start with a stable pinned base Python footprint.
- Rely solely on `uv` to pull dependencies.
- Design execution environments targeting **least-privileged, non-root user accesses** post setup contexts.
- Leverage multi-stage layering caching code properly avoiding runtime dependencies bleeds.
- Pull static FTI pipeline model endpoints directly from pre-existing DVC structures (artifacts are never generated inside standard CI deployments).

### 8.2 Artifact Placement for Docker

`artifacts/` is Gitignored. Valid components rely completely on DVC pulling states correctly. CI routines test image structures utilizing mock placeholders. Real deployments map the container securely over the fetched DVC nodes.

---

## 9. Documentation Rules

### 9.1 The Master Source of Truth

Architectural pivots, complex implementations, and rules updates adhere strictly to standardized pillar structures. Verify their compliance within `reports/docs/` subdirectories **before executing or pushing logic updates**.

| Pillar | Location |
| :--- | :--- |
| **The Why (Decisions)** | `reports/docs/decisions/` |
| **The Map (Architecture)** | `reports/docs/architecture/` |
| **The Rules (Guardrails)** | `reports/docs/runbooks/rules_and_guardrails.md` |
| **The Evals (Quality)** | `reports/docs/evaluations/` |
| **The Workflows (Implementation)** | `reports/docs/workflows/` |

---

## 10. The "Do Not Do This" List

> This is a Hard-Stop reference applicable to ALL contributors & orchestrating Agents. Violations require explicit overrides defined in `reports/docs/decisions/`.

| # | Prohibition | Impact if Violated |
| :--- | :--- | :--- |
| **R-01** | ❌ DO NOT permit the LLM to directly calculate tip amounts without explicitly employing the Prediction Tool. | Hallucinated numeric projections misleading drivers/users. |
| **R-02** | ❌ DO NOT hardcode LLM Model types, credentials, or API Keys directly tracking inside Agent modules. | Security breaches & complete loss of parameter switching elasticity. |
| **R-03** | ❌ DO NOT intertwine FastAPI prediction microservice endpoints with preprocessing tasks inherently tied to pipeline `data_engineering`. | Complete disruption of the FTI architectural standard pattern & severe Training-serving Skew risk occurrences. |
| **R-04** | ❌ DO NOT inject naked dictionaries (`dict`) interfacing models and endpoint calls blindly. | Failure to validate edge cases via the explicit Pydantic standard boundaries. |
| **R-05** | ❌ DO NOT directly embed `.joblib` / `.parquet` massive assets into version control. | Bloated repositories breaking git clone performances permanently (DVC strictly controls components). |
| **R-06** | ❌ DO NOT crash explicitly via silent unhandled API HTTP responses directly up to the Agents execution frame. | Fails completely breaking multi-turn agent execution instead of prompting safe localized reporting constraints correctly. |
| **R-07** | ❌ DO NOT install or migrate core system integrations resolving requirements using traditional `pip` configurations exclusively. | Total bypass of `uv` lockfile resolutions and speed benefits. |
| **R-08** | ❌ DO NOT allow LLMs to directly read `target/labels` in prediction context extraction scenarios. | Severe data leakage impacting validation scores definitively. |

---

## 11. Incident Response Quick Reference

| Symptom | Likely Cause | First Action |
| :--- | :--- | :--- |
| Prediction API unreachable in Agent UI | Backend FastAPI offline / Not Port Forwarded | Validate `uvicorn` instance activity on `:8000` via `/health` |
| Agent completely guesses the projected tip blindly | Did not trigger `TaxiPredictionTool` strictly | Refine the overarching System Prompt or Agent logic forcing tool utilization |
| Pydantic `ValidationError` on Input format | Agent passed raw inputs avoiding strict parameter constraints or out of bounds `day`/`hour` formats | Assess structured tool payload trace requirements specifically around Temporal fields bindings |
| Artifact Missing error starting API | `MODEL_REGISTRY` fails locating Model binary (`.joblib`) | Trigger `uv run dvc repro` fetching remote artifact blocks synchronously prior to inference booting up. |
| Unreproducible results | Pipeline ran natively dropping random seed lock | Explicitly insert hardcoded global `random_state` logic standard variables mapping |
