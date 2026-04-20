# Agentic Layer for Hybrid Book Recommender

## Goal

Transform the Hybrid Book Recommender from a traditional ML pipeline with a search UI into an **agentic system** where a conversational AI agent reasons about user preferences and orchestrates the `HybridRecommender` engine as a deterministic tool. This directly addresses §3.7 of the codebase review and aligns with the Agentic Data Scientist framework, Brain vs. Brawn separation, and Structured Output Enforcement.

> **Confirmed Decisions**:
> - Framework: `pydantic-ai`
> - LLM Provider: Google Gemini

---

## Architecture Overview

The design follows the **"Agent as Tool" pattern** and the **Brain vs. Brawn** separation:

```
┌─────────────────────────────────────────────────┐
│                  Gradio UI                      │
│  ┌──────────────┐    ┌───────────────────────┐  │
│  │ Search Tab   │    │  Chat Tab (NEW)       │  │
│  │ (existing)   │    │  Conversational Agent │  │
│  └──────────────┘    └───────┬───────────────┘  │
│                              │                  │
└──────────────────────────────┼──────────────────┘
                               │ Structured I/O
                    ┌──────────▼──────────┐
                    │   Agent (Brain)     │
                    │   LLM Reasoning     │
                    │   Prompt Templates  │
                    └──────────┬──────────┘
                               │ Tool Calls
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌────────────┐   ┌────────────┐   ┌────────────┐
     │ search_    │   │ get_book_  │   │ get_       │
     │ books()    │   │ details()  │   │ categories │
     │ (Tool)     │   │ (Tool)     │   │ (Tool)     │
     └─────┬──────┘   └─────┬──────┘   └────────────┘
           │                │
           ▼                ▼
     HybridRecommender (Brawn)
         ChromaDB + Pandas
```

---

## Proposed Changes

### Dependencies

#### [MODIFY] `pyproject.toml`
- Add `pydantic-ai>=0.2.0` to production dependencies (Option A) **OR** `langgraph>=0.4.0` (Option B).

---

### Agent Package (NEW)

#### [NEW] `src/agent/__init__.py`
- Package marker.

#### [NEW] `src/agent/schemas.py`
- **Structured output models** for agent responses:
  - `BookRecommendation(BaseModel)`: title, authors, description, rating, mood_score, thumbnail, match_reason.
  - `AgentResponse(BaseModel)`: message (str), recommendations (list[BookRecommendation]), follow_up_suggestions (list[str]).
- All models enforce `extra="forbid"` per project convention.

#### [NEW] `src/agent/prompts.py`
- **Versioned system prompt** as a constant string, not hardcoded inline.
- The prompt instructs the agent to:
  1. Interpret user preferences (genre, mood, themes).
  2. Use the `search_books` tool to query the recommender.
  3. Synthesize results into a conversational, personalized response.
  4. Suggest follow-up refinements (narrower genre, different mood).

#### [NEW] `src/agent/tools.py`
- **Deterministic tools** wrapping the `HybridRecommender`:
  - `search_books(query: str, category: str | None, tone: str | None) -> list[BookRecommendation]`: Calls `recommender.recommend()`, maps `RecommendationResult` → `BookRecommendation`.
  - `get_available_categories() -> list[str]`: Returns the configured category list from `params.yaml`.
  - `get_available_tones() -> list[str]`: Returns the configured tone list from `params.yaml`.
- Tools have **rich docstrings** so the LLM understands their capabilities.
- Input validation via Pydantic `BaseModel` for tool inputs.

#### [NEW] `src/agent/agent.py`
- Core agent setup:
  - Initializes the LLM client (Gemini Flash for cost efficiency).
  - Registers tools from `tools.py`.
  - Applies the system prompt from `prompts.py`.
  - Exposes a `chat(user_message: str, history: list) -> AgentResponse` function.

---

### Configuration

#### [MODIFY] `config/params.yaml`
- Add an `agent:` section:
  ```yaml
  agent:
    model_name: gemini-2.0-flash
    temperature: 0.7
    max_results_per_search: 5
  ```

#### [MODIFY] `src/entity/config_entity.py`
- Add `AgentConfig(BaseModel)` with `extra="forbid"`:
  - `model_name: str`
  - `temperature: float`
  - `max_results_per_search: int`

#### [MODIFY] `src/config/configuration.py`
- Add `get_agent_config() -> AgentConfig` method.

---

### Gradio UI Integration

#### [MODIFY] `src/app/main.py`
- Add a **second Gradio tab** ("AI Book Assistant") alongside the existing search tab.
- The new tab contains a `gr.Chatbot` component for conversational interaction.
- The chat handler calls `agent.chat()` and renders `AgentResponse` with book cards.

---

### Environment

#### [MODIFY] `.env.example`
- Ensure `GOOGLE_API_KEY` is documented (already present — the agent reuses it).

---

### Tests

#### [NEW] `tests/unit/test_agent.py`
- Mock-based unit tests (no API keys required in CI):
  - `test_search_books_tool`: Mocks `HybridRecommender.recommend()`, verifies `search_books` returns structured `BookRecommendation` objects.
  - `test_get_categories_tool`: Verifies categories are loaded from config.
  - `test_agent_response_schema`: Validates that `AgentResponse` enforces `extra="forbid"` and rejects malformed data.
  - `test_agent_response_structure`: Validates the Pydantic schema accepts well-formed data.

---

## Taks

### Phase 1: Implementation
- [ ] Add `pydantic-ai` dependency to [pyproject.toml](file:///c:/Users/sebas/Desktop/hybrid-book-recommender/pyproject.toml)
- [ ] Create `src/agent/` package with agentic components
  - [ ] [__init__.py](file:///c:/Users/sebas/Desktop/hybrid-book-recommender/src/__init__.py)
  - [ ] `tools.py` — Deterministic tools wrapping [HybridRecommender](file:///c:/Users/sebas/Desktop/hybrid-book-recommender/src/models/hybrid_recommender.py#21-196)
  - [ ] `agent.py` — Agent definition with system prompt and tool registration
  - [ ] `prompts.py` — Versioned, templated system prompts (Rule 1.5)
  - [ ] `schemas.py` — Pydantic structured output models (Rule 1.4)
- [ ] Add agent config section to [config/params.yaml](file:///c:/Users/sebas/Desktop/hybrid-book-recommender/config/params.yaml)
- [ ] Add `AgentConfig` entity to [config_entity.py](file:///c:/Users/sebas/Desktop/hybrid-book-recommender/src/entity/config_entity.py)
- [ ] Add `get_agent_config()` to [configuration.py](file:///c:/Users/sebas/Desktop/hybrid-book-recommender/src/config/configuration.py)
- [ ] Integrate agent chat tab into Gradio UI ([src/app/main.py](file:///c:/Users/sebas/Desktop/hybrid-book-recommender/src/app/main.py))
- [ ] Update [.env.example](file:///c:/Users/sebas/Desktop/hybrid-book-recommender/.env.example) with required agent env vars

### Phase 2: Testing
- [ ] Create `tests/unit/test_agent.py` with mock-based tests
- [ ] Run [validate_recommender.bat](file:///c:/Users/sebas/Desktop/hybrid-book-recommender/validate_recommender.bat) (Pyright + Ruff + Pytest)

---

## Verification Plan

### Automated Tests

1. **Existing test suite** (must remain green):
   ```
   uv run pytest tests/ -v --cov=src
   ```

2. **New agent unit tests**:
   ```
   uv run pytest tests/unit/test_agent.py -v
   ```

3. **Static quality gates** (full validation):
   ```
   .\validate_recommender.bat
   ```
   Must pass all 4 pillars: Pyright (0 errors), Ruff (all checks), Pytest (all pass), DVC (status OK).

### Manual Verification

> [!NOTE]
> After implementation, please launch the app with `.\launch_recommender.bat` and:
> 1. Open `http://localhost:7860` in your browser.
> 2. Navigate to the **"AI Book Assistant"** tab.
> 3. Type: *"I'm looking for a dark, suspenseful thriller set in a small town"*.
> 4. Verify the agent responds with book recommendations including titles, authors, ratings, and mood scores.
> 5. Send a follow-up: *"Something lighter and happier instead"*.
> 6. Verify the agent adjusts its search (switching tone filter).
