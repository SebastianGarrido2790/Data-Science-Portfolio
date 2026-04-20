## Agentic Framework Decision

> [!IMPORTANT]
> **Framework Decision:** Two viable approaches are presented below. Please select your preferred framework before implementation begins.

### Option A: `pydantic-ai` (Recommended)
- **Pros:** Lightweight, native Pydantic integration for structured outputs, minimal abstraction, aligns with the project's existing Pydantic-first philosophy (`extra="forbid"` entities). Fast deterministic schema validation.
- **Cons:** Newer ecosystem, fewer community examples than LangChain/LangGraph.

### Option B: LangGraph + LangChain
- **Pros:** Already a project dependency (`langchain>=1.1.2`). Rich ecosystem, battle-tested ReAct agent. Integrates naturally with existing `langchain_chroma` and `langchain_huggingface`.
- **Cons:** Heavier abstraction layer, more boilerplate. LangGraph adds a stateful graph layer that may be overkill for a single-agent tool-calling scenario.

> [!IMPORTANT]
> **LLM Provider:** The agent needs an LLM for reasoning. The project already has `google-generativeai` and `GOOGLE_API_KEY` configured. The plan uses **Google Gemini** (e.g., `gemini-2.0-flash`) as the default agent model. Please confirm this is acceptable.
