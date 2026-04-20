# Implementation Plan — UI & Agentic Layer

*Last updated: 2026-03-26 | System version: v1.3*

---

## Overview

This document covers the design and implementation of the `src/app/` layer, which serves as the **Storefront** of the Hybrid Book Recommender. As of v1.3, the application exposes **two distinct interfaces** via Gradio:

1. **Search Tab** — Direct semantic search with category and tone filters.
2. **AI Book Assistant Tab** — Conversational agentic interface powered by `pydantic-ai` + Google Gemini.

> [!IMPORTANT]
> **Framework:** Gradio `Blocks` API with the `Glass` theme. We switched from Streamlit to Gradio to better align with ML model showcasing and to unlock the `gr.Chatbot` component needed for the agentic interface.

---

## Architecture (`src/app/`)

The application strictly separates concerns across three modules:

| Module | Role |
|---|---|
| `main.py` | Gradio `Blocks` lifecycle, event handlers, tab composition |
| `styles.py` | HTML/CSS templates (`get_book_details_html`, `get_chat_book_card_html`) |
| `data_loaders.py` | Engine initialization, image helpers, config loading |

The **Agentic Layer** (`src/agent/`) is a separate package that `main.py` imports. See [Agentic Layer Architecture](../architecture/agentic_layer.md) for its full breakdown.

---

## Tab 1 — Search Interface

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  📚 Semantic Book Recommender                               │
│  Discover your next favorite read using AI-powered search.  │
│                                                             │
│  [Query Textbox] [Category ▼] [Tone ▼] [Find Books]        │
│                                                             │
│  ## Recommendations                                         │
│  [Status Markdown]                                          │
│  [Gallery — 8 columns × 6 rows]                             │
│  [Book Details HTML — triggered on gallery click]           │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Component | ID | Source |
|---|---|---|
| `gr.Textbox` | `user_query` | User natural language input |
| `gr.Dropdown` | `category_dropdown` | Values from `params.yaml` (candidates) |
| `gr.Dropdown` | `tone_dropdown` | Values from `params.yaml` (tone_map display names) |
| `gr.Button` | `submit_button` | "Find recommendations" |
| `gr.Gallery` | `output_gallery` | `(image_url, caption)` tuples |
| `gr.HTML` | `details_output` | Selected book card (glassmorphism HTML) |
| `gr.State` | `results_state` | Full data list shared between Gallery and Details |

### Event Handlers

| Trigger | Handler | Outputs |
|---|---|---|
| `submit_button.click` | `recommend_books()` | `output_gallery`, `results_state`, `status_output` |
| `user_query.submit` | `recommend_books()` | `output_gallery`, `results_state`, `status_output` |
| `output_gallery.select` | `on_select()` | `details_output` |

---

## Tab 2 — AI Book Assistant (v1.3)

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  🤖 AI Book Assistant                                       │
│  Chat with an AI-powered book assistant...                  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  gr.Chatbot (height=500, type="messages")            │   │
│  │                                                      │   │
│  │  User: "I want a dark thriller in a small town"      │   │
│  │  Assistant: Here are some recommendations...        │   │
│  │    📖 **Book Title** by *Author*                    │   │
│  │    ⭐ 4.3 · 🎭 0.89 · 📂 Thriller                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  [Message Textbox .......................] [Send]            │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Component | ID | Source |
|---|---|---|
| `gr.Chatbot` | `chatbot` | `messages` format; renders markdown + book cards |
| `gr.Textbox` | `chat_input` | User message input |
| `gr.Button` | `chat_submit` | "Send" |

### Event Handlers

| Trigger | Handler | Outputs |
|---|---|---|
| `chat_submit.click` | `agent_chat()` | `chatbot`, `chat_input` (cleared) |
| `chat_input.submit` | `agent_chat()` | `chatbot`, `chat_input` (cleared) |

### `agent_chat()` Flow

```python
def agent_chat(user_message, history) -> tuple[history, ""]
    1. Guard: skip empty messages
    2. Lazy-init agent deps via _get_agent_deps() (cached singleton)
    3. Append user message to history
    4. Call chat(user_message, deps) → AgentResponse
    5. Format AgentResponse → Markdown via _format_agent_response()
    6. Append assistant markdown to history
    7. Return (history, "")  ← empty string clears the input
```

---

## Styling

The UI uses `gr.themes.Glass()` as a base, supplemented by:

- **`get_book_details_html(book)`** — Full-width glassmorphism card with thumbnail, rating badge, and mood badge for the Search tab's detail view.
- **`get_chat_book_card_html(book)`** — Compact markdown block for the Chat tab's inline recommendations:
  ```
  📖 **Title** by *Author*
  ⭐ 4.3 · 🎭 0.89 · 📂 Fiction
  > Truncated description (max 200 chars)...
  ```

---

## Launch

```bash
# Local development
.\launch_recommender.bat

# Or directly:
uv run python -m src.app.main
```

Opens at `http://localhost:7860`.

> [!IMPORTANT]
> The AI Book Assistant tab requires `GOOGLE_API_KEY` to be set in `.env`. Without it, the agent will initialize but fail on the first message. The Search tab is unaffected.

---

## Agent Dependencies — Lazy Initialization

The `_agent_deps` global singleton is initialized lazily on the **first chat request**:

```python
_agent_deps: AgentDependencies | None = None

def _get_agent_deps() -> AgentDependencies | None:
    global _agent_deps
    if _agent_deps is None:
        _agent_deps = create_agent_dependencies()
    return _agent_deps
```

This avoids blocking the Gradio startup — the ChromaDB connection and CSV load happen only when a user actually opens the chat tab.

---

## Verification

### Automated Tests

```bash
# All unit tests (9 total: 7 agent + 2 recommender)
uv run pytest tests/unit/ -v

# Full validation with coverage
uv run pytest tests/ -v -m "not integration" --cov=src --cov-report=term-missing

# Full health check (Pyright + Ruff + Pytest + DVC)
.\validate_recommender.bat
```

### Manual Verification Checklist

**Search tab (existing):**
1. Run `.\launch_recommender.bat` and open `http://localhost:7860`.
2. Select **"🔍 Search"** tab.
3. Enter *"A story about forgiveness"*, click **Find recommendations**.
4. Verify gallery populates; click a book to see the detail card.
5. Filter by *"Fiction"* + *"Happy"* and confirm results respect both filters.

**AI Book Assistant tab (new):**
1. Open the **"🤖 AI Book Assistant"** tab.
2. Type: *"I want a dark, suspenseful thriller set in a small town"* and press Enter.
3. Verify the chatbot responds with:
   - A conversational message.
   - 1–5 book cards (title, author, rating, mood, category, truncated description).
   - 2–3 follow-up suggestion bullets.
4. Send a follow-up: *"Something lighter and happier instead"*.
5. Verify the agent switches its search tone filter.
6. Test with a very vague query (e.g., *"a good book"*) — verify the agent asks for clarification or broadens the search.
