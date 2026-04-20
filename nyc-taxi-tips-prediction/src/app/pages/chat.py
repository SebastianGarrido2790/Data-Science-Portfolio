"""
Interactive chat page for the NYC Taxi Tips Streamlit app.

This module handles the UI and logic for interacting with the Agentic Taxi Analyst.
"""

import streamlit as st
from dotenv import load_dotenv

from src.agents.taxi_analyst_agent import AgentConfigError, get_taxi_analyst_agent


def render_chat() -> None:
    """Renders the Agentic Chat UI explicitly.

    This function manages the session state for messages, initialises the LangGraph
    agent, and handles the user input and agent response lifecycle.
    """
    load_dotenv()  # Ensure .env is loaded (idempotent; safe to call multiple times)

    st.header("🤖 Agentic Taxi Analyst")
    st.markdown(
        "Chat naturally with the **Agentic Taxi Analyst** — describe your ride and get an "
        "ML-powered tip prediction, or ask any NYC taxi question."
    )

    # --- Session State Initialisation ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Lazy-load the compiled LangGraph agent once per browser session
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = get_taxi_analyst_agent()
        except AgentConfigError as cfg_err:
            st.session_state.agent = None
            st.session_state.agent_error = str(cfg_err)

    agent = st.session_state.get("agent")

    # --- API Key Missing Banner ---
    if agent is None:
        st.error(
            f"⚠️ **Agent not initialised:** "
            f"{st.session_state.get('agent_error', 'Unknown configuration error.')}  \n"
            "Open `.env`, set `GOOGLE_API_KEY=AIza<your-key>`, and restart the app.",
            icon="🔑",
        )
        return

    # --- Toolbar ---
    _, col_clear = st.columns([5, 1])
    with col_clear:
        if st.button("🗑️ Clear Chat", width="stretch"):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")

    # --- Chat History ---
    messages: list[dict[str, str]] = st.session_state.messages
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Chat Input ---
    user_input = st.chat_input(
        "Describe your ride… e.g. 'Predict a tip for a 5-mile JFK trip at 3 PM, $25 fare.'"
    )

    if user_input:
        # Append and render the user's message immediately
        messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # --- Agent Invocation ---
        with st.chat_message("assistant"):
            with st.spinner("Analysing your request…"):
                try:
                    # Construct the full conversation history for LangGraph's state
                    langchain_history = []
                    for m in messages:
                        if m["role"] == "user":
                            langchain_history.append(("human", m["content"]))
                        elif m["role"] == "assistant":
                            langchain_history.append(("ai", m["content"]))

                    result = agent.invoke({"messages": langchain_history})

                    # The last message in the graph output is the assistant's reply
                    msg = result["messages"][-1]
                    raw_content = msg.content

                    if isinstance(raw_content, list):
                        # Extract and join all 'text' parts (handles Gemini's block-based format)
                        assistant_reply = "".join(
                            [
                                part.get("text", "")
                                for part in raw_content
                                if isinstance(part, dict) and part.get("type") == "text"
                            ]
                        )
                    else:
                        assistant_reply = str(raw_content)
                except Exception as agent_err:
                    err_name = type(agent_err).__name__
                    if (
                        "RateLimitError" in err_name
                        or "quota" in str(agent_err).lower()
                        or "RESOURCE_EXHAUSTED" in str(agent_err)
                    ):
                        assistant_reply = (
                            f"⚠️ **Brain Error (Google Quota):**\n\n"
                            f"`{err_name}: {agent_err}`\n\n"
                            "It looks like your **Google API Key** has hit a rate limit. "
                            "Please check your usage at [Google AI Studio](https://aistudio.google.com/app/apikey)."
                        )
                    elif "ConnectionError" in err_name or "localhost:8000" in str(agent_err):
                        assistant_reply = (
                            f"⚠️ **Brawn Error (FastAPI Offline):**\n\n"
                            f"`{err_name}: {agent_err}`\n\n"
                            "The analyst can't reach the tip prediction model. Please ensure the "
                            "**FastAPI backend** is running "
                            "(`uv run uvicorn src.api.predict_api:app --reload`)."
                        )
                    else:
                        assistant_reply = (
                            f"⚠️ **The analyst encountered an unexpected error:**\n\n"
                            f"`{err_name}: {agent_err}`\n\n"
                            "Check your `.env` or ensure all backend services are active."
                        )

            st.markdown(assistant_reply)

        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
