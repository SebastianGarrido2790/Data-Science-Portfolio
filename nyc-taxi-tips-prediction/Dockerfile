# -----------------------------------------------------------------------------
# FTI ARCHITECTURE & AGENTIC ORCHESTRATION LAYER
# -----------------------------------------------------------------------------
# This repository follows a strict decoupled "Brain vs. Brawn" architecture
# using LangGraph and Gemini-2.5-flash for the natural language UI.
#
# DO NOT use this root Dockerfile directly.
# The application is split into specialized microservices to highlight the agentic workflow:
#
# 1. The Brain (Agentic Frontend UI)
#    - See: docker/frontend.Dockerfile
#    - Orchestrates the ReAct pattern using LangGraph.
#    - Needs GOOGLE_API_KEY for the Gemini-2.5-flash model.
#
# 2. The Brawn (Model Serving API)
#    - See: docker/backend.Dockerfile
#    - Deterministic execution and FastAPI inference.
#
# Use `docker-compose up --build` to launch the full Agentic Chat UI.
# -----------------------------------------------------------------------------

# Ensure the file validates cleanly if inadvertently built
FROM scratch
