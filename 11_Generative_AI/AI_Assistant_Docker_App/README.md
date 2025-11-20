# AI Chat App with Memory

![AI Assistant with Memory.png](AI%20Assistant%20with%20Memory.png)

A production-ready AI chat application built with Docker, Streamlit, and LangChain. It supports seamless switching between a local open-source LLM (via Docker Model Runner) and a cloud-based model (via OpenRouter), with full conversation history retention for context-aware responses.

## Features

- **Local and Cloud LLMs**: Run local models (e.g., gemma3) via Docker or switch to cloud models (e.g., openai/gpt-oss-20b) with a checkbox.
- **Streamlit Interface**: Clean, user-friendly chat UI with message history display.
- **Context-Aware Responses**: Uses LangChain's `ConversationBufferMemory` to maintain full chat context across model switches.
- **Containerized Deployment**: Fully containerized with Docker Compose for easy setup and scalability.
- **Robust Error Handling**: Includes logging and health checks for reliability.
- **Conversation Persistence**: Save chat history to a JSON file for future reference.
- **Dependency Management**: Uses `uv` for fast, reproducible package installation.

## Models

- **Local Model**: `gemma3`
  - **Description**: Google’s latest Gemma, small yet strong for chat and generation.
- **Cloud Model**: `openai/gpt-oss-20b`
  - **Description**: gpt-oss-20b is an open-weight 21B parameter model released by OpenAI under the Apache 2.0 license. It uses a Mixture-of-Experts (MoE) architecture with 3.6B active parameters per forward pass, optimized for lower-latency inference and deployability on consumer or single-GPU hardware. The model is trained in OpenAI’s Harmony response format and supports reasoning level configuration, fine-tuning, and agentic capabilities, including function calling, tool use, and structured outputs.

## Prerequisites

- Docker and Docker Compose
- An OpenRouter API key (for cloud model access)
- A system with sufficient resources to run local LLMs (e.g., 8GB RAM for gemma3)

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SebastianGarrido2790/Data-Science-Portfolio.git
   cd ai-chat-app
   ```

2. **Configure Environment Variables**:
   - Copy the `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and replace `YOUR_OPENROUTER_API_KEY` with your OpenRouter API key:
     ```plaintext
     LOCAL_BASE_URL=http://host.docker.internal:8080/engines/llama.cpp/v1
     REMOTE_BASE_URL=https://openrouter.ai/api/v1
     LOCAL_MODEL_NAME=ai/gemma3
     REMOTE_MODEL_NAME=openai/gpt-oss-20b
     OPENROUTER_API_KEY=your-api-key-here
     ```

3. **Build and Run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```
   This starts the Streamlit app and the local LLM service.

4. **Access the Application**:
   - Open your browser and navigate to `http://localhost:8501`.
   - The app will display a chat interface with a checkbox to toggle between local and cloud models.

## Usage

- **Chat Interface**:
  - Type your message in the input box at the bottom.
  - Use the "Use cloud model (Think harder...)" checkbox to switch between the local LLM (faster, lighter) and the cloud LLM (more powerful).
  - View the conversation history displayed above the input box.
- **Save Conversation**:
  - Click the "Save Conversation" button to store the chat history in `conversation_history.json`.
- **Monitoring**:
  - Check `app.log` for application logs, including errors and model usage details.

## Project Structure

- `app.py`: Core application logic, including LLM initialization, configuration, and conversation handling.
- `gui.py`: Streamlit GUI for rendering the chat interface and handling user interactions.
- `Dockerfile`: Defines the Docker image with `uv` for dependency management and Python 3.12.
- `docker-compose.yaml`: Configures the app and LLM services with health checks.
- `pyproject.toml`: Specifies dependencies and `uv` configuration.
- `.env`: Environment variables for model configuration and API keys.
- `conversation_history.json`: Stores saved chat history (generated on save).
- `app.log`: Logs application events and errors.

## Development Notes

### Design Considerations

- **Reliability**: Error handling for configuration, LLM initialization, and API calls. Health checks ensure service availability.
- **Scalability**: LangChain's `ConversationChain` optimizes conversation handling. Docker Compose supports multi-container scaling.
- **Maintainability**: Separated core logic (`app.py`) from UI (`gui.py`) for modular development. Logging aids debugging.
- **Adaptability**: Environment variables allow flexible model configuration. The `AIChatApp` class can be reused in other frontends (e.g., API).

### Extending the Project

- **Add New Models**: Update `LOCAL_MODEL_NAME` or `REMOTE_MODEL_NAME` in `.env` to use different LLMs.
- **Custom Memory**: Extend `ConversationBufferMemory` with custom summarization or truncation for long conversations.
- **API Integration**: Expose `AIChatApp` methods via a FastAPI endpoint for programmatic access.
- **UI Enhancements**: Add Streamlit components for richer interactions (e.g., file uploads, model parameter tuning).

### Troubleshooting

- **Docker Issues**: Ensure Docker Desktop is running and `host.docker.internal` resolves correctly. On Linux, replace with `172.17.0.1`.
- **API Key Errors**: Verify the OpenRouter API key in `.env`. Check logs in `app.log` for details.
- **LLM Unresponsive**: Confirm the local LLM service is healthy via `docker-compose logs llm`. Adjust health check intervals if needed.

## Dependencies

- Python 3.12
- LangChain (`langchain`, `langchain-openai`)
- Streamlit
- Docker Model Runner (for local LLMs)
- `uv` (for package management)

## License

MIT License. See [LICENSE](LICENSE.txt) for details.

## Contact

For issues or contributions, please open a GitHub issue or contact the maintainer at `sebastiangarrido2790@gmail.com`.