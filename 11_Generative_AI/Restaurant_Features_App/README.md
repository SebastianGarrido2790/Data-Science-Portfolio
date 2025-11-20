# Restaurant Features Generator

A Streamlit web application that leverages LangChain to generate a fancy restaurant name, menu items, ambiance description, and a signature dish for a user-specified cuisine, with support for dietary preferences. The app is designed for **reliability** (error handling and retries), **scalability** (context retention via memory), **maintainability** (modular code), and **adaptability** (customizable inputs).

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project uses LangChain with OpenAI's `gpt-3.5-turbo` model to generate creative restaurant concepts based on user-selected cuisines and optional dietary preferences (e.g., vegan, gluten-free). The app employs a sequential chain of prompts to produce:
- A fancy restaurant name.
- A list of menu items.
- A description of the restaurant’s ambiance.
- A highlighted signature dish with a brief description.

The application is built with Streamlit for an intuitive user interface and uses `uv` for efficient package management.

## Features
- **Cuisine Selection**: Choose from predefined cuisines (Indian, Italian, Mexican, Arabic, American) or enter a custom cuisine.
- **Dietary Preferences**: Customize menu items, ambiance, and signature dish with user-defined preferences.
- **Context Retention**: LangChain’s `ConversationBufferMemory` retains generation history for consistent outputs.
- **Error Handling**: Robust validation and retry mechanisms ensure reliable API interactions.
- **Responsive UI**: Streamlit interface with sidebar inputs and clear output sections.

## Installation
### Prerequisites
- Python 3.9 or higher (but less than 4.0)
- `uv` package manager
- OpenAI API key

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SebastianGarrido2790/Data-Science-Portfolio
   cd restaurant-features-app
   ```

2. **Initialize Virtual Environment and Install Dependencies**:
   ```bash
   uv sync
   ```
   This uses the `pyproject.toml` to install dependencies in a `.venv` environment.

3. **Set Up Environment Variables**:
   Create a `.env` file in the project root with your OpenAI API key:
   ```bash
   echo OPENAI_API_KEY=your_key_here > .env
   ```

## Usage
1. **Run the Application**:
   ```bash
   uv run streamlit run main.py
   ```
   This starts the Streamlit server, typically accessible at `http://localhost:8501`.

2. **Interact with the App**:
   - Select a cuisine from the sidebar dropdown or enter a custom cuisine.
   - Optionally specify dietary preferences (e.g., "vegan", "gluten-free").
   - Click "Generate Restaurant" to view the restaurant name, menu items, ambiance, signature dish, and conversation history.

3. **Example Output**:
   - **Cuisine**: Italian
   - **Preferences**: Vegan options preferred
   - **Output**:
     - **Restaurant Name**: Trattoria Verde
     - **Menu Items**: Vegan Margherita Pizza, Eggplant Parmesan, Mushroom Risotto
     - **Ambiance**: A cozy setting with warm lighting and rustic wooden decor.
     - **Signature Dish**: Vegan Tiramisu - A creamy, coffee-infused dessert with plant-based mascarpone.

## Project Structure
```
restaurant-features-app/
├── .venv/                  # Virtual environment (created by uv)
├── .env                    # Environment variables (not tracked)
├── langchain_helper.py     # LangChain logic for generating restaurant details
├── main.py                 # Streamlit app for user interface
├── pyproject.toml          # Project metadata and dependencies
└── README.md               # Project documentation
```

## Dependencies
Specified in `pyproject.toml`:
- `streamlit==1.38.0`: Web app framework
- `langchain==0.3.1`: Language model chaining
- `langchain-openai==0.2.1`: OpenAI integration
- `python-dotenv==1.0.1`: Environment variable management
- `tenacity==8.3.0`: Retry mechanism for API calls

Install with:
```bash
uv sync
```

## Configuration
- **OpenAI API Key**: Store in `.env` as `OPENAI_API_KEY`.
- **Python Version**: Ensure Python `>=3.9,<4.0` (configured in `pyproject.toml`).
- **uv**: Managed via `[tool.uv]` in `pyproject.toml`.

## Contributing
1. Fork the repository (if applicable).
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request with a clear description of changes.

Please ensure code adheres to:
- PEP 8 style guidelines.
- Modular design for maintainability.
- Comprehensive error handling.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE.txt) for details.
