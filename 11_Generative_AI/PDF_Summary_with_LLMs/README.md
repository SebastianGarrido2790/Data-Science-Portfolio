# PDF Summarization Script

This project provides two Python scripts to automate the summarization of PDF files, particularly Machine Learning papers, using large language models (LLMs) from OpenAI (GPT-3.5-turbo) or Anthropic (Claude-3.5-Haiku-latest). The scripts process PDFs in a specified directory, generate summaries, and save them to a customizable output file in either text or JSON format.

- **Simple Version (`main.py`)**: A lightweight script with basic functionality for quick summarization using a fixed `map_reduce` chain.
- **Enhanced Version (`custom_prompt/main.py`)**: An advanced script with additional features like customizable summarization chains, structured summary formats, and support for `token_max` configuration, tailored for data scientists summarizing Machine Learning papers.

## Features

### Simple Version (`main.py`)
- **Configuration**: Loads environment variables (API keys, default model) from a `.env` file using `python-dotenv`.
- **Command-Line Arguments**: Supports model selection (`openai` or `anthropic`), PDF directory, output file name, and output format (`text` or `json`) using `argparse`.
- **PDF Processing**: Scans the specified directory for `.pdf` files and processes each one using `PyPDFLoader` to extract text.
- **Model Selection**: Choose between OpenAI's `GPT-3.5-turbo` or Anthropic's `Claude-3.5-Haiku-latest` via command-line arguments or environment variables.
- **Summarization**: Uses LangChain's `map_reduce` summarization chain to generate summaries, handling errors gracefully.
- **Flexible Output**: Saves summaries to a custom file name:
  - **Text Format**: Appends each summary to a `.txt` file with headers and separators.
  - **JSON Format**: Collects summaries in a dictionary and saves them as a `.json` file.
- **Error Handling**: Gracefully handles errors (e.g., invalid PDFs) and includes error messages in the output.
- **Console Feedback**: Prints progress, summaries, and the final output file location.

### Enhanced Version (`custom_prompt/main.py`)
- **All Simple Version Features**: Inherits all the functionality of the simple version.
- **Customizable Chain Type**: Choose between `stuff`, `map_reduce`, or `refine` chains via `--chain-type`.
- **Custom Prompt Support**: Load custom prompt templates from files using `--map-prompt-file`, `--combine-prompt-file`, and `--refine-prompt-file`.
- **Verbose Logging**: Enable detailed logging with `--verbose` for debugging the summarization process.
- **Token Max Support**: Configure the maximum tokens per chunk in the `map_reduce` chain with `--token-max`.
- **Structured Summary Format**: Use `--summary-style=structured` to generate summaries in a formal format for data scientists:
  - **Main Idea**: A single sentence (≤50 words) summarizing the paper’s primary contribution.
  - **Key Points**: Up to 3 practical applications (e.g., model deployment, real-world use cases).
  - **Conclusion**: A sentence on the practical impact for data scientists.
- **Formal Tone for Data Scientists**: Summaries are tailored for Machine Learning papers, focusing on practical applications.
- **Modular Design**: Core functionality is separated into `custom_prompt/src/helper.py` for better maintainability.

## Requirements

- Python 3.8+
- Required Python packages (install via `uv`):
  ```bash
  uv add langchain openai anthropic pypdf python-dotenv
  ```
- API keys for OpenAI and/or Anthropic (depending on the model used).
- PDF files to summarize.

## Setup

**1. Clone the Repository (if applicable):**
```bash
git clone https://github.com/SebastianGarrido2790/Data-Science-Portfolio.git
cd Data-Science-Portfolio/11_PDF_Summary_with_LLMs
```

**2. Install Dependencies:**
- Initialize the project with `uv` (already done if you ran `uv init`):
  ```bash
  uv init
  ```
- Activate the virtual environment:
  ```bash
  .venv\Scripts\activate
  ```
- Install dependencies using `uv`:
  ```bash
  uv add langchain openai anthropic pypdf python-dotenv
  ```

**Packages and Versions:**
- `langchain==0.2.16`: For the summarization chain and document loading utilities.
- `openai==1.35.13`: For accessing OpenAI's GPT-3.5-turbo model.
- `anthropic==0.28.0`: For accessing Anthropic's Claude-3.5-Haiku-latest model.
- `pypdf==4.2.0`: For reading and parsing PDF files.
- `python-dotenv==1.0.1`: For loading environment variables from a `.env` file.

**3. Configure Environment Variables:**
Create a `.env` file in the project root with the following content:
```plain
OPENAI_API_KEY="your-openai-api-key"
ANTHROPIC_API_KEY="your-anthropic-api-key"
LLM_MODEL="openai"  # Optional: default model ("openai" or "anthropic")
```

Replace `your-openai-api-key` and `your-anthropic-api-key` with your actual API keys. The `LLM_MODEL` variable is optional and sets the default model if not specified via command-line.

**4. Prepare PDFs:**
Place the PDF files you want to summarize in a directory (e.g., the project root or a subfolder like `pdfs`).

## Usage

### Simple Version (`main.py`)
Run the script using the `python main.py` command with optional arguments to customize its behavior.

**Command-Line Arguments:**
- `--model`: Specify the LLM to use (`openai` or `anthropic`). Defaults to `LLM_MODEL` environment variable or `openai`.
- `--directory`: Directory containing PDF files. Defaults to the current directory (`.`).
- `--output`: Output file name (without extension). Defaults to `summaries`.
- `--format`: Output format (`text` or `json`). Defaults to `text`.

**Examples:**

**1. Summarize PDFs in the current directory using OpenAI, saving as text:**
```bash
python main.py
```

Output file: `summaries.txt`

**2. Use Anthropic, specify a directory, and save as JSON:**
```bash
python main.py --model anthropic --directory ./pdfs --output my_summaries --format json
```

Output file: `my_summaries.json`

**3. Use OpenAI, custom output file, text format:**
```bash
python main.py --model openai --directory ./pdfs --output results
```

Output file: `results.txt`

### Enhanced Version (`custom_prompt/main.py`)
Run the script using `python custom_prompt/main.py` with additional arguments for advanced configuration, particularly for summarizing Machine Learning papers.

**Command-Line Arguments:**
- Inherits all arguments from the simple version.
- `--chain-type`: Summarization chain type (`stuff`, `map_reduce`, or `refine`). Defaults to `map_reduce`.
- `--verbose`: Enable verbose logging for the summarization chain.
- `--map-prompt-file`: Path to a file containing a custom map prompt template (for `map_reduce` or `stuff`).
- `--combine-prompt-file`: Path to a file containing a custom combine prompt template (for `map_reduce`).
- `--refine-prompt-file`: Path to a file containing a custom refine prompt template (for `refine`).
- `--token-max`: Maximum tokens per chunk for `map_reduce` chain (e.g., `1000`). Defaults to None.
- `--summary-style`: Summary style (`paragraph` or `structured`). Defaults to `paragraph`.

**Examples:**

**1. Summarize PDFs with the default `map_reduce` chain, structured format:**
```bash
python custom_prompt/main.py --summary-style structured
```

Output file: `summaries.txt`

**2. Use Anthropic, `refine` chain, custom prompts, and JSON output:**
```bash
python custom_prompt/main.py --model anthropic --directory ./pdfs --output my_summaries --format json --chain-type refine --refine-prompt-file refine_prompt.txt --summary-style structured
```

Output file: `my_summaries.json`

**3. Use OpenAI, `map_reduce` chain, verbose mode, and `token-max`:**
```bash
python custom_prompt/main.py --model openai --directory ./pdfs --output results --chain-type map_reduce --verbose --token-max 1000 --summary-style structured
```

Output file: `results.txt`

## Output

### Simple Version Output
- **Console Output**: Prints the model used, each PDF being processed, its summary, and the final output file location.
- **Text Format** (e.g., `summaries.txt`):
```plain
Summary for document1.pdf:
This document discusses the impact of climate change on coastal ecosystems.
==================================================

Summary for document2.pdf:
Error processing document2.pdf: Invalid PDF format
==================================================

Summary for document3.pdf:
The report covers advancements in renewable energy technologies.
==================================================
```
- **JSON Format** (e.g., `summaries.json`):
```json
{
  "document1.pdf": "This document discusses the impact of climate change on coastal ecosystems.",
  "document2.pdf": "Error processing document2.pdf: Invalid PDF format",
  "document3.pdf": "The report covers advancements in renewable energy technologies."
}
```

### Enhanced Version Output
- **Console Output**: Includes additional details like chain type, verbose mode, token max, and summary style.
- **Text Format (Structured, e.g., `summaries.txt`)**:
```plain
Summary for document1.pdf:
Main Idea: This paper introduces a novel ML model for real-time fraud detection.
Key Points:
- Enhances fraud detection in banking systems.
- Reduces false positives by 15%.
- Deployable on edge devices.
Conclusion: Offers practical advancements for secure financial transactions.
==================================================
```
- **JSON Format (Structured, e.g., `summaries.json`)**:
```json
{
  "document1.pdf": {
    "main_idea": "This paper introduces a novel ML model for real-time fraud detection.",
    "key_points": [
      "Enhances fraud detection in banking systems.",
      "Reduces false positives by 15%.",
      "Deployable on edge devices."
    ],
    "conclusion": "Offers practical advancements for secure financial transactions."
  }
}
```

## Directory Structure
```
custom_prompt/
├── main.py            # Enhanced version of the script
├── src/
│   └── helper.py      # Helper functions for the enhanced version
├── .env
├── .python-version
├── main.py            # Simple version of the script
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Notes

- **Choosing a Version**: Use `main.py` for simple summarization tasks, or `custom_prompt/main.py` for advanced features like structured summaries and customizable chains.
- **API Keys**: Ensure the appropriate API key is set in `.env` for the selected model. Missing keys will cause authentication errors.
- **File Overwrite**: The output file is overwritten each time the script runs. To append instead, modify the script by removing the `os.remove(output_file)` line.
- **Error Handling**: If a PDF fails to process (e.g., corrupt file), the error is included in the output, and the script continues with the next file.
- **Prompt Design**: The default prompts are tailored for Machine Learning papers, but users can override them with custom prompt files.
- **Custom Prompts (Enhanced Version)**: Ensure prompt files contain valid templates with appropriate variables (e.g., `{text}` for map prompts). Custom prompts should be tested to ensure they produce meaningful summaries.
- **Token Max (Enhanced Version)**: The `--token-max` parameter applies only to the `map_reduce` chain. Set it based on your LLM’s token limits.
- **Extending the Script**:
  - Add more output formats (e.g., CSV) by extending the `save_summary_*` functions.
  - Support additional LLMs by updating the `get_llm` function in `helper.py`.
  - Enable parallel processing for faster handling of large PDF collections.

## Troubleshooting

- **ModuleNotFoundError**: Ensure all dependencies are installed (`uv add langchain openai anthropic pypdf python-dotenv`).
- **Authentication Errors**: Verify that API keys in `.env` are correct and match the selected model.
- **No PDFs Found**: Check that the specified directory contains `.pdf` files and the path is correct.
- **Invalid PDF**: If a PDF fails to process, the error will be included in the output, and other PDFs will still be processed.
- **Prompt Errors (Enhanced Version)**: Ensure custom prompt files are correctly formatted and contain the required variables.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE.txt) for details.

## Contact
For questions or contributions, please open an issue or submit a pull request on the repository.
