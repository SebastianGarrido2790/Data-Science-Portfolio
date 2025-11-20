import json
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.prompts import PromptTemplate


def get_llm(model_name):
    """Initialize the language model based on the provided name."""
    if model_name.lower() == "openai":
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    elif model_name.lower() == "anthropic":
        return ChatAnthropic(temperature=0, model_name="claude-3-5-haiku-latest")
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Choose 'openai' or 'anthropic'."
        )


def load_prompt_from_file(file_path):
    """Load a prompt template from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return PromptTemplate.from_template(f.read().strip())
    except Exception as e:
        raise ValueError(f"Error loading prompt from {file_path}: {str(e)}")


def format_summary(summary, style):
    """Format the summary based on the specified style."""
    if style == "structured":
        # Parse the summary assuming it follows the expected format
        lines = summary.strip().split("\n")
        main_idea = ""
        key_points = []
        conclusion = ""
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("Main Idea:"):
                main_idea = line.replace("Main Idea:", "").strip()
                current_section = None
            elif line.startswith("- "):
                key_points.append(line[2:].strip())
                current_section = "key_points"
            elif line.startswith("Conclusion:"):
                conclusion = line.replace("Conclusion:", "").strip()
                current_section = "conclusion"
            elif current_section == "key_points" and line:
                key_points.append(line)
            elif current_section == "conclusion" and line:
                conclusion += " " + line
        return {
            "main_idea": main_idea,
            "key_points": key_points,
            "conclusion": conclusion.strip(),
        }
    return summary


def summarize_pdf(
    file_path,
    model_name,
    chain_type="map_reduce",
    verbose=False,
    map_prompt=None,
    combine_prompt=None,
    refine_prompt=None,
    token_max=None,
):
    """
    Summarize a PDF file using the specified LLM and chain configuration.

    Args:
        file_path (str): Path to the PDF file.
        model_name (str): LLM to use ('openai' or 'anthropic').
        chain_type (str): Summarization chain type ('stuff', 'map_reduce', 'refine').
        verbose (bool): Enable verbose logging.
        map_prompt (PromptTemplate): Custom prompt for map step (map_reduce or stuff).
        combine_prompt (PromptTemplate): Custom prompt for combine step (map_reduce).
        refine_prompt (PromptTemplate): Custom prompt for refine chain.
        token_max (int): Maximum tokens per chunk for map_reduce chain.

    Returns:
        str: Summary text or error message.
    """
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load_and_split()
        llm = get_llm(model_name)

        # Define default prompts for Machine Learning papers
        default_map_prompt = PromptTemplate.from_template(
            "Summarize this Machine Learning paper excerpt in one sentence (max 50 words), focusing on practical applications: {text}"
        )
        default_combine_prompt = PromptTemplate.from_template(
            """Combine these summaries into a formal summary for data scientists, focusing on practical Machine Learning applications:
Main Idea: [One sentence, max 50 words]
Key Points:
- [Point 1]
- [Point 2]
- [Point 3]
Conclusion: [One sentence on practical impact]
{text}"""
        )
        default_refine_prompt = PromptTemplate.from_template(
            """Refine this summary for a Machine Learning paper, keeping it formal, max 50 words, focusing on practical applications:
Existing summary: {existing_answer}
New text: {text}
Provide an updated summary in this format:
Main Idea: [One sentence]
Key Points:
- [Point 1]
- [Point 2]
- [Point 3]
Conclusion: [One sentence]"""
        )

        # Prepare kwargs for load_summarize_chain
        chain_kwargs = {"verbose": verbose}
        if chain_type == "map_reduce":
            chain_kwargs["map_prompt"] = map_prompt or default_map_prompt
            chain_kwargs["combine_prompt"] = combine_prompt or default_combine_prompt
            if token_max:
                chain_kwargs["token_max"] = token_max
        elif chain_type == "refine":
            chain_kwargs["refine_prompt"] = refine_prompt or default_refine_prompt
        elif chain_type == "stuff":
            chain_kwargs["prompt"] = map_prompt or default_map_prompt
        else:
            raise ValueError(
                f"Unsupported chain_type: {chain_type}. Choose 'stuff', 'map_reduce', or 'refine'."
            )

        # Load and run the summarization chain
        chain = load_summarize_chain(llm, chain_type=chain_type, **chain_kwargs)
        summary = chain.invoke(docs)
        return summary["output_text"]
    except Exception as e:
        return f"Error processing {file_path}: {str(e)}"


def save_summary_text(file_name, summary, output_file, summary_style):
    """Save summary to a text file, handling structured or paragraph format."""
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"\nSummary for {file_name}:\n")
        if summary_style == "structured":
            formatted_summary = format_summary(summary, summary_style)
            f.write(f"Main Idea: {formatted_summary['main_idea']}\n")
            f.write("Key Points:\n")
            for point in formatted_summary["key_points"]:
                f.write(f"- {point}\n")
            f.write(f"Conclusion: {formatted_summary['conclusion']}\n")
        else:
            f.write(f"{summary}\n")
        f.write("=" * 50 + "\n")


def save_summary_json(summaries, output_file, summary_style):
    """Save summaries to a JSON file, handling structured or paragraph format."""
    formatted_summaries = {
        file: (
            format_summary(summary, summary_style)
            if summary_style == "structured"
            else summary
        )
        for file, summary in summaries.items()
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_summaries, f, indent=2)
