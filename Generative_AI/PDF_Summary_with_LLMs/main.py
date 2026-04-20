import os
import argparse
import json
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI, ChatAnthropic

load_dotenv()


def get_llm(model_name):
    if model_name.lower() == "openai":
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    elif model_name.lower() == "anthropic":
        return ChatAnthropic(temperature=0, model_name="claude-3-5-haiku-latest")
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Choose 'openai' or 'anthropic'."
        )


def summarize_pdf(file_path, model_name):
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load_and_split()
        llm = get_llm(model_name)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.invoke(docs)
        return summary["output_text"]
    except Exception as e:
        return f"Error processing {file_path}: {str(e)}"


def save_summary_text(file_name, summary, output_file):
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"\nSummary for {file_name}:\n")
        f.write(f"{summary}\n")
        f.write("=" * 50 + "\n")


def save_summary_json(summaries, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Summarize PDF files using a specified LLM."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("LLM_MODEL", "openai"),
        choices=["openai", "anthropic"],
        help="LLM model to use: 'openai' or 'anthropic' (default: env var LLM_MODEL or 'openai')",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=".",
        help="Directory containing PDF files (default: current directory)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="summaries",
        help="Output file name without extension (default: summaries)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=["text", "json"],
        help="Output format: 'text' or 'json' (default: text)",
    )
    args = parser.parse_args()

    # Set output file extension based on format
    output_file = f"{args.output}.{args.format}"

    # Clear the output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Collect all PDF files
    pdf_files = [f for f in os.listdir(args.directory) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in directory: {args.directory}")
    else:
        print(f"Using model: {args.model}")
        print(f"Saving summaries to: {output_file} (format: {args.format})")

        # Store summaries for JSON output
        summaries = {}

        for file in pdf_files:
            print(f"\nProcessing {file}...")
            file_path = os.path.join(args.directory, file)
            summary = summarize_pdf(file_path, args.model)
            print(f"Summary for {file}:")
            print(summary)

            if args.format == "text":
                save_summary_text(file, summary, output_file)
            else:  # JSON format
                summaries[file] = summary

        # Save JSON summaries if format is JSON
        if args.format == "json":
            save_summary_json(summaries, output_file)

        print(f"\nSummaries saved to {output_file}")
