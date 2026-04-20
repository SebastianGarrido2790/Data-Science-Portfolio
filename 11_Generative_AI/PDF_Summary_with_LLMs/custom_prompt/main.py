import os
import argparse
from dotenv import load_dotenv
from src.helper import (
    get_llm,
    load_prompt_from_file,
    summarize_pdf,
    save_summary_text,
    save_summary_json,
)

load_dotenv()

if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Summarize Machine Learning PDF papers using a specified LLM."
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
    parser.add_argument(
        "--chain-type",
        type=str,
        default="map_reduce",
        choices=["stuff", "map_reduce", "refine"],
        help="Summarization chain type: 'stuff', 'map_reduce', or 'refine' (default: map_reduce)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for the summarization chain",
    )
    parser.add_argument(
        "--map-prompt-file",
        type=str,
        help="Path to file containing custom map prompt template (for map_reduce or stuff)",
    )
    parser.add_argument(
        "--combine-prompt-file",
        type=str,
        help="Path to file containing custom combine prompt template (for map_reduce)",
    )
    parser.add_argument(
        "--refine-prompt-file",
        type=str,
        help="Path to file containing custom refine prompt template (for refine)",
    )
    parser.add_argument(
        "--token-max",
        type=int,
        default=None,
        help="Maximum tokens per chunk for map_reduce chain (default: None)",
    )
    parser.add_argument(
        "--summary-style",
        type=str,
        default="paragraph",
        choices=["paragraph", "structured"],
        help="Summary style: 'paragraph' or 'structured' (default: paragraph)",
    )
    args = parser.parse_args()

    # Set output file extension based on format
    output_file = f"{args.output}.{args.format}"

    # Load custom prompts if provided
    map_prompt = (
        load_prompt_from_file(args.map_prompt_file) if args.map_prompt_file else None
    )
    combine_prompt = (
        load_prompt_from_file(args.combine_prompt_file)
        if args.combine_prompt_file
        else None
    )
    refine_prompt = (
        load_prompt_from_file(args.refine_prompt_file)
        if args.refine_prompt_file
        else None
    )

    # Clear the output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Collect all PDF files
    pdf_files = [f for f in os.listdir(args.directory) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in directory: {args.directory}")
    else:
        print(f"Using model: {args.model}")
        print(f"Using chain type: {args.chain_type}")
        print(f"Verbose mode: {'enabled' if args.verbose else 'disabled'}")
        print(f"Token max: {args.token_max if args.token_max else 'default'}")
        print(f"Summary style: {args.summary_style}")
        print(f"Saving summaries to: {output_file} (format: {args.format})")

        # Store summaries for JSON output
        summaries = {}

        for file in pdf_files:
            print(f"\nProcessing {file}...")
            file_path = os.path.join(args.directory, file)
            summary = summarize_pdf(
                file_path,
                args.model,
                chain_type=args.chain_type,
                verbose=args.verbose,
                map_prompt=map_prompt,
                combine_prompt=combine_prompt,
                refine_prompt=refine_prompt,
                token_max=args.token_max,
            )
            print(f"Summary for {file}:")
            print(summary)

            if args.format == "text":
                save_summary_text(file, summary, output_file, args.summary_style)
            else:  # JSON format
                summaries[file] = summary

        # Save JSON summaries if format is JSON
        if args.format == "json":
            save_summary_json(summaries, output_file, args.summary_style)

        print(f"\nSummaries saved to {output_file}")
