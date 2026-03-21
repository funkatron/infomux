#!/usr/bin/env python3
"""
Basic OpenAI smoke test for transcript summarization.

Usage:
    uv run python smoke_openai_summary.py /path/to/transcript.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running from repo root without package install.
REPO_SRC = Path(__file__).resolve().parent / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from infomux.ext.openai import OpenAIError, chat_completion
from infomux.env import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call OpenAI and summarize a text file.",
    )
    parser.add_argument("input_file", type=Path, help="Path to text file to summarize")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("INFOMUX_OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model (default: INFOMUX_OPENAI_MODEL or gpt-4o-mini)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("INFOMUX_OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI API base URL (default: INFOMUX_OPENAI_BASE_URL or official API)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="Max chars to send from input file (default: 12000)",
    )
    return parser.parse_args()


def main() -> int:
    # Match CLI behavior: load .env from current working directory.
    load_dotenv()

    args = parse_args()

    api_key = os.environ.get("INFOMUX_OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: INFOMUX_OPENAI_API_KEY is not set", file=sys.stderr)
        return 1

    if not args.input_file.exists():
        print(f"ERROR: input file not found: {args.input_file}", file=sys.stderr)
        return 1

    text = args.input_file.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        print("ERROR: input file is empty", file=sys.stderr)
        return 1

    text = text[: args.max_chars]

    system_prompt = (
        "You summarize transcripts clearly and concisely. "
        "Do not fabricate details."
    )
    user_prompt = (
        "Summarize the following file content.\n\n"
        "Return:\n"
        "1) A 1-2 sentence overview\n"
        "2) 3-7 bullet key points\n\n"
        "CONTENT:\n"
        f"{text}"
    )

    try:
        summary, output_tokens = chat_completion(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            top_p=0.9,
            max_tokens=700,
            seed=42,
        )
    except OpenAIError as e:
        print(f"ERROR: OpenAI request failed: {e}", file=sys.stderr)
        return 2

    print("=== OpenAI Smoke Test Summary ===")
    print(f"model: {args.model}")
    print(f"input_file: {args.input_file}")
    print(f"input_chars_sent: {len(text)}")
    print(f"output_tokens: {output_tokens}")
    print()
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

