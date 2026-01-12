"""
Summarize step: generate summary from transcript using Ollama.

Uses Ollama for local LLM inference. Supports configurable models
and records all parameters for reproducibility.

Requires:
- Ollama running locally (ollama serve)
- Model pulled (e.g., ollama pull qwen2.5:14b-instruct)

Environment:
- INFOMUX_OLLAMA_MODEL: Model to use (default: qwen2.5:7b-instruct)
- INFOMUX_OLLAMA_URL: Ollama API URL (default: http://localhost:11434)
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from infomux.llm import (
    DEFAULT_SUMMARIZE_PARAMS,
    GenerationParams,
    ModelInfo,
    StepModelRecord,
    hash_text,
)
from infomux.log import get_logger
from infomux.steps import StepError, StepResult, register_step

logger = get_logger(__name__)

# Environment variables
ENV_OLLAMA_MODEL = "INFOMUX_OLLAMA_MODEL"
ENV_OLLAMA_URL = "INFOMUX_OLLAMA_URL"
ENV_CONTENT_TYPE_HINT = "INFOMUX_CONTENT_TYPE_HINT"

# Defaults
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Chunking settings
DEFAULT_CHUNK_SIZE = 12000  # chars (~3k tokens)
CHUNK_OVERLAP = 500  # chars overlap between chunks
MIN_CHUNK_THRESHOLD = 15000  # Only chunk if longer than this

# Output filename
SUMMARY_FILENAME = "summary.md"

# System prompt for summarization
SUMMARIZE_SYSTEM_PROMPT = """You extract key information from transcripts of audio/video recordings.

Output format (use ALL sections, write "None identified" if empty):

## Overview
One sentence: what this recording is, who is speaking/present, and when (if mentioned).

## Action Items
CRITICAL: Extract EVERY task, to-do, or commitment mentioned. Never skip action items.
- [ ] **[Person]**: [Task] (due: [date if mentioned])

## Key Takeaways
Main insights, lessons, or important points from the content.
- [Specific takeaway with context]

## Decisions Made
- [Decision with context/reasoning]

## Dates & Events Mentioned
- [Date]: [Event/deadline/meeting/reference]

## People Mentioned
- **[Name]**: [Role or context]

## Notable Quotes
Memorable or important statements worth preserving.
- "[Quote]" - [Speaker if known]

## Open Questions / Follow-ups
- [Unresolved question or topic needing follow-up]

Rules:
- Analyze the ENTIRE transcript from beginning to end
- Extract EVERY action item - this is the most important requirement
- Use exact names from the transcript
- Be specific with details, names, and numbers
- Adapt emphasis based on content type (meeting vs talk vs interview)"""

# Content type templates for the hint system
CONTENT_TYPE_HINTS = {
    "meeting": "This is a work meeting. Focus on: action items, decisions, assignments, deadlines.",
    "talk": "This is a conference talk or presentation. Focus on: key concepts, takeaways, quotes.",
    "podcast": "This is a podcast or interview. Focus on: main topics, guest insights, recommendations.",
    "lecture": "This is an educational lecture. Focus on: concepts taught, examples, key definitions.",
    "standup": "This is a standup or status meeting. Focus on: blockers, progress updates, next steps.",
    "1on1": "This is a 1-on-1 meeting. Focus on: feedback, goals, action items, concerns raised.",
}

# Prompt for extracting info from each chunk
CHUNK_EXTRACT_PROMPT = """Extract key information from this SECTION of a longer transcript.
This is section {chunk_num} of {total_chunks}.
{content_context}
TRANSCRIPT SECTION:
{chunk_text}

---

Extract ALL of the following from this section. Write "None in this section" if not found:

**Action Items:** (tasks, to-dos, commitments - include WHO and WHEN if mentioned)
**Decisions:** (conclusions reached, choices made)
**Dates/Events:** (any dates, deadlines, scheduled events mentioned)
**People:** (names mentioned with context)
**Key Points:** (main topics, important statements)
**Notable Quotes:** (memorable or important statements)
**Questions/Follow-ups:** (unresolved items)"""

# Prompt for combining chunk summaries
COMBINE_PROMPT = """You have extracted information from {total_chunks} sections of a transcript.
Combine these into a single coherent summary.
{content_context}
EXTRACTED INFORMATION FROM ALL SECTIONS:

{chunk_summaries}

---

Now create a unified summary. Merge duplicates, resolve conflicts, and organize coherently.
Use EXACTLY these headings:

## Overview
One sentence: what this recording is, who is speaking/present, and when (if mentioned).

## Action Items
CRITICAL: Include EVERY action item from all sections. Never skip action items.
- [ ] **[Person]**: [Task] (due: [date if mentioned])

## Key Takeaways
Main insights from the ENTIRE recording, not just one section.

## Decisions Made

## Dates & Events Mentioned

## People Mentioned

## Notable Quotes

## Open Questions / Follow-ups"""


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks.

    Tries to split at sentence boundaries for cleaner chunks.

    Args:
        text: Text to split.
        chunk_size: Target size per chunk in characters.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If this isn't the last chunk, try to find a good break point
        if end < len(text):
            # Look for sentence end (.!?) in the last 20% of the chunk
            search_start = end - int(chunk_size * 0.2)
            search_region = text[search_start:end]

            # Find last sentence-ending punctuation
            best_break = -1
            for punct in [". ", ".\n", "! ", "!\n", "? ", "?\n"]:
                pos = search_region.rfind(punct)
                if pos > best_break:
                    best_break = pos

            if best_break > 0:
                end = search_start + best_break + 2  # Include the punctuation and space

        chunks.append(text[start:end].strip())

        # Move start, accounting for overlap
        start = end - overlap
        if start >= len(text):
            break

    return chunks


@register_step
@dataclass
class SummarizeStep:
    """
    Pipeline step to summarize a transcript using Ollama.

    Records model name, seed, and parameters for reproducibility.
    """

    name: str = "summarize"
    model: str | None = None
    params: GenerationParams | None = None

    def execute(
        self, input_path: Path, output_dir: Path
    ) -> tuple[list[Path], StepModelRecord]:
        """
        Summarize the transcript.

        Args:
            input_path: Path to the transcript file.
            output_dir: Directory to write the summary.

        Returns:
            Tuple of (output paths, model record for reproducibility).

        Raises:
            StepError: If Ollama is not available or summarization fails.
        """
        # Get configuration
        model_name = self.model or os.environ.get(ENV_OLLAMA_MODEL, DEFAULT_MODEL)
        ollama_url = os.environ.get(ENV_OLLAMA_URL, DEFAULT_OLLAMA_URL)
        content_hint = os.environ.get(ENV_CONTENT_TYPE_HINT, "")

        # Get or create generation params with seed
        params = self.params or DEFAULT_SUMMARIZE_PARAMS
        params = params.with_seed()  # Ensure we have a seed

        # Read transcript
        if not input_path.exists():
            raise StepError(self.name, f"transcript not found: {input_path}")

        transcript = input_path.read_text()
        if not transcript.strip():
            raise StepError(self.name, "transcript is empty")

        logger.info("summarizing transcript (%d chars)", len(transcript))
        logger.debug("using model: %s", model_name)
        logger.debug("using seed: %d", params.seed)
        if content_hint:
            logger.debug("content type hint: %s", content_hint)

        # Create model record for reproducibility
        model_info = ModelInfo(
            name=model_name,
            provider="ollama",
        )

        # Build content type context
        content_context = ""
        if content_hint:
            # Check if it's a known type with a template
            if content_hint.lower() in CONTENT_TYPE_HINTS:
                content_context = CONTENT_TYPE_HINTS[content_hint.lower()]
            else:
                # Use the hint directly as a description
                content_context = f"Content type: {content_hint}"
            content_context = f"CONTENT CONTEXT: {content_context}"

        # Decide whether to use chunking
        total_tokens = 0
        if len(transcript) > MIN_CHUNK_THRESHOLD:
            # Use chunked summarization for long transcripts
            response_text, total_tokens = self._summarize_chunked(
                transcript=transcript,
                ollama_url=ollama_url,
                model_name=model_name,
                params=params,
                content_context=content_context,
            )
        else:
            # Direct summarization for shorter transcripts
            response_text, total_tokens = self._summarize_direct(
                transcript=transcript,
                ollama_url=ollama_url,
                model_name=model_name,
                params=params,
                content_context=content_context,
            )

        # Write output
        output_path = output_dir / SUMMARY_FILENAME
        output_path.write_text(response_text)

        logger.info(
            "summary written: %s (%d chars)", output_path.name, len(response_text)
        )

        # Create full model record
        model_record = StepModelRecord(
            model=model_info,
            params=params,
            input_hash=hash_text(transcript),
            output_tokens=total_tokens,
        )

        return [output_path], model_record

    def _summarize_direct(
        self,
        transcript: str,
        ollama_url: str,
        model_name: str,
        params: GenerationParams,
        content_context: str,
    ) -> tuple[str, int]:
        """
        Summarize a short transcript directly.

        Args:
            transcript: Full transcript text.
            ollama_url: Ollama API URL.
            model_name: Model to use.
            params: Generation parameters.
            content_context: Content type context string.

        Returns:
            Tuple of (summary text, output tokens).
        """
        context_line = f"\n{content_context}\n" if content_context else ""
        prompt = f"""Extract key information from this transcript using the EXACT format below.
{context_line}
TRANSCRIPT:
{transcript}

---

Now provide the structured summary. Use EXACTLY these headings:

## Overview

## Action Items

## Key Takeaways

## Decisions Made

## Dates & Events Mentioned

## People Mentioned

## Notable Quotes

## Open Questions / Follow-ups"""

        return self._call_ollama(ollama_url, model_name, prompt, params)

    def _summarize_chunked(
        self,
        transcript: str,
        ollama_url: str,
        model_name: str,
        params: GenerationParams,
        content_context: str,
    ) -> tuple[str, int]:
        """
        Summarize a long transcript by chunking.

        Splits the transcript into chunks, extracts key info from each,
        then combines into a final summary.

        Args:
            transcript: Full transcript text.
            ollama_url: Ollama API URL.
            model_name: Model to use.
            params: Generation parameters.
            content_context: Content type context string.

        Returns:
            Tuple of (combined summary text, total output tokens).
        """
        # Split into chunks
        chunks = _chunk_text(transcript, DEFAULT_CHUNK_SIZE, CHUNK_OVERLAP)
        total_chunks = len(chunks)
        logger.info("splitting into %d chunks for processing", total_chunks)

        # Process each chunk
        chunk_summaries = []
        total_tokens = 0

        for i, chunk in enumerate(chunks, 1):
            logger.info("processing chunk %d/%d (%d chars)", i, total_chunks, len(chunk))

            prompt = CHUNK_EXTRACT_PROMPT.format(
                chunk_num=i,
                total_chunks=total_chunks,
                content_context=content_context,
                chunk_text=chunk,
            )

            summary, tokens = self._call_ollama(ollama_url, model_name, prompt, params)
            chunk_summaries.append(f"=== Section {i}/{total_chunks} ===\n{summary}")
            total_tokens += tokens

        # Combine all chunk summaries
        logger.info("combining %d chunk summaries", total_chunks)
        combined_extracts = "\n\n".join(chunk_summaries)

        context_line = f"\n{content_context}\n" if content_context else ""
        combine_prompt = COMBINE_PROMPT.format(
            total_chunks=total_chunks,
            content_context=context_line,
            chunk_summaries=combined_extracts,
        )

        final_summary, final_tokens = self._call_ollama(
            ollama_url, model_name, combine_prompt, params
        )
        total_tokens += final_tokens

        return final_summary, total_tokens

    def _call_ollama(
        self,
        base_url: str,
        model: str,
        prompt: str,
        params: GenerationParams,
    ) -> tuple[str, int]:
        """
        Call the Ollama API.

        Args:
            base_url: Ollama API base URL.
            model: Model name.
            prompt: User prompt.
            params: Generation parameters.

        Returns:
            Tuple of (response text, token count).

        Raises:
            StepError: If the API call fails.
        """
        url = f"{base_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "system": params.system_prompt or SUMMARIZE_SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "seed": params.seed,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "top_k": params.top_k,
                "num_predict": params.max_tokens,
            },
        }

        if params.stop:
            payload["options"]["stop"] = params.stop

        logger.debug("ollama request: %s", url)

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode())

            response_text = result.get("response", "")
            # Ollama returns eval_count for output tokens
            output_tokens = result.get("eval_count", 0)

            return response_text, output_tokens

        except urllib.error.URLError as e:
            if "Connection refused" in str(e):
                raise StepError(
                    self.name,
                    "Ollama not running. Start with: ollama serve",
                )
            raise StepError(self.name, f"connection error: {e}")
        except json.JSONDecodeError as e:
            raise StepError(self.name, f"invalid response: {e}")


def run(input_path: Path, output_dir: Path) -> StepResult:
    """
    Convenience function to run the summarize step.

    Args:
        input_path: Path to transcript file.
        output_dir: Directory for output artifacts.

    Returns:
        StepResult with execution details and model info.
    """
    step = SummarizeStep()
    start_time = time.monotonic()

    try:
        outputs, model_record = step.execute(input_path, output_dir)
        duration = time.monotonic() - start_time
        return StepResult(
            name=step.name,
            success=True,
            outputs=outputs,
            duration_seconds=duration,
            model_info=model_record.to_dict(),
        )
    except StepError as e:
        duration = time.monotonic() - start_time
        return StepResult(
            name=step.name,
            success=False,
            outputs=[],
            duration_seconds=duration,
            error=str(e),
        )
