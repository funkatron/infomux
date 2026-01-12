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

# Defaults
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Output filename
SUMMARY_FILENAME = "summary.md"

# System prompt for summarization
SUMMARIZE_SYSTEM_PROMPT = """You extract actionable information from transcripts of meetings, calls, and recordings.

Output format (use ALL sections, write "None" if empty):

## Overview
One sentence: what this recording is, who is present, and when (if mentioned).

## Action Items
- [ ] **[Person]**: [Task] (due: [date if mentioned])
- [ ] **[Person]**: [Task]

## Decisions Made
- [Decision with context]

## Dates & Events Mentioned
- [Date]: [Event/deadline/meeting]

## Key Points
- [Specific point with names and details]

## Open Questions / Follow-ups
- [Unresolved question or topic needing follow-up]

Rules:
- Extract EVERY action item, decision, and date mentioned
- Use exact names from the transcript
- Be specific: "John will send the report" not "someone will send something"
- Include context for decisions (why it was decided)
- If a date/time is mentioned, capture it"""


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

        # Create model record for reproducibility
        model_info = ModelInfo(
            name=model_name,
            provider="ollama",
        )

        # Build the prompt
        prompt = f"""Extract actionable information from this transcript using the EXACT format below.

TRANSCRIPT:
{transcript}

---

Now provide the structured summary. Use EXACTLY these headings:

## Overview

## Action Items

## Decisions Made

## Dates & Events Mentioned

## Key Points

## Open Questions / Follow-ups"""

        # Call Ollama API
        # _call_ollama raises StepError on failure
        response_text, output_tokens = self._call_ollama(
            ollama_url,
            model_name,
            prompt,
            params,
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
            output_tokens=output_tokens,
        )

        return [output_path], model_record

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
