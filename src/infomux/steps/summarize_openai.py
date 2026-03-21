"""
Summarize step: summarize a transcript using OpenAI.

This step is intentionally explicit about external API usage.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from infomux.cache import get_openai_cache_dir
from infomux.ext.openai import OpenAIError, chat_completion
from infomux.llm import (
    DEFAULT_SUMMARIZE_PARAMS,
    GenerationParams,
    ModelInfo,
    StepModelRecord,
    hash_text,
)
from infomux.log import get_logger
from infomux.steps import StepError, StepResult, register_step
from infomux.steps.summarize import (
    CONTENT_TYPE_HINTS,
    MIN_CHUNK_THRESHOLD,
    SUMMARIZE_SYSTEM_PROMPT,
    SUMMARY_FILENAME,
    _chunk_text,
)

logger = get_logger(__name__)

ENV_OPENAI_API_KEY = "INFOMUX_OPENAI_API_KEY"
ENV_OPENAI_MODEL = "INFOMUX_OPENAI_MODEL"
ENV_OPENAI_BASE_URL = "INFOMUX_OPENAI_BASE_URL"
ENV_OPENAI_CACHE = "INFOMUX_OPENAI_CACHE"
ENV_CONTENT_TYPE_HINT = "INFOMUX_CONTENT_TYPE_HINT"

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _env_bool(name: str, default: bool = True) -> bool:
    """Parse a boolean environment variable with sensible defaults."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    return value not in {"0", "false", "no", "off"}


@register_step
@dataclass
class SummarizeOpenAIStep:
    """
    Pipeline step to summarize a transcript using OpenAI.

    Records model name, seed, and generation parameters for reproducibility.
    """

    name: str = "summarize_openai"
    model: str | None = None
    params: GenerationParams | None = None

    def execute(
        self, input_path: Path, output_dir: Path
    ) -> tuple[list[Path], StepModelRecord]:
        """
        Summarize the transcript.

        Args:
            input_path: Path to transcript file.
            output_dir: Directory to write outputs.

        Returns:
            Tuple of (output paths, model record).
        """
        api_key = os.environ.get(ENV_OPENAI_API_KEY, "").strip()
        if not api_key:
            raise StepError(
                self.name,
                f"{ENV_OPENAI_API_KEY} not set (required for external OpenAI summarization)",
            )

        model_name = self.model or os.environ.get(
            ENV_OPENAI_MODEL, DEFAULT_OPENAI_MODEL
        )
        base_url = os.environ.get(ENV_OPENAI_BASE_URL, DEFAULT_OPENAI_BASE_URL)
        content_hint = os.environ.get(ENV_CONTENT_TYPE_HINT, "")

        params = self.params or DEFAULT_SUMMARIZE_PARAMS
        params = params.with_seed()

        if not input_path.exists():
            raise StepError(self.name, f"transcript not found: {input_path}")

        transcript = input_path.read_text()
        if not transcript.strip():
            raise StepError(self.name, "transcript is empty")

        logger.info("summarizing transcript via OpenAI (%d chars)", len(transcript))
        logger.debug("using OpenAI model: %s", model_name)
        logger.debug("using seed: %d", params.seed)

        content_context = ""
        if content_hint:
            if content_hint.lower() in CONTENT_TYPE_HINTS:
                content_context = CONTENT_TYPE_HINTS[content_hint.lower()]
            else:
                content_context = f"Content type: {content_hint}"
            content_context = f"CONTENT CONTEXT: {content_context}"

        total_tokens = 0
        if len(transcript) > MIN_CHUNK_THRESHOLD:
            response_text, total_tokens = self._summarize_chunked(
                transcript=transcript,
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                params=params,
                content_context=content_context,
            )
        else:
            response_text, total_tokens = self._summarize_direct(
                transcript=transcript,
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                params=params,
                content_context=content_context,
            )

        output_path = output_dir / SUMMARY_FILENAME
        output_path.write_text(response_text)
        logger.info(
            "summary written: %s (%d chars)", output_path.name, len(response_text)
        )

        model_record = StepModelRecord(
            model=ModelInfo(name=model_name, provider="openai"),
            params=params,
            input_hash=hash_text(transcript),
            output_tokens=total_tokens,
        )
        return [output_path], model_record

    def _summarize_direct(
        self,
        *,
        transcript: str,
        base_url: str,
        api_key: str,
        model_name: str,
        params: GenerationParams,
        content_context: str,
    ) -> tuple[str, int]:
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
        return self._call_openai(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            prompt=prompt,
            params=params,
        )

    def _summarize_chunked(
        self,
        *,
        transcript: str,
        base_url: str,
        api_key: str,
        model_name: str,
        params: GenerationParams,
        content_context: str,
    ) -> tuple[str, int]:
        chunks = _chunk_text(transcript)
        logger.info("long transcript detected, summarizing in %d chunks", len(chunks))

        chunk_summaries: list[str] = []
        total_tokens = 0

        for i, chunk in enumerate(chunks):
            logger.info(
                "summarizing chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk)
            )
            context_line = f"\n{content_context}\n" if content_context else ""
            prompt = f"""Summarize this transcript chunk concisely.
{context_line}
CHUNK {i + 1} OF {len(chunks)}:
{chunk}

Focus on key points, decisions, action items, dates, and people mentioned."""

            summary, tokens = self._call_openai(
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                prompt=prompt,
                params=params,
            )
            chunk_summaries.append(summary)
            total_tokens += tokens

        combined = "\n\n---\n\n".join(
            f"CHUNK {i + 1} SUMMARY:\n{s}" for i, s in enumerate(chunk_summaries)
        )
        combine_prompt = f"""Combine these chunk summaries into one cohesive summary.

Use EXACTLY these headings:

## Overview

## Action Items

## Key Takeaways

## Decisions Made

## Dates & Events Mentioned

## People Mentioned

## Notable Quotes

## Open Questions / Follow-ups

CHUNK SUMMARIES:
{combined}"""

        final_summary, final_tokens = self._call_openai(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            prompt=combine_prompt,
            params=params,
        )
        total_tokens += final_tokens
        return final_summary, total_tokens

    def _call_openai(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        prompt: str,
        params: GenerationParams,
    ) -> tuple[str, int]:
        cache_enabled = _env_bool(ENV_OPENAI_CACHE, default=True)
        system_prompt = params.system_prompt or SUMMARIZE_SYSTEM_PROMPT

        cache_payload = {
            "base_url": base_url.rstrip("/"),
            "model_name": model_name,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
            # Intentionally exclude seed from cache key so repeated summaries
            # of identical input/context can reuse cached responses across runs.
        }
        cache_key = hash_text(json.dumps(cache_payload, sort_keys=True))
        cache_file = get_openai_cache_dir() / f"{cache_key}.json"

        if cache_enabled and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text())
                response = cached.get("response")
                output_tokens = cached.get("output_tokens", 0)
                if isinstance(response, str):
                    logger.info("using cached OpenAI response: %s", cache_file.name)
                    if not isinstance(output_tokens, int):
                        output_tokens = 0
                    return response, output_tokens
            except Exception as e:
                logger.warning("failed reading OpenAI cache %s: %s", cache_file, e)

        try:
            response, output_tokens = chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model_name,
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=params.temperature,
                top_p=params.top_p,
                max_tokens=params.max_tokens,
                seed=params.seed,
            )
            if cache_enabled:
                try:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(
                        json.dumps(
                            {
                                "response": response,
                                "output_tokens": output_tokens,
                                "created_at": time.time(),
                                "model_name": model_name,
                            }
                        )
                    )
                except Exception as e:
                    logger.warning("failed writing OpenAI cache %s: %s", cache_file, e)
            return response, output_tokens
        except OpenAIError as e:
            raise StepError(self.name, str(e))


def run(input_path: Path, output_dir: Path) -> StepResult:
    """
    Convenience function to run the summarize_openai step.
    """
    step = SummarizeOpenAIStep()
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
