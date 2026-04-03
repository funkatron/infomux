"""
Tests for the summarize_openai step.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from infomux.llm import DEFAULT_SUMMARIZE_PARAMS
from infomux.steps.summarize_openai import SummarizeOpenAIStep


class TestSummarizeOpenAIStep:
    """Tests for SummarizeOpenAIStep."""

    def test_step_name(self) -> None:
        """Step has expected registry name."""
        step = SummarizeOpenAIStep()
        assert step.name == "summarize_openai"

    def test_requires_api_key(self, tmp_path: Path) -> None:
        """Step fails when OpenAI API key is missing."""
        transcript_path = tmp_path / "transcript.txt"
        transcript_path.write_text("hello world")

        step = SummarizeOpenAIStep()
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(Exception) as exc_info:
                step.execute(transcript_path, tmp_path)

        assert "INFOMUX_OPENAI_API_KEY" in str(exc_info.value)

    @patch.object(SummarizeOpenAIStep, "_call_openai")
    def test_writes_summary_file(self, mock_call, tmp_path: Path) -> None:
        """Step writes summary.md when API call succeeds."""
        mock_call.return_value = ("## Overview\nTest summary", 42)

        transcript_path = tmp_path / "transcript.txt"
        transcript_path.write_text("some transcript content")

        step = SummarizeOpenAIStep()
        with patch.dict("os.environ", {"INFOMUX_OPENAI_API_KEY": "test-key"}):
            outputs, record = step.execute(transcript_path, tmp_path)

        assert outputs[0].name == "summary.md"
        assert outputs[0].read_text() == "## Overview\nTest summary"
        assert record.model.provider == "openai"
        assert record.output_tokens == 42

    @patch("infomux.steps.summarize_openai.chat_completion")
    def test_openai_response_cache_hit_skips_api_call(
        self, mock_chat, tmp_path: Path
    ) -> None:
        """Second identical request should be served from local cache."""
        mock_chat.return_value = ("cached summary", 11)
        step = SummarizeOpenAIStep()

        env = {
            "INFOMUX_OPENAI_CACHE": "1",
            "INFOMUX_OPENAI_CACHE_DIR": str(tmp_path / "cache"),
        }
        with patch.dict("os.environ", env, clear=False):
            params = DEFAULT_SUMMARIZE_PARAMS.with_seed()
            first = step._call_openai(
                base_url="https://api.openai.com/v1",
                api_key="test-key",
                model_name="gpt-4o-mini",
                prompt="Summarize this text.",
                params=params,
            )
            second = step._call_openai(
                base_url="https://api.openai.com/v1",
                api_key="test-key",
                model_name="gpt-4o-mini",
                prompt="Summarize this text.",
                params=params,
            )

        assert first == second
        assert mock_chat.call_count == 1
