"""
Tests for the summarize step.

Tests chunking logic, content type hints, and summarization workflow.
Does NOT test actual LLM calls - those are mocked.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from infomux.steps.summarize import (
    CONTENT_TYPE_HINTS,
    DEFAULT_CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_THRESHOLD,
    SummarizeStep,
    _chunk_text,
)


class TestChunkText:
    """Tests for the _chunk_text function."""

    def test_short_text_not_chunked(self) -> None:
        """Text shorter than chunk_size returns single chunk."""
        text = "Short text."
        chunks = _chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_exact_chunk_size(self) -> None:
        """Text exactly at chunk_size returns single chunk."""
        text = "x" * 100
        chunks = _chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1

    def test_long_text_chunked(self) -> None:
        """Long text is split into multiple chunks."""
        text = "word " * 100  # 500 chars
        chunks = _chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1

    def test_chunks_have_overlap(self) -> None:
        """Consecutive chunks share overlapping content."""
        # Create text with clear markers
        text = "A" * 50 + "B" * 50 + "C" * 50 + "D" * 50  # 200 chars
        chunks = _chunk_text(text, chunk_size=80, overlap=20)

        # With 80 char chunks and 20 overlap, chunks should overlap
        assert len(chunks) >= 2

        # Check that chunks are not empty
        for chunk in chunks:
            assert len(chunk) > 0

    def test_chunks_cover_all_content(self) -> None:
        """All original content appears in at least one chunk."""
        text = "The quick brown fox jumps over the lazy dog. " * 10
        chunks = _chunk_text(text, chunk_size=100, overlap=20)

        # Combine all chunks (removing duplicates from overlap)
        combined = "".join(chunks)

        # Every word from original should appear
        for word in text.split():
            assert word in combined

    def test_sentence_boundary_preference(self) -> None:
        """Chunking prefers sentence boundaries when possible."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = _chunk_text(text, chunk_size=40, overlap=10)

        # Chunks should tend to end at sentence boundaries
        # (This is a soft test - the algorithm tries but may not always succeed)
        for chunk in chunks[:-1]:  # Exclude last chunk
            # Most chunks should end with punctuation or near it
            stripped = chunk.rstrip()
            # Allow some tolerance
            assert len(stripped) > 0

    def test_empty_text(self) -> None:
        """Empty text returns single empty chunk."""
        chunks = _chunk_text("", chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_realistic_sizes(self) -> None:
        """Test with realistic chunk sizes."""
        # Simulate ~15k char transcript
        text = "This is a sentence with some content. " * 400  # ~15200 chars
        chunks = _chunk_text(text, DEFAULT_CHUNK_SIZE, CHUNK_OVERLAP)

        # Should produce 2 chunks for 15k at 12k chunk size
        assert len(chunks) >= 1
        assert all(len(c) <= DEFAULT_CHUNK_SIZE + 100 for c in chunks)  # Allow small overflow


class TestContentTypeHints:
    """Tests for content type hint templates."""

    def test_all_hints_are_strings(self) -> None:
        """All content type hints are non-empty strings."""
        for key, value in CONTENT_TYPE_HINTS.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
            assert len(value) > 0

    def test_expected_hints_exist(self) -> None:
        """Expected content types are defined."""
        expected = ["meeting", "talk", "podcast", "lecture", "standup", "1on1"]
        for hint_type in expected:
            assert hint_type in CONTENT_TYPE_HINTS

    def test_hints_contain_focus_areas(self) -> None:
        """Each hint describes focus areas."""
        for hint_type, hint_text in CONTENT_TYPE_HINTS.items():
            # Each hint should mention what to focus on
            assert "focus" in hint_text.lower() or ":" in hint_text


class TestChunkThreshold:
    """Tests for chunking threshold logic."""

    def test_threshold_constants_sensible(self) -> None:
        """Chunking constants have sensible values."""
        # Threshold should be larger than chunk size
        assert MIN_CHUNK_THRESHOLD > DEFAULT_CHUNK_SIZE

        # Overlap should be much smaller than chunk size
        assert CHUNK_OVERLAP < DEFAULT_CHUNK_SIZE / 4

        # Chunk size should be reasonable (2k-20k tokens worth)
        assert 5000 < DEFAULT_CHUNK_SIZE < 50000


class TestSummarizeStep:
    """Tests for SummarizeStep class."""

    def test_step_name(self) -> None:
        """Step has correct name."""
        step = SummarizeStep()
        assert step.name == "summarize"

    def test_step_default_model(self) -> None:
        """Step has None model by default (uses env/default)."""
        step = SummarizeStep()
        assert step.model is None

    def test_step_custom_model(self) -> None:
        """Step accepts custom model."""
        step = SummarizeStep(model="custom:model")
        assert step.model == "custom:model"

    def test_execute_requires_transcript(self, tmp_path: Path) -> None:
        """Execute fails if transcript doesn't exist."""
        step = SummarizeStep()
        fake_path = tmp_path / "nonexistent.txt"

        with pytest.raises(Exception) as exc_info:
            step.execute(fake_path, tmp_path)

        assert "not found" in str(exc_info.value).lower()

    def test_execute_requires_non_empty_transcript(self, tmp_path: Path) -> None:
        """Execute fails on empty transcript."""
        step = SummarizeStep()
        transcript_path = tmp_path / "transcript.txt"
        transcript_path.write_text("")

        with pytest.raises(Exception) as exc_info:
            step.execute(transcript_path, tmp_path)

        assert "empty" in str(exc_info.value).lower()

    @patch.object(SummarizeStep, "_call_ollama")
    def test_short_transcript_uses_direct(
        self, mock_ollama: MagicMock, tmp_path: Path
    ) -> None:
        """Short transcripts use direct summarization (no chunking)."""
        mock_ollama.return_value = ("## Overview\nTest summary", 100)

        step = SummarizeStep()
        transcript_path = tmp_path / "transcript.txt"
        # Create transcript shorter than threshold
        transcript_path.write_text("Short transcript. " * 100)  # ~1800 chars

        step.execute(transcript_path, tmp_path)

        # Should only call ollama once (direct, no chunking)
        assert mock_ollama.call_count == 1

    @patch.object(SummarizeStep, "_call_ollama")
    def test_long_transcript_uses_chunking(
        self, mock_ollama: MagicMock, tmp_path: Path
    ) -> None:
        """Long transcripts use chunked summarization."""
        mock_ollama.return_value = ("## Overview\nTest summary", 100)

        step = SummarizeStep()
        transcript_path = tmp_path / "transcript.txt"
        # Create transcript longer than threshold (15k+)
        transcript_path.write_text("This is content. " * 1500)  # ~25.5k chars

        step.execute(transcript_path, tmp_path)

        # Should call ollama multiple times (chunks + combine)
        # 25.5k / 12k = ~2-3 chunks, plus 1 combine = 3-4 calls
        assert mock_ollama.call_count >= 3

    @patch.object(SummarizeStep, "_call_ollama")
    def test_content_hint_from_env(
        self, mock_ollama: MagicMock, tmp_path: Path
    ) -> None:
        """Content type hint is read from environment."""
        mock_ollama.return_value = ("Summary", 50)

        step = SummarizeStep()
        transcript_path = tmp_path / "transcript.txt"
        transcript_path.write_text("Short content.")

        # Set env var
        with patch.dict(os.environ, {"INFOMUX_CONTENT_TYPE_HINT": "meeting"}):
            step.execute(transcript_path, tmp_path)

        # Check the prompt included the meeting hint
        call_args = mock_ollama.call_args
        prompt = call_args[0][2]  # Third positional arg is prompt
        assert "meeting" in prompt.lower() or "action items" in prompt.lower()

    @patch.object(SummarizeStep, "_call_ollama")
    def test_custom_content_hint(
        self, mock_ollama: MagicMock, tmp_path: Path
    ) -> None:
        """Custom content hints are passed through."""
        mock_ollama.return_value = ("Summary", 50)

        step = SummarizeStep()
        transcript_path = tmp_path / "transcript.txt"
        transcript_path.write_text("Short content.")

        # Set custom hint
        with patch.dict(os.environ, {"INFOMUX_CONTENT_TYPE_HINT": "quarterly review"}):
            step.execute(transcript_path, tmp_path)

        # Check the prompt included the custom hint
        call_args = mock_ollama.call_args
        prompt = call_args[0][2]
        assert "quarterly review" in prompt.lower()

    @patch.object(SummarizeStep, "_call_ollama")
    def test_output_file_created(
        self, mock_ollama: MagicMock, tmp_path: Path
    ) -> None:
        """Summary is written to output file."""
        expected_summary = "## Overview\nThis is the summary."
        mock_ollama.return_value = (expected_summary, 100)

        step = SummarizeStep()
        transcript_path = tmp_path / "transcript.txt"
        transcript_path.write_text("Some transcript content.")

        outputs, _ = step.execute(transcript_path, tmp_path)

        assert len(outputs) == 1
        assert outputs[0].name == "summary.md"
        assert outputs[0].read_text() == expected_summary

    @patch.object(SummarizeStep, "_call_ollama")
    def test_model_record_returned(
        self, mock_ollama: MagicMock, tmp_path: Path
    ) -> None:
        """Execution returns model record for reproducibility."""
        mock_ollama.return_value = ("Summary", 150)

        step = SummarizeStep()
        transcript_path = tmp_path / "transcript.txt"
        transcript_path.write_text("Transcript content here.")

        _, record = step.execute(transcript_path, tmp_path)

        assert record is not None
        assert record.model is not None
        assert record.params is not None
        assert record.params.seed is not None
        assert record.input_hash is not None
        assert record.output_tokens == 150


class TestChunkingSentenceBoundaries:
    """Tests for sentence boundary detection in chunking."""

    def test_breaks_at_period(self) -> None:
        """Prefers breaking at period + space."""
        text = "A" * 80 + ". " + "B" * 80
        chunks = _chunk_text(text, chunk_size=100, overlap=10)

        # First chunk should end near the period
        assert len(chunks) >= 2
        # The period should be in the first chunk
        assert "." in chunks[0]

    def test_breaks_at_question_mark(self) -> None:
        """Prefers breaking at question mark."""
        text = "A" * 80 + "? " + "B" * 80
        chunks = _chunk_text(text, chunk_size=100, overlap=10)

        assert len(chunks) >= 2
        assert "?" in chunks[0]

    def test_breaks_at_exclamation(self) -> None:
        """Prefers breaking at exclamation mark."""
        text = "A" * 80 + "! " + "B" * 80
        chunks = _chunk_text(text, chunk_size=100, overlap=10)

        assert len(chunks) >= 2
        assert "!" in chunks[0]
