"""
Tests for the transcribe_timed step.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from infomux.steps.transcribe_timed import (
    TRANSCRIPT_TIMED_FILENAME,
    TranscribeTimedStep,
    _detect_model_type,
    run,
)
from infomux.steps import StepError


class TestDetectModelType:
    """Tests for _detect_model_type helper function."""

    @pytest.mark.parametrize(
        "model_name,expected_type",
        [
            ("ggml-tiny.en.bin", "tiny.en"),
            ("ggml-tiny.bin", "tiny"),
            ("ggml-base.en.bin", "base.en"),
            ("ggml-base.bin", "base"),
            ("ggml-small.en.bin", "small.en"),
            ("ggml-small.bin", "small"),
            ("ggml-medium.en.bin", "medium.en"),
            ("ggml-medium.bin", "medium"),
            ("ggml-large.bin", "large"),
            ("ggml-large-v2.bin", "large"),
        ],
    )
    def test_detects_model_types(self, model_name: str, expected_type: str) -> None:
        """Correctly identifies model type from filename."""
        model_path = Path(f"/models/{model_name}")
        assert _detect_model_type(model_path) == expected_type

    def test_defaults_to_base_en(self) -> None:
        """Defaults to base.en for unknown model names."""
        model_path = Path("/models/unknown-model.bin")
        assert _detect_model_type(model_path) == "base.en"

    def test_case_insensitive(self) -> None:
        """Model type detection is case-insensitive."""
        model_path = Path("/models/GGML-SMALL.EN.BIN")
        assert _detect_model_type(model_path) == "small.en"


class TestTranscribeTimedStep:
    """Tests for TranscribeTimedStep."""

    def test_step_name(self) -> None:
        """Step has correct name."""
        step = TranscribeTimedStep()
        assert step.name == "transcribe_timed"

    def test_raises_when_whisper_cli_not_found(self, tmp_path: Path) -> None:
        """Raises StepError when whisper-cli is not available."""
        input_file = tmp_path / "audio.wav"
        input_file.touch()

        with patch("infomux.steps.transcribe_timed.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(
                whisper_cli=None,
                whisper_model=Path("/path/to/model.bin"),
            )

            step = TranscribeTimedStep()
            with pytest.raises(StepError) as exc_info:
                step.execute(input_file, tmp_path)

            assert "whisper-cli not found" in str(exc_info.value)

    def test_raises_when_model_not_found(self, tmp_path: Path) -> None:
        """Raises StepError when whisper model is not available."""
        input_file = tmp_path / "audio.wav"
        input_file.touch()

        with patch("infomux.steps.transcribe_timed.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(
                whisper_cli=Path("/usr/bin/whisper-cli"),
                whisper_model=None,
            )

            step = TranscribeTimedStep()
            with pytest.raises(StepError) as exc_info:
                step.execute(input_file, tmp_path)

            assert "model not found" in str(exc_info.value).lower()

    def test_builds_command_with_dtw(self, tmp_path: Path) -> None:
        """Builds command with DTW flag for word-level timing."""
        input_file = tmp_path / "audio.wav"
        input_file.touch()

        model_path = Path("/path/to/ggml-base.en.bin")

        with patch("infomux.steps.transcribe_timed.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(
                whisper_cli=Path("/usr/bin/whisper-cli"),
                whisper_model=model_path,
            )

            with patch("infomux.steps.transcribe_timed.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                # Create expected output files
                (tmp_path / "transcript.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")
                (tmp_path / "transcript.vtt").write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello")
                (tmp_path / "transcript.json").write_text('{"text": "Hello"}')

                step = TranscribeTimedStep()
                step.execute(input_file, tmp_path)

                # Verify DTW flag is present
                mock_run.assert_called_once()
                cmd = mock_run.call_args[0][0]

                assert "-dtw" in cmd
                assert "base.en" in cmd  # Model type for DTW
                assert "-osrt" in cmd  # SRT output
                assert "-ovtt" in cmd  # VTT output
                assert "-ojf" in cmd  # JSON output

    def test_returns_multiple_outputs(self, tmp_path: Path) -> None:
        """Returns list of all output files (SRT, VTT, JSON)."""
        input_file = tmp_path / "audio.wav"
        input_file.touch()

        with patch("infomux.steps.transcribe_timed.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(
                whisper_cli=Path("/usr/bin/whisper-cli"),
                whisper_model=Path("/path/to/ggml-base.en.bin"),
            )

            with patch("infomux.steps.transcribe_timed.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                # Create all output files
                (tmp_path / "transcript.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")
                (tmp_path / "transcript.vtt").write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello")
                (tmp_path / "transcript.json").write_text('{"text": "Hello"}')

                step = TranscribeTimedStep()
                outputs = step.execute(input_file, tmp_path)

                assert len(outputs) == 3
                output_names = {p.name for p in outputs}
                assert "transcript.srt" in output_names
                assert "transcript.vtt" in output_names
                assert "transcript.json" in output_names

    def test_raises_when_no_outputs_created(self, tmp_path: Path) -> None:
        """Raises StepError when no output files are created."""
        input_file = tmp_path / "audio.wav"
        input_file.touch()

        with patch("infomux.steps.transcribe_timed.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(
                whisper_cli=Path("/usr/bin/whisper-cli"),
                whisper_model=Path("/path/to/model.bin"),
            )

            with patch("infomux.steps.transcribe_timed.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                # Don't create any output files

                step = TranscribeTimedStep()
                with pytest.raises(StepError) as exc_info:
                    step.execute(input_file, tmp_path)

                assert "no output files created" in str(exc_info.value)


class TestRunFunction:
    """Tests for the run() convenience function."""

    def test_returns_step_result_with_dtw_info(self, tmp_path: Path) -> None:
        """run() includes DTW model type in model info."""
        input_file = tmp_path / "audio.wav"
        input_file.touch()

        model_path = Path("/path/to/ggml-small.en.bin")

        with patch("infomux.steps.transcribe_timed.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(
                whisper_cli=Path("/usr/bin/whisper-cli"),
                whisper_model=model_path,
            )

            with patch("infomux.steps.transcribe_timed.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                (tmp_path / "transcript.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nTest")
                (tmp_path / "transcript.vtt").write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nTest")
                (tmp_path / "transcript.json").write_text('{"text": "Test"}')

                result = run(input_file, tmp_path)

                assert result.name == "transcribe_timed"
                assert result.success is True
                assert len(result.outputs) == 3
                assert result.model_info is not None
                assert result.model_info["params"]["dtw"] == "small.en"

    def test_captures_error_on_failure(self, tmp_path: Path) -> None:
        """run() captures error in StepResult on failure."""
        input_file = tmp_path / "audio.wav"
        input_file.touch()

        with patch("infomux.steps.transcribe_timed.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(
                whisper_cli=None,
                whisper_model=None,
            )

            result = run(input_file, tmp_path)

            assert result.name == "transcribe_timed"
            assert result.success is False
            assert result.error is not None
