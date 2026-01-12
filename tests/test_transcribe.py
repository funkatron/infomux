"""
Tests for the transcribe step.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from infomux.steps.transcribe import (
    TRANSCRIPT_FILENAME,
    TranscribeStep,
    run,
)
from infomux.steps import StepError


class TestTranscribeStep:
    """Tests for TranscribeStep."""

    def test_step_name(self) -> None:
        """Step has correct name."""
        step = TranscribeStep()
        assert step.name == "transcribe"

    def test_raises_when_whisper_cli_not_found(self, tmp_path: Path) -> None:
        """Raises StepError when whisper-cli is not available."""
        input_file = tmp_path / "audio.wav"
        input_file.touch()

        with patch("infomux.steps.transcribe.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(
                whisper_cli=None,
                whisper_model=Path("/path/to/model.bin"),
            )

            step = TranscribeStep()
            with pytest.raises(StepError) as exc_info:
                step.execute(input_file, tmp_path)

            assert "whisper-cli not found" in str(exc_info.value)

    def test_raises_when_model_not_found(self, tmp_path: Path) -> None:
        """Raises StepError when whisper model is not available."""
        input_file = tmp_path / "audio.wav"
        input_file.touch()

        with patch("infomux.steps.transcribe.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(
                whisper_cli=Path("/usr/bin/whisper-cli"),
                whisper_model=None,
            )

            step = TranscribeStep()
            with pytest.raises(StepError) as exc_info:
                step.execute(input_file, tmp_path)

            assert "Whisper model not found" in str(exc_info.value)

    def test_builds_correct_command(self, tmp_path: Path) -> None:
        """Builds correct whisper-cli command with expected flags."""
        input_file = tmp_path / "audio.wav"
        input_file.touch()

        model_path = Path("/path/to/ggml-base.en.bin")

        with patch("infomux.steps.transcribe.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(
                whisper_cli=Path("/usr/bin/whisper-cli"),
                whisper_model=model_path,
            )

            with patch("infomux.steps.transcribe.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                # Create expected output file
                output_file = tmp_path / TRANSCRIPT_FILENAME
                output_file.write_text("Hello world")

                step = TranscribeStep()
                step.execute(input_file, tmp_path)

                # Verify command structure
                mock_run.assert_called_once()
                cmd = mock_run.call_args[0][0]

                assert cmd[0] == "/usr/bin/whisper-cli"
                assert "-m" in cmd  # Model flag
                assert str(model_path) in cmd  # Model path
                assert "-f" in cmd  # Input file flag
                assert str(input_file) in cmd  # Input file
                assert "-of" in cmd  # Output prefix flag
                assert "-otxt" in cmd  # Text output
                assert "-np" in cmd  # No progress

    def test_raises_on_whisper_failure(self, tmp_path: Path) -> None:
        """Raises StepError when whisper-cli returns non-zero exit code."""
        input_file = tmp_path / "audio.wav"
        input_file.touch()

        with patch("infomux.steps.transcribe.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(
                whisper_cli=Path("/usr/bin/whisper-cli"),
                whisper_model=Path("/path/to/model.bin"),
            )

            with patch("infomux.steps.transcribe.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=1, stderr="error", stdout=""
                )

                step = TranscribeStep()
                with pytest.raises(StepError) as exc_info:
                    step.execute(input_file, tmp_path)

                assert "whisper-cli failed" in str(exc_info.value)

    def test_raises_when_output_not_created(self, tmp_path: Path) -> None:
        """Raises StepError when whisper succeeds but transcript file missing."""
        input_file = tmp_path / "audio.wav"
        input_file.touch()

        with patch("infomux.steps.transcribe.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(
                whisper_cli=Path("/usr/bin/whisper-cli"),
                whisper_model=Path("/path/to/model.bin"),
            )

            with patch("infomux.steps.transcribe.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                # Don't create output file

                step = TranscribeStep()
                with pytest.raises(StepError) as exc_info:
                    step.execute(input_file, tmp_path)

                assert "transcript not created" in str(exc_info.value)

    def test_returns_output_path_on_success(self, tmp_path: Path) -> None:
        """Returns list containing transcript file path on success."""
        input_file = tmp_path / "audio.wav"
        input_file.touch()

        with patch("infomux.steps.transcribe.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(
                whisper_cli=Path("/usr/bin/whisper-cli"),
                whisper_model=Path("/path/to/model.bin"),
            )

            with patch("infomux.steps.transcribe.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                # Create output file
                output_file = tmp_path / TRANSCRIPT_FILENAME
                output_file.write_text("Hello, this is a test transcription.")

                step = TranscribeStep()
                outputs = step.execute(input_file, tmp_path)

                assert len(outputs) == 1
                assert outputs[0] == output_file
                assert outputs[0].name == TRANSCRIPT_FILENAME


class TestRunFunction:
    """Tests for the run() convenience function."""

    def test_returns_step_result_with_model_info(self, tmp_path: Path) -> None:
        """run() returns StepResult with model information."""
        input_file = tmp_path / "audio.wav"
        input_file.touch()

        model_path = Path("/path/to/ggml-base.en.bin")

        with patch("infomux.steps.transcribe.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(
                whisper_cli=Path("/usr/bin/whisper-cli"),
                whisper_model=model_path,
            )

            with patch("infomux.steps.transcribe.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                output_file = tmp_path / TRANSCRIPT_FILENAME
                output_file.write_text("Test transcript")

                result = run(input_file, tmp_path)

                assert result.name == "transcribe"
                assert result.success is True
                assert len(result.outputs) == 1
                assert result.duration_seconds >= 0
                assert result.error is None
                assert result.model_info is not None
                assert result.model_info["model"]["provider"] == "whisper.cpp"

    def test_captures_error_on_failure(self, tmp_path: Path) -> None:
        """run() captures error in StepResult on failure."""
        input_file = tmp_path / "audio.wav"
        input_file.touch()

        with patch("infomux.steps.transcribe.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(
                whisper_cli=None,
                whisper_model=None,
            )

            result = run(input_file, tmp_path)

            assert result.name == "transcribe"
            assert result.success is False
            assert result.outputs == []
            assert result.error is not None

    def test_model_info_none_when_no_model(self, tmp_path: Path) -> None:
        """run() handles missing model gracefully."""
        input_file = tmp_path / "audio.wav"
        input_file.touch()

        with patch("infomux.steps.transcribe.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(
                whisper_cli=Path("/usr/bin/whisper-cli"),
                whisper_model=None,
            )

            result = run(input_file, tmp_path)

            # Should fail due to missing model, but model_info should still be None
            assert result.success is False
            assert result.model_info is None
