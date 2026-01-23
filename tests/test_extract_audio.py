"""
Tests for the extract_audio step.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from infomux.steps.extract_audio import (
    AUDIO_FILENAME,
    ExtractAudioStep,
    run,
)
from infomux.steps import StepError


class TestExtractAudioStep:
    """Tests for ExtractAudioStep."""

    def test_step_name(self) -> None:
        """Step has correct name."""
        step = ExtractAudioStep()
        assert step.name == "extract_audio"

    def test_raises_when_ffmpeg_not_found(self, tmp_path: Path) -> None:
        """Raises StepError when ffmpeg is not available."""
        input_file = tmp_path / "input.mp4"
        input_file.touch()

        # Mock get_tool_paths to return no ffmpeg
        with patch("infomux.steps.extract_audio.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=None)

            step = ExtractAudioStep()
            with pytest.raises(StepError) as exc_info:
                step.execute(input_file, tmp_path)

            assert "ffmpeg not found" in str(exc_info.value)

    def test_builds_correct_command(self, tmp_path: Path) -> None:
        """Builds correct ffmpeg command with expected flags."""
        input_file = tmp_path / "input.mp4"
        input_file.touch()
        output_file = tmp_path / AUDIO_FILENAME

        with patch("infomux.steps.extract_audio.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.extract_audio.subprocess.run") as mock_run:
                # Simulate successful ffmpeg run
                mock_run.return_value = MagicMock(returncode=0)
                
                # Create output file after subprocess.run (simulate ffmpeg creating it)
                def create_output(*args, **kwargs):
                    output_file.write_bytes(b"fake wav data")
                    return MagicMock(returncode=0)
                mock_run.side_effect = create_output

                step = ExtractAudioStep()
                step.execute(input_file, tmp_path)

                # Verify command structure
                mock_run.assert_called_once()
                cmd = mock_run.call_args[0][0]

                assert cmd[0] == "/usr/bin/ffmpeg"
                assert "-y" in cmd  # Overwrite
                assert "-i" in cmd  # Input flag
                assert str(input_file) in cmd  # Input file
                assert "-vn" in cmd  # No video
                assert "-ac" in cmd and "1" in cmd  # Mono
                assert "-ar" in cmd and "16000" in cmd  # 16kHz
                assert "pcm_s16le" in cmd  # PCM format
    
    def test_skips_extraction_if_already_exists(self, tmp_path: Path) -> None:
        """Skips extraction if audio.wav already exists."""
        input_file = tmp_path / "input.mp4"
        input_file.touch()
        
        # Create output file before extraction
        output_file = tmp_path / AUDIO_FILENAME
        output_file.write_bytes(b"existing audio data")

        with patch("infomux.steps.extract_audio.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.extract_audio.subprocess.run") as mock_run:
                step = ExtractAudioStep()
                outputs = step.execute(input_file, tmp_path)

                # Should not call ffmpeg
                mock_run.assert_not_called()
                
                # Should return existing file
                assert len(outputs) == 1
                assert outputs[0] == output_file

    def test_raises_on_ffmpeg_failure(self, tmp_path: Path) -> None:
        """Raises StepError when ffmpeg returns non-zero exit code."""
        input_file = tmp_path / "input.mp4"
        input_file.touch()

        with patch("infomux.steps.extract_audio.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.extract_audio.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stderr="error")

                step = ExtractAudioStep()
                with pytest.raises(StepError) as exc_info:
                    step.execute(input_file, tmp_path)

                assert "ffmpeg failed" in str(exc_info.value)

    def test_raises_when_output_not_created(self, tmp_path: Path) -> None:
        """Raises StepError when ffmpeg succeeds but output file is missing."""
        input_file = tmp_path / "input.mp4"
        input_file.touch()

        with patch("infomux.steps.extract_audio.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.extract_audio.subprocess.run") as mock_run:
                # ffmpeg "succeeds" but doesn't create output
                mock_run.return_value = MagicMock(returncode=0)

                step = ExtractAudioStep()
                with pytest.raises(StepError) as exc_info:
                    step.execute(input_file, tmp_path)

                assert "not created" in str(exc_info.value)

    def test_returns_output_path_on_success(self, tmp_path: Path) -> None:
        """Returns list containing output file path on success."""
        input_file = tmp_path / "input.mp4"
        input_file.touch()

        with patch("infomux.steps.extract_audio.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.extract_audio.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                # Create output file
                output_file = tmp_path / AUDIO_FILENAME
                output_file.write_bytes(b"fake wav data")

                step = ExtractAudioStep()
                outputs = step.execute(input_file, tmp_path)

                assert len(outputs) == 1
                assert outputs[0] == output_file
                assert outputs[0].name == AUDIO_FILENAME


class TestRunFunction:
    """Tests for the run() convenience function."""

    def test_returns_step_result(self, tmp_path: Path) -> None:
        """run() returns StepResult with correct fields."""
        input_file = tmp_path / "input.mp4"
        input_file.touch()

        with patch("infomux.steps.extract_audio.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.extract_audio.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                output_file = tmp_path / AUDIO_FILENAME
                output_file.write_bytes(b"fake wav data")

                result = run(input_file, tmp_path)

                assert result.name == "extract_audio"
                assert result.success is True
                assert len(result.outputs) == 1
                assert result.duration_seconds >= 0
                assert result.error is None

    def test_captures_error_on_failure(self, tmp_path: Path) -> None:
        """run() captures error in StepResult on failure."""
        input_file = tmp_path / "input.mp4"
        input_file.touch()

        with patch("infomux.steps.extract_audio.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=None)

            result = run(input_file, tmp_path)

            assert result.name == "extract_audio"
            assert result.success is False
            assert result.outputs == []
            assert result.error is not None
            assert "ffmpeg not found" in result.error
