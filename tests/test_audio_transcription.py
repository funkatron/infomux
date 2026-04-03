"""
Tests for transcription-audio preparation helpers.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from infomux.audio_transcription import (
    TRANSCRIPTION_AUDIO_FILENAME,
    ensure_transcription_audio,
)


class TestEnsureTranscriptionAudio:
    """Tests for ensure_transcription_audio()."""

    def test_converts_to_mono_16khz_wav(self, tmp_path: Path) -> None:
        """Runs ffmpeg conversion and returns output path."""
        input_file = tmp_path / "audio.wav"
        input_file.write_bytes(b"fake audio")
        expected_output = tmp_path / TRANSCRIPTION_AUDIO_FILENAME
        expected_output.write_bytes(b"converted audio")

        with patch("infomux.audio_transcription.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            output = ensure_transcription_audio(
                input_path=input_file,
                output_dir=tmp_path,
                ffmpeg_path=Path("/usr/bin/ffmpeg"),
            )

        assert output == expected_output
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/ffmpeg"
        assert "-ac" in cmd and "1" in cmd
        assert "-ar" in cmd and "16000" in cmd
        assert "-c:a" in cmd and "pcm_s16le" in cmd

    def test_raises_when_ffmpeg_fails(self, tmp_path: Path) -> None:
        """Raises runtime error with ffmpeg failure output."""
        input_file = tmp_path / "audio.wav"
        input_file.write_bytes(b"fake audio")

        with patch("infomux.audio_transcription.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stderr="conversion failed",
                stdout="",
            )
            with pytest.raises(RuntimeError, match="conversion failed"):
                ensure_transcription_audio(
                    input_path=input_file,
                    output_dir=tmp_path,
                    ffmpeg_path=Path("/usr/bin/ffmpeg"),
                )

    def test_raises_when_output_missing(self, tmp_path: Path) -> None:
        """Raises runtime error if conversion claims success but file missing."""
        input_file = tmp_path / "audio.wav"
        input_file.write_bytes(b"fake audio")

        with patch("infomux.audio_transcription.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            with pytest.raises(RuntimeError, match="transcription audio not created"):
                ensure_transcription_audio(
                    input_path=input_file,
                    output_dir=tmp_path,
                    ffmpeg_path=Path("/usr/bin/ffmpeg"),
                )
