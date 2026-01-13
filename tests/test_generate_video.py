"""
Tests for the generate_video step.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from infomux.steps.generate_video import GenerateVideoStep, run
from infomux.steps import StepError


class TestGenerateVideoStep:
    """Tests for GenerateVideoStep."""

    def test_step_name(self) -> None:
        """Step has correct name."""
        step = GenerateVideoStep()
        assert step.name == "generate_video"

    def test_default_background_color(self) -> None:
        """Default background is black."""
        step = GenerateVideoStep()
        assert step.background_color == "black"

    def test_default_video_size(self) -> None:
        """Default video size is 1920x1080."""
        step = GenerateVideoStep()
        assert step.video_size == "1920x1080"

    def test_raises_when_ffmpeg_not_found(self, tmp_path: Path) -> None:
        """Raises StepError when ffmpeg is not available."""
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.generate_video.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=None)

            step = GenerateVideoStep()
            with pytest.raises(StepError) as exc_info:
                step.execute(audio_file, tmp_path)

            assert "ffmpeg not found" in str(exc_info.value)

    def test_raises_when_no_subtitle_file(self, tmp_path: Path) -> None:
        """Raises StepError when no subtitle file exists."""
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()
        # No .srt file

        step = GenerateVideoStep()
        with pytest.raises(StepError) as exc_info:
            step.execute(audio_file, tmp_path)

        assert "No subtitle file found" in str(exc_info.value)

    def test_finds_transcript_srt(self, tmp_path: Path) -> None:
        """Finds transcript.srt in output directory."""
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.generate_video.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.generate_video.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                output_file = tmp_path / "audio_with_subs.mp4"
                output_file.write_bytes(b"fake video")

                step = GenerateVideoStep()
                outputs = step.execute(audio_file, tmp_path)

                # Verify it used transcript.srt
                cmd = mock_run.call_args[0][0]
                assert str(srt_file) in " ".join(str(c) for c in cmd)

    def test_solid_color_command(self, tmp_path: Path) -> None:
        """Builds correct command for solid color background."""
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.generate_video.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.generate_video.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                output_file = tmp_path / "audio_with_subs.mp4"
                output_file.write_bytes(b"fake video")

                step = GenerateVideoStep(background_color="blue", video_size="1280x720")
                step.execute(audio_file, tmp_path)

                cmd = mock_run.call_args[0][0]

                # Check for solid color input
                assert "-f" in cmd
                assert "lavfi" in cmd
                color_index = cmd.index("color=c=blue:s=1280x720")
                assert color_index > 0

    def test_image_background_command(self, tmp_path: Path) -> None:
        """Builds correct command for image background."""
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")
        bg_image = tmp_path / "background.png"
        bg_image.write_bytes(b"fake image")

        with patch("infomux.steps.generate_video.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.generate_video.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                output_file = tmp_path / "audio_with_subs.mp4"
                output_file.write_bytes(b"fake video")

                step = GenerateVideoStep(background_image=bg_image)
                step.execute(audio_file, tmp_path)

                cmd = mock_run.call_args[0][0]

                # Check for image input
                assert "-loop" in cmd
                assert "1" in cmd
                assert str(bg_image) in cmd

    def test_output_filename(self, tmp_path: Path) -> None:
        """Output filename uses audio stem with _with_subs suffix."""
        audio_file = tmp_path / "my_audio.wav"
        audio_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.generate_video.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.generate_video.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                output_file = tmp_path / "my_audio_with_subs.mp4"
                output_file.write_bytes(b"fake video")

                step = GenerateVideoStep()
                outputs = step.execute(audio_file, tmp_path)

                assert len(outputs) == 1
                assert outputs[0].name == "my_audio_with_subs.mp4"

    def test_raises_on_ffmpeg_failure(self, tmp_path: Path) -> None:
        """Raises StepError when ffmpeg fails."""
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.generate_video.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.generate_video.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stderr="error")

                step = GenerateVideoStep()
                with pytest.raises(StepError) as exc_info:
                    step.execute(audio_file, tmp_path)

                assert "ffmpeg failed" in str(exc_info.value)

    def test_subtitle_styling(self, tmp_path: Path) -> None:
        """Custom subtitle styling is included in command."""
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.generate_video.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.generate_video.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                output_file = tmp_path / "audio_with_subs.mp4"
                output_file.write_bytes(b"fake video")

                step = GenerateVideoStep(
                    subtitle_style={"fontsize": 24, "fontname": "Arial"}
                )
                step.execute(audio_file, tmp_path)

                cmd = mock_run.call_args[0][0]
                cmd_str = " ".join(str(c) for c in cmd)

                assert "FontSize=24" in cmd_str
                assert "FontName=Arial" in cmd_str


class TestRunFunction:
    """Tests for the run() convenience function."""

    def test_returns_step_result(self, tmp_path: Path) -> None:
        """run() returns StepResult with correct fields."""
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.generate_video.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.generate_video.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                output_file = tmp_path / "audio_with_subs.mp4"
                output_file.write_bytes(b"fake video")

                result = run(audio_file, tmp_path)

                assert result.name == "generate_video"
                assert result.success is True
                assert len(result.outputs) == 1
                assert result.duration_seconds >= 0
                assert result.error is None

    def test_background_image_parameter(self, tmp_path: Path) -> None:
        """run() accepts background_image parameter."""
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")
        bg_image = tmp_path / "bg.png"
        bg_image.write_bytes(b"fake")

        with patch("infomux.steps.generate_video.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.generate_video.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                output_file = tmp_path / "audio_with_subs.mp4"
                output_file.write_bytes(b"fake video")

                result = run(audio_file, tmp_path, background_image=str(bg_image))

                # Should use image background
                cmd = mock_run.call_args[0][0]
                assert "-loop" in cmd

    def test_background_color_parameter(self, tmp_path: Path) -> None:
        """run() accepts background_color parameter."""
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.generate_video.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.generate_video.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                output_file = tmp_path / "audio_with_subs.mp4"
                output_file.write_bytes(b"fake video")

                result = run(audio_file, tmp_path, background_color="blue")

                # Should use blue background
                cmd = mock_run.call_args[0][0]
                assert "color=c=blue" in " ".join(str(c) for c in cmd)

    def test_video_size_parameter(self, tmp_path: Path) -> None:
        """run() accepts video_size parameter."""
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.generate_video.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.generate_video.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                output_file = tmp_path / "audio_with_subs.mp4"
                output_file.write_bytes(b"fake video")

                result = run(audio_file, tmp_path, video_size="1280x720")

                # Should use custom size
                cmd = mock_run.call_args[0][0]
                assert "1280x720" in " ".join(str(c) for c in cmd)

    def test_captures_error_on_failure(self, tmp_path: Path) -> None:
        """run() captures error in StepResult on failure."""
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()
        # No subtitle file

        result = run(audio_file, tmp_path)

        assert result.name == "generate_video"
        assert result.success is False
        assert result.outputs == []
        assert result.error is not None
