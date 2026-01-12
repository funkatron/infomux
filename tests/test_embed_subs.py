"""
Tests for the embed_subs step.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from infomux.steps.embed_subs import EmbedSubsStep, run
from infomux.steps import StepError


class TestEmbedSubsStep:
    """Tests for EmbedSubsStep."""

    def test_step_name(self) -> None:
        """Step has correct name."""
        step = EmbedSubsStep()
        assert step.name == "embed_subs"

    def test_default_is_soft_subs(self) -> None:
        """Default mode is soft subtitles (not burned in)."""
        step = EmbedSubsStep()
        assert step.burn_in is False

    def test_default_language(self) -> None:
        """Default language is English."""
        step = EmbedSubsStep()
        assert step.language == "eng"

    def test_raises_when_ffmpeg_not_found(self, tmp_path: Path) -> None:
        """Raises StepError when ffmpeg is not available."""
        video_file = tmp_path / "video.mp4"
        video_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.embed_subs.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=None)

            step = EmbedSubsStep()
            with pytest.raises(StepError) as exc_info:
                step.execute(video_file, tmp_path)

            assert "ffmpeg not found" in str(exc_info.value)

    def test_raises_when_no_subtitle_file(self, tmp_path: Path) -> None:
        """Raises StepError when no subtitle file exists."""
        video_file = tmp_path / "video.mp4"
        video_file.touch()
        # No .srt file

        step = EmbedSubsStep()
        with pytest.raises(StepError) as exc_info:
            step.execute(video_file, tmp_path)

        assert "No subtitle file found" in str(exc_info.value)

    def test_finds_transcript_srt(self, tmp_path: Path) -> None:
        """Finds transcript.srt in output directory."""
        video_file = tmp_path / "video.mp4"
        video_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.embed_subs.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.embed_subs.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                # Create output file
                output_file = tmp_path / "video_captioned.mp4"
                output_file.write_bytes(b"fake video")

                step = EmbedSubsStep()
                outputs = step.execute(video_file, tmp_path)

                # Verify it used transcript.srt
                cmd = mock_run.call_args[0][0]
                assert str(srt_file) in cmd

    def test_finds_any_srt_file(self, tmp_path: Path) -> None:
        """Falls back to any .srt file if transcript.srt doesn't exist."""
        video_file = tmp_path / "video.mp4"
        video_file.touch()
        srt_file = tmp_path / "custom_subtitles.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.embed_subs.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.embed_subs.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                output_file = tmp_path / "video_captioned.mp4"
                output_file.write_bytes(b"fake video")

                step = EmbedSubsStep()
                outputs = step.execute(video_file, tmp_path)

                # Should still succeed
                assert len(outputs) == 1

    def test_soft_sub_command(self, tmp_path: Path) -> None:
        """Builds correct command for soft subtitles."""
        video_file = tmp_path / "video.mp4"
        video_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.embed_subs.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.embed_subs.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                output_file = tmp_path / "video_captioned.mp4"
                output_file.write_bytes(b"fake video")

                step = EmbedSubsStep(burn_in=False)
                step.execute(video_file, tmp_path)

                cmd = mock_run.call_args[0][0]

                # Soft sub specific flags
                assert "-c" in cmd and "copy" in cmd  # Copy streams
                assert "mov_text" in cmd  # Subtitle codec

    def test_burn_in_command(self, tmp_path: Path) -> None:
        """Builds correct command for burned-in subtitles."""
        video_file = tmp_path / "video.mp4"
        video_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.embed_subs.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.embed_subs.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                output_file = tmp_path / "video_burned.mp4"
                output_file.write_bytes(b"fake video")

                step = EmbedSubsStep(burn_in=True)
                step.execute(video_file, tmp_path)

                cmd = mock_run.call_args[0][0]

                # Burn-in specific flags
                assert "-vf" in cmd  # Video filter
                vf_index = cmd.index("-vf")
                vf_value = cmd[vf_index + 1]
                assert "subtitles=" in vf_value

    def test_output_filename_soft(self, tmp_path: Path) -> None:
        """Soft sub output has _captioned suffix."""
        video_file = tmp_path / "my_video.mp4"
        video_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.embed_subs.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.embed_subs.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                # Create correct output file
                output_file = tmp_path / "my_video_captioned.mp4"
                output_file.write_bytes(b"fake video")

                step = EmbedSubsStep(burn_in=False)
                outputs = step.execute(video_file, tmp_path)

                assert len(outputs) == 1
                assert outputs[0].name == "my_video_captioned.mp4"

    def test_output_filename_burned(self, tmp_path: Path) -> None:
        """Burn-in output has _burned suffix."""
        video_file = tmp_path / "my_video.mp4"
        video_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.embed_subs.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.embed_subs.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                output_file = tmp_path / "my_video_burned.mp4"
                output_file.write_bytes(b"fake video")

                step = EmbedSubsStep(burn_in=True)
                outputs = step.execute(video_file, tmp_path)

                assert len(outputs) == 1
                assert outputs[0].name == "my_video_burned.mp4"

    def test_raises_on_ffmpeg_failure(self, tmp_path: Path) -> None:
        """Raises StepError when ffmpeg fails."""
        video_file = tmp_path / "video.mp4"
        video_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.embed_subs.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.embed_subs.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stderr="error")

                step = EmbedSubsStep()
                with pytest.raises(StepError) as exc_info:
                    step.execute(video_file, tmp_path)

                assert "ffmpeg failed" in str(exc_info.value)

    def test_raises_when_video_not_found(self, tmp_path: Path) -> None:
        """Raises StepError when video file doesn't exist."""
        video_file = tmp_path / "nonexistent.mp4"  # Does not exist
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.embed_subs.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            step = EmbedSubsStep()
            with pytest.raises(StepError) as exc_info:
                step.execute(video_file, tmp_path)

            assert "video not found" in str(exc_info.value)


class TestRunFunction:
    """Tests for the run() convenience function."""

    def test_returns_step_result(self, tmp_path: Path) -> None:
        """run() returns StepResult with correct fields."""
        video_file = tmp_path / "video.mp4"
        video_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.embed_subs.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.embed_subs.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                output_file = tmp_path / "video_captioned.mp4"
                output_file.write_bytes(b"fake video")

                result = run(video_file, tmp_path, burn_in=False)

                assert result.name == "embed_subs"
                assert result.success is True
                assert len(result.outputs) == 1
                assert result.duration_seconds >= 0
                assert result.error is None

    def test_burn_in_parameter(self, tmp_path: Path) -> None:
        """run() passes burn_in parameter correctly."""
        video_file = tmp_path / "video.mp4"
        video_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.embed_subs.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.embed_subs.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                output_file = tmp_path / "video_burned.mp4"
                output_file.write_bytes(b"fake video")

                result = run(video_file, tmp_path, burn_in=True)

                # Should use burn-in command
                cmd = mock_run.call_args[0][0]
                assert "-vf" in cmd

    def test_captures_error_on_failure(self, tmp_path: Path) -> None:
        """run() captures error in StepResult on failure."""
        video_file = tmp_path / "video.mp4"
        video_file.touch()
        # No subtitle file

        result = run(video_file, tmp_path)

        assert result.name == "embed_subs"
        assert result.success is False
        assert result.outputs == []
        assert result.error is not None


class TestSubtitleStyling:
    """Tests for subtitle styling options."""

    def test_custom_style_in_burn_in(self, tmp_path: Path) -> None:
        """Custom styling is applied to burn-in command."""
        video_file = tmp_path / "video.mp4"
        video_file.touch()
        srt_file = tmp_path / "transcript.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        with patch("infomux.steps.embed_subs.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            with patch("infomux.steps.embed_subs.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                output_file = tmp_path / "video_burned.mp4"
                output_file.write_bytes(b"fake video")

                step = EmbedSubsStep(
                    burn_in=True,
                    subtitle_style={"fontsize": 24, "fontname": "Arial"},
                )
                step.execute(video_file, tmp_path)

                cmd = mock_run.call_args[0][0]
                vf_index = cmd.index("-vf")
                vf_value = cmd[vf_index + 1]

                assert "force_style=" in vf_value
                assert "FontSize=24" in vf_value
                assert "FontName=Arial" in vf_value
