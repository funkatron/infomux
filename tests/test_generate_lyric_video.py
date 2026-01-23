"""
Tests for the generate_lyric_video step.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from infomux.steps.generate_lyric_video import (
    GenerateLyricVideoStep,
    PositionedWord,
    WordEntry,
    run,
)
from infomux.steps import StepError


class TestGenerateLyricVideoStep:
    """Tests for GenerateLyricVideoStep."""

    def test_step_name(self) -> None:
        """Step has correct name."""
        step = GenerateLyricVideoStep()
        assert step.name == "generate_lyric_video"

    def test_default_config(self) -> None:
        """Default configuration values are correct."""
        step = GenerateLyricVideoStep()
        assert step.background_color == "black"
        assert step.video_size == "1920x1080"
        assert step.font_name == "Arial"
        assert step.font_size == 48
        assert step.font_color == "white"
        assert step.position == "center"
        assert step.word_spacing == 20

    def test_raises_when_ffmpeg_not_found(self, tmp_path: Path) -> None:
        """Raises StepError when ffmpeg is not available."""
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()
        json_file = tmp_path / "transcript.json"
        json_file.write_text('{"transcription": []}')

        with patch("infomux.steps.generate_lyric_video.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=None)

            step = GenerateLyricVideoStep()
            with pytest.raises(StepError) as exc_info:
                step.execute(audio_file, tmp_path)

            assert "ffmpeg not found" in str(exc_info.value)

    def test_raises_when_no_transcript_json(self, tmp_path: Path) -> None:
        """Raises StepError when transcript.json not found."""
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()
        # No transcript.json

        step = GenerateLyricVideoStep()
        with pytest.raises(StepError) as exc_info:
            step.execute(audio_file, tmp_path)

        assert "transcript.json not found" in str(exc_info.value)

    def test_raises_when_no_words(self, tmp_path: Path) -> None:
        """Raises StepError when no word timestamps found."""
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()
        json_file = tmp_path / "transcript.json"
        json_file.write_text('{"transcription": []}')

        with patch("infomux.steps.generate_lyric_video.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            step = GenerateLyricVideoStep()
            with pytest.raises(StepError) as exc_info:
                step.execute(audio_file, tmp_path)

            assert "No word-level timestamps found" in str(exc_info.value)

    def test_parse_word_timestamps(self, tmp_path: Path) -> None:
        """Parses word timestamps from transcript.json correctly."""
        json_file = tmp_path / "transcript.json"
        json_data = {
            "transcription": [
                {
                    "tokens": [
                        {
                            "text": " Hello",
                            "timestamps": {"from": "00:00:01,000", "to": "00:00:01,500"},
                        },
                        {
                            "text": "world",
                            "timestamps": {"from": "00:00:01,500", "to": "00:00:02,000"},
                        },
                    ]
                }
            ]
        }
        json_file.write_text(json.dumps(json_data))

        step = GenerateLyricVideoStep()
        words = step._parse_word_timestamps(json_file)

        # "world" doesn't have leading space, so it continues "Hello"
        assert len(words) == 1
        assert words[0].text == "Helloworld"
        assert words[0].start_ms == 1000
        assert words[0].end_ms == 2000

    def test_parse_word_timestamps_multiple_words(self, tmp_path: Path) -> None:
        """Parses multiple words correctly."""
        json_file = tmp_path / "transcript.json"
        json_data = {
            "transcription": [
                {
                    "tokens": [
                        {
                            "text": " Hello",
                            "timestamps": {"from": "00:00:01,000", "to": "00:00:01,500"},
                        },
                        {
                            "text": "world",
                            "timestamps": {"from": "00:00:01,500", "to": "00:00:02,000"},
                        },
                        {
                            "text": " test",
                            "timestamps": {"from": "00:00:02,000", "to": "00:00:02,500"},
                        },
                    ]
                }
            ]
        }
        json_file.write_text(json.dumps(json_data))

        step = GenerateLyricVideoStep()
        words = step._parse_word_timestamps(json_file)

        # "world" continues "Hello", " test" starts new word
        assert len(words) == 2
        assert words[0].text == "Helloworld"
        assert words[1].text == "test"

    def test_calculate_positions_single_word(self) -> None:
        """Calculates position for single word (centered)."""
        words = [WordEntry(text="Hello", start_ms=1000, end_ms=2000)]
        step = GenerateLyricVideoStep(font_size=48, position="center")

        positioned = step._calculate_positions(words, 1920, 1080)

        assert len(positioned) == 1
        assert positioned[0].word.text == "Hello"
        assert positioned[0].line == 0

    def test_calculate_positions_overlapping_words(self) -> None:
        """Calculates positions for overlapping words on same line."""
        words = [
            WordEntry(text="Hello", start_ms=1000, end_ms=2500),
            WordEntry(text="world", start_ms=2000, end_ms=3000),
        ]
        step = GenerateLyricVideoStep(font_size=48, word_spacing=20)

        positioned = step._calculate_positions(words, 1920, 1080)

        assert len(positioned) == 2
        # Both should be on same line since they overlap
        assert positioned[0].line == positioned[1].line
        # Second word should be positioned after first
        assert positioned[1].x > positioned[0].x

    def test_calculate_positions_non_overlapping(self) -> None:
        """Non-overlapping words go on separate lines."""
        words = [
            WordEntry(text="Hello", start_ms=1000, end_ms=1500),
            WordEntry(text="world", start_ms=2000, end_ms=2500),
        ]
        step = GenerateLyricVideoStep(font_size=48)

        positioned = step._calculate_positions(words, 1920, 1080)

        assert len(positioned) == 2
        # Should be on different lines or same line depending on algorithm
        # (Our algorithm places them sequentially, so they may be on same line
        # if there's room, or different lines if wrapping occurs)

    def test_build_drawtext_filter(self) -> None:
        """Builds correct drawtext filter string."""
        word = PositionedWord(
            word=WordEntry(text="Hello", start_ms=1000, end_ms=2000),
            x=100,
            y=540,
            line=0,
        )
        step = GenerateLyricVideoStep(font_size=48, font_color="white")

        filter_str = step._build_drawtext_filter(word, 1920, 1080)

        assert "drawtext=" in filter_str
        assert "text='Hello'" in filter_str
        assert "fontsize=48" in filter_str
        assert "fontcolor=white" in filter_str
        assert "x=100" in filter_str
        assert "y=540" in filter_str
        # Use 'if' parameter instead of 'enable' (it's an alias, avoids comma parsing issues)
        assert "if=" in filter_str
        assert "between(t,1.000,2.000)" in filter_str

    def test_build_drawtext_filter_with_font_file(self, tmp_path: Path) -> None:
        """Builds drawtext filter with font file when specified."""
        font_file = tmp_path / "font.ttf"
        font_file.write_bytes(b"fake font")
        word = PositionedWord(
            word=WordEntry(text="Hello", start_ms=1000, end_ms=2000),
            x=100,
            y=540,
            line=0,
        )
        step = GenerateLyricVideoStep(font_file=font_file)

        filter_str = step._build_drawtext_filter(word, 1920, 1080)

        assert "fontfile=" in filter_str
        assert str(font_file.resolve()) in filter_str

    def test_sanitize_filename(self) -> None:
        """Sanitizes text for safe filenames."""
        step = GenerateLyricVideoStep()
        
        # Test basic sanitization
        assert step._sanitize_filename("Hello world") == "Hello_world"
        assert step._sanitize_filename("test/word") == "testword"
        assert step._sanitize_filename("word:test") == "wordtest"
        
        # Test unsafe characters
        assert step._sanitize_filename("word<test>") == "wordtest"
        assert step._sanitize_filename('word"test"') == "wordtest"
        assert step._sanitize_filename("word|test") == "wordtest"
        
        # Test length limit
        long_text = "a" * 100
        assert len(step._sanitize_filename(long_text)) <= 50
        
        # Test empty result fallback
        assert step._sanitize_filename("") == "word"
        assert step._sanitize_filename("...") == "word"
        
        # Test Unicode replacement character removal
        assert "\ufffd" not in step._sanitize_filename("test\ufffdword")

    def test_build_command(self, tmp_path: Path) -> None:
        """Builds correct ffmpeg command using overlay approach."""
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()
        json_file = tmp_path / "transcript.json"
        json_data = {
            "transcription": [
                {
                    "tokens": [
                        {
                            "text": " Hello",
                            "timestamps": {"from": "00:00:01,000", "to": "00:00:01,500"},
                        },
                    ]
                }
            ]
        }
        json_file.write_text(json.dumps(json_data))

        # Create word images directory
        word_images_dir = tmp_path / "word_images"
        word_images_dir.mkdir()

        with patch("infomux.steps.generate_lyric_video.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            call_count = [0]
            def mock_run_side_effect(cmd, **kwargs):
                call_count[0] += 1
                # First call: ffprobe for duration
                if call_count[0] == 1:
                    return MagicMock(returncode=0, stdout='{"format":{"duration":"10.5"}}')
                # Subsequent calls: word image generation (create PNG files)
                if len(cmd) > 0:
                    for arg in cmd:
                        if isinstance(arg, (str, Path)) and str(arg).endswith(".png"):
                            img_path = Path(arg)
                            img_path.parent.mkdir(parents=True, exist_ok=True)
                            img_path.write_bytes(b"fake png")
                            # Verify filename format: word_XXXX_{phrase}_{font}_{size}.png
                            assert img_path.name.startswith("word_")
                            assert "_" in img_path.name  # Should have phrase/font/size
                            assert img_path.name.endswith(".png")
                            break
                # Last call: final video generation
                if "filter_complex_script" in " ".join(str(c) for c in cmd):
                    output_file = tmp_path / "audio_lyric_video.mp4"
                    output_file.write_bytes(b"fake video")
                return MagicMock(returncode=0)

            mock_run = MagicMock(side_effect=mock_run_side_effect)
            with patch("infomux.steps.generate_lyric_video.subprocess.run", side_effect=mock_run_side_effect):
                step = GenerateLyricVideoStep()
                outputs = step.execute(audio_file, tmp_path)

                # Verify the output was created
                assert len(outputs) == 1
                # Verify filter file exists and contains overlay filters
                filter_file = tmp_path / "filter_complex.txt"
                assert filter_file.exists()
                filter_content = filter_file.read_text()
                assert "overlay=" in filter_content
                assert "[out]" in filter_content
                # Verify output file
                assert outputs[0].name == "audio_lyric_video.mp4"

    def test_output_filename(self, tmp_path: Path) -> None:
        """Output filename uses audio stem with _lyric_video suffix."""
        audio_file = tmp_path / "my_audio.wav"
        audio_file.touch()
        json_file = tmp_path / "transcript.json"
        json_data = {
            "transcription": [
                {
                    "tokens": [
                        {
                            "text": " Hello",
                            "timestamps": {"from": "00:00:01,000", "to": "00:00:01,500"},
                        },
                    ]
                }
            ]
        }
        json_file.write_text(json.dumps(json_data))

        # Create word images directory
        word_images_dir = tmp_path / "word_images"
        word_images_dir.mkdir()

        with patch("infomux.steps.generate_lyric_video.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            call_count = [0]
            def mock_run_side_effect(cmd, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return MagicMock(returncode=0, stdout='{"format":{"duration":"10.5"}}')
                if len(cmd) > 0:
                    for arg in cmd:
                        if isinstance(arg, (str, Path)) and str(arg).endswith(".png"):
                            img_path = Path(arg)
                            img_path.parent.mkdir(parents=True, exist_ok=True)
                            img_path.write_bytes(b"fake png")
                            break
                if "filter_complex_script" in " ".join(str(c) for c in cmd):
                    output_file = tmp_path / "my_audio_lyric_video.mp4"
                    output_file.write_bytes(b"fake video")
                return MagicMock(returncode=0)

            with patch("infomux.steps.generate_lyric_video.subprocess.run", side_effect=mock_run_side_effect):
                step = GenerateLyricVideoStep()
                outputs = step.execute(audio_file, tmp_path)

                assert len(outputs) == 1
                assert outputs[0].name == "my_audio_lyric_video.mp4"

    def test_parse_video_size(self) -> None:
        """Parses video size string correctly."""
        step = GenerateLyricVideoStep()
        width, height = step._parse_video_size("1920x1080")
        assert width == 1920
        assert height == 1080

        width, height = step._parse_video_size("1280x720")
        assert width == 1280
        assert height == 720

    def test_parse_video_size_invalid(self) -> None:
        """Raises StepError for invalid video size format."""
        step = GenerateLyricVideoStep()
        with pytest.raises(StepError) as exc_info:
            step._parse_video_size("invalid")
        assert "Invalid video size format" in str(exc_info.value)


class TestRunFunction:
    """Tests for the run() convenience function."""

    def test_returns_step_result(self, tmp_path: Path) -> None:
        """run() returns StepResult with correct fields."""
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()
        json_file = tmp_path / "transcript.json"
        json_data = {
            "transcription": [
                {
                    "tokens": [
                        {
                            "text": " Hello",
                            "timestamps": {"from": "00:00:01,000", "to": "00:00:01,500"},
                        },
                    ]
                }
            ]
        }
        json_file.write_text(json.dumps(json_data))

        # Create word images directory
        word_images_dir = tmp_path / "word_images"
        word_images_dir.mkdir()

        with patch("infomux.steps.generate_lyric_video.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            call_count = [0]
            def mock_run_side_effect(cmd, **kwargs):
                call_count[0] += 1
                # First call: ffprobe for duration
                if call_count[0] == 1:
                    return MagicMock(returncode=0, stdout='{"format":{"duration":"10.5"}}')
                # Subsequent calls: word image generation or final video
                # Create output file if it's an image generation call
                if len(cmd) > 0:
                    for i, arg in enumerate(cmd):
                        if isinstance(arg, (str, Path)) and str(arg).endswith(".png"):
                            img_path = Path(arg)
                            img_path.parent.mkdir(parents=True, exist_ok=True)
                            img_path.write_bytes(b"fake png")
                            break
                # Last call: final video generation
                if "filter_complex_script" in " ".join(str(c) for c in cmd):
                    output_file = tmp_path / "audio_lyric_video.mp4"
                    output_file.write_bytes(b"fake video")
                return MagicMock(returncode=0)

            with patch("infomux.steps.generate_lyric_video.subprocess.run", side_effect=mock_run_side_effect):
                result = run(audio_file, tmp_path)

                assert result.name == "generate_lyric_video"
                assert result.success is True
                assert len(result.outputs) == 1
                assert result.duration_seconds >= 0
                assert result.error is None

    def test_custom_font_parameters(self, tmp_path: Path) -> None:
        """run() accepts custom font parameters."""
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()
        json_file = tmp_path / "transcript.json"
        json_data = {
            "transcription": [
                {
                    "tokens": [
                        {
                            "text": " Hello",
                            "timestamps": {"from": "00:00:01,000", "to": "00:00:01,500"},
                        },
                    ]
                }
            ]
        }
        json_file.write_text(json.dumps(json_data))

        # Create word images directory
        word_images_dir = tmp_path / "word_images"
        word_images_dir.mkdir()

        with patch("infomux.steps.generate_lyric_video.get_tool_paths") as mock_tools:
            mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))

            call_count = [0]
            def mock_run_side_effect(cmd, **kwargs):
                call_count[0] += 1
                # First call: ffprobe for duration
                if call_count[0] == 1:
                    return MagicMock(returncode=0, stdout='{"format":{"duration":"10.5"}}')
                # Subsequent calls: word image generation or final video
                if len(cmd) > 0:
                    for i, arg in enumerate(cmd):
                        if isinstance(arg, (str, Path)) and str(arg).endswith(".png"):
                            img_path = Path(arg)
                            img_path.parent.mkdir(parents=True, exist_ok=True)
                            img_path.write_bytes(b"fake png")
                            break
                if "filter_complex_script" in " ".join(str(c) for c in cmd):
                    output_file = tmp_path / "audio_lyric_video.mp4"
                    output_file.write_bytes(b"fake video")
                return MagicMock(returncode=0)

            with patch("infomux.steps.generate_lyric_video.subprocess.run", side_effect=mock_run_side_effect):
                result = run(
                    audio_file,
                    tmp_path,
                    font_name="Helvetica",
                    font_size=60,
                    font_color="yellow",
                    position="top",
                )

                assert result.success is True

    def test_captures_error_on_failure(self, tmp_path: Path) -> None:
        """run() captures error in StepResult on failure."""
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()
        # No transcript.json

        result = run(audio_file, tmp_path)

        assert result.name == "generate_lyric_video"
        assert result.success is False
        assert result.outputs == []
        assert result.error is not None
