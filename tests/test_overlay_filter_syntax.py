"""
Test overlay filter syntax to catch FFmpeg compatibility issues.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from infomux.steps.generate_lyric_video import (
    GenerateLyricVideoStep,
    PositionedWord,
    WordEntry,
)


class TestOverlayFilterSyntax:
    """Test that overlay filter syntax is correct for FFmpeg."""

    def test_overlay_uses_enable_not_if(self, tmp_path: Path) -> None:
        """
        CRITICAL: Overlay filter must use 'enable', not 'if'.
        
        FFmpeg's overlay filter does NOT support 'if' parameter.
        This test will fail if we accidentally use 'if' instead of 'enable'.
        """
        step = GenerateLyricVideoStep()

        # Create test positioned words
        positioned_words = [
            PositionedWord(
                word=WordEntry(text="Hello", start_ms=1000, end_ms=2000),
                x=100,
                y=540,
                line=0,
            ),
            PositionedWord(
                word=WordEntry(text="world", start_ms=2000, end_ms=3000),
                x=200,
                y=540,
                line=0,
            ),
        ]

        word_image_paths = [
            (pw, tmp_path / f"word_{i:04d}.png")
            for i, pw in enumerate(positioned_words)
        ]

        # Create mock image files
        for _, img_path in word_image_paths:
            img_path.write_bytes(b"fake png")

        cmd = step._build_overlay_command(
            Path("/usr/bin/ffmpeg"),
            tmp_path / "audio.wav",
            tmp_path / "output.mp4",
            positioned_words,
            word_image_paths,
            10.0,
            1920,
            1080,
        )

        # Extract filter content
        assert "-filter_complex_script" in cmd
        filter_script_idx = cmd.index("-filter_complex_script")
        filter_file = Path(cmd[filter_script_idx + 1])
        assert filter_file.exists()

        filter_content = filter_file.read_text()

        # CRITICAL ASSERTIONS - these will catch the regression
        assert "enable=" in filter_content, (
            "Overlay filter MUST use 'enable' parameter. "
            "Found filter content: " + filter_content[:200]
        )

        assert "if=" not in filter_content or "enable=" in filter_content, (
            "Overlay filter does NOT support 'if' parameter. "
            "Use 'enable' instead. Found filter content: " + filter_content[:200]
        )

        # Verify the format is correct
        assert "overlay=" in filter_content
        assert "between(t," in filter_content

    def test_no_double_brackets(self, tmp_path: Path) -> None:
        """
        CRITICAL: No double brackets in filter chain.
        
        Filter labels should be [v1], [v2], etc., not [[v1]], [[v2]].
        """
        step = GenerateLyricVideoStep()

        # Create many words to test chaining
        positioned_words = [
            PositionedWord(
                word=WordEntry(text=f"word{i}", start_ms=1000 + i * 100, end_ms=1500 + i * 100),
                x=100,
                y=540 + i * 10,
                line=i,
            )
            for i in range(10)
        ]

        word_image_paths = [
            (pw, tmp_path / f"word_{i:04d}.png")
            for i, pw in enumerate(positioned_words)
        ]

        # Create mock image files
        for _, img_path in word_image_paths:
            img_path.write_bytes(b"fake png")

        cmd = step._build_overlay_command(
            Path("/usr/bin/ffmpeg"),
            tmp_path / "audio.wav",
            tmp_path / "output.mp4",
            positioned_words,
            word_image_paths,
            10.0,
            1920,
            1080,
        )

        # Extract filter content
        filter_script_idx = cmd.index("-filter_complex_script")
        filter_file = Path(cmd[filter_script_idx + 1])
        filter_content = filter_file.read_text()

        # CRITICAL: No double brackets
        assert "[[" not in filter_content, (
            "Found double brackets in filter chain! "
            "This will cause FFmpeg 'Invalid argument' error. "
            f"Filter content: {filter_content[:500]}"
        )

        # Verify correct format
        assert "[1:v]" in filter_content  # First input
        assert "[v1]" in filter_content  # Intermediate label
        assert "[out]" in filter_content  # Final output

    def test_filter_chain_format_with_many_words(self, tmp_path: Path) -> None:
        """
        Test filter chain format with many words (like real-world usage).
        
        This simulates the actual error case with 162 words.
        """
        step = GenerateLyricVideoStep()

        # Create 162 words (same as the error case)
        positioned_words = [
            PositionedWord(
                word=WordEntry(
                    text=f"word{i}",
                    start_ms=1000 + i * 100,
                    end_ms=1500 + i * 100,
                ),
                x=100 + (i % 20) * 50,
                y=540 + (i // 20) * 50,
                line=i // 20,
            )
            for i in range(162)
        ]

        word_image_paths = [
            (pw, tmp_path / f"word_{i:04d}_{pw.word.text}_{step.font_name}_{step.font_size}.png")
            for i, pw in enumerate(positioned_words)
        ]

        # Create mock image files
        for _, img_path in word_image_paths:
            img_path.write_bytes(b"fake png")

        cmd = step._build_overlay_command(
            Path("/usr/bin/ffmpeg"),
            tmp_path / "audio.wav",
            tmp_path / "output.mp4",
            positioned_words,
            word_image_paths,
            225.0,  # Duration from error case
            1920,
            1080,
        )

        # Extract filter content
        filter_script_idx = cmd.index("-filter_complex_script")
        filter_file = Path(cmd[filter_script_idx + 1])
        filter_content = filter_file.read_text()

        # Verify all critical requirements
        assert "[[" not in filter_content, "Double brackets found!"
        assert "enable=" in filter_content, "Missing 'enable' parameter!"
        assert "if=" not in filter_content or "enable=" in filter_content, "Found 'if=' instead of 'enable'!"
        assert "[1:v]" in filter_content, "Missing background video input!"
        assert "[out]" in filter_content, "Missing output label!"

        # Verify we have the right number of overlays
        overlay_count = filter_content.count("overlay=")
        assert overlay_count == 162, f"Expected 162 overlays, found {overlay_count}"

        # Verify intermediate labels are correct
        for i in range(1, 162):
            assert f"[v{i}]" in filter_content, f"Missing intermediate label [v{i}]"

        # Verify last overlay outputs to [out]
        assert filter_content.endswith("[out]") or "[out]" in filter_content.split(";")[-1]

    def test_filter_syntax_matches_ffmpeg_requirements(self, tmp_path: Path) -> None:
        """
        Test that filter syntax matches FFmpeg's exact requirements.
        
        This is a comprehensive syntax check.
        """
        step = GenerateLyricVideoStep()

        positioned_words = [
            PositionedWord(
                word=WordEntry(text="test", start_ms=1000, end_ms=2000),
                x=100,
                y=540,
                line=0,
            )
        ]

        word_image_paths = [
            (positioned_words[0], tmp_path / "word_0000.png")
        ]
        word_image_paths[0][1].write_bytes(b"fake png")

        cmd = step._build_overlay_command(
            Path("/usr/bin/ffmpeg"),
            tmp_path / "audio.wav",
            tmp_path / "output.mp4",
            positioned_words,
            word_image_paths,
            10.0,
            1920,
            1080,
        )

        filter_script_idx = cmd.index("-filter_complex_script")
        filter_file = Path(cmd[filter_script_idx + 1])
        filter_content = filter_file.read_text()

        # Expected format: [1:v][2:v]overlay=x:y:enable='between(t,START,END)'[out]
        expected_patterns = [
            "[1:v]",  # Background video input
            "[2:v]",  # First word image input
            "overlay=",  # Overlay filter
            "enable=",  # Enable parameter (NOT if=)
            "between(t,",  # Time expression
            "[out]",  # Output label
        ]

        for pattern in expected_patterns:
            assert pattern in filter_content, (
                f"Missing required pattern '{pattern}' in filter. "
                f"Filter content: {filter_content}"
            )

        # Must NOT have these
        forbidden_patterns = [
            "[[",  # Double brackets
            "if=",  # 'if' parameter (overlay doesn't support it)
        ]

        for pattern in forbidden_patterns:
            assert pattern not in filter_content, (
                f"Found forbidden pattern '{pattern}' in filter. "
                f"This will cause FFmpeg errors. Filter content: {filter_content}"
            )
