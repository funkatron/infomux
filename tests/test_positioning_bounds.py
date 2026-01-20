"""
Test that word positioning stays within video bounds.
"""

from __future__ import annotations

import pytest

from infomux.steps.generate_lyric_video import (
    GenerateLyricVideoStep,
    PositionedWord,
    WordEntry,
)


class TestPositioningBounds:
    """Test that positioning algorithm keeps words within video bounds."""

    def test_y_positions_stay_within_bounds(self) -> None:
        """
        CRITICAL: Y positions must stay within video height.
        
        This test ensures words don't go off-screen vertically.
        """
        # Create many words that would cause line wrapping
        words = [
            WordEntry(text=f"word{i}", start_ms=1000 + i * 100, end_ms=1500 + i * 100)
            for i in range(200)  # Many words to force many lines
        ]

        step = GenerateLyricVideoStep(font_size=48, position="center")
        video_width = 1920
        video_height = 1080

        positioned = step._calculate_positions(words, video_width, video_height)

        # Verify all Y positions are within bounds
        for pw in positioned:
            assert pw.y >= 0, f"Y position {pw.y} is negative for word '{pw.word.text}'"
            assert pw.y <= video_height, (
                f"Y position {pw.y} exceeds video height {video_height} "
                f"for word '{pw.word.text}'"
            )
            # Y should be at least font_size from top and bottom
            assert pw.y >= step.font_size, (
                f"Y position {pw.y} is too close to top (font_size={step.font_size}) "
                f"for word '{pw.word.text}'"
            )
            assert pw.y <= video_height - step.font_size, (
                f"Y position {pw.y} is too close to bottom "
                f"(video_height={video_height}, font_size={step.font_size}) "
                f"for word '{pw.word.text}'"
            )

    def test_x_positions_stay_within_bounds(self) -> None:
        """X positions should stay within video width."""
        words = [
            WordEntry(text="Hello", start_ms=1000, end_ms=2000),
            WordEntry(text="world", start_ms=1500, end_ms=2500),  # Overlaps
            WordEntry(text="test", start_ms=2000, end_ms=3000),  # Overlaps with world
        ]

        step = GenerateLyricVideoStep(font_size=48)
        video_width = 1920
        video_height = 1080

        positioned = step._calculate_positions(words, video_width, video_height)

        # Verify all X positions are within bounds
        for pw in positioned:
            assert pw.x >= 0, f"X position {pw.x} is negative for word '{pw.word.text}'"
            # X + word_width should fit (we'll check approximate)
            char_width = int(step.font_size * 0.6)
            word_width = len(pw.word.text) * char_width + step.word_spacing
            assert pw.x + word_width <= video_width + 100, (  # Allow some margin
                f"Word '{pw.word.text}' at x={pw.x} with width ~{word_width} "
                f"exceeds video width {video_width}"
            )

    def test_overlapping_words_placed_horizontally(self) -> None:
        """Overlapping words should be placed horizontally, not vertically stacked."""
        words = [
            WordEntry(text="Hello", start_ms=1000, end_ms=2500),
            WordEntry(text="world", start_ms=2000, end_ms=3000),  # Overlaps
            WordEntry(text="test", start_ms=2500, end_ms=3500),  # Overlaps with world
        ]

        step = GenerateLyricVideoStep(font_size=48)
        video_width = 1920
        video_height = 1080

        positioned = step._calculate_positions(words, video_width, video_height)

        # Overlapping words should be on same line (same Y)
        # Check if words overlap in time
        for i, pw1 in enumerate(positioned):
            for j, pw2 in enumerate(positioned[i+1:], i+1):
                # Check if they overlap in time
                overlap = not (
                    pw1.word.end_ms <= pw2.word.start_ms
                    or pw2.word.end_ms <= pw1.word.start_ms
                )
                if overlap:
                    # They should be on same line (same Y) or different lines if width exceeded
                    # But at least X should be different if on same line
                    if pw1.y == pw2.y:
                        assert pw1.x != pw2.x, (
                            f"Overlapping words '{pw1.word.text}' and '{pw2.word.text}' "
                            f"have same X position {pw1.x}"
                        )

    def test_words_are_centered_horizontally(self) -> None:
        """Words on each line should be centered horizontally."""
        words = [
            WordEntry(text="Hello", start_ms=1000, end_ms=2000),
            WordEntry(text="world", start_ms=1500, end_ms=2500),  # Overlaps
            WordEntry(text="test", start_ms=2000, end_ms=3000),  # Overlaps with world
        ]

        step = GenerateLyricVideoStep(font_size=48)
        video_width = 1920
        video_height = 1080

        positioned = step._calculate_positions(words, video_width, video_height)

        # Group words by line
        lines: dict[int, list[PositionedWord]] = {}
        for pw in positioned:
            if pw.line not in lines:
                lines[pw.line] = []
            lines[pw.line].append(pw)

        # Check that each line is centered
        for line_num, line_words in lines.items():
            if len(line_words) == 1:
                # Single word should be centered
                pw = line_words[0]
                word_width_est = len(pw.word.text) * int(step.font_size * 0.6)
                expected_x = (video_width - word_width_est) // 2
                # Allow some tolerance
                assert abs(pw.x - expected_x) < 50, (
                    f"Single word '{pw.word.text}' not centered. "
                    f"Expected x≈{expected_x}, got x={pw.x}"
                )
            else:
                # Multiple words: first and last should be positioned such that
                # the line center is near video center
                line_words.sort(key=lambda pw: pw.x)
                first_x = line_words[0].x
                last_word = line_words[-1]
                last_x = last_word.x
                last_width = len(last_word.word.text) * int(step.font_size * 0.6)
                line_center_x = first_x + (last_x + last_width - first_x) // 2
                video_center_x = video_width // 2
                # Allow tolerance
                assert abs(line_center_x - video_center_x) < 100, (
                    f"Line {line_num} not centered. "
                    f"Line center={line_center_x}, video center={video_center_x}"
                )

    def test_many_words_with_gaps(self) -> None:
        """
        Test with many words that have gaps (like real-world scenario).
        
        This simulates the actual error case where words have gaps in timing.
        """
        # Create words with some gaps (simulating missing words)
        words = []
        time = 1000
        for i in range(162):  # Same as error case
            if i == 42:
                # Simulate a gap (missing words)
                time += 23000  # 23 second gap
            elif i == 76:
                time += 1500  # 1.5 second gap
            elif i == 137:
                time += 24000  # 24 second gap
            
            words.append(
                WordEntry(
                    text=f"word{i}",
                    start_ms=time,
                    end_ms=time + 500,
                )
            )
            time += 1000  # Next word starts 1 second after previous ends

        step = GenerateLyricVideoStep(font_size=48, position="center")
        video_width = 1920
        video_height = 1080

        positioned = step._calculate_positions(words, video_width, video_height)

        # Verify all positions are within bounds
        assert len(positioned) == len(words), "All words should be positioned"

        for pw in positioned:
            assert 0 <= pw.x < video_width, f"X {pw.x} out of bounds for '{pw.word.text}'"
            assert step.font_size <= pw.y <= video_height - step.font_size, (
                f"Y {pw.y} out of bounds for '{pw.word.text}' "
                f"(font_size={step.font_size}, height={video_height})"
            )
