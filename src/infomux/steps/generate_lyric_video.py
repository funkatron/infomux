"""
Generate lyric video with word-level burned subtitles.

Creates a video file from an audio file by displaying each word from
transcript.json at its exact timing, with support for multiple simultaneous
words positioned horizontally.

Command:
    ffmpeg -i audio.wav -f lavfi -i color=c=black:s=1920x1080:d=10.5 \\
           -vf "[drawtext filters with enable expressions]" \\
           -shortest output.mp4
"""

from __future__ import annotations

import json
import subprocess
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import ftfy

from infomux.config import get_tool_paths
from infomux.log import get_logger
from infomux.steps import StepError, StepResult, register_step

logger = get_logger(__name__)


@dataclass
class WordEntry:
    """Represents a word with its timing information."""

    text: str
    start_ms: int
    end_ms: int


@dataclass
class PositionedWord:
    """Represents a word with its position on screen."""

    word: WordEntry
    x: int
    y: int
    line: int


@register_step
@dataclass
class GenerateLyricVideoStep:
    """
    Pipeline step to generate a lyric video with word-level burned subtitles.

    Creates a video file by:
    1. Taking audio as input
    2. Parsing word-level timestamps from transcript.json
    3. Calculating positions for overlapping words
    4. Using ffmpeg drawtext filters with enable expressions for precise timing
    5. Generating video with words displayed at their exact timestamps

    Each word appears on screen for its exact duration, and multiple words
    can be visible simultaneously, positioned horizontally.
    """

    name: str = "generate_lyric_video"
    background_color: str = "black"  # Background color (solid color, or gradient like "gradient:purple:black")
    background_image: Path | None = None  # Optional background image (scaled to fit)
    background_gradient: str | None = None  # Gradient spec: "vertical:color1:color2" or "horizontal:color1:color2"
    video_size: str = "1920x1080"  # Width x Height
    font_name: str = "Arial"  # Font family name
    font_file: Path | None = None  # Optional explicit font file path
    font_size: int = 48  # Font size in pixels (default: 48pt)
    font_color: str = "white"  # Text color
    position: str = "center"  # Vertical position: top, center, bottom
    word_spacing: int = 20  # Horizontal spacing between words in pixels
    max_words_per_line: int | None = None  # Auto-calculate if None

    def execute(
        self,
        input_path: Path,
        output_dir: Path,
    ) -> list[Path]:
        """
        Generate lyric video from audio with word-level burned subtitles.

        Args:
            input_path: Path to the input audio file.
            output_dir: Directory containing transcript.json and for output.

        Returns:
            List containing the path to the generated video.

        Raises:
            StepError: If generation fails or required files not found.
        """
        tools = get_tool_paths()

        if not tools.ffmpeg:
            raise StepError(self.name, "ffmpeg not found")

        if not input_path.exists():
            raise StepError(self.name, f"audio not found: {input_path}")

        # Ensure we're using the original audio (not isolated vocals)
        # The isolated vocals (audio_vocals_only.wav) should only be used for transcription
        if input_path.name == "audio_vocals_only.wav":
            logger.warning(
                "generate_lyric_video received isolated vocals instead of original audio. "
                "This will result in video without background music. "
                "Check pipeline configuration - should use extract_audio output, not isolate_vocals."
            )

        # Find transcript.json in output_dir
        json_path = output_dir / "transcript.json"
        if not json_path.exists():
            raise StepError(
                self.name,
                f"transcript.json not found in {output_dir}. "
                "Run transcribe_timed step first.",
            )

        # Find both SRT files for dual subtitle tracks
        # Line-level subtitles (transcript.srt) - normal captions
        line_srt_path = output_dir / "transcript.srt"
        if not line_srt_path.exists():
            logger.warning("transcript.srt not found, line-level subtitles will be skipped")
            line_srt_path = None
        else:
            logger.info("found transcript.srt for line-level subtitles")
        
        # Word-level subtitles (transcript_words.srt) - word-by-word captions
        word_srt_path = output_dir / "transcript_words.srt"
        if not word_srt_path.exists():
            logger.warning("transcript_words.srt not found, word-level subtitles will be skipped")
            word_srt_path = None
        else:
            logger.info("found transcript_words.srt for word-level subtitles")

        # Parse word timestamps from transcript.json
        words = self._parse_word_timestamps(json_path)
        if not words:
            raise StepError(
                self.name,
                "No word-level timestamps found in transcript.json. "
                "Ensure transcribe_timed was run with word-level support.",
            )

        logger.info("parsed %d words from transcript.json", len(words))
        
        # Check for potential timing issues
        if words:
            first_word_start = words[0].start_ms / 1000.0
            last_word_end = words[-1].end_ms / 1000.0
            
            # Check for large gaps in timing
            gaps = []
            for i in range(len(words) - 1):
                gap_start = words[i].end_ms / 1000.0
                gap_end = words[i + 1].start_ms / 1000.0
                gap_duration = gap_end - gap_start
                if gap_duration > 2.0:  # Gaps longer than 2 seconds
                    gaps.append((gap_start, gap_end, gap_duration))
            
            logger.info(
                "word timing range: %.3f - %.3f seconds (%.2f seconds total)",
                first_word_start,
                last_word_end,
                last_word_end - first_word_start,
            )
            
            if gaps:
                logger.warning(
                    "found %d large timing gaps (>2s) where no words will appear:",
                    len(gaps)
                )
                for gap_start, gap_end, duration in gaps[:5]:  # Show first 5 gaps
                    logger.warning(
                        "  gap: %.2f - %.2f seconds (%.2f seconds, no words)",
                        gap_start, gap_end, duration
                    )
                if len(gaps) > 5:
                    logger.warning("  ... and %d more gaps", len(gaps) - 5)
            
            # Check if first word starts at 0 (might indicate timing offset)
            if first_word_start > 0.1:
                logger.debug(
                    "first word starts at %.3fs (not at 0.0s) - this is normal if transcription had initial silence",
                    first_word_start,
                )

        # Get audio duration
        audio_duration = None
        try:
            audio_duration = self._get_audio_duration(tools.ffmpeg, input_path)
            logger.debug("audio duration: %.2f seconds", audio_duration)
            
            # Warn if word timings extend beyond audio duration
            if words and audio_duration:
                last_word_end = words[-1].end_ms / 1000.0
                if last_word_end > audio_duration + 0.5:
                    logger.warning(
                        "word timings extend beyond audio duration: last word ends at %.2fs but audio is %.2fs",
                        last_word_end,
                        audio_duration,
                    )
        except StepError as e:
            logger.warning(
                "Could not determine audio duration: %s. Using -shortest fallback.", e
            )

        # Parse video dimensions
        width, height = self._parse_video_size(self.video_size)

        # Calculate positions for words
        positioned_words = self._calculate_positions(words, width, height)
        
        # Log positioning statistics for debugging
        if positioned_words:
            y_positions = [pw.y for pw in positioned_words]
            x_positions = [pw.x for pw in positioned_words]
            lines_used = len(set(pw.line for pw in positioned_words))
            logger.info(
                "positioned %d words: Y range [%d-%d], X range [%d-%d], %d lines",
                len(positioned_words),
                min(y_positions),
                max(y_positions),
                min(x_positions),
                max(x_positions),
                lines_used,
            )
            
            # Check for words that might be off-screen
            off_screen = [
                pw for pw in positioned_words
                if pw.y < self.font_size or pw.y > height - self.font_size
            ]
            if off_screen:
                logger.warning(
                    "%d words positioned outside safe Y bounds (font_size=%d, height=%d)",
                    len(off_screen),
                    self.font_size,
                    height,
                )

        # Generate word images with alpha transparency
        word_images_dir = output_dir / "word_images"
        word_images_dir.mkdir(exist_ok=True)
        
        logger.info("generating %d word images with alpha transparency", len(positioned_words))
        word_image_paths = self._generate_word_images(
            tools.ffmpeg,
            positioned_words,
            word_images_dir,
            width,
            height,
        )

        # Build ffmpeg command using overlay filters
        output_path = output_dir / f"{input_path.stem}_lyric_video.mp4"

        logger.info(
            "generating lyric video: %s (%d words, %dx%d)",
            input_path.name,
            len(words),
            width,
            height,
        )

        cmd = self._build_overlay_command(
            tools.ffmpeg,
            input_path,
            output_path,
            positioned_words,
            word_image_paths,
            audio_duration,
            width,
            height,
            line_srt_path=line_srt_path,
            word_srt_path=word_srt_path,
        )

        logger.debug("running: %s", " ".join(str(c) for c in cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # Log subprocess output (goes to log file if configured)
            if result.stderr:
                logger.debug("ffmpeg stderr: %s", result.stderr)
            if result.stdout:
                logger.debug("ffmpeg stdout: %s", result.stdout)

            if result.returncode != 0:
                logger.error("ffmpeg stderr: %s", result.stderr[-500:])
                raise StepError(
                    self.name,
                    f"ffmpeg failed: {result.returncode}",
                )

            if not output_path.exists():
                raise StepError(self.name, f"output not created: {output_path}")

            output_size = output_path.stat().st_size
            logger.info(
                "created: %s (%.1f MB)",
                output_path.name,
                output_size / 1024 / 1024,
            )

            return [output_path]

        except FileNotFoundError:
            raise StepError(self.name, f"ffmpeg not found: {tools.ffmpeg}")
        except subprocess.SubprocessError as e:
            raise StepError(self.name, f"subprocess error: {e}")

    def _parse_timestamp(self, timestamp_str: str) -> int:
        """
        Parse timestamp format (HH:MM:SS,mmm) to milliseconds.

        Args:
            timestamp_str: Timestamp string like "00:00:01,234"

        Returns:
            Milliseconds as integer.
        """
        # Replace comma with period for parsing
        time_str = timestamp_str.replace(",", ".")
        parts = time_str.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return int((hours * 3600 + minutes * 60 + seconds) * 1000)

    def _is_word_boundary(self, text: str) -> tuple[bool, str]:
        """
        Check if text indicates a word boundary and extract clean text.

        Args:
            text: Token text (may have leading spaces).

        Returns:
            Tuple of (is_boundary, clean_text).
        """
        has_leading_space = text.startswith(" ")
        clean_text = text.strip()

        if not clean_text:
            return (True, "")

        # Skip Unicode replacement characters and other problematic characters
        if clean_text == "\ufffd" or (len(clean_text) == 1 and unicodedata.category(clean_text) == "So"):
            # Unicode replacement character or other symbols that might be encoding artifacts
            return (True, "")

        # Punctuation-only tokens are boundaries
        if clean_text in [",", ".", "!", "?", ";", ":", "-"]:
            return (True, clean_text)

        # Special tokens are boundaries
        if clean_text.startswith("[") or clean_text.startswith("</"):
            return (True, "")

        return (has_leading_space, clean_text)

    def _parse_word_timestamps(self, json_path: Path) -> list[WordEntry]:
        """
        Parse word-level timestamps from transcript.json.

        Args:
            json_path: Path to transcript.json file.

        Returns:
            List of WordEntry objects with text and timing.
        """
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except UnicodeDecodeError as e:
            # Try reading with error handling for invalid UTF-8 bytes
            logger.warning(
                "UTF-8 decode error reading transcript.json at position %d-%d, "
                "attempting recovery with error replacement",
                e.start,
                e.end,
            )
            try:
                with open(json_path, encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
            except json.JSONDecodeError as json_err:
                raise StepError(
                    self.name,
                    f"Failed to parse transcript.json after encoding recovery: {json_err}",
                ) from json_err
        except json.JSONDecodeError as e:
            raise StepError(
                self.name,
                f"Failed to parse transcript.json: {e}. "
                "The file may be corrupted or not valid JSON.",
            ) from e

        transcription = data.get("transcription", [])
        if not transcription:
            logger.warning("no transcription data in JSON")
            return []

        words = []
        for segment in transcription:
            tokens = segment.get("tokens", [])
            current_word = ""
            word_start_ms = None
            word_end_ms = None

            for token in tokens:
                raw_text = token.get("text", "")
                # Fix Unicode encoding issues (mojibake, replacement characters)
                raw_text = ftfy.fix_text(raw_text, normalization="NFC")
                # Remove Unicode replacement characters that couldn't be fixed
                raw_text = raw_text.replace("\ufffd", "")
                is_boundary, clean_text = self._is_word_boundary(raw_text)

                # Skip empty or special tokens
                if not clean_text:
                    continue

                timestamps = token.get("timestamps", {})
                if not timestamps:
                    continue

                from_ts = timestamps.get("from", "")
                to_ts = timestamps.get("to", "")

                if not from_ts or not to_ts:
                    continue

                # Parse timestamps
                try:
                    from_ms = self._parse_timestamp(from_ts)
                    to_ms = self._parse_timestamp(to_ts)
                except (ValueError, IndexError):
                    continue

                # Skip if duration is 0 or negative
                if to_ms <= from_ms:
                    continue

                if is_boundary:
                    # Save current word if we have one
                    if (
                        current_word
                        and word_start_ms is not None
                        and word_end_ms is not None
                    ):
                        word_text = current_word.strip()
                        if word_text:
                            words.append(
                                WordEntry(
                                    text=word_text,
                                    start_ms=word_start_ms,
                                    end_ms=word_end_ms,
                                )
                            )
                        current_word = ""
                        word_start_ms = None
                        word_end_ms = None

                    # If it's punctuation, create a separate entry for it
                    if clean_text in [",", ".", "!", "?", ";", ":", "-"]:
                        # Punctuation gets very short duration
                        words.append(
                            WordEntry(
                                text=clean_text,
                                start_ms=from_ms,
                                end_ms=min(from_ms + 50, to_ms),
                            )
                        )
                    else:
                        # Start new word
                        word_start_ms = from_ms
                        word_end_ms = to_ms
                        current_word = clean_text
                else:
                    # Continue current word
                    if word_start_ms is None:
                        word_start_ms = from_ms
                    word_end_ms = to_ms
                    current_word += clean_text

            # Save final word if exists
            if current_word and word_start_ms is not None and word_end_ms is not None:
                word_text = current_word.strip()
                if word_text:
                    words.append(
                        WordEntry(
                            text=word_text,
                            start_ms=word_start_ms,
                            end_ms=word_end_ms,
                        )
                    )

        return words

    def _parse_video_size(self, size_str: str) -> tuple[int, int]:
        """
        Parse video size string to width and height.

        Args:
            size_str: Size string like "1920x1080"

        Returns:
            Tuple of (width, height).
        """
        parts = size_str.split("x")
        if len(parts) != 2:
            raise StepError(self.name, f"Invalid video size format: {size_str}")
        return (int(parts[0]), int(parts[1]))

    def _calculate_positions(
        self, words: list[WordEntry], video_width: int, video_height: int
    ) -> list[PositionedWord]:
        """
        Calculate horizontal positions for overlapping words.

        Words overlapping in time are placed on the same line in start-time order.
        Wraps to next line when exceeding width. Ensures Y positions stay within video bounds.

        Args:
            words: List of words with timing.
            video_width: Video width in pixels.
            video_height: Video height in pixels.

        Returns:
            List of PositionedWord objects with x, y, line positions.
        """
        if not words:
            return []

        # Estimate character width (approximate, based on font size)
        # Average character width is roughly 0.6 * font_size for most fonts
        char_width = int(self.font_size * 0.6)
        line_height = self.font_size + 10  # Height per line including spacing

        # Calculate vertical position based on position setting
        if self.position == "top":
            base_y = self.font_size + 20
        elif self.position == "bottom":
            base_y = video_height - self.font_size - 20
        else:  # center (default)
            base_y = video_height // 2

        # Calculate max lines that fit on screen
        max_lines_up = (base_y - self.font_size) // line_height if base_y > self.font_size else 0
        max_lines_down = (video_height - base_y - self.font_size) // line_height
        max_total_lines = max_lines_up + max_lines_down + 1  # +1 for center line

        # Build timeline of active words at each time point
        # Group words by their active time ranges
        positioned = []
        # Track rightmost x position for each line
        line_rightmost: dict[int, int] = {}
        # Track which line number we're using (can be negative for lines above center)
        line_numbers_used: set[int] = {0}  # Start with line 0 (center)

        # Sort words by start time
        sorted_words = sorted(words, key=lambda w: w.start_ms)

        # For each word, find which line it should be on
        # by checking overlap with words already placed
        for word in sorted_words:
            # Find all words that overlap with this word's time range
            overlapping = [
                pw
                for pw in positioned
                if not (
                    pw.word.end_ms <= word.start_ms or pw.word.start_ms >= word.end_ms
                )
            ]

            # Estimate word width
            word_width = len(word.text) * char_width + self.word_spacing

            # Try to place on existing line if there's overlap
            placed = False
            for pw in overlapping:
                # Find rightmost position on this line
                line_right_x = line_rightmost.get(pw.line, 0)
                # Check if we can fit on this line
                if line_right_x + word_width <= video_width:
                    # Place on same line as overlapping word, after rightmost word
                    x = line_right_x
                    y = base_y + (pw.line * line_height)
                    # Ensure Y stays within bounds
                    if y < self.font_size:
                        y = self.font_size
                    elif y > video_height - self.font_size:
                        y = video_height - self.font_size
                    positioned.append(
                        PositionedWord(word=word, x=x, y=y, line=pw.line)
                    )
                    # Update rightmost position for this line
                    line_rightmost[pw.line] = x + word_width
                    placed = True
                    break

            if not placed:
                # Start new line - find the best available line
                # Try to use lines near center first
                best_line = None
                for line_offset in range(max_total_lines):
                    # Try lines: 0, 1, -1, 2, -2, 3, -3, ...
                    for sign in [1, -1] if line_offset > 0 else [0]:
                        if sign == 0:
                            test_line = 0
                        else:
                            test_line = sign * line_offset
                        
                        # Check if this line has room
                        line_right_x = line_rightmost.get(test_line, 0)
                        if line_right_x + word_width <= video_width:
                            best_line = test_line
                            break
                    
                    if best_line is not None:
                        break
                
                # If no line has room, use the line with most space
                if best_line is None:
                    best_line = min(line_rightmost.keys(), key=lambda l: line_rightmost.get(l, 0), default=0)
                    # If even the emptiest line doesn't fit, start a new line
                    if line_rightmost.get(best_line, 0) + word_width > video_width:
                        # Find next unused line number
                        for line_num in range(-max_lines_up, max_lines_down + 1):
                            if line_num not in line_numbers_used:
                                best_line = line_num
                                line_numbers_used.add(line_num)
                                break
                        else:
                            # All lines used, use the one with least content
                            best_line = min(line_rightmost.keys(), key=lambda l: line_rightmost.get(l, 0), default=0)

                x = line_rightmost.get(best_line, 0)
                y = base_y + (best_line * line_height)
                # Ensure Y stays within bounds
                if y < self.font_size:
                    y = self.font_size
                elif y > video_height - self.font_size:
                    y = video_height - self.font_size
                
                positioned.append(PositionedWord(word=word, x=x, y=y, line=best_line))
                line_rightmost[best_line] = x + word_width
                line_numbers_used.add(best_line)

        # Center all lines horizontally
        # Group words by line and calculate total width for each line
        lines: dict[int, list[PositionedWord]] = {}
        for pw in positioned:
            if pw.line not in lines:
                lines[pw.line] = []
            lines[pw.line].append(pw)

        # Calculate centering offsets for each line
        line_offsets: dict[int, int] = {}
        for line_num, line_words in lines.items():
            if not line_words:
                continue
            
            # Sort words on this line by x position
            line_words.sort(key=lambda pw: pw.x)
            
            # Calculate total width of all words on this line
            # (including spacing between words)
            first_word = line_words[0]
            last_word = line_words[-1]
            last_word_width = len(last_word.word.text) * char_width
            total_line_width = last_word.x + last_word_width - first_word.x
            
            # Calculate center offset to center the line
            center_x = video_width // 2
            line_center_x = first_word.x + (total_line_width // 2)
            offset_x = center_x - line_center_x
            line_offsets[line_num] = offset_x

        # Apply centering offsets while preserving original word order
        centered_positioned = []
        for pw in positioned:
            offset_x = line_offsets.get(pw.line, 0)
            new_x = pw.x + offset_x
            # Ensure X stays within bounds
            if new_x < 0:
                new_x = 0
            word_width_est = len(pw.word.text) * char_width
            if new_x + word_width_est > video_width:
                new_x = max(0, video_width - word_width_est)
            
            # Create new PositionedWord with centered x
            centered_positioned.append(
                PositionedWord(word=pw.word, x=new_x, y=pw.y, line=pw.line)
            )

        return centered_positioned

    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """
        Sanitize text for use in filenames.

        Args:
            text: Text to sanitize.
            max_length: Maximum length of sanitized text (default: 50).

        Returns:
            Sanitized text safe for filenames.
        """
        import re
        
        # Remove Unicode replacement characters
        text = text.replace("\ufffd", "")
        
        # Replace spaces with underscores
        text = text.replace(" ", "_")
        
        # Remove or replace unsafe filename characters
        # Keep alphanumeric, underscores, hyphens, and basic Unicode letters
        # Remove: / \ : * ? " < > |
        text = re.sub(r'[<>:"/\\|?*]', '', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f]', '', text)
        
        # Limit length
        if len(text) > max_length:
            text = text[:max_length]
        
        # Remove leading/trailing dots and spaces (Windows issue)
        text = text.strip('. ')
        
        # If empty after sanitization, use a fallback
        if not text:
            text = "word"
        
        return text

    def _escape_text_for_drawtext(self, text: str) -> str:
        """
        Escape text for use in FFmpeg drawtext filter.

        Args:
            text: Text to escape.

        Returns:
            Escaped text safe for drawtext filter.
        """
        # Remove Unicode replacement characters (from encoding recovery)
        # These can cause FFmpeg filter parsing issues
        text = text.replace("\ufffd", "")  # Unicode replacement character
        
        # Escape special characters for FFmpeg drawtext
        # Order matters: escape backslashes FIRST, then other characters
        # Backslashes need to be escaped (do this first!)
        text = text.replace("\\", "\\\\")
        # Single quotes need to be escaped
        text = text.replace("'", "\\'")
        # Colons need to be escaped (they're filter parameter separators)
        text = text.replace(":", "\\:")
        # Square brackets need to be escaped (they're used for input labels)
        text = text.replace("[", "\\[")
        text = text.replace("]", "\\]")
        return text

    def _get_background_input(
        self,
        ffmpeg: Path,
        duration: float | None,
        output_dir: Path,
    ) -> tuple[list[str], str]:
        """
        Get the ffmpeg input arguments for the background.

        Supports:
        - Solid color: "black", "#FF0000", etc.
        - Gradient: "vertical:purple:black" or "horizontal:blue:cyan"
        - Image: Path to an image file (scaled/cropped to fit)

        Args:
            ffmpeg: Path to ffmpeg executable.
            duration: Duration in seconds (None if unknown).
            output_dir: Directory for temporary files.

        Returns:
            Tuple of (input_args, filter_prefix) where:
            - input_args: List of ffmpeg arguments to add before filter
            - filter_prefix: Filter to apply to background (may be empty)
        """
        width, height = map(int, self.video_size.split("x"))
        duration_str = f":d={duration}" if duration else ""

        # Priority: background_image > background_gradient > background_color
        if self.background_image and Path(self.background_image).exists():
            # Use image as background (loop it to match audio duration)
            img_path = Path(self.background_image)
            logger.info("using background image: %s", img_path.name)
            
            # Loop image and scale/crop to fit video size
            input_args = ["-loop", "1", "-i", str(img_path)]
            # Scale to cover, then crop to exact size
            filter_prefix = f"[1:v]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}[bg];"
            return input_args, filter_prefix

        elif self.background_gradient:
            # Parse gradient spec: "direction:color1:color2"
            # direction: vertical, horizontal, radial
            parts = self.background_gradient.split(":")
            if len(parts) >= 3:
                direction = parts[0].lower()
                color1 = parts[1]
                color2 = parts[2]
            elif len(parts) == 2:
                direction = "vertical"
                color1 = parts[0]
                color2 = parts[1]
            else:
                logger.warning("invalid gradient spec, using solid color")
                direction = None

            if direction:
                logger.info("using %s gradient: %s -> %s", direction, color1, color2)
                
                # Generate gradient using geq filter
                # For vertical gradient: darker at top, lighter at bottom
                if direction == "vertical":
                    # Create gradient using gradients filter (FFmpeg 5.0+)
                    # Fallback: use geq for older ffmpeg
                    gradient_filter = (
                        f"gradients=s={width}x{height}:c0={color1}:c1={color2}:"
                        f"x0=0:y0=0:x1=0:y1={height}{duration_str}"
                    )
                elif direction == "horizontal":
                    gradient_filter = (
                        f"gradients=s={width}x{height}:c0={color1}:c1={color2}:"
                        f"x0=0:y0=0:x1={width}:y1=0{duration_str}"
                    )
                elif direction == "radial":
                    # Radial gradient from center
                    gradient_filter = (
                        f"gradients=s={width}x{height}:c0={color1}:c1={color2}:"
                        f"x0={width//2}:y0={height//2}:x1=0:y1=0:type=radial{duration_str}"
                    )
                else:
                    # Default to vertical
                    gradient_filter = (
                        f"gradients=s={width}x{height}:c0={color1}:c1={color2}:"
                        f"x0=0:y0=0:x1=0:y1={height}{duration_str}"
                    )
                
                input_args = ["-f", "lavfi", "-i", gradient_filter]
                return input_args, ""

        # Default: solid color
        color_filter = f"color=c={self.background_color}:s={self.video_size}{duration_str}"
        input_args = ["-f", "lavfi", "-i", color_filter]
        return input_args, ""

    def _generate_word_images(
        self,
        ffmpeg: Path,
        positioned_words: list[PositionedWord],
        output_dir: Path,
        video_width: int,
        video_height: int,
    ) -> list[tuple[PositionedWord, Path]]:
        """
        Generate PNG images for each word with alpha transparency.

        Args:
            ffmpeg: Path to ffmpeg executable.
            positioned_words: List of positioned words.
            output_dir: Directory to save word images.
            video_width: Video width.
            video_height: Video height.

        Returns:
            List of (PositionedWord, image_path) tuples.
        """
        word_image_paths: list[tuple[PositionedWord, Path]] = []
        
        # Sanitize font name for filename
        font_name_safe = self._sanitize_filename(self.font_name, max_length=20)
        
        for i, pw in enumerate(positioned_words):
            # Sanitize word text for filename
            word_text_safe = self._sanitize_filename(pw.word.text, max_length=30)
            
            # Generate image filename with phrase, font, and size
            # Format: word_0001_{phrase}_{font}_{size}.png
            image_path = output_dir / f"word_{i:04d}_{word_text_safe}_{font_name_safe}_{self.font_size}.png"
            
            # Escape text for FFmpeg
            text_escaped = self._escape_text_for_drawtext(pw.word.text)
            
            # Build font specification
            if self.font_file and self.font_file.exists():
                font_path = str(self.font_file.resolve()).replace("\\", "\\\\").replace("'", "\\'")
                font_spec = f"fontfile='{font_path}'"
            else:
                font_name_escaped = self.font_name.replace("'", "\\'").replace(":", "\\:")
                font_spec = f"font='{font_name_escaped}'"
            
            # Estimate word dimensions (add padding for safety)
            char_width = int(self.font_size * 0.6)
            word_width = len(pw.word.text) * char_width + 20  # Add padding
            word_height = self.font_size + 20  # Add padding
            
            # Build ffmpeg command to generate word image
            # Use color source with alpha=0 (transparent) and drawtext on it
            drawtext_filter = (
                f"drawtext={font_spec}:"
                f"text='{text_escaped}':"
                f"fontsize={self.font_size}:"
                f"fontcolor={self.font_color}:"
                f"x=10:y=10"  # Position text in image (with padding)
            )
            
            cmd = [
                str(ffmpeg),
                "-y",
                "-loglevel", "warning",  # Show warnings and errors, suppress info/stats
                "-f", "lavfi",
                "-i", f"color=c=black@0:s={word_width}x{word_height}:d=0.1",  # Transparent background (alpha=0)
                "-vf", drawtext_filter,
                "-frames:v", "1",  # Single frame
                "-pix_fmt", "rgba",  # RGBA format for alpha transparency (no background)
                str(image_path),
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                
                if result.returncode != 0:
                    logger.warning(
                        "Failed to generate word image for '%s': %s",
                        pw.word.text,
                        result.stderr[-200:] if result.stderr else "unknown error",
                    )
                    continue
                
                if image_path.exists():
                    word_image_paths.append((pw, image_path))
                    logger.debug("generated word image: %s", image_path.name)
                else:
                    logger.warning("Word image not created: %s", image_path)
                    
            except Exception as e:
                logger.warning("Error generating word image for '%s': %s", pw.word.text, e)
                continue
        
        logger.info("generated %d word images", len(word_image_paths))
        return word_image_paths

    def _build_overlay_command(
        self,
        ffmpeg: Path,
        audio: Path,
        output: Path,
        positioned_words: list[PositionedWord],
        word_image_paths: list[tuple[PositionedWord, Path]],
        duration: float | None,
        width: int,
        height: int,
        line_srt_path: Path | None = None,
        word_srt_path: Path | None = None,
    ) -> list[str]:
        """
        Build ffmpeg command using overlay filters for word images.

        Args:
            ffmpeg: Path to ffmpeg executable.
            audio: Path to audio file.
            output: Path to output video file.
            positioned_words: List of positioned words.
            word_image_paths: List of (PositionedWord, image_path) tuples.
            duration: Audio duration in seconds (None if unknown).
            width: Video width.
            height: Video height.

        Returns:
            List of command arguments.
        """
        cmd = [
            str(ffmpeg),
            "-y",  # Overwrite output
            "-loglevel", "warning",  # Show warnings and errors, suppress info/stats
            "-nostats",  # Don't print encoding statistics (prevents broken pipe errors)
        ]

        # Add audio input
        cmd.extend(["-i", str(audio)])

        # Add background (solid color, gradient, or image)
        bg_input_args, bg_filter_prefix = self._get_background_input(
            ffmpeg, duration, output.parent
        )
        cmd.extend(bg_input_args)

        # Add all word images as inputs
        input_idx = 2  # Start after audio (0) and background (1)
        for pw, img_path in word_image_paths:
            cmd.extend(["-i", str(img_path)])
            input_idx += 1

        # Add line-level SRT subtitle file (if available)
        line_subtitle_idx = None
        if line_srt_path and line_srt_path.exists():
            cmd.extend(["-i", str(line_srt_path)])
            line_subtitle_idx = input_idx
            input_idx += 1
            logger.debug("adding line-level subtitles from: %s", line_srt_path.name)

        # Add word-level SRT subtitle file (if available)
        word_subtitle_idx = None
        if word_srt_path and word_srt_path.exists():
            cmd.extend(["-i", str(word_srt_path)])
            word_subtitle_idx = input_idx
            input_idx += 1
            logger.debug("adding word-level subtitles from: %s", word_srt_path.name)

        if not word_image_paths:
            raise StepError(
                self.name,
                "No word images generated - cannot create lyric video",
            )

        # Build overlay filter chain
        # Start with background video
        filter_chains = []
        
        # Add background filter prefix if needed (for image backgrounds)
        if bg_filter_prefix:
            filter_chains.append(bg_filter_prefix.rstrip(";"))
            current_input = "[bg]"
        else:
            current_input = "[1:v]"
        
        # Overlay each word image at the right time and position
        for i, (pw, img_path) in enumerate(word_image_paths):
            img_input_idx = 2 + i  # Image input index (after audio=0, background=1)
            start_sec = pw.word.start_ms / 1000.0
            end_sec = pw.word.end_ms / 1000.0
            
            # Log first few words for timing debugging
            if i < 10:
                logger.debug(
                    "word %d: '%s' at (x=%d, y=%d) timing %.3f-%.3f (from JSON: %d-%d ms)",
                    i + 1,
                    pw.word.text,
                    pw.x,
                    pw.y,
                    start_sec,
                    end_sec,
                    pw.word.start_ms,
                    pw.word.end_ms,
                )
            
            # Check for overlapping words at the same time
            if i > 0:
                prev_pw = word_image_paths[i - 1][0]
                prev_end = prev_pw.word.end_ms / 1000.0
                if start_sec < prev_end:
                    logger.debug(
                        "word '%s' overlaps with previous word '%s' (%.3f < %.3f)",
                        pw.word.text,
                        prev_pw.word.text,
                        start_sec,
                        prev_end,
                    )
            
            # Build overlay filter with enable expression
            # overlay=x=X:y=Y:format=auto:enable='between(t,START,END)'
            # format=auto is required to properly blend RGBA images with alpha
            # Note: current_input already contains brackets (e.g., [1:v] or [v155])
            # Note: overlay filter uses 'enable', not 'if' (unlike drawtext)
            overlay_filter = (
                f"{current_input}[{img_input_idx}:v]"
                f"overlay={pw.x}:{pw.y}:format=auto:"
                f"enable='between(t,{start_sec:.3f},{end_sec:.3f})'"
            )
            
            if i == len(word_image_paths) - 1:
                # Last overlay outputs to [out]
                filter_chains.append(f"{overlay_filter}[out]")
            else:
                # Intermediate overlays use [vN] labels
                next_label = f"[v{i+1}]"
                filter_chains.append(f"{overlay_filter}{next_label}")
                current_input = next_label
        
        filter_chain = ";".join(filter_chains)
        
        # Write filter to file
        filter_file = output.parent / "filter_complex.txt"
        filter_file.write_text(filter_chain, encoding="utf-8")
        logger.debug("wrote overlay filter complex to file: %s (%d chars)", filter_file.name, len(filter_chain))
        
        # Log filter for debugging (truncate if too long)
        if len(filter_chain) > 500:
            logger.debug("filter complex (truncated): %s...", filter_chain[:500])
        else:
            logger.debug("filter complex: %s", filter_chain)

        # Map streams and use filter_complex_script
        cmd.extend(
            [
                "-filter_complex_script",
                str(filter_file),
                "-map",
                "[out]",  # Map output from filter complex
                "-map",
                "0:a",  # Audio from input 0
            ]
        )

        # Map line-level subtitle stream (first subtitle track)
        subtitle_track = 0
        if line_subtitle_idx is not None:
            cmd.extend(
                [
                    "-map",
                    f"{line_subtitle_idx}:s",  # Map line-level subtitles
                ]
            )
            subtitle_track += 1

        # Map word-level subtitle stream (second subtitle track)
        if word_subtitle_idx is not None:
            cmd.extend(
                [
                    "-map",
                    f"{word_subtitle_idx}:s",  # Map word-level subtitles
                ]
            )
            subtitle_track += 1

        # Only use -shortest if we couldn't determine duration
        if duration is None:
            cmd.append("-shortest")

        # Add encoding options
        cmd.extend(
            [
                "-c:v",
                "libx264",  # H.264 video codec
                "-pix_fmt",
                "yuv420p",  # Standard pixel format for compatibility
                "-c:a",
                "aac",  # Encode audio as AAC for MP4
                "-b:a",
                "192k",  # Audio bitrate
                "-r",
                "30",  # Frame rate for stable timing
                "-vsync",
                "cfr",  # Constant frame rate
            ]
        )

        # Add subtitle codec and metadata for all subtitle tracks
        if line_subtitle_idx is not None or word_subtitle_idx is not None:
            cmd.extend(["-c:s", "mov_text"])  # Subtitle codec for MP4 (soft subtitles - toggleable)
            
            # Set metadata for line-level subtitles (track 0)
            if line_subtitle_idx is not None:
                cmd.extend(
                    [
                        "-metadata:s:s:0",
                        "language=eng",
                        "-metadata:s:s:0",
                        "title=English",  # Display name for line-level track
                        "-disposition:s:0",
                        "default",  # Make line-level subtitles default
                    ]
                )
            
            # Set metadata for word-level subtitles (track 1)
            if word_subtitle_idx is not None:
                track_num = 1 if line_subtitle_idx is not None else 0
                cmd.extend(
                    [
                        f"-metadata:s:s:{track_num}",
                        "language=eng",
                        f"-metadata:s:s:{track_num}",
                        "title=English - Word Level",  # Display name for word-level track
                    ]
                )

        cmd.append(str(output))

        return cmd

    def _build_drawtext_filter(
        self, word: PositionedWord, video_width: int, video_height: int
    ) -> str:
        """
        Build a single drawtext filter for one word.

        Args:
            word: PositionedWord with word and position.
            video_width: Video width (for centering calculations).
            video_height: Video height (for positioning).

        Returns:
            Drawtext filter string.
        """
        # Convert milliseconds to seconds for enable expression
        start_sec = word.word.start_ms / 1000.0
        end_sec = word.word.end_ms / 1000.0

        # Escape text for FFmpeg
        text_escaped = self._escape_text_for_drawtext(word.word.text)

        # Build font specification
        if self.font_file and self.font_file.exists():
            font_path = str(self.font_file.resolve()).replace("\\", "\\\\").replace("'", "\\'")
            font_spec = f"fontfile='{font_path}'"
        else:
            # Escape font name
            font_name_escaped = self.font_name.replace("'", "\\'").replace(":", "\\:")
            font_spec = f"font='{font_name_escaped}'"

        # Build drawtext filter
        # Use 'if' expression instead of 'enable' to avoid comma parsing issues
        # Format: if='between(t,START,END)', which doesn't require commas in the expression
        # Actually, we still need commas. Let's try using double quotes for the enable expression
        # or use if expressions: if='between(t,START,END)'
        # FFmpeg's filter parser should handle commas inside single-quoted strings correctly
        # But to be safe, let's use the 'if' parameter which is an alias for 'enable'
        # and ensure proper quoting
        
        # Use if='between(t,START,END)' - the if parameter should handle this better
        enable_expr = f"between(t,{start_sec:.3f},{end_sec:.3f})"
        
        filter_parts = [
            font_spec,
            f"text='{text_escaped}'",
            f"fontsize={self.font_size}",
            f"fontcolor={self.font_color}",
            f"x={word.x}",
            f"y={word.y}",
            f"if='{enable_expr}'",  # Use 'if' instead of 'enable' - it's an alias
        ]

        return "drawtext=" + ":".join(filter_parts)

    def _build_command(
        self,
        ffmpeg: Path,
        audio: Path,
        output: Path,
        positioned_words: list[PositionedWord],
        duration: float | None,
        width: int,
        height: int,
    ) -> list[str]:
        """
        Build ffmpeg command for lyric video generation.

        Args:
            ffmpeg: Path to ffmpeg executable.
            audio: Path to audio file.
            output: Path to output video file.
            positioned_words: List of positioned words.
            duration: Audio duration in seconds (None if unknown).
            width: Video width.
            height: Video height.

        Returns:
            List of command arguments.
        """
        cmd = [
            str(ffmpeg),
            "-y",  # Overwrite output
            "-loglevel", "warning",  # Show warnings and errors, suppress info/stats
            "-nostats",  # Don't print encoding statistics (prevents broken pipe errors)
        ]

        # Add audio input
        cmd.extend(["-i", str(audio)])

        # Add solid color background
        if duration is not None:
            cmd.extend(
                [
                    "-f",
                    "lavfi",
                    "-i",
                    f"color=c={self.background_color}:s={self.video_size}:d={duration}",
                ]
            )
        else:
            cmd.extend(
                [
                    "-f",
                    "lavfi",
                    "-i",
                    f"color=c={self.background_color}:s={self.video_size}",
                ]
            )

        # Build video filter chain: background + all drawtext filters
        # FFmpeg syntax: [input_label]filter1,filter2,filter3
        filter_parts = []

        # Add all drawtext filters
        for pw in positioned_words:
            drawtext_filter = self._build_drawtext_filter(pw, width, height)
            filter_parts.append(drawtext_filter)

        if not filter_parts:
            raise StepError(
                self.name,
                "No words to display - cannot generate empty lyric video",
            )

        # Validate all filters are non-empty
        valid_filters = [f for f in filter_parts if f and f.strip()]
        if len(valid_filters) != len(filter_parts):
            raise StepError(
                self.name,
                f"Some drawtext filters are empty ({len(filter_parts)} total, {len(valid_filters)} valid)",
            )

        # Write filter complex to a file to avoid shell quoting/escaping issues
        # This is more reliable for complex filters with many parameters
        filter_file = output.parent / "filter_complex.txt"
        
        # Build filter chain using semicolons with intermediate labels
        filter_chains = []
        current_input = "[1:v]"
        
        for i, filter_str in enumerate(valid_filters):
            if i == len(valid_filters) - 1:
                # Last filter outputs to [out]
                filter_chains.append(f"{current_input}{filter_str}[out]")
            else:
                # Intermediate filters use [vN] labels
                next_label = f"[v{i+1}]"
                filter_chains.append(f"{current_input}{filter_str}{next_label}")
                current_input = next_label
        
        filter_chain = ";".join(filter_chains)
        
        # Write filter to file
        filter_file.write_text(filter_chain, encoding="utf-8")
        logger.debug("wrote filter complex to file: %s (%d chars)", filter_file.name, len(filter_chain))
        
        # Log filter for debugging (truncate if too long)
        if len(filter_chain) > 500:
            logger.debug("filter complex (truncated): %s...", filter_chain[:500])
        else:
            logger.debug("filter complex: %s", filter_chain)

        # Map streams and use filter_complex_script (reads from file)
        cmd.extend(
            [
                "-filter_complex_script",
                str(filter_file),
                "-map",
                "[out]",  # Map output from filter complex
                "-map",
                "0:a",  # Audio from input 0
            ]
        )

        # Only use -shortest if we couldn't determine duration
        if duration is None:
            cmd.append("-shortest")

        # Add encoding options
        cmd.extend(
            [
                "-c:a",
                "aac",  # Encode audio as AAC for MP4
                "-b:a",
                "192k",  # Audio bitrate
                "-r",
                "30",  # Frame rate for stable timing
                "-vsync",
                "cfr",  # Constant frame rate
                str(output),
            ]
        )

        return cmd

    def _get_audio_duration(self, ffmpeg: Path, audio_path: Path) -> float:
        """
        Get the duration of an audio file using ffprobe.

        Args:
            ffmpeg: Path to ffmpeg (we use ffprobe which is usually in same dir).
            audio_path: Path to the audio file.

        Returns:
            Duration in seconds as a float.

        Raises:
            StepError: If ffprobe fails or duration cannot be determined.
        """
        # Try ffprobe (usually in same directory as ffmpeg)
        ffprobe = ffmpeg.parent / "ffprobe"
        if not ffprobe.exists():
            # Fallback: try ffprobe in PATH
            ffprobe = Path("ffprobe")

        cmd = [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(audio_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            data = json.loads(result.stdout)
            duration = float(data.get("format", {}).get("duration", 0))

            if duration <= 0:
                raise StepError(
                    self.name,
                    f"Could not determine audio duration from {audio_path}",
                )

            return duration

        except (
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
        ) as e:
            raise StepError(
                self.name,
                f"Could not determine audio duration: {e}",
            )


def run(
    input_path: Path,
    output_dir: Path,
    background_color: str = "black",
    background_image: str | Path | None = None,
    background_gradient: str | None = None,
    video_size: str = "1920x1080",
    font_name: str = "Arial",
    font_file: str | Path | None = None,
    font_size: int = 48,
    font_color: str = "white",
    position: str = "center",
    word_spacing: int = 20,
    max_words_per_line: int | None = None,
) -> StepResult:
    """
    Run the generate_lyric_video step.

    Args:
        input_path: Path to audio file.
        output_dir: Directory for output (must contain transcript.json).
        background_color: Background color (default: "black").
        background_image: Optional path to background image.
        background_gradient: Optional gradient spec (e.g., "vertical:purple:black").
        video_size: Video dimensions as "WxH" (default: "1920x1080").
        font_name: Font family name (default: "Arial").
        font_file: Optional explicit font file path.
        font_size: Font size in pixels (default: 48).
        font_color: Text color (default: "white").
        position: Vertical position: top, center, bottom (default: "center").
        word_spacing: Horizontal spacing between words in pixels (default: 20).
        max_words_per_line: Maximum words per line (None = auto-calculate).

    Returns:
        StepResult with execution details.
    """
    font_file_path = None
    if font_file:
        font_file_path = Path(font_file) if isinstance(font_file, str) else font_file

    bg_image_path = None
    if background_image:
        bg_image_path = Path(background_image) if isinstance(background_image, str) else background_image

    step = GenerateLyricVideoStep(
        background_color=background_color,
        background_image=bg_image_path,
        background_gradient=background_gradient,
        video_size=video_size,
        font_name=font_name,
        font_file=font_file_path,
        font_size=font_size,
        font_color=font_color,
        position=position,
        word_spacing=word_spacing,
        max_words_per_line=max_words_per_line,
    )
    start_time = time.monotonic()

    try:
        outputs = step.execute(input_path, output_dir)
        duration = time.monotonic() - start_time
        return StepResult(
            name=step.name,
            success=True,
            outputs=outputs,
            duration_seconds=duration,
        )
    except StepError as e:
        duration = time.monotonic() - start_time
        return StepResult(
            name=step.name,
            success=False,
            outputs=[],
            duration_seconds=duration,
            error=str(e),
        )
