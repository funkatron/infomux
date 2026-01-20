"""
Transcribe with word-level timestamps for caption generation.

Uses whisper-cli with Dynamic Time Warping (-dtw) for precise word-level
alignment. Outputs subtitle files (SRT, VTT) and JSON with timing data.

Command:
    whisper-cli -m <model> -f <audio.wav> -of <prefix> -osrt -ovtt -ojf -dtw <type>

Output files:
    - transcript.srt: SRT subtitles
    - transcript.vtt: VTT subtitles
    - transcript.json: Full JSON with word-level timestamps
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from infomux.config import get_tool_paths
from infomux.log import get_logger
from infomux.steps import StepError, StepResult, register_step

logger = get_logger(__name__)

# Output filename prefix and primary output (for pipeline input resolution)
OUTPUT_PREFIX = "transcript"
TRANSCRIPT_TIMED_FILENAME = "transcript.srt"  # Primary output for downstream steps


def _parse_timestamp(timestamp_str: str) -> int:
    """
    Parse SRT timestamp format (HH:MM:SS,mmm) to milliseconds.

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


def _format_timestamp_srt(ms: int) -> str:
    """
    Format milliseconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        ms: Milliseconds.

    Returns:
        Formatted timestamp string.
    """
    total_seconds = ms // 1000
    milliseconds = ms % 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _format_timestamp_vtt(ms: int) -> str:
    """
    Format milliseconds to VTT timestamp format (HH:MM:SS.mmm).

    Args:
        ms: Milliseconds.

    Returns:
        Formatted timestamp string.
    """
    total_seconds = ms // 1000
    milliseconds = ms % 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _is_word_boundary(text: str) -> tuple[bool, str]:
    """
    Check if text indicates a word boundary and extract clean text.

    Args:
        text: Token text (may have leading spaces).

    Returns:
        Tuple of (is_boundary, clean_text).
        is_boundary: True if this starts a new word or is punctuation.
        clean_text: Cleaned text without leading spaces.
    """
    # Leading space indicates word boundary
    has_leading_space = text.startswith(" ")
    clean_text = text.strip()

    if not clean_text:
        return (True, "")

    # Punctuation-only tokens are boundaries
    if clean_text in [",", ".", "!", "?", ";", ":", "-"]:
        return (True, clean_text)

    # Special tokens are boundaries
    if clean_text.startswith("[") or clean_text.startswith("</"):
        return (True, "")

    return (has_leading_space, clean_text)


def _generate_word_level_srt(json_path: Path, output_path: Path) -> None:
    """
    Generate word-level SRT file from JSON transcription.

    Each word appears when it's spoken and disappears when it's done.
    Combines sub-word tokens into complete words.

    Args:
        json_path: Path to transcript.json file.
        output_path: Path to write word-level SRT file.
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
            logger.error("Failed to parse transcript.json after encoding recovery: %s", json_err)
            return
    except json.JSONDecodeError as e:
        logger.error("Failed to parse transcript.json: %s", e)
        return

    transcription = data.get("transcription", [])
    if not transcription:
        logger.warning("no transcription data in JSON, skipping word-level SRT")
        return

    entries = []
    entry_num = 1

    for segment in transcription:
        tokens = segment.get("tokens", [])
        current_word = ""
        word_start_ms = None
        word_end_ms = None

        for token in tokens:
            raw_text = token.get("text", "")
            is_boundary, clean_text = _is_word_boundary(raw_text)

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
                from_ms = _parse_timestamp(from_ts)
                to_ms = _parse_timestamp(to_ts)
            except (ValueError, IndexError):
                continue

            # Skip if duration is 0 or negative
            if to_ms <= from_ms:
                continue

            if is_boundary:
                # Save current word if we have one
                if current_word and word_start_ms is not None and word_end_ms is not None:
                    word_text = current_word.strip()
                    if word_text:
                        entries.append({
                            "num": entry_num,
                            "from": word_start_ms,
                            "to": word_end_ms,
                            "text": word_text,
                        })
                        entry_num += 1
                    current_word = ""
                    word_start_ms = None
                    word_end_ms = None

                # If it's punctuation, create a separate entry for it
                if clean_text in [",", ".", "!", "?", ";", ":", "-"]:
                    # Punctuation gets very short duration
                    entries.append({
                        "num": entry_num,
                        "from": from_ms,
                        "to": min(from_ms + 50, to_ms),  # Max 50ms for punctuation
                        "text": clean_text,
                    })
                    entry_num += 1
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
                entries.append({
                    "num": entry_num,
                    "from": word_start_ms,
                    "to": word_end_ms,
                    "text": word_text,
                })
                entry_num += 1

    # Write SRT file
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(f"{entry['num']}\n")
            f.write(
                f"{_format_timestamp_srt(entry['from'])} --> "
                f"{_format_timestamp_srt(entry['to'])}\n"
            )
            f.write(f"{entry['text']}\n")
            f.write("\n")

    logger.info(
        "generated word-level SRT: %s (%d words)",
        output_path.name,
        len(entries),
    )


def _generate_word_level_vtt(json_path: Path, output_path: Path) -> None:
    """
    Generate word-level VTT file from JSON transcription.

    Each word appears when it's spoken and disappears when it's done.
    Combines sub-word tokens into complete words.

    Args:
        json_path: Path to transcript.json file.
        output_path: Path to write word-level VTT file.
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
            logger.error("Failed to parse transcript.json after encoding recovery: %s", json_err)
            return
    except json.JSONDecodeError as e:
        logger.error("Failed to parse transcript.json: %s", e)
        return

    transcription = data.get("transcription", [])
    if not transcription:
        logger.warning("no transcription data in JSON, skipping word-level VTT")
        return

    entries = []

    for segment in transcription:
        tokens = segment.get("tokens", [])
        current_word = ""
        word_start_ms = None
        word_end_ms = None

        for token in tokens:
            raw_text = token.get("text", "")
            is_boundary, clean_text = _is_word_boundary(raw_text)

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
                from_ms = _parse_timestamp(from_ts)
                to_ms = _parse_timestamp(to_ts)
            except (ValueError, IndexError):
                continue

            # Skip if duration is 0 or negative
            if to_ms <= from_ms:
                continue

            if is_boundary:
                # Save current word if we have one
                if current_word and word_start_ms is not None and word_end_ms is not None:
                    word_text = current_word.strip()
                    if word_text:
                        entries.append({
                            "from": word_start_ms,
                            "to": word_end_ms,
                            "text": word_text,
                        })
                    current_word = ""
                    word_start_ms = None
                    word_end_ms = None

                # If it's punctuation, create a separate entry for it
                if clean_text in [",", ".", "!", "?", ";", ":", "-"]:
                    # Punctuation gets very short duration
                    entries.append({
                        "from": from_ms,
                        "to": min(from_ms + 50, to_ms),  # Max 50ms for punctuation
                        "text": clean_text,
                    })
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
                entries.append({
                    "from": word_start_ms,
                    "to": word_end_ms,
                    "text": word_text,
                })

    # Write VTT file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for entry in entries:
            f.write(
                f"{_format_timestamp_vtt(entry['from'])} --> "
                f"{_format_timestamp_vtt(entry['to'])}\n"
            )
            f.write(f"{entry['text']}\n\n")

    logger.info(
        "generated word-level VTT: %s (%d words)",
        output_path.name,
        len(entries),
    )


def _detect_model_type(model_path: Path) -> str:
    """
    Detect whisper model type from filename for DTW.

    Args:
        model_path: Path to the model file.

    Returns:
        Model type string for -dtw flag (e.g., 'base.en', 'small', 'medium').
    """
    name = model_path.stem.lower()

    # Map ggml model names to DTW model types
    if "tiny.en" in name:
        return "tiny.en"
    elif "tiny" in name:
        return "tiny"
    elif "base.en" in name:
        return "base.en"
    elif "base" in name:
        return "base"
    elif "small.en" in name:
        return "small.en"
    elif "small" in name:
        return "small"
    elif "medium.en" in name:
        return "medium.en"
    elif "medium" in name:
        return "medium"
    elif "large" in name:
        return "large"
    else:
        # Default to base.en
        return "base.en"


@register_step
@dataclass
class TranscribeTimedStep:
    """
    Pipeline step for word-level timestamped transcription.

    Uses whisper.cpp's DTW (Dynamic Time Warping) for precise word alignment.
    Produces SRT, VTT, and JSON output files suitable for captioning.

    By default, generates sentence-level subtitles. Set generate_word_level=True
    to also generate word-level subtitles where each word appears individually.
    """

    name: str = "transcribe_timed"
    generate_word_level: bool = False  # Generate word-level subtitles (optional)

    def execute(self, input_path: Path, output_dir: Path) -> list[Path]:
        """
        Transcribe audio with word-level timestamps.

        Args:
            input_path: Path to input audio file (16kHz mono WAV).
            output_dir: Directory to write outputs.

        Returns:
            List of output file paths (SRT, VTT, JSON).

        Raises:
            StepError: If transcription fails.
        """
        tools = get_tool_paths()

        if not tools.whisper_cli:
            raise StepError(
                self.name,
                "whisper-cli not found. Install via: brew install whisper-cpp",
            )

        if not tools.whisper_model:
            raise StepError(self.name, "Whisper model not found")

        # Detect model type for DTW
        model_type = _detect_model_type(tools.whisper_model)
        output_prefix = output_dir / OUTPUT_PREFIX

        logger.info("transcribing with word-level timestamps: %s", input_path.name)
        logger.debug("model: %s, dtw type: %s", tools.whisper_model.name, model_type)

        # Build command with DTW for word-level timestamps
        cmd = [
            str(tools.whisper_cli),
            "-m", str(tools.whisper_model),
            "-f", str(input_path),
            "-of", str(output_prefix),
            "-osrt",      # SRT subtitles
            "-ovtt",      # VTT subtitles
            "-ojf",       # Full JSON with timestamps
            "-dtw", model_type,  # Enable DTW for word-level alignment
            "-np",        # No progress output
        ]

        logger.debug("running: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                logger.error("whisper-cli stderr: %s", result.stderr)
                raise StepError(
                    self.name,
                    f"whisper-cli failed: {result.returncode}",
                )

            # Collect output files
            outputs = []
            json_path = None
            for ext in [".srt", ".vtt", ".json"]:
                path = Path(str(output_prefix) + ext)
                if path.exists():
                    outputs.append(path)
                    size = path.stat().st_size
                    logger.info("created: %s (%d bytes)", path.name, size)
                    if ext == ".json":
                        json_path = path

            if not outputs:
                raise StepError(self.name, "no output files created")

            # Generate word-level subtitles from JSON (if enabled)
            if self.generate_word_level and json_path and json_path.exists():
                word_srt_path = output_dir / "transcript_words.srt"
                word_vtt_path = output_dir / "transcript_words.vtt"

                try:
                    _generate_word_level_srt(json_path, word_srt_path)
                    if word_srt_path.exists():
                        outputs.append(word_srt_path)
                        # Count entries (each entry is 4 lines: number, timestamp, text, blank)
                        with open(word_srt_path) as f:
                            line_count = sum(1 for _ in f)
                            word_count = line_count // 4
                        logger.info(
                            "generated word-level SRT: %s (%d words)",
                            word_srt_path.name,
                            word_count,
                        )

                    _generate_word_level_vtt(json_path, word_vtt_path)
                    if word_vtt_path.exists():
                        outputs.append(word_vtt_path)
                        logger.info(
                            "generated word-level VTT: %s",
                            word_vtt_path.name,
                        )
                except Exception as e:
                    # Log but don't fail - word-level is optional enhancement
                    logger.warning("failed to generate word-level subtitles: %s", e)

            return outputs

        except FileNotFoundError:
            raise StepError(self.name, f"whisper-cli not found: {tools.whisper_cli}")
        except subprocess.SubprocessError as e:
            raise StepError(self.name, f"subprocess error: {e}")


def run(
    input_path: Path,
    output_dir: Path,
    generate_word_level: bool = False,
) -> StepResult:
    """
    Run the transcribe_timed step.

    Args:
        input_path: Path to input audio file.
        output_dir: Directory for output artifacts.
        generate_word_level: If True, also generate word-level subtitles.

    Returns:
        StepResult with execution details.
    """
    step = TranscribeTimedStep(generate_word_level=generate_word_level)
    start_time = time.monotonic()

    tools = get_tool_paths()
    model_info = None
    if tools.whisper_model:
        model_info = {
            "model": {
                "name": tools.whisper_model.name,
                "provider": "whisper.cpp",
                "path": str(tools.whisper_model),
            },
            "params": {
                "dtw": _detect_model_type(tools.whisper_model),
            },
        }

    try:
        outputs = step.execute(input_path, output_dir)
        duration = time.monotonic() - start_time
        return StepResult(
            name=step.name,
            success=True,
            outputs=outputs,
            duration_seconds=duration,
            model_info=model_info,
        )
    except StepError as e:
        duration = time.monotonic() - start_time
        return StepResult(
            name=step.name,
            success=False,
            outputs=[],
            duration_seconds=duration,
            error=str(e),
            model_info=model_info,
        )
