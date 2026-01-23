"""
Extract audio step: extract audio track from media file.

Uses ffmpeg to extract audio as 16kHz mono PCM WAV, which is the
format required by whisper.cpp for transcription.

Command:
    ffmpeg -y -i <input> -vn -ac 1 -ar 16000 -c:a pcm_s16le <output>

Flags:
    -y          Overwrite output without asking
    -i          Input file
    -vn         No video (audio only)
    -ac 1       Mono audio (1 channel)
    -ar 16000   16kHz sample rate
    -c:a pcm_s16le  16-bit PCM little-endian (uncompressed WAV)
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from infomux.config import get_tool_paths
from infomux.log import get_logger
from infomux.steps import StepError, StepResult, register_step

logger = get_logger(__name__)

# Output filename for extracted audio (full mix)
AUDIO_FILENAME = "audio_full.wav"


@register_step
@dataclass
class ExtractAudioStep:
    """
    Pipeline step to extract audio from a media file.

    Produces a 16kHz mono WAV file suitable for whisper.cpp transcription.
    """

    name: str = "extract_audio"

    def execute(self, input_path: Path, output_dir: Path) -> list[Path]:
        """
        Extract audio from the input media file.

        Args:
            input_path: Path to the input media file (audio or video).
            output_dir: Directory to write the extracted audio.

        Returns:
            List containing the path to the extracted audio file.

        Raises:
            StepError: If ffmpeg is not found or extraction fails.
        """
        # Find ffmpeg
        tools = get_tool_paths()
        if not tools.ffmpeg:
            raise StepError(
                self.name, "ffmpeg not found. Install via: brew install ffmpeg"
            )

        output_path = output_dir / AUDIO_FILENAME

        # Skip extraction if audio already exists (e.g., from isolate_vocals)
        if output_path.exists():
            size = output_path.stat().st_size
            logger.info("audio already exists: %s (%d bytes), skipping extraction", output_path.name, size)
            return [output_path]

        logger.info("extracting audio: %s -> %s", input_path.name, output_path.name)

        # Build ffmpeg command
        cmd = [
            str(tools.ffmpeg),
            "-y",                   # Overwrite without asking
            "-i", str(input_path),  # Input file
            "-vn",                  # No video
            "-ac", "1",             # Mono
            "-ar", "16000",         # 16kHz sample rate
            "-c:a", "pcm_s16le",    # 16-bit PCM
            str(output_path),
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
                # ffmpeg writes to stderr even on success, so only log on failure
                logger.error("ffmpeg stderr: %s", result.stderr)
                raise StepError(
                    self.name,
                    f"ffmpeg failed with exit code {result.returncode}",
                )

            # Verify output was created
            if not output_path.exists():
                raise StepError(self.name, f"output file not created: {output_path}")

            size = output_path.stat().st_size
            logger.info("extracted audio: %s (%d bytes)", output_path.name, size)
            return [output_path]

        except FileNotFoundError:
            raise StepError(self.name, f"ffmpeg not executable: {tools.ffmpeg}")
        except subprocess.SubprocessError as e:
            raise StepError(self.name, f"subprocess error: {e}")


def run(input_path: Path, output_dir: Path) -> StepResult:
    """
    Convenience function to run the extract_audio step.

    Args:
        input_path: Path to input media file.
        output_dir: Directory for output artifacts.

    Returns:
        StepResult with execution details.
    """
    step = ExtractAudioStep()
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
