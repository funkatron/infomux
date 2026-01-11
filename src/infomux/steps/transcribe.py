"""
Transcribe step: transcribe audio to text using whisper.cpp.

Uses whisper-cli (from whisper-cpp Homebrew formula) for local,
GPU-accelerated transcription on Apple Silicon.

Command:
    whisper-cli -m <model> -f <input.wav> -of <output_prefix> -otxt -np

Flags:
    -m      Path to GGML model file
    -f      Input audio file (must be 16kHz mono WAV)
    -of     Output file prefix (whisper-cli appends .txt)
    -otxt   Output as plain text
    -np     No progress output (quiet mode)
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

# Output filename for transcript
TRANSCRIPT_PREFIX = "transcript"
TRANSCRIPT_FILENAME = f"{TRANSCRIPT_PREFIX}.txt"


@register_step
@dataclass
class TranscribeStep:
    """
    Pipeline step to transcribe audio using whisper.cpp.

    Requires:
    - whisper-cli installed (brew install whisper-cpp)
    - GGML model file (e.g., ggml-base.en.bin)
    - Input audio in 16kHz mono WAV format
    """

    name: str = "transcribe"

    def execute(self, input_path: Path, output_dir: Path) -> list[Path]:
        """
        Transcribe an audio file to text.

        Args:
            input_path: Path to the input audio file (16kHz mono WAV).
            output_dir: Directory to write the transcript.

        Returns:
            List containing the path to the transcript file.

        Raises:
            StepError: If whisper-cli or model not found, or transcription fails.
        """
        # Validate tools
        tools = get_tool_paths()

        if not tools.whisper_cli:
            raise StepError(
                self.name,
                "whisper-cli not found. Install via: brew install whisper-cpp",
            )

        if not tools.whisper_model:
            raise StepError(
                self.name,
                "Whisper model not found. Set INFOMUX_WHISPER_MODEL or run: "
                "curl -L -o ~/.local/share/infomux/models/whisper/ggml-base.en.bin "
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
            )

        # whisper-cli uses -of for output prefix, then appends .txt
        output_prefix = output_dir / TRANSCRIPT_PREFIX
        output_path = output_dir / TRANSCRIPT_FILENAME

        logger.info("transcribing: %s", input_path.name)
        logger.debug("using model: %s", tools.whisper_model)

        # Build whisper-cli command
        cmd = [
            str(tools.whisper_cli),
            "-m", str(tools.whisper_model),
            "-f", str(input_path),
            "-of", str(output_prefix),  # Output file prefix
            "-otxt",                     # Plain text output
            "-np",                       # No progress (quiet)
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
                logger.error("whisper-cli stdout: %s", result.stdout)
                raise StepError(
                    self.name,
                    f"whisper-cli failed with exit code {result.returncode}",
                )

            # Verify output was created
            if not output_path.exists():
                raise StepError(self.name, f"transcript not created: {output_path}")

            transcript_size = output_path.stat().st_size
            logger.info("transcribed: %s (%d bytes)", output_path.name, transcript_size)

            # Log first line of transcript for verification
            with open(output_path) as f:
                first_line = f.readline().strip()
                if first_line:
                    if len(first_line) > 80:
                        preview = first_line[:80] + "..."
                    else:
                        preview = first_line
                    logger.debug("transcript preview: %s", preview)

            return [output_path]

        except FileNotFoundError:
            raise StepError(
                self.name, f"whisper-cli not executable: {tools.whisper_cli}"
            )
        except subprocess.SubprocessError as e:
            raise StepError(self.name, f"subprocess error: {e}")


def run(input_path: Path, output_dir: Path) -> StepResult:
    """
    Convenience function to run the transcribe step.

    Args:
        input_path: Path to input audio file (16kHz mono WAV).
        output_dir: Directory for output artifacts.

    Returns:
        StepResult with execution details.
    """
    step = TranscribeStep()
    start_time = time.monotonic()

    # Get model info for recording
    tools = get_tool_paths()
    model_info = None
    if tools.whisper_model:
        model_info = {
            "model": {
                "name": tools.whisper_model.name,
                "provider": "whisper.cpp",
                "path": str(tools.whisper_model),
            },
            "params": {},  # whisper.cpp doesn't have generation params
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
