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
    """

    name: str = "transcribe_timed"

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
            for ext in [".srt", ".vtt", ".json"]:
                path = Path(str(output_prefix) + ext)
                if path.exists():
                    outputs.append(path)
                    size = path.stat().st_size
                    logger.info("created: %s (%d bytes)", path.name, size)

            if not outputs:
                raise StepError(self.name, "no output files created")

            return outputs

        except FileNotFoundError:
            raise StepError(self.name, f"whisper-cli not found: {tools.whisper_cli}")
        except subprocess.SubprocessError as e:
            raise StepError(self.name, f"subprocess error: {e}")


def run(input_path: Path, output_dir: Path) -> StepResult:
    """
    Run the transcribe_timed step.

    Args:
        input_path: Path to input audio file.
        output_dir: Directory for output artifacts.

    Returns:
        StepResult with execution details.
    """
    step = TranscribeTimedStep()
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
