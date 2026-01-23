"""
Isolate vocals from music using source separation.

Uses Demucs or Spleeter to extract vocal track from mixed audio.
This can improve transcription timing accuracy by removing background music.

Command (Demucs):
    demucs --two-stems=vocals <input.wav>

Command (Spleeter):
    spleeter separate -i <input.wav> -o <output_dir> -p spleeter:2stems

Output:
    audio_vocals_only.wav - Isolated vocal track
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

# Output filename for isolated vocals
VOCALS_FILENAME = "audio_vocals_only.wav"

# Register output filename for pipeline input resolution
OUTPUT_FILENAME = VOCALS_FILENAME


@register_step
@dataclass
class IsolateVocalsStep:
    """
    Pipeline step to isolate vocals from mixed audio.

    Uses source separation (Demucs or Spleeter) to extract vocal track,
    which can improve transcription timing accuracy by removing background music.
    """

    name: str = "isolate_vocals"
    tool: str = "demucs"  # "demucs" or "spleeter"
    model: str | None = None  # Model name (e.g., "htdemucs_ft" for Demucs, "spleeter:2stems" for Spleeter)

    def execute(self, input_path: Path, output_dir: Path) -> list[Path]:
        """
        Isolate vocals from mixed audio.

        Args:
            input_path: Path to the input audio file.
            output_dir: Directory to write the isolated vocals.

        Returns:
            List containing the path to the isolated vocal audio file.

        Raises:
            StepError: If tool is not found or isolation fails.
        """
        if self.tool == "demucs":
            result = self._isolate_with_demucs(input_path, output_dir)
        elif self.tool == "spleeter":
            result = self._isolate_with_spleeter(input_path, output_dir)
        else:
            raise StepError(
                self.name,
                f"Unknown tool: {self.tool}. Use 'demucs' or 'spleeter'.",
            )
        
        # Also extract full audio if it doesn't exist (avoids re-extracting later)
        self._extract_full_audio_if_needed(input_path, output_dir)
        
        return result
    
    def _extract_full_audio_if_needed(self, input_path: Path, output_dir: Path) -> None:
        """
        Extract full audio as a side effect if it doesn't already exist.
        This avoids re-extracting the same file later in the pipeline.
        
        Args:
            input_path: Path to the input audio file.
            output_dir: Directory to write the extracted audio.
        """
        audio_output = output_dir / "audio_full.wav"
        if audio_output.exists():
            return  # Already exists, skip
        
        tools = get_tool_paths()
        if not tools.ffmpeg:
            return  # Can't extract without ffmpeg
        
        logger.debug("extracting full audio as side effect: %s", audio_output.name)
        extract_cmd = [
            str(tools.ffmpeg),
            "-y",
            "-i",
            str(input_path),
            "-vn",  # No video
            "-ac",
            "1",  # Mono
            "-ar",
            "16000",  # 16kHz
            "-c:a",
            "pcm_s16le",  # 16-bit PCM
            str(audio_output),
        ]
        extract_result = subprocess.run(
            extract_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if extract_result.returncode == 0 and audio_output.exists():
            logger.debug("extracted full audio: %s", audio_output.name)
        # Don't fail if extraction fails - it's just a side effect

    def _isolate_with_demucs(self, input_path: Path, output_dir: Path) -> list[Path]:
        """Isolate vocals using Demucs."""
        # Check if demucs is available
        try:
            result = subprocess.run(
                ["demucs", "--help"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            raise StepError(
                self.name,
                "demucs not found. Install via: uv pip install demucs",
            )

        logger.info("isolating vocals with Demucs: %s", input_path.name)

        # Use temporary directory for Demucs output (it creates its own structure)
        temp_output = output_dir / "demucs_temp"
        temp_output.mkdir(exist_ok=True)

        # Build demucs command
        # --two-stems=vocals extracts vocals vs everything else
        # --mp3 output format avoids torchcodec dependency issues
        cmd = [
            "demucs",
            "--two-stems=vocals",
            "--mp3",  # Use MP3 output to avoid torchcodec issues
            "-o",
            str(temp_output),
        ]

        # Add model if specified
        if self.model:
            cmd.extend(["-n", self.model])

        # Add input file
        cmd.append(str(input_path))

        logger.debug("running: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                cwd=output_dir,
            )

            if result.returncode != 0:
                error_msg = result.stderr[-500:] if result.stderr else ""
                stdout_msg = result.stdout[-500:] if result.stdout else ""
                combined_error = (error_msg + " " + stdout_msg).lower()
                logger.error("demucs stderr: %s", error_msg)
                if stdout_msg:
                    logger.error("demucs stdout: %s", stdout_msg)
                
                # Check for torchcodec error and suggest fallback
                if "torchcodec" in combined_error:
                    logger.warning(
                        "Demucs requires torchcodec. Falling back to Spleeter or use: uv pip install torchcodec"
                    )
                    # Try Spleeter as fallback if available
                    try:
                        return self._isolate_with_spleeter(input_path, output_dir)
                    except StepError:
                        raise StepError(
                            self.name,
                            "demucs failed (torchcodec required). Install torchcodec: uv pip install torchcodec, "
                            "or install Spleeter: uv pip install spleeter",
                        )
                
                raise StepError(
                    self.name,
                    f"demucs failed with exit code {result.returncode}",
                )

            # Demucs creates: <output>/<model>/<input_name>/vocals.wav
            # With --two-stems, it's directly in temp_output, not in a "separated" subdirectory
            # Find the vocals file
            vocals_file = None
            
            # Try both structures: with and without "separated" subdirectory
            separated_dir = temp_output / "separated"
            if separated_dir.exists():
                # Standard structure: separated/<model>/<input_name>/vocals.wav
                model_dirs = [d for d in separated_dir.iterdir() if d.is_dir()]
                if model_dirs:
                    model_dir = model_dirs[0]
                    track_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
                    if track_dirs:
                        track_dir = track_dirs[0]
                        vocals_file = track_dir / "vocals.wav"
                        if not vocals_file.exists():
                            vocals_file = track_dir / "vocals.mp3"
            else:
                # Direct structure (with --two-stems): <model>/<input_name>/vocals.wav
                model_dirs = [d for d in temp_output.iterdir() if d.is_dir()]
                if model_dirs:
                    model_dir = model_dirs[0]
                    track_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
                    if track_dirs:
                        track_dir = track_dirs[0]
                        vocals_file = track_dir / "vocals.wav"
                        if not vocals_file.exists():
                            vocals_file = track_dir / "vocals.mp3"
                    else:
                        # Try direct vocals file in model directory
                        vocals_file = model_dir / "vocals.wav"
                        if not vocals_file.exists():
                            vocals_file = model_dir / "vocals.mp3"

            if not vocals_file or not vocals_file.exists():
                # List what was actually created for debugging
                created_files = []
                if temp_output.exists():
                    for root, dirs, files in temp_output.rglob("*"):
                        if files:
                            created_files.extend([str(Path(root) / f) for f in files])
                error_msg = f"demucs did not create vocals file. Check: {temp_output}"
                if created_files:
                    error_msg += f"\nFound files: {', '.join(created_files[:5])}"
                raise StepError(self.name, error_msg)

            # Copy to final location and convert to 16kHz mono if needed
            output_path = output_dir / VOCALS_FILENAME
            tools = get_tool_paths()

            if not tools.ffmpeg:
                # Just copy if ffmpeg not available
                import shutil

                shutil.copy2(vocals_file, output_path)
            else:
                # Convert to 16kHz mono WAV (same format as extract_audio)
                convert_cmd = [
                    str(tools.ffmpeg),
                    "-y",
                    "-i",
                    str(vocals_file),
                    "-ac",
                    "1",  # Mono
                    "-ar",
                    "16000",  # 16kHz
                    "-c:a",
                    "pcm_s16le",  # 16-bit PCM
                    str(output_path),
                ]

                convert_result = subprocess.run(
                    convert_cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if convert_result.returncode != 0:
                    logger.warning(
                        "ffmpeg conversion failed, copying original: %s",
                        convert_result.stderr[-200:],
                    )
                    import shutil

                    shutil.copy2(vocals_file, output_path)

            # Clean up temporary directory
            import shutil

            if temp_output.exists():
                shutil.rmtree(temp_output)

            if not output_path.exists():
                raise StepError(self.name, f"output file not created: {output_path}")

            size = output_path.stat().st_size
            logger.info("isolated vocals: %s (%d bytes)", output_path.name, size)
            return [output_path]

        except FileNotFoundError:
            raise StepError(self.name, "demucs not found. Install via: uv pip install demucs")
        except subprocess.SubprocessError as e:
            raise StepError(self.name, f"subprocess error: {e}")

    def _isolate_with_spleeter(self, input_path: Path, output_dir: Path) -> list[Path]:
        """Isolate vocals using Spleeter."""
        # Check if spleeter is available
        try:
            result = subprocess.run(
                ["spleeter", "--help"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            raise StepError(
                self.name,
                "spleeter not found. Install via: uv pip install spleeter",
            )

        logger.info("isolating vocals with Spleeter: %s", input_path.name)

        # Use temporary directory for Spleeter output
        temp_output = output_dir / "spleeter_temp"
        temp_output.mkdir(exist_ok=True)

        # Build spleeter command
        model = self.model or "spleeter:2stems"  # Default to 2-stem (vocals + accompaniment)
        cmd = [
            "spleeter",
            "separate",
            "-i",
            str(input_path),
            "-o",
            str(temp_output),
            "-p",
            model,
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
                logger.error("spleeter stderr: %s", result.stderr[-500:])
                raise StepError(
                    self.name,
                    f"spleeter failed with exit code {result.returncode}",
                )

            # Spleeter creates: <output_dir>/<input_name>/vocals.wav
            input_stem = input_path.stem
            vocals_file = temp_output / input_stem / "vocals.wav"

            if not vocals_file.exists():
                # Try alternative location
                vocals_file = temp_output / "vocals.wav"
                if not vocals_file.exists():
                    raise StepError(
                        self.name,
                        f"spleeter did not create vocals file. Check: {temp_output}",
                    )

            # Copy to final location and convert to 16kHz mono if needed
            output_path = output_dir / VOCALS_FILENAME
            tools = get_tool_paths()

            if not tools.ffmpeg:
                # Just copy if ffmpeg not available
                import shutil

                shutil.copy2(vocals_file, output_path)
            else:
                # Convert to 16kHz mono WAV (same format as extract_audio)
                convert_cmd = [
                    str(tools.ffmpeg),
                    "-y",
                    "-i",
                    str(vocals_file),
                    "-ac",
                    "1",  # Mono
                    "-ar",
                    "16000",  # 16kHz
                    "-c:a",
                    "pcm_s16le",  # 16-bit PCM
                    str(output_path),
                ]

                convert_result = subprocess.run(
                    convert_cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if convert_result.returncode != 0:
                    logger.warning(
                        "ffmpeg conversion failed, copying original: %s",
                        convert_result.stderr[-200:],
                    )
                    import shutil

                    shutil.copy2(vocals_file, output_path)

            # Clean up temporary directory
            import shutil

            if temp_output.exists():
                shutil.rmtree(temp_output)

            if not output_path.exists():
                raise StepError(self.name, f"output file not created: {output_path}")

            size = output_path.stat().st_size
            logger.info("isolated vocals: %s (%d bytes)", output_path.name, size)
            return [output_path]

        except FileNotFoundError:
            raise StepError(
                self.name, "spleeter not found. Install via: uv pip install spleeter"
            )
        except subprocess.SubprocessError as e:
            raise StepError(self.name, f"subprocess error: {e}")


def run(
    input_path: Path,
    output_dir: Path,
    tool: str = "demucs",
    model: str | None = None,
) -> StepResult:
    """
    Convenience function to run the isolate_vocals step.

    Args:
        input_path: Path to input audio file.
        output_dir: Directory for output artifacts.
        tool: Tool to use ("demucs" or "spleeter").
        model: Optional model name.

    Returns:
        StepResult with execution details.
    """
    step = IsolateVocalsStep(tool=tool, model=model)
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
