"""
Forced alignment step: align official lyrics to audio for precise word-level timing.

Uses aeneas or similar forced alignment tools to synchronize known lyrics text
with audio, producing word-level timestamps. This is more accurate than
transcription-based timing when you have the official lyrics.

Command (aeneas):
    python -m aeneas.tools.execute_task audio.wav lyrics.txt config syncmap.json

Output:
    transcript.json - Word-level timestamps matching official lyrics
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from infomux.config import get_tool_paths
from infomux.log import get_logger
from infomux.steps import StepError, StepResult, register_step

logger = get_logger(__name__)

# Output filename for aligned transcript
TRANSCRIPT_JSON_FILENAME = "transcript.json"

# Register output filename for pipeline input resolution
OUTPUT_FILENAME = TRANSCRIPT_JSON_FILENAME


@register_step
@dataclass
class AlignLyricsStep:
    """
    Pipeline step to align official lyrics to audio using forced alignment.

    Takes official lyrics text and audio, and produces word-level timestamps
    that precisely match the lyrics to the audio. More accurate than transcription
    when you have the exact lyrics.
    """

    name: str = "align_lyrics"
    lyrics_file: str | Path | None = None  # Path to lyrics text file (if None, looks for lyrics.txt in output_dir)
    language: str = "eng"  # Language code for alignment

    def execute(self, input_path: Path, output_dir: Path) -> list[Path]:
        """
        Align lyrics to audio using forced alignment.

        Args:
            input_path: Path to the input audio file.
            output_dir: Directory containing lyrics file and for output.

        Returns:
            List containing the path to transcript.json with word-level timestamps.

        Raises:
            StepError: If alignment fails or required files not found.
        """
        # Find lyrics file
        lyrics_path = None
        
        # If lyrics_file is provided in config, use it
        if self.lyrics_file:
            # Handle both string and Path types
            lyrics_file_str = str(self.lyrics_file)
            lyrics_path = Path(lyrics_file_str)
            
            # If relative path, try relative to output_dir first, then current working directory
            if not lyrics_path.is_absolute():
                # Try in output_dir first
                candidate = output_dir / lyrics_path
                if candidate.exists():
                    lyrics_path = candidate
                elif lyrics_path.exists():
                    # Use as-is if it exists relative to current directory
                    lyrics_path = lyrics_path.resolve()
                else:
                    raise StepError(self.name, f"lyrics file not found: {self.lyrics_file}")
            elif not lyrics_path.exists():
                raise StepError(self.name, f"lyrics file not found: {lyrics_path}")
        
        # If not found yet, look for common lyrics file names in output_dir
        if lyrics_path is None:
            for name in ["lyrics.txt", "lyrics", "official_lyrics.txt"]:
                candidate = output_dir / name
                if candidate.exists():
                    lyrics_path = candidate
                    break
        
        if lyrics_path is None:
            raise StepError(
                self.name,
                "lyrics file not found. Provide lyrics_file in config (--lyrics-file) or place lyrics.txt in output directory.",
            )

        logger.info("aligning lyrics: %s -> %s", lyrics_path.name, input_path.name)

        # Read lyrics text
        try:
            lyrics_text = lyrics_path.read_text(encoding="utf-8", errors="replace").strip()
            if not lyrics_text:
                raise StepError(self.name, "lyrics file is empty")
        except Exception as e:
            raise StepError(self.name, f"failed to read lyrics file: {e}")

        # Check if aeneas is available
        try:
            result = subprocess.run(
                ["python", "-m", "aeneas.tools.execute_task", "--help"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            raise StepError(
                self.name,
                "aeneas not found. Install via: uv pip install aeneas",
            )

        # Prepare temporary files
        temp_syncmap = output_dir / "syncmap.json"
        
        # Build aeneas command
        # Use word-level alignment with non-speech masking for music
        # Use macOS built-in TTS if available, otherwise fall back to espeak
        import platform
        if platform.system() == "Darwin":
            tts_engine = "macos"  # Use macOS built-in TTS (no espeak needed)
        else:
            tts_engine = "espeak"  # Default to espeak on Linux
        
        config_string = (
            f"task_language={self.language}|"
            "is_text_type=plain|"
            "os_task_file_format=json|"
            f"tts={tts_engine}|"
            "mfcc_mask_nonspeech=True|"
            "mfcc_mask_nonspeech_l3=True"
        )

        cmd = [
            "python",
            "-m",
            "aeneas.tools.execute_task",
            str(input_path),
            str(lyrics_path),
            config_string,
            str(temp_syncmap),
            "--presets-word",  # Word-level alignment
        ]

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
                logger.error("aeneas stderr: %s", result.stderr[-500:])
                raise StepError(
                    self.name,
                    f"aeneas alignment failed with exit code {result.returncode}",
                )

            if not temp_syncmap.exists():
                raise StepError(
                    self.name,
                    f"aeneas did not create syncmap file: {temp_syncmap}",
                )

            # Convert aeneas syncmap to transcript.json format
            output_path = output_dir / TRANSCRIPT_JSON_FILENAME
            self._convert_syncmap_to_transcript_json(temp_syncmap, lyrics_text, output_path)

            # Clean up temporary file
            if temp_syncmap.exists():
                temp_syncmap.unlink()

            if not output_path.exists():
                raise StepError(self.name, f"output file not created: {output_path}")

            size = output_path.stat().st_size
            logger.info("aligned lyrics: %s (%d bytes, %d words)", output_path.name, size, len(lyrics_text.split()))
            return [output_path]

        except FileNotFoundError:
            raise StepError(
                self.name, "aeneas not found. Install via: uv pip install aeneas"
            )
        except subprocess.SubprocessError as e:
            raise StepError(self.name, f"subprocess error: {e}")

    def _convert_syncmap_to_transcript_json(
        self, syncmap_path: Path, lyrics_text: str, output_path: Path
    ) -> None:
        """
        Convert aeneas syncmap JSON to transcript.json format.

        Args:
            syncmap_path: Path to aeneas syncmap JSON file.
            lyrics_text: Original lyrics text.
            output_path: Path to write transcript.json.
        """
        try:
            with open(syncmap_path, encoding="utf-8") as f:
                syncmap_data = json.load(f)
        except Exception as e:
            raise StepError(
                self.name, f"failed to parse aeneas syncmap: {e}"
            )

        # Parse lyrics into words (preserve line breaks for better alignment)
        # Split by whitespace but keep track of line structure
        lines = [line.strip() for line in lyrics_text.split("\n") if line.strip()]
        all_words = []
        for line in lines:
            words_in_line = line.split()
            all_words.extend(words_in_line)
        
        # Extract fragments from syncmap
        fragments = syncmap_data.get("fragments", [])
        
        if len(fragments) != len(all_words):
            logger.warning(
                "syncmap has %d fragments but lyrics has %d words - alignment may be incomplete",
                len(fragments),
                len(all_words),
            )
            # Use minimum to avoid index errors
            num_words = min(len(fragments), len(all_words))
        else:
            num_words = len(all_words)

        # Build transcript.json structure
        # Format matches what transcribe_timed produces
        transcription = []
        current_segment = {
            "id": 0,
            "start": "00:00:00,000",
            "end": "00:00:00,000",
            "text": "",
            "tokens": [],
        }

        for i, fragment in enumerate(fragments):
            if i >= num_words:
                break
                
            word_text = all_words[i]
            begin_time = fragment.get("begin", "0.000")
            end_time = fragment.get("end", "0.000")
            
            # Convert seconds to milliseconds
            try:
                begin_ms = int(float(begin_time) * 1000)
                end_ms = int(float(end_time) * 1000)
            except (ValueError, TypeError):
                logger.warning("invalid timing for word %d: begin=%s, end=%s", i, begin_time, end_time)
                continue

            # Format timestamps as HH:MM:SS,mmm
            def format_timestamp(ms: int) -> str:
                total_seconds = ms // 1000
                milliseconds = ms % 1000
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

            # Create token entry
            token = {
                "text": f" {word_text}",  # Leading space indicates word boundary
                "timestamps": {
                    "from": format_timestamp(begin_ms),
                    "to": format_timestamp(end_ms),
                },
            }

            current_segment["tokens"].append(token)
            current_segment["text"] += word_text + " "

            # Update segment end time
            current_segment["end"] = format_timestamp(end_ms)

        # Add final segment
        if current_segment["tokens"]:
            transcription.append(current_segment)

        # Build full transcript.json structure
        transcript_data = {
            "transcription": transcription,
        }

        # Write output
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)

        logger.debug("converted syncmap to transcript.json format: %d segments", len(transcription))


def run(
    input_path: Path,
    output_dir: Path,
    lyrics_file: Path | None = None,
    language: str = "eng",
) -> StepResult:
    """
    Convenience function to run the align_lyrics step.

    Args:
        input_path: Path to input audio file.
        output_dir: Directory containing lyrics file and for output.
        lyrics_file: Optional path to lyrics file (if None, looks for lyrics.txt).
        language: Language code for alignment (default: "eng").

    Returns:
        StepResult with execution details.
    """
    step = AlignLyricsStep(lyrics_file=lyrics_file, language=language)
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
