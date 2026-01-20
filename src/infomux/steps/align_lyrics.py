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
        
        # Convert lyrics to MPLAIN format for better word-level alignment
        # MPLAIN: paragraphs separated by blank lines, sentences on separate lines
        # For lyrics, treat each line as a sentence, group stanzas as paragraphs
        lines = [line.strip() for line in lyrics_text.split("\n") if line.strip()]
        mplain_text = "\n".join(lines)  # Each line is a sentence, blank lines separate paragraphs

        # Convert audio to a format aeneas can read (44.1kHz mono 16-bit PCM)
        # aeneas prefers higher sample rates and may have issues with 16kHz
        tools = get_tool_paths()
        if not tools.ffmpeg:
            raise StepError(self.name, "ffmpeg not found")
        
        # Create a temporary converted audio file for aeneas
        temp_audio = output_dir / "audio_for_alignment.wav"
        convert_cmd = [
            str(tools.ffmpeg),
            "-y",
            "-i", str(input_path),
            "-ac", "1",  # Mono
            "-ar", "44100",  # 44.1kHz (aeneas prefers this)
            "-c:a", "pcm_s16le",  # 16-bit PCM
            str(temp_audio),
        ]
        
        logger.debug("converting audio for aeneas: %s", " ".join(convert_cmd))
        convert_result = subprocess.run(
            convert_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        
        if convert_result.returncode != 0:
            logger.error("ffmpeg conversion failed: %s", convert_result.stderr[-500:])
            raise StepError(
                self.name,
                f"failed to convert audio for aeneas: {convert_result.returncode}",
            )
        
        if not temp_audio.exists():
            raise StepError(self.name, "converted audio file not created")
        
        # Use the converted audio file for alignment
        audio_for_alignment = temp_audio

        # Check if aeneas is available (use sys.executable to use the same Python as the script)
        import sys
        python_exe = sys.executable
        try:
            result = subprocess.run(
                [python_exe, "-m", "aeneas.tools.execute_task", "--help"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            raise StepError(
                self.name,
                "aeneas not found. Install via: uv pip install numpy && uv pip install aeneas",
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
        
        # Use MPLAIN (multilevel plain) format for better word-level alignment
        # MPLAIN: paragraphs separated by blank lines, sentences on separate lines
        # This gives us multilevel fragments (paragraph -> sentence -> word)
        config_string = (
            f"task_language={self.language}|"
            "is_text_type=mplain|"  # Multilevel plain text for word-level alignment
            "os_task_file_format=json|"
            f"tts={tts_engine}|"
            "mfcc_mask_nonspeech=True|"
            "mfcc_mask_nonspeech_l3=True"
        )

        # Write MPLAIN formatted text to temporary file
        temp_lyrics = output_dir / "lyrics_mplain.txt"
        temp_lyrics.write_text(mplain_text, encoding="utf-8")
        
        cmd = [
            python_exe,  # Use the same Python executable
            "-m",
            "aeneas.tools.execute_task",
            str(audio_for_alignment),  # Use converted audio
            str(temp_lyrics),  # Use MPLAIN formatted lyrics
            config_string,
            str(temp_syncmap),
            "--presets-word",  # Word-level alignment
        ]

        logger.debug("running: %s", " ".join(cmd))

        # Set environment variables for aeneas (UTF-8 encoding, etc.)
        import os
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "UTF-8"
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                cwd=output_dir,
                env=env,
            )

            if result.returncode != 0:
                error_msg = result.stderr[-1000:] if result.stderr else ""
                stdout_msg = result.stdout[-1000:] if result.stdout else ""
                logger.error("aeneas failed (exit code %d)", result.returncode)
                if error_msg:
                    logger.error("aeneas stderr: %s", error_msg)
                if stdout_msg:
                    logger.error("aeneas stdout: %s", stdout_msg)
                combined_error = (error_msg + " " + stdout_msg).strip()
                if not combined_error:
                    combined_error = "No error output from aeneas"
                raise StepError(
                    self.name,
                    f"aeneas alignment failed: {combined_error[:200]}",
                )

            if not temp_syncmap.exists():
                raise StepError(
                    self.name,
                    f"aeneas did not create syncmap file: {temp_syncmap}",
                )

            # Convert aeneas syncmap to transcript.json format
            output_path = output_dir / TRANSCRIPT_JSON_FILENAME
            self._convert_syncmap_to_transcript_json(temp_syncmap, lyrics_text, output_path)

            # Clean up temporary files
            if temp_syncmap.exists():
                temp_syncmap.unlink()
            if temp_audio.exists():
                temp_audio.unlink()
            if temp_lyrics.exists():
                temp_lyrics.unlink()

            if not output_path.exists():
                raise StepError(self.name, f"output file not created: {output_path}")

            size = output_path.stat().st_size
            logger.info("aligned lyrics: %s (%d bytes, %d words)", output_path.name, size, len(lyrics_text.split()))
            return [output_path]

        except FileNotFoundError:
            raise StepError(
                self.name, "aeneas not found. Install via: uv pip install numpy && uv pip install aeneas"
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

        # Extract fragments from syncmap
        fragments = syncmap_data.get("fragments", [])
        
        # Check if fragments have children (multilevel/word-level alignment)
        has_children = any(frag.get("children") for frag in fragments)
        
        # Build transcript.json structure
        transcription = []
        current_segment = {
            "id": 0,
            "start": "00:00:00,000",
            "end": "00:00:00,000",
            "text": "",
            "tokens": [],
        }

        def format_timestamp(ms: int) -> str:
            """Format milliseconds as HH:MM:SS,mmm"""
            total_seconds = ms // 1000
            milliseconds = ms % 1000
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

        if has_children:
            # Multilevel alignment: extract word-level fragments from children
            word_fragments = []
            for frag in fragments:
                children = frag.get("children", [])
                for child in children:
                    grand_children = child.get("children", [])
                    if grand_children:
                        # Three levels: paragraph -> sentence -> word
                        word_fragments.extend(grand_children)
                    else:
                        # Two levels: sentence -> word
                        word_fragments.append(child)
            
            # Use word fragments directly
            fragments = word_fragments
            logger.debug("using multilevel alignment: %d word fragments", len(fragments))

        # Process fragments - each fragment may contain one or more words
        for fragment in fragments:
            # Get text from fragment (from 'lines' field)
            fragment_lines = fragment.get("lines", [])
            fragment_text = " ".join(fragment_lines).strip()
            
            if not fragment_text:
                continue
            
            # Split fragment text into words
            words_in_fragment = fragment_text.split()
            
            begin_time = fragment.get("begin", "0.000")
            end_time = fragment.get("end", "0.000")
            
            # Convert seconds to milliseconds
            try:
                begin_ms = int(float(begin_time) * 1000)
                end_ms = int(float(end_time) * 1000)
            except (ValueError, TypeError):
                logger.warning("invalid timing: begin=%s, end=%s", begin_time, end_time)
                continue
            
            # If fragment has multiple words, split timing proportionally
            if len(words_in_fragment) > 1:
                duration_ms = end_ms - begin_ms
                time_per_word = duration_ms / len(words_in_fragment)
                
                for i, word_text in enumerate(words_in_fragment):
                    word_begin_ms = begin_ms + int(i * time_per_word)
                    word_end_ms = begin_ms + int((i + 1) * time_per_word)
                    
                    # Last word gets the exact end time
                    if i == len(words_in_fragment) - 1:
                        word_end_ms = end_ms
                    
                    token = {
                        "text": f" {word_text}" if i > 0 else word_text,  # Leading space for word boundaries
                        "timestamps": {
                            "from": format_timestamp(word_begin_ms),
                            "to": format_timestamp(word_end_ms),
                        },
                    }
                    
                    current_segment["tokens"].append(token)
                    current_segment["text"] += word_text + " "
            else:
                # Single word fragment
                word_text = words_in_fragment[0] if words_in_fragment else fragment_text
                
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

        total_words = sum(len(seg.get("tokens", [])) for seg in transcription)
        logger.debug("converted syncmap to transcript.json: %d segments, %d words", len(transcription), total_words)


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
