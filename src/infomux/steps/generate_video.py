"""
Generate video from audio with subtitles (burned-in + soft stream).

Creates a video file from an audio file by combining it with a static
background image (or solid color), burning in subtitles, and adding a
soft subtitle stream for toggleable display.

By default includes both:
- Burned-in subtitles (permanently rendered on video)
- Soft subtitle stream (toggleable in video players)

Command:
    ffmpeg -i audio.wav -f lavfi -i color=c=black:s=1920x1080 -i subs.srt \\
           -map 0:a -map 1:v -map 2:s \\
           -vf "[1:v]subtitles='subs.srt'" \\
           -c:s mov_text -shortest output.mp4
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from infomux.config import get_tool_paths
from infomux.log import get_logger
from infomux.steps import StepError, StepResult, register_step

logger = get_logger(__name__)


@register_step
@dataclass
class GenerateVideoStep:
    """
    Pipeline step to generate a video from audio with subtitles.

    Creates a video file by:
    1. Taking audio as input
    2. Finding subtitle file (transcript.srt or transcript_words.srt) in output_dir
    3. Using a static background (image or solid color)
    4. Burning subtitles into the video (always visible)
    5. Adding soft subtitle stream (toggleable in video players)

    By default, includes BOTH:
    - Burned-in subtitles (permanently rendered on video)
    - Soft subtitle stream (can be toggled on/off in players)

    When used in a pipeline, this step:
    - Takes audio file as input (from extract_audio)
    - Finds the .srt file in the output directory (from transcribe_timed)
    - Prefers transcript_words.srt (word-level) if available, falls back to transcript.srt
    """

    name: str = "generate_video"
    background_image: Path | None = None  # If None, uses solid color
    background_color: str = "black"  # Used if background_image is None
    video_size: str = "1920x1080"  # Width x Height
    subtitle_style: dict = field(default_factory=dict)

    def execute(
        self,
        input_path: Path,
        output_dir: Path,
    ) -> list[Path]:
        """
        Generate video from audio with burned subtitles.

        Args:
            input_path: Path to the input audio file.
            output_dir: Directory containing subtitle file and for output.

        Returns:
            List containing the path to the generated video.
        """
        # Find subtitle file in output_dir
        # Prefer word-level subtitles if available (optional feature)
        word_srt_path = output_dir / "transcript_words.srt"
        if word_srt_path.exists():
            subtitle_path = word_srt_path
            logger.debug("using word-level subtitles (if available)")
        else:
            # Use regular sentence-level subtitles (default)
            srt_path = output_dir / "transcript.srt"
            if srt_path.exists():
                subtitle_path = srt_path
            else:
                # Look for any .srt file
                srt_files = list(output_dir.glob("*.srt"))
                if srt_files:
                    subtitle_path = srt_files[0]
                else:
                    raise StepError(
                        self.name, "No subtitle file found in output directory"
                    )

        return self._generate(input_path, subtitle_path, output_dir)

    def _generate(
        self,
        audio_path: Path,
        subtitle_path: Path,
        output_dir: Path,
    ) -> list[Path]:
        """
        Generate video from audio with burned subtitles.

        Args:
            audio_path: Path to the input audio file.
            subtitle_path: Path to the subtitle file (SRT or VTT).
            output_dir: Directory to write the output video.

        Returns:
            List containing the path to the output video.

        Raises:
            StepError: If ffmpeg fails or files not found.
        """
        tools = get_tool_paths()

        if not tools.ffmpeg:
            raise StepError(self.name, "ffmpeg not found")

        if not audio_path.exists():
            raise StepError(self.name, f"audio not found: {audio_path}")

        if not subtitle_path.exists():
            raise StepError(self.name, f"subtitles not found: {subtitle_path}")

        # Get audio duration to generate video of exact length
        # Fallback to None if we can't determine duration (will use -shortest)
        audio_duration = None
        try:
            audio_duration = self._get_audio_duration(tools.ffmpeg, audio_path)
            logger.debug("audio duration: %.2f seconds", audio_duration)
        except StepError as e:
            logger.warning("Could not determine audio duration: %s. Using -shortest fallback.", e)
            # Continue with duration=None, will use -shortest in command

        # Output filename
        output_path = output_dir / f"{audio_path.stem}_with_subs.mp4"

        logger.info(
            "generating video from audio: %s (background: %s)",
            audio_path.name,
            self.background_image.name if self.background_image else f"solid {self.background_color}",
        )

        cmd = self._build_command(
            tools.ffmpeg, audio_path, subtitle_path, output_path, audio_duration
        )

        logger.debug("running: %s", " ".join(str(c) for c in cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

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
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
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

        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError) as e:
            # Re-raise as StepError so caller can handle fallback
            raise StepError(
                self.name,
                f"Could not determine audio duration: {e}",
            )

    def _build_command(
        self,
        ffmpeg: Path,
        audio: Path,
        subs: Path,
        output: Path,
        duration: float | None,
    ) -> list:
        """Build ffmpeg command for video generation."""
        cmd = [str(ffmpeg), "-y"]  # Overwrite output

        # Add audio input
        cmd.extend(["-i", str(audio)])

        # Add background (image or solid color)
        input_count = 1  # Track input number
        if self.background_image and self.background_image.exists():
            # Use image as background
            cmd.extend(["-loop", "1", "-i", str(self.background_image)])
            video_input = "[1:v]"
            input_count = 2
        else:
            # Use solid color background
            if duration is not None:
                # Generate video of the exact audio length
                cmd.extend([
                    "-f", "lavfi",
                    "-i", f"color=c={self.background_color}:s={self.video_size}:d={duration}",
                ])
            else:
                # Fallback: infinite stream, will use -shortest
                cmd.extend([
                    "-f", "lavfi",
                    "-i", f"color=c={self.background_color}:s={self.video_size}",
                ])
            video_input = "[1:v]"
            input_count = 2

        # Add subtitle file as input for soft subtitle stream
        cmd.extend(["-i", str(subs)])
        subtitle_input_idx = input_count  # Subtitle is input at this index (2 for color, 2 for image)

        # Build subtitle filter
        # Use absolute path and escape properly for ffmpeg
        subs_abs = subs.resolve()
        # Escape colons, backslashes, and single quotes for ffmpeg filter
        subs_escaped = str(subs_abs).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        sub_filter = f"subtitles='{subs_escaped}'"

        # Add optional styling
        if self.subtitle_style:
            style_parts = []
            if "fontsize" in self.subtitle_style:
                style_parts.append(f"FontSize={self.subtitle_style['fontsize']}")
            if "fontname" in self.subtitle_style:
                style_parts.append(f"FontName={self.subtitle_style['fontname']}")
            if "primarycolor" in self.subtitle_style:
                style_parts.append(
                    f"PrimaryColour={self.subtitle_style['primarycolor']}"
                )

            if style_parts:
                sub_filter += f":force_style='{','.join(style_parts)}'"

        # Build video filter
        if self.background_image:
            # Image background: scale image, then add subtitles
            vf = f"{video_input}scale={self.video_size},{sub_filter}"
        else:
            # Solid color: reference video stream, then add subtitles
            vf = f"{video_input}{sub_filter}"

        # Map streams explicitly
        # Video: from background input (1), filtered with burned-in subtitles
        # Audio: from audio input (0)
        # Subtitles: from subtitle input (2) - soft, toggleable subtitles
        # Result: Both burned-in subtitles (always visible) AND soft subtitle stream (toggleable)
        cmd.extend([
            "-map", "0:a",  # Map audio from input 0
            "-map", "1:v",  # Map video from input 1 (will be filtered with burned-in subs)
            f"-map", f"{subtitle_input_idx}:s",  # Map subtitles from subtitle input (soft subs)
            "-vf", vf,  # Apply video filter (burns in subtitles on video)
        ])
        
        # Only use -shortest if we couldn't determine duration
        if duration is None:
            cmd.append("-shortest")  # End when shortest input ends (audio)
        
        # Add encoding options
        cmd.extend([
            "-c:a", "aac",  # Encode audio as AAC for MP4
            "-b:a", "192k",  # Audio bitrate
            "-c:s", "mov_text",  # Subtitle codec for MP4 (soft subtitles - toggleable)
            "-metadata:s:s:0", "language=eng",  # Set subtitle language
            str(output),
        ])

        return cmd


def run(
    input_path: Path,
    output_dir: Path,
    background_image: str | Path | None = None,
    background_color: str = "black",
    video_size: str = "1920x1080",
    subtitle_style: dict | None = None,
) -> StepResult:
    """
    Run the generate_video step.

    Args:
        input_path: Path to audio file.
        output_dir: Directory for output (must contain transcript.srt).
        background_image: Optional path to background image file.
        background_color: Color name for solid background (default: "black").
        video_size: Video dimensions as "WxH" (default: "1920x1080").
        subtitle_style: Optional dict with fontsize, fontname, primarycolor.

    Returns:
        StepResult with execution details.
    """
    # Convert background_image string to Path if provided
    bg_image_path = None
    if background_image:
        bg_image_path = Path(background_image) if isinstance(background_image, str) else background_image

    step = GenerateVideoStep(
        background_image=bg_image_path,
        background_color=background_color,
        video_size=video_size,
        subtitle_style=subtitle_style or {},
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
